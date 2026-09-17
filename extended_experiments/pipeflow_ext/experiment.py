from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import build_features, build_inference_features, dataset_fingerprint, sample_dict
from .model import BranchCorrectionNet, ModelConfig, decode_prediction
from .physics import baseline_initializer, newton_refine, physical_residual, residual_scale


@dataclass
class TrainConfig:
    epochs: int = 300
    patience: int = 35
    batch_size: int = 512
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    device: str = "cpu"
    num_workers: int = 0


class Standardizer:
    def __init__(self, mean=None, std=None):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self.std = None if std is None else np.asarray(std, dtype=np.float64)

    def fit(self, x: np.ndarray) -> "Standardizer":
        axes = tuple(range(x.ndim - 1))
        self.mean = np.mean(x, axis=axes, keepdims=False)
        self.std = np.std(x, axis=axes, keepdims=False)
        self.std = np.where(self.std < 1e-10, 1.0, self.std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _arrays(data: Mapping[str, np.ndarray], input_mode: str,
            scalers: tuple[Standardizer, Standardizer, Standardizer] | None = None,
            target_mode: str = "correction", flow_parameterization: str = "logit"):
    branch, glob, target = build_features(data, input_mode, target_mode, flow_parameterization)
    if scalers is None:
        scalers = (Standardizer().fit(branch), Standardizer().fit(glob), Standardizer().fit(target))
    bs, gs, ts = scalers
    return bs.transform(branch), gs.transform(glob), ts.transform(target), scalers


def train_model(train_data: Mapping[str, np.ndarray], val_data: Mapping[str, np.ndarray],
                input_mode: str, seed: int, output: str | Path,
                train_config: TrainConfig, model_config: ModelConfig,
                target_mode: str = "correction",
                flow_parameterization: str = "logit",
                suite_id: str = "unregistered") -> Path:
    set_seed(seed)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tr_b, tr_g, tr_t, scalers = _arrays(
        train_data, input_mode, target_mode=target_mode,
        flow_parameterization=flow_parameterization)
    va_b, va_g, va_t, _ = _arrays(
        val_data, input_mode, scalers, target_mode, flow_parameterization)
    train_ds = TensorDataset(torch.from_numpy(tr_b), torch.from_numpy(tr_g), torch.from_numpy(tr_t))
    val_ds = TensorDataset(torch.from_numpy(va_b), torch.from_numpy(va_g), torch.from_numpy(va_t))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=train_config.batch_size, shuffle=True,
                              num_workers=train_config.num_workers, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=train_config.batch_size, shuffle=False,
                            num_workers=train_config.num_workers)
    device = torch.device(train_config.device)
    model = BranchCorrectionNet(tr_b.shape[-1], tr_g.shape[-1], model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate,
                                  weight_decay=train_config.weight_decay)
    loss_fn = torch.nn.SmoothL1Loss(beta=0.1)
    best, best_epoch, best_state, wait = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, train_config.epochs + 1):
        model.train()
        train_sum = 0.0
        for b, g, target in train_loader:
            b, g, target = b.to(device), g.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(b, g), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_sum += float(loss.item()) * len(b)
        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for b, g, target in val_loader:
                b, g, target = b.to(device), g.to(device), target.to(device)
                val_sum += float(loss_fn(model(b, g), target).item()) * len(b)
        tr_loss = train_sum / len(train_ds)
        va_loss = val_sum / len(val_ds)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})
        if va_loss < best - 1e-8:
            best, best_epoch = va_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= train_config.patience:
                break
    if best_state is None:
        raise RuntimeError("No valid checkpoint was produced")
    branch_scaler, global_scaler, target_scaler = scalers
    checkpoint = {
        "state_dict": best_state,
        "input_mode": input_mode,
        "target_mode": target_mode,
        "flow_parameterization": flow_parameterization,
        "suite_id": suite_id,
        "train_dataset_id": dataset_fingerprint(train_data),
        "val_dataset_id": dataset_fingerprint(val_data),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best,
        "model": model.metadata(),
        "train_config": asdict(train_config),
        "branch_scaler": branch_scaler.state(),
        "global_scaler": global_scaler.state(),
        "target_scaler": target_scaler.state(),
        "history": history,
        "format_version": 3,
    }
    torch.save(checkpoint, output)
    return output


def load_checkpoint(path: str | Path, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["model"]
    model = BranchCorrectionNet(meta["branch_dim"], meta["global_dim"],
                                ModelConfig(**meta["model_config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    scalers = tuple(Standardizer(**ckpt[name]) for name in
                    ["branch_scaler", "global_scaler", "target_scaler"])
    return ckpt, model, scalers


def predict(checkpoint: str | Path, data: Mapping[str, np.ndarray], device: str = "cpu",
            batch_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    ckpt, model, (bs, gs, ts) = load_checkpoint(checkpoint, device)
    branch, glob = build_inference_features(data, ckpt["input_mode"])
    branch, glob = bs.transform(branch), gs.transform(glob)
    q0 = np.asarray(data["q_baseline"], dtype=np.float32)
    x0 = np.asarray(data["x_baseline"], dtype=np.float32)
    qt = np.asarray(data["Q_total"], dtype=np.float32)
    mean = torch.tensor(ts.mean, dtype=torch.float32, device=device)
    std = torch.tensor(ts.std, dtype=torch.float32, device=device)
    target_mode = ckpt.get("target_mode", "correction")
    flow_parameterization = ckpt.get("flow_parameterization", "logit")
    q_out, x_out = [], []
    with torch.inference_mode():
        for start in range(0, len(qt), batch_size):
            sl = slice(start, start + batch_size)
            pred = model(torch.from_numpy(branch[sl]).to(device), torch.from_numpy(glob[sl]).to(device))
            q, x = decode_prediction(
                pred, torch.from_numpy(q0[sl]).to(device),
                torch.from_numpy(x0[sl]).to(device),
                torch.from_numpy(qt[sl]).to(device), mean, std,
                target_mode, flow_parameterization)
            q_out.append(q.cpu().numpy())
            x_out.append(x.cpu().numpy())
    return np.concatenate(q_out), np.concatenate(x_out)


def _direct_metrics(q: np.ndarray, x: np.ndarray, data: Mapping[str, np.ndarray]) -> dict:
    q_true = np.asarray(data["q_target"])
    x_true = np.asarray(data["x_target"])
    qt = np.asarray(data["Q_total"])[:, None]
    residuals = []
    for i in range(len(q)):
        p = sample_dict(data, i)
        q0, x0 = baseline_initializer(p)
        scale = residual_scale(p, q0, x0)
        residuals.append(np.max(np.abs(physical_residual(q[i], x[i], p) / scale)))
    return {
        "q_relative_rmse": float(np.sqrt(np.mean(((q - q_true) / qt) ** 2))),
        "x_rmse": float(np.sqrt(np.mean((x - x_true) ** 2))),
        "direct_residual_mean": float(np.mean(residuals)),
        "direct_residual_p90": float(np.percentile(residuals, 90)),
    }


def evaluate_initializer(q: np.ndarray, x: np.ndarray, data: Mapping[str, np.ndarray],
                         tol: float = 1e-10, max_iter: int = 30) -> dict:
    metrics = _direct_metrics(q, x, data)
    iterations, converged, final_residual = [], [], []
    q_refined, x_refined = [], []
    for i in range(len(q)):
        result = newton_refine(q[i], x[i], sample_dict(data, i), tol=tol, max_iter=max_iter)
        iterations.append(result.iterations)
        converged.append(result.converged)
        final_residual.append(result.residual_inf)
        q_refined.append(result.q)
        x_refined.append(result.x)
    q_refined, x_refined = np.asarray(q_refined), np.asarray(x_refined)
    qt = np.asarray(data["Q_total"])[:, None]
    metrics.update({
        "newton_iterations_mean": float(np.mean(iterations)),
        "newton_iterations_median": float(np.median(iterations)),
        "newton_iterations_p90": float(np.percentile(iterations, 90)),
        "convergence_ratio": float(np.mean(converged)),
        "final_residual_mean": float(np.mean(final_residual)),
        "refined_q_relative_rmse": float(np.sqrt(np.mean(((q_refined - data["q_target"]) / qt) ** 2))),
        "refined_x_rmse": float(np.sqrt(np.mean((x_refined - data["x_target"]) ** 2))),
    })
    return metrics


def evaluate_checkpoint(checkpoint: str | Path, data: Mapping[str, np.ndarray],
                        device: str = "cpu", tol: float = 1e-10,
                        max_iter: int = 30) -> list[dict]:
    q0, x0 = np.asarray(data["q_baseline"]), np.asarray(data["x_baseline"])
    qn, xn = predict(checkpoint, data, device=device)
    base = {"method": "baseline_plus_newton", **evaluate_initializer(q0, x0, data, tol, max_iter)}
    neural = {"method": "neural_plus_newton", **evaluate_initializer(qn, xn, data, tol, max_iter)}
    return [base, neural]


def _sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_runtime(checkpoint: str | Path, data: Mapping[str, np.ndarray], device: str = "cpu",
                      samples: int = 500, warmup: int = 2, repeats: int = 10,
                      tol: float = 1e-10, max_iter: int = 30,
                      inference_batch_size: int = 0) -> dict:
    n = min(samples, len(data["Q_total"]))
    subset = {k: (v[:n] if np.asarray(v).ndim > 0 and len(np.asarray(v).shape) > 0 and np.asarray(v).shape[0] == len(data["Q_total"]) else v)
              for k, v in data.items()}
    ckpt, model, (bs, gs, ts) = load_checkpoint(checkpoint, device)
    target_mean = torch.tensor(ts.mean, dtype=torch.float32, device=device)
    target_std = torch.tensor(ts.std, dtype=torch.float32, device=device)
    target_mode = ckpt.get("target_mode", "correction")
    flow_parameterization = ckpt.get("flow_parameterization", "logit")

    def predict_loaded(runtime_data):
        branch, glob = build_inference_features(runtime_data, ckpt["input_mode"])
        branch, glob = bs.transform(branch), gs.transform(glob)
        q0 = np.asarray(runtime_data["q_baseline"], dtype=np.float32)
        x0 = np.asarray(runtime_data["x_baseline"], dtype=np.float32)
        qt = np.asarray(runtime_data["Q_total"], dtype=np.float32)
        chunk = len(qt) if inference_batch_size <= 0 else inference_batch_size
        q_out, x_out = [], []
        with torch.inference_mode():
            for start in range(0, len(qt), chunk):
                sl = slice(start, start + chunk)
                pred = model(torch.from_numpy(branch[sl]).to(device),
                             torch.from_numpy(glob[sl]).to(device))
                q, x = decode_prediction(
                    pred, torch.from_numpy(q0[sl]).to(device),
                    torch.from_numpy(x0[sl]).to(device),
                    torch.from_numpy(qt[sl]).to(device), target_mean, target_std,
                    target_mode, flow_parameterization)
                q_out.append(q.cpu().numpy()); x_out.append(x.cpu().numpy())
        return np.concatenate(q_out), np.concatenate(x_out)

    def baseline_pipeline():
        for i in range(n):
            p = sample_dict(subset, i)
            q0, x0 = baseline_initializer(p)
            newton_refine(q0, x0, p, tol=tol, max_iter=max_iter)

    def neural_pipeline():
        pipeline_chunk = n if inference_batch_size <= 0 else inference_batch_size
        for start in range(0, n, pipeline_chunk):
            stop = min(start + pipeline_chunk, n)
            runtime_data = {
                key: (value[start:stop] if np.asarray(value).ndim > 0
                      and np.asarray(value).shape[0] == n else value)
                for key, value in subset.items()
            }
            baselines = [baseline_initializer(sample_dict(subset, i))
                         for i in range(start, stop)]
            runtime_data["q_baseline"] = np.asarray([value[0] for value in baselines])
            runtime_data["x_baseline"] = np.asarray([value[1] for value in baselines])
            qn, xn = predict_loaded(runtime_data)
            for local_i, global_i in enumerate(range(start, stop)):
                newton_refine(qn[local_i], xn[local_i], sample_dict(subset, global_i),
                              tol=tol, max_iter=max_iter)

    for _ in range(warmup):
        baseline_pipeline(); neural_pipeline()
    base_times, neural_times = [], []
    def timed(fn):
        _sync(device); start = time.perf_counter(); fn(); _sync(device)
        return time.perf_counter() - start

    for repeat in range(repeats):
        if repeat % 2 == 0:
            base_times.append(timed(baseline_pipeline))
            neural_times.append(timed(neural_pipeline))
        else:
            neural_times.append(timed(neural_pipeline))
            base_times.append(timed(baseline_pipeline))
    b = 1000.0 * np.asarray(base_times) / n
    h = 1000.0 * np.asarray(neural_times) / n
    return {
        "samples": n, "repeats": repeats, "device": device,
        "inference_batch_size": n if inference_batch_size <= 0 else min(inference_batch_size, n),
        "baseline_ms_per_sample_median": float(np.median(b)),
        "baseline_ms_per_sample_iqr": float(np.percentile(b, 75) - np.percentile(b, 25)),
        "neural_ms_per_sample_median": float(np.median(h)),
        "neural_ms_per_sample_iqr": float(np.percentile(h, 75) - np.percentile(h, 25)),
        "speedup_baseline_over_neural": float(np.median(b) / np.median(h)),
        "baseline_repeat_ms_per_sample": json.dumps(b.tolist()),
        "neural_repeat_ms_per_sample": json.dumps(h.tolist()),
    }


def save_rows(path: str | Path, rows: list[dict]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
