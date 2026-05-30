#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_colebrook_original_refine.py

Colebrook-like original equation:
    F(x; a, b) = x + 2*log10(a + b*x) = 0

이 스크립트는 refinement를 coeffs 기반 Taylor polynomial이 아니라,
원래 식 F(x;a,b)에 대해 수행한다.
즉 direct prediction 뒤에 "진짜 Newton refinement"를 붙인다.

입력 NPZ key:
- coeffs
- center
- a
- b
- root

모델 checkpoint:
- train_colebrook_root.py 로 만든 best_model.pt
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn


LN10 = math.log(10.0)


# =========================================================
# Original equation
# =========================================================
def F_np(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x + 2.0 * np.log10(a + b * x)


def dF_np(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 + 2.0 * b / ((a + b * x) * LN10)


def safe_domain_left_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return -a / b + eps


def project_to_domain_np(x: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    left = safe_domain_left_np(a, b, eps=eps)
    return np.maximum(x, left)


# =========================================================
# Metrics
# =========================================================
def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def summarize_predictions(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    root_ref: np.ndarray = None,
    newton_iters: np.ndarray = None,
    converged: np.ndarray = None,
) -> Dict[str, float]:
    inside = (a + b * y_pred) > 0
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(r2_score_np(y_true, y_pred))
    valid_ratio = float(np.mean(inside))

    if np.any(inside):
        residual = np.abs(F_np(y_pred[inside], a[inside], b[inside]))
        residual_mean = float(np.mean(residual))
        residual_median = float(np.median(residual))
        residual_p90 = float(np.percentile(residual, 90))
    else:
        residual_mean = float("inf")
        residual_median = float("inf")
        residual_p90 = float("inf")

    out = {
        "name": name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "valid_ratio": valid_ratio,
        "residual_mean": residual_mean,
        "residual_median": residual_median,
        "residual_p90": residual_p90,
    }

    if newton_iters is not None:
        out["newton_iter_mean"] = float(np.mean(newton_iters))
        out["newton_iter_median"] = float(np.median(newton_iters))
        out["newton_iter_p90"] = float(np.percentile(newton_iters, 90))
    if converged is not None:
        out["newton_converged_ratio"] = float(np.mean(converged))
    if root_ref is not None:
        out["max_abs_error"] = float(np.max(np.abs(root_ref - y_pred)))

    return out


# =========================================================
# Feature builder (must match training code)
# =========================================================
def build_features_from_checkpoint_args(
    coeffs: np.ndarray,
    center: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    ckpt_args: Dict,
) -> np.ndarray:
    use_log_ab = bool(ckpt_args.get("use_log_ab", False))
    include_center = bool(ckpt_args.get("include_center", True))
    include_ab = bool(ckpt_args.get("include_ab", True))

    feats = [coeffs.astype(np.float32)]

    if include_center:
        feats.append(center.reshape(-1, 1).astype(np.float32))

    if include_ab:
        if use_log_ab:
            feats.append(np.log(a).reshape(-1, 1).astype(np.float32))
            feats.append(np.log(b).reshape(-1, 1).astype(np.float32))
        else:
            feats.append(a.reshape(-1, 1).astype(np.float32))
            feats.append(b.reshape(-1, 1).astype(np.float32))

    return np.concatenate(feats, axis=1).astype(np.float32)


def apply_scaler(X: np.ndarray, scaler_dict: Dict) -> np.ndarray:
    mean = np.array(scaler_dict["mean"], dtype=np.float32)
    std = np.array(scaler_dict["std"], dtype=np.float32)
    return (X - mean) / std


# =========================================================
# Model
# =========================================================
class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_model_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)

    input_dim = ckpt["input_dim"]
    hidden_dims = ckpt["hidden_dims"]
    dropout = 0.0
    if "args" in ckpt and isinstance(ckpt["args"], dict):
        dropout = float(ckpt["args"].get("dropout", 0.0))

    model = MLPRegressor(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return ckpt, model


def predict_with_checkpoint(
    coeffs: np.ndarray,
    center: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    ckpt_path: str,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    ckpt, model = load_model_checkpoint(ckpt_path, device)

    X = build_features_from_checkpoint_args(coeffs, center, a, b, ckpt["args"])
    X = apply_scaler(X, ckpt["scaler"])

    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite features detected after scaling for checkpoint: {ckpt_path}")

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).to(device)
            yb = model(xb).squeeze(-1).cpu().numpy()
            preds.append(yb)

    pred = np.concatenate(preds, axis=0)
    if not np.isfinite(pred).all():
        raise ValueError(f"Non-finite predictions detected for checkpoint: {ckpt_path}")

    return pred


# =========================================================
# Baseline initial guess
# =========================================================
def heuristic_init_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x0 = -2.0 * np.log10(a)
    x0 = project_to_domain_np(x0, a, b, eps=1e-10)
    return x0


# =========================================================
# Newton refinement on ORIGINAL equation
# =========================================================
def newton_refine_original_vectorized(
    x0: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 20,
    damping: float = 1.0,
    deriv_eps: float = 1e-14,
    max_step: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = project_to_domain_np(x0.copy(), a, b, eps=1e-10)
    n = len(x)
    iters_used = np.zeros(n, dtype=np.int32)
    converged = np.zeros(n, dtype=bool)
    active = np.ones(n, dtype=bool)

    for k in range(1, max_iter + 1):
        if not np.any(active):
            break

        idx = np.where(active)[0]
        xa = x[idx]
        aa = a[idx]
        bb = b[idx]

        xa = project_to_domain_np(xa, aa, bb, eps=1e-10)

        fx = F_np(xa, aa, bb)
        dfx = dF_np(xa, aa, bb)

        small = np.abs(dfx) < deriv_eps
        dfx[small] = np.where(dfx[small] >= 0, deriv_eps, -deriv_eps)

        step = damping * fx / dfx
        step = np.clip(step, -max_step, max_step)
        x_new = xa - step
        x_new = project_to_domain_np(x_new, aa, bb, eps=1e-10)

        # residual이 악화되면 half-step 한 번 더 시도
        fx_new = np.abs(F_np(x_new, aa, bb))
        worse = fx_new > np.abs(fx)
        if np.any(worse):
            x_half = xa[worse] - 0.5 * step[worse]
            x_half = project_to_domain_np(x_half, aa[worse], bb[worse], eps=1e-10)
            fx_half = np.abs(F_np(x_half, aa[worse], bb[worse]))
            better_half = fx_half < fx_new[worse]
            x_new[worse] = np.where(better_half, x_half, x_new[worse])

        x[idx] = x_new
        iters_used[idx] = k

        fx_now = np.abs(F_np(x_new, aa, bb))
        done = fx_now <= tol
        converged[idx[done]] = True
        active[idx[done]] = False

    return x, iters_used, converged


# =========================================================
# I/O
# =========================================================
def parse_model_pairs(items: List[str]) -> List[Tuple[str, str]]:
    out = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--model must be NAME=PATH format, got: {item}")
        name, path = item.split("=", 1)
        out.append((name, path))
    return out


def save_csv(path: Path, rows: List[Dict]):
    if not rows:
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            full_row = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(full_row)


def save_report(path: Path, rows: List[Dict], args: argparse.Namespace):
    lines = []
    lines.append("=== Evaluation Report (ORIGINAL equation refinement) ===")
    lines.append("")
    lines.append("[Args]")
    for k, v in vars(args).items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("[Summary]")
    for row in rows:
        lines.append("-" * 80)
        for k, v in row.items():
            lines.append(f"{k}: {v}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_npz", type=str, required=True)
    parser.add_argument("--model", action="append", default=[], help="NAME=PATH to best_model.pt ; repeatable")
    parser.add_argument("--max_newton_iter", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(args.test_npz)
    required_keys = ["coeffs", "center", "a", "b", "root"]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"'{k}' not found in test npz. keys={list(data.keys())}")

    coeffs = data["coeffs"].astype(np.float32)
    center = data["center"].astype(np.float32)
    a = data["a"].astype(np.float64)
    b = data["b"].astype(np.float64)
    root_true = data["root"].astype(np.float64)

    model_pairs = parse_model_pairs(args.model)

    summary_rows = []
    pred_bank = {
        "root_true": root_true,
        "a": a,
        "b": b,
        "center": center,
    }

    # zero baseline
    zero_pred = np.zeros_like(root_true)
    zero_pred = project_to_domain_np(zero_pred, a, b, eps=1e-10)
    summary_rows.append(summarize_predictions("zero_init_direct", root_true, zero_pred, a, b, root_true))
    pred_bank["zero_init_direct"] = zero_pred

    zero_newton, zero_iters, zero_conv = newton_refine_original_vectorized(
        zero_pred, a, b, tol=args.tol, max_iter=args.max_newton_iter, damping=args.damping
    )
    summary_rows.append(summarize_predictions(
        "zero_init_plus_newton", root_true, zero_newton, a, b, root_true, zero_iters, zero_conv
    ))
    pred_bank["zero_init_plus_newton"] = zero_newton
    pred_bank["zero_init_plus_newton_iters"] = zero_iters
    pred_bank["zero_init_plus_newton_converged"] = zero_conv.astype(np.int32)

    # heuristic baseline
    heur_pred = heuristic_init_np(a, b)
    summary_rows.append(summarize_predictions("heuristic_direct", root_true, heur_pred, a, b, root_true))
    pred_bank["heuristic_direct"] = heur_pred

    heur_newton, heur_iters, heur_conv = newton_refine_original_vectorized(
        heur_pred, a, b, tol=args.tol, max_iter=args.max_newton_iter, damping=args.damping
    )
    summary_rows.append(summarize_predictions(
        "heuristic_plus_newton", root_true, heur_newton, a, b, root_true, heur_iters, heur_conv
    ))
    pred_bank["heuristic_plus_newton"] = heur_newton
    pred_bank["heuristic_plus_newton_iters"] = heur_iters
    pred_bank["heuristic_plus_newton_converged"] = heur_conv.astype(np.int32)

    # model(s)
    for model_name, model_path in model_pairs:
        pred = predict_with_checkpoint(
            coeffs=coeffs,
            center=center,
            a=a,
            b=b,
            ckpt_path=model_path,
            device=device,
            batch_size=args.batch_size,
        ).astype(np.float64)

        summary_rows.append(summarize_predictions(
            f"{model_name}_direct", root_true, pred, a, b, root_true
        ))
        pred_bank[f"{model_name}_direct"] = pred

        pred_newton, pred_iters, pred_conv = newton_refine_original_vectorized(
            pred, a, b, tol=args.tol, max_iter=args.max_newton_iter, damping=args.damping
        )
        summary_rows.append(summarize_predictions(
            f"{model_name}_plus_newton", root_true, pred_newton, a, b, root_true, pred_iters, pred_conv
        ))
        pred_bank[f"{model_name}_plus_newton"] = pred_newton
        pred_bank[f"{model_name}_plus_newton_iters"] = pred_iters
        pred_bank[f"{model_name}_plus_newton_converged"] = pred_conv.astype(np.int32)

    save_csv(out_dir / "summary_metrics.csv", summary_rows)
    np.savez_compressed(out_dir / "per_sample_predictions.npz", **pred_bank)
    save_report(out_dir / "report.txt", summary_rows, args)

    print(f"[DONE] Saved to: {out_dir.resolve()}")
    print("  - summary_metrics.csv")
    print("  - per_sample_predictions.npz")
    print("  - report.txt")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_colebrook_original_refine.py

Colebrook-like original equation:
    F(x; a, b) = x + 2*log10(a + b*x) = 0

이 스크립트는 refinement를 coeffs 기반 Taylor polynomial이 아니라,
원래 식 F(x;a,b)에 대해 수행한다.
즉 direct prediction 뒤에 "진짜 Newton refinement"를 붙인다.

입력 NPZ key:
- coeffs
- center
- a
- b
- root

모델 checkpoint:
- train_colebrook_root.py 로 만든 best_model.pt
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn


LN10 = math.log(10.0)


# =========================================================
# Original equation
# =========================================================
def F_np(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x + 2.0 * np.log10(a + b * x)


def dF_np(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 + 2.0 * b / ((a + b * x) * LN10)


def safe_domain_left_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return -a / b + eps


def project_to_domain_np(x: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    left = safe_domain_left_np(a, b, eps=eps)
    return np.maximum(x, left)


# =========================================================
# Metrics
# =========================================================
def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def summarize_predictions(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    root_ref: np.ndarray = None,
    newton_iters: np.ndarray = None,
    converged: np.ndarray = None,
) -> Dict[str, float]:
    inside = (a + b * y_pred) > 0
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(r2_score_np(y_true, y_pred))
    valid_ratio = float(np.mean(inside))

    if np.any(inside):
        residual = np.abs(F_np(y_pred[inside], a[inside], b[inside]))
        residual_mean = float(np.mean(residual))
        residual_median = float(np.median(residual))
        residual_p90 = float(np.percentile(residual, 90))
    else:
        residual_mean = float("inf")
        residual_median = float("inf")
        residual_p90 = float("inf")

    out = {
        "name": name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "valid_ratio": valid_ratio,
        "residual_mean": residual_mean,
        "residual_median": residual_median,
        "residual_p90": residual_p90,
    }

    if newton_iters is not None:
        out["newton_iter_mean"] = float(np.mean(newton_iters))
        out["newton_iter_median"] = float(np.median(newton_iters))
        out["newton_iter_p90"] = float(np.percentile(newton_iters, 90))
    if converged is not None:
        out["newton_converged_ratio"] = float(np.mean(converged))
    if root_ref is not None:
        out["max_abs_error"] = float(np.max(np.abs(root_ref - y_pred)))

    return out


# =========================================================
# Feature builder (must match training code)
# =========================================================
def build_features_from_checkpoint_args(
    coeffs: np.ndarray,
    center: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    ckpt_args: Dict,
) -> np.ndarray:
    use_log_ab = bool(ckpt_args.get("use_log_ab", False))
    include_center = bool(ckpt_args.get("include_center", True))
    include_ab = bool(ckpt_args.get("include_ab", True))

    feats = [coeffs.astype(np.float32)]

    if include_center:
        feats.append(center.reshape(-1, 1).astype(np.float32))

    if include_ab:
        if use_log_ab:
            feats.append(np.log(a).reshape(-1, 1).astype(np.float32))
            feats.append(np.log(b).reshape(-1, 1).astype(np.float32))
        else:
            feats.append(a.reshape(-1, 1).astype(np.float32))
            feats.append(b.reshape(-1, 1).astype(np.float32))

    return np.concatenate(feats, axis=1).astype(np.float32)


def apply_scaler(X: np.ndarray, scaler_dict: Dict) -> np.ndarray:
    mean = np.array(scaler_dict["mean"], dtype=np.float32)
    std = np.array(scaler_dict["std"], dtype=np.float32)
    return (X - mean) / std


# =========================================================
# Model
# =========================================================
class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_model_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)

    input_dim = ckpt["input_dim"]
    hidden_dims = ckpt["hidden_dims"]
    dropout = 0.0
    if "args" in ckpt and isinstance(ckpt["args"], dict):
        dropout = float(ckpt["args"].get("dropout", 0.0))

    model = MLPRegressor(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return ckpt, model


def predict_with_checkpoint(
    coeffs: np.ndarray,
    center: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    ckpt_path: str,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    ckpt, model = load_model_checkpoint(ckpt_path, device)

    X = build_features_from_checkpoint_args(coeffs, center, a, b, ckpt["args"])
    X = apply_scaler(X, ckpt["scaler"])

    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite features detected after scaling for checkpoint: {ckpt_path}")

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).to(device)
            yb = model(xb).squeeze(-1).cpu().numpy()
            preds.append(yb)

    pred = np.concatenate(preds, axis=0)
    if not np.isfinite(pred).all():
        raise ValueError(f"Non-finite predictions detected for checkpoint: {ckpt_path}")

    return pred


# =========================================================
# Baseline initial guess
# =========================================================
def heuristic_init_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x0 = -2.0 * np.log10(a)
    x0 = project_to_domain_np(x0, a, b, eps=1e-10)
    return x0


# =========================================================
# Newton refinement on ORIGINAL equation
# =========================================================
def newton_refine_original_vectorized(
    x0: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 20,
    damping: float = 1.0,
    deriv_eps: float = 1e-14,
    max_step: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = project_to_domain_np(x0.copy(), a, b, eps=1e-10)
    n = len(x)
    iters_used = np.zeros(n, dtype=np.int32)
    converged = np.zeros(n, dtype=bool)
    active = np.ones(n, dtype=bool)

    for k in range(1, max_iter + 1):
        if not np.any(active):
            break

        idx = np.where(active)[0]
        xa = x[idx]
        aa = a[idx]
        bb = b[idx]

        xa = project_to_domain_np(xa, aa, bb, eps=1e-10)

        fx = F_np(xa, aa, bb)
        dfx = dF_np(xa, aa, bb)

        small = np.abs(dfx) < deriv_eps
        dfx[small] = np.where(dfx[small] >= 0, deriv_eps, -deriv_eps)

        step = damping * fx / dfx
        step = np.clip(step, -max_step, max_step)
        x_new = xa - step
        x_new = project_to_domain_np(x_new, aa, bb, eps=1e-10)

        # residual이 악화되면 half-step 한 번 더 시도
        fx_new = np.abs(F_np(x_new, aa, bb))
        worse = fx_new > np.abs(fx)
        if np.any(worse):
            x_half = xa[worse] - 0.5 * step[worse]
            x_half = project_to_domain_np(x_half, aa[worse], bb[worse], eps=1e-10)
            fx_half = np.abs(F_np(x_half, aa[worse], bb[worse]))
            better_half = fx_half < fx_new[worse]
            x_new[worse] = np.where(better_half, x_half, x_new[worse])

        x[idx] = x_new
        iters_used[idx] = k

        fx_now = np.abs(F_np(x_new, aa, bb))
        done = fx_now <= tol
        converged[idx[done]] = True
        active[idx[done]] = False

    return x, iters_used, converged


# =========================================================
# I/O
# =========================================================
def parse_model_pairs(items: List[str]) -> List[Tuple[str, str]]:
    out = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--model must be NAME=PATH format, got: {item}")
        name, path = item.split("=", 1)
        out.append((name, path))
    return out


def save_csv(path: Path, rows: List[Dict]):
    if not rows:
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            full_row = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(full_row)


def save_report(path: Path, rows: List[Dict], args: argparse.Namespace):
    lines = []
    lines.append("=== Evaluation Report (ORIGINAL equation refinement) ===")
    lines.append("")
    lines.append("[Args]")
    for k, v in vars(args).items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("[Summary]")
    for row in rows:
        lines.append("-" * 80)
        for k, v in row.items():
            lines.append(f"{k}: {v}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_npz", type=str, required=True)
    parser.add_argument("--model", action="append", default=[], help="NAME=PATH to best_model.pt ; repeatable")
    parser.add_argument("--max_newton_iter", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(args.test_npz)
    required_keys = ["coeffs", "center", "a", "b", "root"]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"'{k}' not found in test npz. keys={list(data.keys())}")

    coeffs = data["coeffs"].astype(np.float32)
    center = data["center"].astype(np.float32)
    a = data["a"].astype(np.float64)
    b = data["b"].astype(np.float64)
    root_true = data["root"].astype(np.float64)

    model_pairs = parse_model_pairs(args.model)

    summary_rows = []
    pred_bank = {
        "root_true": root_true,
        "a": a,
        "b": b,
        "center": center,
    }

    # zero baseline
    zero_pred = np.zeros_like(root_true)
    zero_pred = project_to_domain_np(zero_pred, a, b, eps=1e-10)
    summary_rows.append(summarize_predictions("zero_init_direct", root_true, zero_pred, a, b, root_true))
    pred_bank["zero_init_direct"] = zero_pred

    zero_newton, zero_iters, zero_conv = newton_refine_original_vectorized(
        zero_pred, a, b, tol=args.tol, max_iter=args.max_newton_iter, damping=args.damping
    )
    summary_rows.append(summarize_predictions(
        "zero_init_plus_newton", root_true, zero_newton, a, b, root_true, zero_iters, zero_conv
    ))
    pred_bank["zero_init_plus_newton"] = zero_newton
    pred_bank["zero_init_plus_newton_iters"] = zero_iters
    pred_bank["zero_init_plus_newton_converged"] = zero_conv.astype(np.int32)

    # heuristic baseline
    heur_pred = heuristic_init_np(a, b)
    summary_rows.append(summarize_predictions("heuristic_direct", root_true, heur_pred, a, b, root_true))
    pred_bank["heuristic_direct"] = heur_pred

    heur_newton, heur_iters, heur_conv = newton_refine_original_vectorized(
        heur_pred, a, b, tol=args.tol, max_iter=args.max_newton_iter, damping=args.damping
    )
    summary_rows.append(summarize_predictions(
        "heuristic_plus_newton", root_true, heur_newton, a, b, root_true, heur_iters, heur_conv
    ))
    pred_bank["heuristic_plus_newton"] = heur_newton
    pred_bank["heuristic_plus_newton_iters"] = heur_iters
    pred_bank["heuristic_plus_newton_converged"] = heur_conv.astype(np.int32)

    # model(s)
    for model_name, model_path in model_pairs:
        pred = predict_with_checkpoint(
            coeffs=coeffs,
            center=center,
            a=a,
            b=b,
            ckpt_path=model_path,
            device=device,
            batch_size=args.batch_size,
        ).astype(np.float64)

        summary_rows.append(summarize_predictions(
            f"{model_name}_direct", root_true, pred, a, b, root_true
        ))
        pred_bank[f"{model_name}_direct"] = pred

        pred_newton, pred_iters, pred_conv = newton_refine_original_vectorized(
            pred, a, b, tol=args.tol, max_iter=args.max_newton_iter, damping=args.damping
        )
        summary_rows.append(summarize_predictions(
            f"{model_name}_plus_newton", root_true, pred_newton, a, b, root_true, pred_iters, pred_conv
        ))
        pred_bank[f"{model_name}_plus_newton"] = pred_newton
        pred_bank[f"{model_name}_plus_newton_iters"] = pred_iters
        pred_bank[f"{model_name}_plus_newton_converged"] = pred_conv.astype(np.int32)

    save_csv(out_dir / "summary_metrics.csv", summary_rows)
    np.savez_compressed(out_dir / "per_sample_predictions.npz", **pred_bank)
    save_report(out_dir / "report.txt", summary_rows, args)

    print(f"[DONE] Saved to: {out_dir.resolve()}")
    print("  - summary_metrics.csv")
    print("  - per_sample_predictions.npz")
    print("  - report.txt")


if __name__ == "__main__":
    main()
