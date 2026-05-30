#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
measure_hybrid_inference_time_cpu_only.py

목적
----
이미 학습된 hybrid correction 모델(best_model.pt)을 불러와서
degree/model/run별 inference time을 완전히 CPU 기준으로 측정한다.

기존 GPU/CPU 혼합 측정 코드와의 차이
-----------------------------------
기존 코드:
- preprocessing: CPU
- neural forward: GPU 가능
- Newton refinement: CPU
- end-to-end: CPU + GPU 혼합

이 코드:
- preprocessing: CPU
- neural forward: CPU
- Newton refinement: CPU
- end-to-end: CPU only

즉, 논문에서 "CPU runtime" 또는 "CPU-only inference time"으로 보고할 수 있는 수치를 생성한다.

대상 구조 예시
--------------
/root/project/dataset/math_03_14/hybrid_params2_runs/
└── deg25/
    └── lstm_trial_030_lstm/
        └── run_042/
            └── train/
                └── best_model.pt

필요한 hybrid script
-------------------
예:
  /root/project/dataset/math_03_14/repeat_experiments_multidim_allinone_v2.py

해당 파일 안에 아래 함수/클래스가 있어야 한다.
- load_npz
- load_model_checkpoint
- build_inputs_and_baseline
- refine_batch

모델 forward는 다음 형태를 가정한다.
  pred, delta_norm, delta_real = model(seq, glob, z0, q_total, delta_scaler_t)

측정 지표
---------
1. preprocessing_cpu_ms_per_sample
   build_inputs_and_baseline + scaling 시간.
   즉 heuristic baseline z0 생성과 입력 scaling 포함.

2. forward_cpu_ms_per_sample
   CPU-only neural forward 시간.
   pred.cpu().numpy() 복사 없이 순수 forward만 측정.

3. forward_with_numpy_cpu_ms_per_sample
   CPU neural forward 후 numpy 변환까지 포함.
   CPU tensor이므로 GPU->CPU copy는 없음.

4. neural_direct_total_cpu_ms_per_sample
   preprocessing + forward_with_numpy.

5. heuristic_newton_cpu_ms_per_sample
   heuristic z0에서 Newton refinement CPU 시간.

6. neural_newton_cpu_ms_per_sample
   neural prediction에서 Newton refinement CPU 시간.

7. neural_end_to_end_cpu_ms_per_sample
   preprocessing + neural forward + neural Newton refinement.

출력
----
out_dir/
  cpu_inference_time_all_raw.csv
  cpu_inference_time_degree_model_summary_raw.csv
  cpu_inference_time_paper_table.csv
  cpu_inference_time_paper_table.md
  cpu_inference_time_paper_table.tex
  cpu_inference_time_best_by_degree.csv
  cpu_inference_time_best_by_degree.md
  cpu_inference_time_best_by_degree.tex
  cpu_inference_time_model_average.csv
  cpu_inference_time_model_average.md
  cpu_inference_time_model_average.tex

권장
----
논문용 CPU timing은 CPU contention을 피하기 위해 병렬 실행하지 말고 단일 프로세스로 측정하는 것을 권장한다.
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch


# =========================================================
# Utilities
# =========================================================
def load_module_from_path(path: str):
    path = str(path)
    spec = importlib.util.spec_from_file_location("hybrid_module_for_cpu_timing", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def safe_float(x, default=float("nan")):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        if math.isnan(v):
            return None
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj
    return obj


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, ensure_ascii=False, indent=2)


def save_csv(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"[WARN] no rows to save: {path}")
        return

    keys = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_sci(x, digits=3):
    x = safe_float(x)
    if not math.isfinite(x):
        return "-"
    return f"{x:.{digits}e}"


def fmt_ms(x, digits=5):
    x = safe_float(x)
    if not math.isfinite(x):
        return "-"
    return f"{x:.{digits}f}"


def fmt_ratio(x, digits=5):
    x = safe_float(x)
    if not math.isfinite(x):
        return "-"
    return f"{x:.{digits}f}"


def markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")

    return "\n".join(lines)


def latex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\%", "%TEMP_PERCENT%")
        .replace("\\", r"\textbackslash{}")
        .replace("%TEMP_PERCENT%", r"\%")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def latex_table(rows: List[Dict[str, Any]], columns: List[str], caption: str, label: str):
    colspec = "l" * len(columns)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + colspec + r"}")
    lines.append(r"\hline")
    lines.append(" & ".join(latex_escape(c) for c in columns) + r" \\")
    lines.append(r"\hline")

    for row in rows:
        vals = [latex_escape(row.get(c, "")) for c in columns]
        lines.append(" & ".join(vals) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# =========================================================
# Path parsing
# =========================================================
def parse_model_path(model_path: Path):
    """
    model_path 예:
    /.../hybrid_params2_runs/deg25/lstm_trial_030_lstm/run_042/train/best_model.pt

    반환:
    degree=25
    model=lstm
    trial_dir=lstm_trial_030_lstm
    run_dir=run_042
    seed=42
    """
    degree = None
    model = None
    trial_dir = None
    run_dir = None
    seed = None

    for part in model_path.parts:
        m = re.match(r"deg(\d+)$", part)
        if m:
            degree = int(m.group(1))

        r = re.match(r"run_(\d+)$", part)
        if r:
            run_dir = part
            seed = int(r.group(1))

    try:
        trial_dir = model_path.parent.parent.parent.name
        model = trial_dir.split("_")[0].lower()
    except Exception:
        pass

    return degree, model, trial_dir, run_dir, seed


def find_model_paths(root: Path, degrees: List[int], models: List[str]):
    all_paths = sorted(root.rglob("best_model.pt"))
    selected = []

    degrees_set = set(int(d) for d in degrees)
    models_set = set(m.lower() for m in models)

    for p in all_paths:
        deg, model, trial_dir, run_dir, seed = parse_model_path(p)

        if deg is None or model is None:
            continue
        if deg not in degrees_set:
            continue
        if model.lower() not in models_set:
            continue

        selected.append(p)

    return selected


def test_npz_for_degree(data_root: Path, degree: int):
    return (
        data_root
        / f"multi_colebrook_data_deg{degree}"
        / f"parallel2_colebrook_deg{degree}_test.npz"
    )


# =========================================================
# Scaling and CPU tensor helpers
# =========================================================
def scaler_transform(X: np.ndarray, scaler: Dict[str, Any]):
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["std"], dtype=np.float64)

    std = np.where(np.abs(std) < 1e-12, 1.0, std)

    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e12, neginf=-1e12)

    Xs = (X - mean) / std
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = np.clip(Xs, -1e6, 1e6)

    return Xs.astype(np.float32)


def make_cpu_batches(seq_x, glob_x, z0, q_total, batch_size: int):
    """
    CPU tensor batch 생성.
    어떤 tensor도 cuda로 올리지 않는다.
    """
    batches = []
    n = len(seq_x)

    for i in range(0, n, batch_size):
        s = torch.from_numpy(seq_x[i:i + batch_size].astype(np.float32))
        g = torch.from_numpy(glob_x[i:i + batch_size].astype(np.float32))
        z = torch.from_numpy(z0[i:i + batch_size].astype(np.float32))
        qt = torch.from_numpy(
            np.asarray(q_total[i:i + batch_size]).astype(np.float32).reshape(-1, 1)
        )

        batches.append((s, g, z, qt))

    return batches


@torch.no_grad()
def forward_cpu_only(model, batches, delta_scaler_t):
    last = None
    for s, g, z, qt in batches:
        pred, delta_norm, delta_real = model(s, g, z, qt, delta_scaler_t)
        last = pred
    return last


@torch.no_grad()
def forward_cpu_with_numpy(model, batches, delta_scaler_t):
    preds = []

    for s, g, z, qt in batches:
        pred, delta_norm, delta_real = model(s, g, z, qt, delta_scaler_t)
        preds.append(pred.detach().numpy())

    return np.concatenate(preds, axis=0)


# =========================================================
# Timing helpers
# =========================================================
def measure_forward_cpu_ms_per_sample(
    model,
    batches,
    delta_scaler_t,
    n_samples: int,
    repeats: int,
    warmup: int,
):
    """
    CPU-only neural forward time.
    numpy 변환 제외.
    """
    for _ in range(warmup):
        forward_cpu_only(model, batches, delta_scaler_t)

    times = []

    for _ in range(repeats):
        t0 = time.perf_counter()
        forward_cpu_only(model, batches, delta_scaler_t)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0 / n_samples)

    return {
        "forward_cpu_ms_per_sample_mean": float(np.mean(times)),
        "forward_cpu_ms_per_sample_std": float(np.std(times)),
        "forward_cpu_ms_per_sample_min": float(np.min(times)),
        "forward_cpu_ms_per_sample_max": float(np.max(times)),
    }


def measure_forward_with_numpy_cpu_ms_per_sample(
    model,
    batches,
    delta_scaler_t,
    n_samples: int,
    repeats: int,
    warmup: int,
):
    """
    CPU neural forward + numpy 변환 포함.
    CPU tensor이므로 GPU-to-CPU copy는 없음.
    """
    for _ in range(warmup):
        _ = forward_cpu_with_numpy(model, batches, delta_scaler_t)

    times = []
    pred_direct = None

    for _ in range(repeats):
        t0 = time.perf_counter()
        pred_direct = forward_cpu_with_numpy(model, batches, delta_scaler_t)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0 / n_samples)

    return {
        "forward_with_numpy_cpu_ms_per_sample_mean": float(np.mean(times)),
        "forward_with_numpy_cpu_ms_per_sample_std": float(np.std(times)),
        "forward_with_numpy_cpu_ms_per_sample_min": float(np.min(times)),
        "forward_with_numpy_cpu_ms_per_sample_max": float(np.max(times)),
        "pred_direct": pred_direct,
    }


def measure_cpu_function_ms_per_sample(fn, n_samples: int, repeats: int):
    times = []
    last = None

    for _ in range(repeats):
        t0 = time.perf_counter()
        last = fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0 / n_samples)

    return {
        "ms_per_sample_mean": float(np.mean(times)),
        "ms_per_sample_std": float(np.std(times)),
        "ms_per_sample_min": float(np.min(times)),
        "ms_per_sample_max": float(np.max(times)),
        "last": last,
    }


# =========================================================
# Metrics
# =========================================================
def vector_metrics(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)

    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return mae, rmse, r2


# =========================================================
# Core CPU-only measurement
# =========================================================
def measure_one_checkpoint_cpu_only(
    mod,
    model_path: Path,
    test_npz: Path,
    batch_size: int,
    repeats: int,
    warmup: int,
    include_newton: bool,
    newton_repeats: int,
    tol: float,
    max_newton_iter: int,
):
    degree, model_name, trial_dir, run_dir, seed = parse_model_path(model_path)

    print("\n" + "=" * 100)
    print(f"[CPU-ONLY MEASURE] degree={degree} model={model_name} seed={seed}")
    print(f"model_path = {model_path}")
    print(f"test_npz   = {test_npz}")

    if not test_npz.exists():
        raise FileNotFoundError(f"Missing test npz: {test_npz}")

    # -----------------------------------------------------
    # CPU settings
    # -----------------------------------------------------
    device = "cpu"

    # -----------------------------------------------------
    # Load data/model on CPU
    # -----------------------------------------------------
    data = mod.load_npz(str(test_npz))

    ckpt, model, seq_scaler, glob_scaler, delta_scaler = mod.load_model_checkpoint(
        str(model_path),
        device=device,
    )

    model.to("cpu")
    model.eval()

    hp = ckpt["hp"]
    use_log_features = bool(hp.get("use_log_features", False))

    # -----------------------------------------------------
    # Preprocessing CPU timing
    # build_inputs_and_baseline + scaler transform
    # -----------------------------------------------------
    def preprocess_once():
        seq_x, glob_x, y_true, z0, extra = mod.build_inputs_and_baseline(
            data,
            use_log_features=use_log_features,
        )

        seq_shape = seq_x.shape

        seq_x = scaler_transform(
            seq_x.reshape(-1, seq_x.shape[-1]),
            seq_scaler,
        ).reshape(seq_shape)

        glob_x = scaler_transform(glob_x, glob_scaler)

        return seq_x, glob_x, y_true, z0, extra

    prep_t0 = time.perf_counter()
    seq_x, glob_x, y_true, z0, extra = preprocess_once()
    prep_t1 = time.perf_counter()

    n_samples = len(seq_x)
    preprocessing_cpu_ms_per_sample = (prep_t1 - prep_t0) * 1000.0 / n_samples

    # -----------------------------------------------------
    # CPU tensors
    # -----------------------------------------------------
    delta_scaler_t = {
        "mean": torch.tensor(
            np.asarray(delta_scaler["mean"], dtype=np.float32),
            device="cpu",
        ),
        "std": torch.tensor(
            np.asarray(delta_scaler["std"], dtype=np.float32),
            device="cpu",
        ),
    }

    q_total = np.asarray(data["Q_total"])

    batches = make_cpu_batches(
        seq_x=seq_x,
        glob_x=glob_x,
        z0=z0,
        q_total=q_total,
        batch_size=batch_size,
    )

    # -----------------------------------------------------
    # CPU neural forward timing
    # -----------------------------------------------------
    forward_cpu_stats = measure_forward_cpu_ms_per_sample(
        model=model,
        batches=batches,
        delta_scaler_t=delta_scaler_t,
        n_samples=n_samples,
        repeats=repeats,
        warmup=warmup,
    )

    forward_numpy_stats = measure_forward_with_numpy_cpu_ms_per_sample(
        model=model,
        batches=batches,
        delta_scaler_t=delta_scaler_t,
        n_samples=n_samples,
        repeats=repeats,
        warmup=warmup,
    )

    pred_direct = forward_numpy_stats.pop("pred_direct")

    neural_direct_total_cpu_ms_per_sample = (
        preprocessing_cpu_ms_per_sample
        + forward_numpy_stats["forward_with_numpy_cpu_ms_per_sample_mean"]
    )

    # -----------------------------------------------------
    # CPU Newton timing
    # -----------------------------------------------------
    heuristic_newton_cpu_ms = float("nan")
    neural_newton_cpu_ms = float("nan")

    heuristic_newton_conv = float("nan")
    neural_newton_conv = float("nan")

    heuristic_newton_iter = float("nan")
    neural_newton_iter = float("nan")

    if include_newton:
        def run_heuristic_newton():
            return mod.refine_batch(
                z0.astype(np.float64),
                data,
                tol=tol,
                max_iter=max_newton_iter,
            )

        h_time = measure_cpu_function_ms_per_sample(
            run_heuristic_newton,
            n_samples=n_samples,
            repeats=newton_repeats,
        )
        href, hit, hconv = h_time["last"]
        heuristic_newton_cpu_ms = h_time["ms_per_sample_mean"]
        heuristic_newton_iter = float(np.mean(hit))
        heuristic_newton_conv = float(np.mean(hconv))

        def run_neural_newton():
            return mod.refine_batch(
                pred_direct.astype(np.float64),
                data,
                tol=tol,
                max_iter=max_newton_iter,
            )

        n_time = measure_cpu_function_ms_per_sample(
            run_neural_newton,
            n_samples=n_samples,
            repeats=newton_repeats,
        )
        nref, nit, nconv = n_time["last"]
        neural_newton_cpu_ms = n_time["ms_per_sample_mean"]
        neural_newton_iter = float(np.mean(nit))
        neural_newton_conv = float(np.mean(nconv))

    neural_end_to_end_cpu_ms_per_sample = (
        preprocessing_cpu_ms_per_sample
        + forward_numpy_stats["forward_with_numpy_cpu_ms_per_sample_mean"]
        + (neural_newton_cpu_ms if math.isfinite(neural_newton_cpu_ms) else 0.0)
    )

    # -----------------------------------------------------
    # Direct metrics for sanity check
    # -----------------------------------------------------
    direct_mae, direct_rmse, direct_r2 = vector_metrics(pred_direct, y_true)

    row = {
        "degree": degree,
        "model": model_name,
        "trial_dir": trial_dir,
        "run_dir": run_dir,
        "seed": seed,
        "n_samples": n_samples,
        "batch_size": batch_size,
        "device": "cpu",
        "model_path": str(model_path),
        "test_npz": str(test_npz),

        "best_epoch": ckpt.get("best_epoch", ""),
        "best_val_rmse": ckpt.get("best_val_rmse", ""),

        "direct_mae": direct_mae,
        "direct_rmse": direct_rmse,
        "direct_r2": direct_r2,

        "preprocessing_cpu_ms_per_sample": preprocessing_cpu_ms_per_sample,

        **forward_cpu_stats,
        **forward_numpy_stats,

        "neural_direct_total_cpu_ms_per_sample": neural_direct_total_cpu_ms_per_sample,

        "heuristic_newton_cpu_ms_per_sample": heuristic_newton_cpu_ms,
        "neural_newton_cpu_ms_per_sample": neural_newton_cpu_ms,
        "neural_end_to_end_cpu_ms_per_sample": neural_end_to_end_cpu_ms_per_sample,

        "heuristic_newton_iter_mean": heuristic_newton_iter,
        "neural_newton_iter_mean": neural_newton_iter,
        "heuristic_newton_converged_ratio": heuristic_newton_conv,
        "neural_newton_converged_ratio": neural_newton_conv,

        "forward_repeats": repeats,
        "forward_warmup": warmup,
        "newton_repeats": newton_repeats if include_newton else 0,
    }

    print("[CPU-ONLY RESULT]")
    print(json.dumps({
        "degree": row["degree"],
        "model": row["model"],
        "seed": row["seed"],
        "direct_rmse": row["direct_rmse"],
        "preprocessing_cpu_ms_per_sample": row["preprocessing_cpu_ms_per_sample"],
        "forward_cpu_ms_per_sample": row["forward_cpu_ms_per_sample_mean"],
        "forward_with_numpy_cpu_ms_per_sample": row["forward_with_numpy_cpu_ms_per_sample_mean"],
        "neural_direct_total_cpu_ms_per_sample": row["neural_direct_total_cpu_ms_per_sample"],
        "neural_newton_cpu_ms_per_sample": row["neural_newton_cpu_ms_per_sample"],
        "neural_end_to_end_cpu_ms_per_sample": row["neural_end_to_end_cpu_ms_per_sample"],
        "neural_iter": row["neural_newton_iter_mean"],
        "neural_conv": row["neural_newton_converged_ratio"],
    }, ensure_ascii=False, indent=2))

    return row


# =========================================================
# Aggregation
# =========================================================
def aggregate_mean_std(rows: List[Dict[str, Any]], group_keys: List[str], metrics: List[str]):
    grouped = {}

    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)

    out = []

    for key, sub in sorted(grouped.items()):
        r = {k: v for k, v in zip(group_keys, key)}
        r["n_runs"] = len(sub)

        for metric in metrics:
            vals = [safe_float(x.get(metric)) for x in sub]
            vals = [v for v in vals if math.isfinite(v)]

            if vals:
                r[f"{metric}_mean"] = float(np.mean(vals))
                r[f"{metric}_std"] = float(np.std(vals))
                r[f"{metric}_min"] = float(np.min(vals))
                r[f"{metric}_max"] = float(np.max(vals))
            else:
                r[f"{metric}_mean"] = float("nan")
                r[f"{metric}_std"] = float("nan")
                r[f"{metric}_min"] = float("nan")
                r[f"{metric}_max"] = float("nan")

        out.append(r)

    return out


def make_paper_rows(summary_rows):
    paper = []

    for r in summary_rows:
        paper.append({
            "Degree": r["degree"],
            "Model": str(r["model"]).upper(),
            "Direct RMSE": fmt_sci(r["direct_rmse_mean"], 3),
            "Preprocess CPU ms/sample": fmt_ms(r["preprocessing_cpu_ms_per_sample_mean"], 5),
            "Forward CPU ms/sample": fmt_ms(r["forward_cpu_ms_per_sample_mean_mean"], 5),
            "Forward+NumPy CPU ms/sample": fmt_ms(r["forward_with_numpy_cpu_ms_per_sample_mean_mean"], 5),
            "Direct Total CPU ms/sample": fmt_ms(r["neural_direct_total_cpu_ms_per_sample_mean"], 5),
            "Neural+Newton CPU ms/sample": fmt_ms(r["neural_end_to_end_cpu_ms_per_sample_mean"], 5),
            "Neural Iter.": fmt_ms(r["neural_newton_iter_mean_mean"], 3),
            "Neural Conv.": fmt_ratio(r["neural_newton_converged_ratio_mean"], 5),
        })

    return paper


def make_model_average_rows(summary_rows):
    def avg_of(rows, key):
        vals = [safe_float(r.get(key)) for r in rows]
        vals = [v for v in vals if math.isfinite(v)]
        if not vals:
            return float("nan")
        return float(np.mean(vals))

    models = sorted(set(str(r.get("model", "")).lower() for r in summary_rows if r.get("model")))

    paper = []

    for model in models:
        sub = [r for r in summary_rows if str(r.get("model", "")).lower() == model]

        paper.append({
            "Model": model.upper(),
            "N Degrees": len(sub),
            "Avg Direct RMSE": fmt_sci(avg_of(sub, "direct_rmse_mean"), 3),
            "Avg Preprocess CPU ms/sample": fmt_ms(avg_of(sub, "preprocessing_cpu_ms_per_sample_mean"), 5),
            "Avg Forward CPU ms/sample": fmt_ms(avg_of(sub, "forward_cpu_ms_per_sample_mean_mean"), 5),
            "Avg Forward+NumPy CPU ms/sample": fmt_ms(avg_of(sub, "forward_with_numpy_cpu_ms_per_sample_mean_mean"), 5),
            "Avg Direct Total CPU ms/sample": fmt_ms(avg_of(sub, "neural_direct_total_cpu_ms_per_sample_mean"), 5),
            "Avg Neural+Newton CPU ms/sample": fmt_ms(avg_of(sub, "neural_end_to_end_cpu_ms_per_sample_mean"), 5),
            "Avg Iter.": fmt_ms(avg_of(sub, "neural_newton_iter_mean_mean"), 3),
            "Avg Conv.": fmt_ratio(avg_of(sub, "neural_newton_converged_ratio_mean"), 5),
        })

    return paper


def choose_best_by_degree(summary_rows, criterion: str):
    lower_is_better = criterion not in [
        "neural_newton_converged_ratio_mean",
    ]

    best = {}

    for r in summary_rows:
        deg = r["degree"]
        val = safe_float(r.get(criterion))

        if not math.isfinite(val):
            continue

        if deg not in best:
            best[deg] = r
            continue

        old = safe_float(best[deg].get(criterion))

        if lower_is_better:
            if val < old:
                best[deg] = r
        else:
            if val > old:
                best[deg] = r

    return [best[d] for d in sorted(best.keys())]


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hybrid_script",
        type=str,
        required=True,
        help="1번 hybrid correction 코드 경로. 예: repeat_experiments_multidim_allinone_v2.py",
    )
    parser.add_argument(
        "--models_root",
        type=str,
        required=True,
        help="best_model.pt들이 들어있는 루트. 예: /root/project/dataset/math_03_14/hybrid_params2_runs",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/project/dataset/math_03_14",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--degrees",
        nargs="+",
        type=int,
        default=[10, 15, 20, 25, 30, 35],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mlp", "lstm", "gru", "transformer"],
    )

    parser.add_argument("--batch_size", type=int, default=4096)

    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)

    parser.add_argument(
        "--include_newton",
        action="store_true",
        help="Newton refinement CPU 시간까지 측정한다.",
    )
    parser.add_argument("--newton_repeats", type=int, default=1)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max_newton_iter", type=int, default=20)

    parser.add_argument(
        "--best_criterion",
        type=str,
        default="neural_direct_total_cpu_ms_per_sample_mean",
        help=(
            "degree별 best 선택 기준. 예: direct_rmse_mean, "
            "forward_cpu_ms_per_sample_mean_mean, "
            "neural_direct_total_cpu_ms_per_sample_mean"
        ),
    )

    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=1,
        help="CPU timing 재현성을 위해 기본 1 권장.",
    )

    args = parser.parse_args()

    # CPU-only 강제
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.cpu_threads)

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(max(1, min(args.cpu_threads, 4)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mod = load_module_from_path(args.hybrid_script)

    models_root = Path(args.models_root)
    data_root = Path(args.data_root)

    model_paths = find_model_paths(
        root=models_root,
        degrees=args.degrees,
        models=args.models,
    )

    if not model_paths:
        raise FileNotFoundError(f"No best_model.pt found under {models_root}")

    print(f"[INFO] CPU-only timing")
    print(f"[INFO] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    print(f"[INFO] torch.cuda.is_available()={torch.cuda.is_available()}")
    print(f"[INFO] torch num threads={torch.get_num_threads()}")
    print(f"[INFO] found {len(model_paths)} model checkpoints")
    for p in model_paths:
        print(" -", p)

    raw_rows = []

    for model_path in model_paths:
        degree, model_name, trial_dir, run_dir, seed = parse_model_path(model_path)
        test_npz = test_npz_for_degree(data_root, degree)

        row = measure_one_checkpoint_cpu_only(
            mod=mod,
            model_path=model_path,
            test_npz=test_npz,
            batch_size=args.batch_size,
            repeats=args.repeats,
            warmup=args.warmup,
            include_newton=args.include_newton,
            newton_repeats=args.newton_repeats,
            tol=args.tol,
            max_newton_iter=args.max_newton_iter,
        )
        raw_rows.append(row)

    # -----------------------------------------------------
    # Save raw
    # -----------------------------------------------------
    raw_rows = sorted(
        raw_rows,
        key=lambda r: (
            r["degree"],
            r["model"],
            r["seed"] if r["seed"] is not None else -1,
        ),
    )

    save_csv(out_dir / "cpu_inference_time_all_raw.csv", raw_rows)
    save_json(out_dir / "cpu_inference_time_all_raw.json", raw_rows)

    # -----------------------------------------------------
    # Aggregate by degree/model
    # -----------------------------------------------------
    metrics = [
        "direct_mae",
        "direct_rmse",
        "direct_r2",

        "preprocessing_cpu_ms_per_sample",

        "forward_cpu_ms_per_sample_mean",
        "forward_with_numpy_cpu_ms_per_sample_mean",
        "neural_direct_total_cpu_ms_per_sample",

        "heuristic_newton_cpu_ms_per_sample",
        "neural_newton_cpu_ms_per_sample",
        "neural_end_to_end_cpu_ms_per_sample",

        "heuristic_newton_iter_mean",
        "neural_newton_iter_mean",
        "heuristic_newton_converged_ratio",
        "neural_newton_converged_ratio",
    ]

    summary_rows = aggregate_mean_std(raw_rows, ["degree", "model"], metrics)
    save_csv(out_dir / "cpu_inference_time_degree_model_summary_raw.csv", summary_rows)

    # -----------------------------------------------------
    # Paper table: all degree/model rows
    # -----------------------------------------------------
    paper_rows = make_paper_rows(summary_rows)

    paper_columns = [
        "Degree",
        "Model",
        "Direct RMSE",
        "Preprocess CPU ms/sample",
        "Forward CPU ms/sample",
        "Forward+NumPy CPU ms/sample",
        "Direct Total CPU ms/sample",
        "Neural+Newton CPU ms/sample",
        "Neural Iter.",
        "Neural Conv.",
    ]

    save_csv(out_dir / "cpu_inference_time_paper_table.csv", paper_rows)
    (out_dir / "cpu_inference_time_paper_table.md").write_text(
        markdown_table(paper_rows, paper_columns),
        encoding="utf-8",
    )
    (out_dir / "cpu_inference_time_paper_table.tex").write_text(
        latex_table(
            paper_rows,
            paper_columns,
            caption=(
                "CPU-only inference-time comparison of the hybrid neural correction initializer "
                "across Taylor degrees and model backbones."
            ),
            label="tab:multidim_cpu_inference_time",
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Best by degree
    # -----------------------------------------------------
    best_degree_raw = choose_best_by_degree(summary_rows, args.best_criterion)
    best_degree_paper = make_paper_rows(best_degree_raw)

    save_csv(out_dir / "cpu_inference_time_best_by_degree_raw.csv", best_degree_raw)
    save_csv(out_dir / "cpu_inference_time_best_by_degree.csv", best_degree_paper)
    (out_dir / "cpu_inference_time_best_by_degree.md").write_text(
        markdown_table(best_degree_paper, paper_columns),
        encoding="utf-8",
    )
    (out_dir / "cpu_inference_time_best_by_degree.tex").write_text(
        latex_table(
            best_degree_paper,
            paper_columns,
            caption=(
                "Best CPU-only inference-time result for each Taylor degree selected by "
                + args.best_criterion.replace("_", r"\_")
                + "."
            ),
            label="tab:multidim_cpu_inference_time_best_by_degree",
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Model average
    # -----------------------------------------------------
    model_avg_paper = make_model_average_rows(summary_rows)

    model_columns = [
        "Model",
        "N Degrees",
        "Avg Direct RMSE",
        "Avg Preprocess CPU ms/sample",
        "Avg Forward CPU ms/sample",
        "Avg Forward+NumPy CPU ms/sample",
        "Avg Direct Total CPU ms/sample",
        "Avg Neural+Newton CPU ms/sample",
        "Avg Iter.",
        "Avg Conv.",
    ]

    save_csv(out_dir / "cpu_inference_time_model_average.csv", model_avg_paper)
    (out_dir / "cpu_inference_time_model_average.md").write_text(
        markdown_table(model_avg_paper, model_columns),
        encoding="utf-8",
    )
    (out_dir / "cpu_inference_time_model_average.tex").write_text(
        latex_table(
            model_avg_paper,
            model_columns,
            caption=(
                "Model-wise average CPU-only inference time of the hybrid neural correction initializer."
            ),
            label="tab:multidim_cpu_inference_time_model_average",
        ),
        encoding="utf-8",
    )

    print("\n[DONE]")
    print("Saved:")
    for name in [
        "cpu_inference_time_all_raw.csv",
        "cpu_inference_time_degree_model_summary_raw.csv",
        "cpu_inference_time_paper_table.csv",
        "cpu_inference_time_paper_table.md",
        "cpu_inference_time_paper_table.tex",
        "cpu_inference_time_best_by_degree.csv",
        "cpu_inference_time_best_by_degree.md",
        "cpu_inference_time_best_by_degree.tex",
        "cpu_inference_time_model_average.csv",
        "cpu_inference_time_model_average.md",
        "cpu_inference_time_model_average.tex",
    ]:
        print(" -", out_dir / name)


if __name__ == "__main__":
    main()