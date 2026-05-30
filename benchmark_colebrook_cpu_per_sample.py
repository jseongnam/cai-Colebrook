#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_colebrook_cpu_per_sample.py

CPU 환경에서 문제 1개당 걸리는 실제 wall-clock time을 분석한다.

비교 대상
---------
1) zero_init_direct
2) zero_init_plus_newton
3) heuristic_direct
4) heuristic_plus_newton
5) neural_direct
6) neural_plus_newton

입력
----
- test npz: colebrook_like_dataset.py 로 생성한 npz
  required keys: coeffs, center, a, b, root
- model checkpoint: train_colebrook_root.py 로 생성한 best_model.pt
예시
----
python benchmark_colebrook_cpu_per_sample.py \
  --test_npz ./colebrook_data/colebrook_deg25_test.npz \
  --model ./runs/colebrook_root_deg25/best_model.pt \
  --num_samples 2000 \
  --repeats 30 \
  --warmup 5 \
  --max_newton_iter 20 \
  --tol 1e-12 \
  --out_dir ./cpu_benchmarks/colebrook
"""

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

LN10 = math.log(10.0)


# =========================================================
# Original equation
# =========================================================
def F_scalar(x: float, a: float, b: float) -> float:
    return x + 2.0 * math.log10(a + b * x)


def dF_scalar(x: float, a: float, b: float) -> float:
    return 1.0 + 2.0 * b / ((a + b * x) * LN10)


def project_to_domain_scalar(x: float, a: float, b: float, eps: float = 1e-10) -> float:
    left = -a / b + eps
    return x if x >= left else left


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


def load_model_checkpoint_cpu(ckpt_path: str):
    device = torch.device("cpu")
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


def build_feature_one(
    coeffs: np.ndarray,
    center: float,
    a: float,
    b: float,
    ckpt_args: Dict,
    scaler: Dict,
) -> np.ndarray:
    use_log_ab = bool(ckpt_args.get("use_log_ab", False))
    include_center = bool(ckpt_args.get("include_center", True))
    include_ab = bool(ckpt_args.get("include_ab", True))

    feats = [coeffs.astype(np.float32)]

    if include_center:
        feats.append(np.array([center], dtype=np.float32))

    if include_ab:
        if use_log_ab:
            feats.append(np.array([math.log(a)], dtype=np.float32))
            feats.append(np.array([math.log(b)], dtype=np.float32))
        else:
            feats.append(np.array([a], dtype=np.float32))
            feats.append(np.array([b], dtype=np.float32))

    x = np.concatenate(feats, axis=0).astype(np.float32)

    mean = np.array(scaler["mean"], dtype=np.float32).reshape(-1)
    std = np.array(scaler["std"], dtype=np.float32).reshape(-1)
    x = (x - mean) / std
    return x


def neural_predict_one(model, feature_vec: np.ndarray) -> float:
    with torch.no_grad():
        x = torch.from_numpy(feature_vec).float().unsqueeze(0)
        y = model(x).squeeze(0).squeeze(0).item()
    return float(y)


# =========================================================
# Initial guess baselines
# =========================================================
def zero_init(a: float, b: float) -> float:
    return project_to_domain_scalar(0.0, a, b)


def heuristic_init(a: float, b: float) -> float:
    x0 = -2.0 * math.log10(a)
    return project_to_domain_scalar(x0, a, b)


# =========================================================
# Newton refinement
# =========================================================
def newton_refine_original(
    x0: float,
    a: float,
    b: float,
    tol: float = 1e-12,
    max_iter: int = 20,
    damping: float = 1.0,
    deriv_eps: float = 1e-14,
    max_step: float = 5.0,
) -> Tuple[float, int, bool]:
    x = project_to_domain_scalar(x0, a, b)

    for k in range(1, max_iter + 1):
        f = F_scalar(x, a, b)
        if abs(f) <= tol:
            return x, k - 1, True

        df = dF_scalar(x, a, b)
        if abs(df) < deriv_eps:
            df = deriv_eps if df >= 0 else -deriv_eps

        step = damping * f / df
        step = max(-max_step, min(max_step, step))

        x_new = project_to_domain_scalar(x - step, a, b)

        # half-step safeguard
        f_new = abs(F_scalar(x_new, a, b))
        if f_new > abs(f):
            x_half = project_to_domain_scalar(x - 0.5 * step, a, b)
            if abs(F_scalar(x_half, a, b)) < f_new:
                x_new = x_half

        if abs(x_new - x) <= 1e-12:
            x = x_new
            break

        x = x_new

    return x, max_iter, abs(F_scalar(x, a, b)) <= tol


# =========================================================
# Timing helpers
# =========================================================
def time_call_ns(fn, *args, **kwargs):
    t0 = time.perf_counter_ns()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter_ns()
    return out, (t1 - t0)


def ns_to_us(arr) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64) / 1_000.0


# =========================================================
# Benchmark per sample
# =========================================================
def benchmark_one_sample(
    coeffs: np.ndarray,
    center: float,
    a: float,
    b: float,
    root_true: float,
    model,
    ckpt_args: Dict,
    scaler: Dict,
    repeats: int,
    warmup: int,
    max_newton_iter: int,
    tol: float,
) -> Dict[str, Dict]:
    feat = build_feature_one(coeffs, center, a, b, ckpt_args, scaler)

    # warmup
    for _ in range(warmup):
        _ = zero_init(a, b)
        _ = heuristic_init(a, b)
        _ = neural_predict_one(model, feat)
        _ = newton_refine_original(zero_init(a, b), a, b, tol=tol, max_iter=max_newton_iter)
        _ = newton_refine_original(heuristic_init(a, b), a, b, tol=tol, max_iter=max_newton_iter)
        _ = newton_refine_original(neural_predict_one(model, feat), a, b, tol=tol, max_iter=max_newton_iter)

    out = {}

    # zero direct
    times, preds = [], []
    for _ in range(repeats):
        pred, t_ns = time_call_ns(zero_init, a, b)
        times.append(t_ns)
        preds.append(pred)
    pred0 = float(preds[-1])
    out["zero_init_direct"] = {
        "pred": pred0,
        "times_ns": np.array(times, dtype=np.int64),
        "mae": abs(pred0 - root_true),
        "residual": abs(F_scalar(pred0, a, b)),
    }

    # zero + newton
    times_total, times_refine, iters, preds_ref, succ = [], [], [], [], []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        x0 = zero_init(a, b)
        (x_ref, k, ok), t_ref = time_call_ns(
            newton_refine_original, x0, a, b, tol=tol, max_iter=max_newton_iter
        )
        t1 = time.perf_counter_ns()
        times_total.append(t1 - t0)
        times_refine.append(t_ref)
        iters.append(k)
        preds_ref.append(x_ref)
        succ.append(int(ok))
    pred0n = float(preds_ref[-1])
    out["zero_init_plus_newton"] = {
        "pred": pred0n,
        "times_ns": np.array(times_total, dtype=np.int64),
        "refine_times_ns": np.array(times_refine, dtype=np.int64),
        "iters": np.array(iters, dtype=np.int64),
        "success": np.array(succ, dtype=np.int64),
        "mae": abs(pred0n - root_true),
        "residual": abs(F_scalar(pred0n, a, b)),
    }

    # heuristic direct
    times, preds = [], []
    for _ in range(repeats):
        pred, t_ns = time_call_ns(heuristic_init, a, b)
        times.append(t_ns)
        preds.append(pred)
    predh = float(preds[-1])
    out["heuristic_direct"] = {
        "pred": predh,
        "times_ns": np.array(times, dtype=np.int64),
        "mae": abs(predh - root_true),
        "residual": abs(F_scalar(predh, a, b)),
    }

    # heuristic + newton
    times_total, times_refine, iters, preds_ref, succ = [], [], [], [], []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        x0 = heuristic_init(a, b)
        (x_ref, k, ok), t_ref = time_call_ns(
            newton_refine_original, x0, a, b, tol=tol, max_iter=max_newton_iter
        )
        t1 = time.perf_counter_ns()
        times_total.append(t1 - t0)
        times_refine.append(t_ref)
        iters.append(k)
        preds_ref.append(x_ref)
        succ.append(int(ok))
    predhn = float(preds_ref[-1])
    out["heuristic_plus_newton"] = {
        "pred": predhn,
        "times_ns": np.array(times_total, dtype=np.int64),
        "refine_times_ns": np.array(times_refine, dtype=np.int64),
        "iters": np.array(iters, dtype=np.int64),
        "success": np.array(succ, dtype=np.int64),
        "mae": abs(predhn - root_true),
        "residual": abs(F_scalar(predhn, a, b)),
    }

    # neural direct
    times, preds = [], []
    for _ in range(repeats):
        pred, t_ns = time_call_ns(neural_predict_one, model, feat)
        times.append(t_ns)
        preds.append(pred)
    predm = float(preds[-1])
    out["neural_direct"] = {
        "pred": predm,
        "times_ns": np.array(times, dtype=np.int64),
        "mae": abs(predm - root_true),
        "residual": abs(F_scalar(predm, a, b)),
    }

    # neural + newton
    times_total, times_pred, times_refine, iters, preds_ref, succ = [], [], [], [], [], []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        pred, t_pred = time_call_ns(neural_predict_one, model, feat)
        (x_ref, k, ok), t_ref = time_call_ns(
            newton_refine_original, pred, a, b, tol=tol, max_iter=max_newton_iter
        )
        t1 = time.perf_counter_ns()
        times_total.append(t1 - t0)
        times_pred.append(t_pred)
        times_refine.append(t_ref)
        iters.append(k)
        preds_ref.append(x_ref)
        succ.append(int(ok))
    predmn = float(preds_ref[-1])
    out["neural_plus_newton"] = {
        "pred": predmn,
        "times_ns": np.array(times_total, dtype=np.int64),
        "pred_times_ns": np.array(times_pred, dtype=np.int64),
        "refine_times_ns": np.array(times_refine, dtype=np.int64),
        "iters": np.array(iters, dtype=np.int64),
        "success": np.array(succ, dtype=np.int64),
        "mae": abs(predmn - root_true),
        "residual": abs(F_scalar(predmn, a, b)),
    }

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_npz", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max_newton_iter", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load dataset
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

    n_total = len(root_true)
    n_use = min(args.num_samples, n_total)
    indices = np.arange(n_total)
    if n_use < n_total:
        indices = np.random.choice(indices, size=n_use, replace=False)
    indices = np.sort(indices)

    # load model
    ckpt, model = load_model_checkpoint_cpu(args.model)

    # per-sample logs
    per_sample_rows = []

    # global arrays
    G = {
        "zero_init_direct_times_ns": [],
        "zero_init_plus_newton_times_ns": [],
        "zero_init_plus_newton_iters": [],
        "zero_init_plus_newton_success": [],
        "heuristic_direct_times_ns": [],
        "heuristic_plus_newton_times_ns": [],
        "heuristic_plus_newton_iters": [],
        "heuristic_plus_newton_success": [],
        "neural_direct_times_ns": [],
        "neural_plus_newton_times_ns": [],
        "neural_plus_newton_pred_times_ns": [],
        "neural_plus_newton_refine_times_ns": [],
        "neural_plus_newton_iters": [],
        "neural_plus_newton_success": [],
        "zero_init_direct_mae": [],
        "zero_init_plus_newton_mae": [],
        "heuristic_direct_mae": [],
        "heuristic_plus_newton_mae": [],
        "neural_direct_mae": [],
        "neural_plus_newton_mae": [],
    }

    for idx in indices:
        res = benchmark_one_sample(
            coeffs=coeffs[idx],
            center=float(center[idx]),
            a=float(a[idx]),
            b=float(b[idx]),
            root_true=float(root_true[idx]),
            model=model,
            ckpt_args=ckpt["args"],
            scaler=ckpt["scaler"],
            repeats=args.repeats,
            warmup=args.warmup,
            max_newton_iter=args.max_newton_iter,
            tol=args.tol,
        )

        row = {
            "sample_id": int(idx),
            "a": float(a[idx]),
            "b": float(b[idx]),
            "root_true": float(root_true[idx]),
            "zero_init_direct_mean_us": float(np.mean(ns_to_us(res["zero_init_direct"]["times_ns"]))),
            "zero_init_plus_newton_mean_us": float(np.mean(ns_to_us(res["zero_init_plus_newton"]["times_ns"]))),
            "zero_init_plus_newton_iter_mean": float(np.mean(res["zero_init_plus_newton"]["iters"])),
            "heuristic_direct_mean_us": float(np.mean(ns_to_us(res["heuristic_direct"]["times_ns"]))),
            "heuristic_plus_newton_mean_us": float(np.mean(ns_to_us(res["heuristic_plus_newton"]["times_ns"]))),
            "heuristic_plus_newton_iter_mean": float(np.mean(res["heuristic_plus_newton"]["iters"])),
            "neural_direct_mean_us": float(np.mean(ns_to_us(res["neural_direct"]["times_ns"]))),
            "neural_plus_newton_mean_us": float(np.mean(ns_to_us(res["neural_plus_newton"]["times_ns"]))),
            "neural_plus_newton_pred_mean_us": float(np.mean(ns_to_us(res["neural_plus_newton"]["pred_times_ns"]))),
            "neural_plus_newton_refine_mean_us": float(np.mean(ns_to_us(res["neural_plus_newton"]["refine_times_ns"]))),
            "neural_plus_newton_iter_mean": float(np.mean(res["neural_plus_newton"]["iters"])),
            "zero_init_direct_mae": float(res["zero_init_direct"]["mae"]),
            "zero_init_plus_newton_mae": float(res["zero_init_plus_newton"]["mae"]),
            "heuristic_direct_mae": float(res["heuristic_direct"]["mae"]),
            "heuristic_plus_newton_mae": float(res["heuristic_plus_newton"]["mae"]),
            "neural_direct_mae": float(res["neural_direct"]["mae"]),
            "neural_plus_newton_mae": float(res["neural_plus_newton"]["mae"]),
        }
        per_sample_rows.append(row)

        G["zero_init_direct_times_ns"].extend(res["zero_init_direct"]["times_ns"].tolist())
        G["zero_init_plus_newton_times_ns"].extend(res["zero_init_plus_newton"]["times_ns"].tolist())
        G["zero_init_plus_newton_iters"].extend(res["zero_init_plus_newton"]["iters"].tolist())
        G["zero_init_plus_newton_success"].extend(res["zero_init_plus_newton"]["success"].tolist())

        G["heuristic_direct_times_ns"].extend(res["heuristic_direct"]["times_ns"].tolist())
        G["heuristic_plus_newton_times_ns"].extend(res["heuristic_plus_newton"]["times_ns"].tolist())
        G["heuristic_plus_newton_iters"].extend(res["heuristic_plus_newton"]["iters"].tolist())
        G["heuristic_plus_newton_success"].extend(res["heuristic_plus_newton"]["success"].tolist())

        G["neural_direct_times_ns"].extend(res["neural_direct"]["times_ns"].tolist())
        G["neural_plus_newton_times_ns"].extend(res["neural_plus_newton"]["times_ns"].tolist())
        G["neural_plus_newton_pred_times_ns"].extend(res["neural_plus_newton"]["pred_times_ns"].tolist())
        G["neural_plus_newton_refine_times_ns"].extend(res["neural_plus_newton"]["refine_times_ns"].tolist())
        G["neural_plus_newton_iters"].extend(res["neural_plus_newton"]["iters"].tolist())
        G["neural_plus_newton_success"].extend(res["neural_plus_newton"]["success"].tolist())

        G["zero_init_direct_mae"].append(res["zero_init_direct"]["mae"])
        G["zero_init_plus_newton_mae"].append(res["zero_init_plus_newton"]["mae"])
        G["heuristic_direct_mae"].append(res["heuristic_direct"]["mae"])
        G["heuristic_plus_newton_mae"].append(res["heuristic_plus_newton"]["mae"])
        G["neural_direct_mae"].append(res["neural_direct"]["mae"])
        G["neural_plus_newton_mae"].append(res["neural_plus_newton"]["mae"])

    # save per-sample csv
    per_sample_csv = out_dir / "per_sample_times.csv"
    with open(per_sample_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_sample_rows)

    summary_rows = []

    def add_summary(method_name: str, times_ns, mae_arr, iters_arr=None, success_arr=None):
        time_us = ns_to_us(np.array(times_ns, dtype=np.int64))
        row = {
            "method": method_name,
            "n_timed_calls": int(len(time_us)),
            "time_mean_us": float(np.mean(time_us)),
            "time_median_us": float(np.median(time_us)),
            "time_p90_us": float(np.percentile(time_us, 90)),
            "time_p95_us": float(np.percentile(time_us, 95)),
            "time_p99_us": float(np.percentile(time_us, 99)),
            "time_min_us": float(np.min(time_us)),
            "time_max_us": float(np.max(time_us)),
            "mae_mean": float(np.mean(mae_arr)),
            "mae_median": float(np.median(mae_arr)),
        }
        if iters_arr is not None:
            iters_arr = np.array(iters_arr, dtype=np.float64)
            row.update({
                "iter_mean": float(np.mean(iters_arr)),
                "iter_median": float(np.median(iters_arr)),
                "iter_p90": float(np.percentile(iters_arr, 90)),
                "iter_p95": float(np.percentile(iters_arr, 95)),
            })
        if success_arr is not None:
            success_arr = np.array(success_arr, dtype=np.float64)
            row["success_rate"] = float(np.mean(success_arr))
        summary_rows.append(row)

    add_summary("zero_init_direct", G["zero_init_direct_times_ns"], G["zero_init_direct_mae"])
    add_summary(
        "zero_init_plus_newton",
        G["zero_init_plus_newton_times_ns"],
        G["zero_init_plus_newton_mae"],
        G["zero_init_plus_newton_iters"],
        G["zero_init_plus_newton_success"],
    )
    add_summary("heuristic_direct", G["heuristic_direct_times_ns"], G["heuristic_direct_mae"])
    add_summary(
        "heuristic_plus_newton",
        G["heuristic_plus_newton_times_ns"],
        G["heuristic_plus_newton_mae"],
        G["heuristic_plus_newton_iters"],
        G["heuristic_plus_newton_success"],
    )
    add_summary("neural_direct", G["neural_direct_times_ns"], G["neural_direct_mae"])
    add_summary(
        "neural_plus_newton",
        G["neural_plus_newton_times_ns"],
        G["neural_plus_newton_mae"],
        G["neural_plus_newton_iters"],
        G["neural_plus_newton_success"],
    )
    add_summary("neural_plus_newton_pred_only", G["neural_plus_newton_pred_times_ns"], G["neural_direct_mae"])
    add_summary(
        "neural_plus_newton_refine_only",
        G["neural_plus_newton_refine_times_ns"],
        G["neural_plus_newton_mae"],
        G["neural_plus_newton_iters"],
        G["neural_plus_newton_success"],
    )

    summary_csv = out_dir / "benchmark_summary.csv"
    fieldnames = []
    seen = set()
    for row in summary_rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in summary_rows:
            full_row = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(full_row)

    report_txt = out_dir / "benchmark_report.txt"
    lines = []
    lines.append("=== CPU Per-Sample Benchmark Report ===")
    lines.append("")
    lines.append("[Args]")
    for k, v in vars(args).items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("[Summary]")
    for row in summary_rows:
        lines.append("-" * 80)
        for k, v in row.items():
            lines.append(f"{k}: {v}")

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[DONE] Saved to: {out_dir.resolve()}")
    print("  - benchmark_summary.csv")
    print("  - benchmark_report.txt")
    print("  - per_sample_times.csv")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_colebrook_cpu_per_sample.py

CPU 환경에서 문제 1개당 걸리는 실제 wall-clock time을 분석한다.

비교 대상
---------
1) zero_init_direct
2) zero_init_plus_newton
3) heuristic_direct
4) heuristic_plus_newton
5) neural_direct
6) neural_plus_newton

입력
----
- test npz: colebrook_like_dataset.py 로 생성한 npz
  required keys: coeffs, center, a, b, root
- model checkpoint: train_colebrook_root.py 로 생성한 best_model.pt
예시
----
python benchmark_colebrook_cpu_per_sample.py \
  --test_npz ./colebrook_data/colebrook_deg25_test.npz \
  --model ./runs/colebrook_root_deg25/best_model.pt \
  --num_samples 2000 \
  --repeats 30 \
  --warmup 5 \
  --max_newton_iter 20 \
  --tol 1e-12 \
  --out_dir ./cpu_benchmarks/colebrook
"""

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

LN10 = math.log(10.0)


# =========================================================
# Original equation
# =========================================================
def F_scalar(x: float, a: float, b: float) -> float:
    return x + 2.0 * math.log10(a + b * x)


def dF_scalar(x: float, a: float, b: float) -> float:
    return 1.0 + 2.0 * b / ((a + b * x) * LN10)


def project_to_domain_scalar(x: float, a: float, b: float, eps: float = 1e-10) -> float:
    left = -a / b + eps
    return x if x >= left else left


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


def load_model_checkpoint_cpu(ckpt_path: str):
    device = torch.device("cpu")
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


def build_feature_one(
    coeffs: np.ndarray,
    center: float,
    a: float,
    b: float,
    ckpt_args: Dict,
    scaler: Dict,
) -> np.ndarray:
    use_log_ab = bool(ckpt_args.get("use_log_ab", False))
    include_center = bool(ckpt_args.get("include_center", True))
    include_ab = bool(ckpt_args.get("include_ab", True))

    feats = [coeffs.astype(np.float32)]

    if include_center:
        feats.append(np.array([center], dtype=np.float32))

    if include_ab:
        if use_log_ab:
            feats.append(np.array([math.log(a)], dtype=np.float32))
            feats.append(np.array([math.log(b)], dtype=np.float32))
        else:
            feats.append(np.array([a], dtype=np.float32))
            feats.append(np.array([b], dtype=np.float32))

    x = np.concatenate(feats, axis=0).astype(np.float32)

    mean = np.array(scaler["mean"], dtype=np.float32).reshape(-1)
    std = np.array(scaler["std"], dtype=np.float32).reshape(-1)
    x = (x - mean) / std
    return x


def neural_predict_one(model, feature_vec: np.ndarray) -> float:
    with torch.no_grad():
        x = torch.from_numpy(feature_vec).float().unsqueeze(0)
        y = model(x).squeeze(0).squeeze(0).item()
    return float(y)


# =========================================================
# Initial guess baselines
# =========================================================
def zero_init(a: float, b: float) -> float:
    return project_to_domain_scalar(0.0, a, b)


def heuristic_init(a: float, b: float) -> float:
    x0 = -2.0 * math.log10(a)
    return project_to_domain_scalar(x0, a, b)


# =========================================================
# Newton refinement
# =========================================================
def newton_refine_original(
    x0: float,
    a: float,
    b: float,
    tol: float = 1e-12,
    max_iter: int = 20,
    damping: float = 1.0,
    deriv_eps: float = 1e-14,
    max_step: float = 5.0,
) -> Tuple[float, int, bool]:
    x = project_to_domain_scalar(x0, a, b)

    for k in range(1, max_iter + 1):
        f = F_scalar(x, a, b)
        if abs(f) <= tol:
            return x, k - 1, True

        df = dF_scalar(x, a, b)
        if abs(df) < deriv_eps:
            df = deriv_eps if df >= 0 else -deriv_eps

        step = damping * f / df
        step = max(-max_step, min(max_step, step))

        x_new = project_to_domain_scalar(x - step, a, b)

        # half-step safeguard
        f_new = abs(F_scalar(x_new, a, b))
        if f_new > abs(f):
            x_half = project_to_domain_scalar(x - 0.5 * step, a, b)
            if abs(F_scalar(x_half, a, b)) < f_new:
                x_new = x_half

        if abs(x_new - x) <= 1e-12:
            x = x_new
            break

        x = x_new

    return x, max_iter, abs(F_scalar(x, a, b)) <= tol


# =========================================================
# Timing helpers
# =========================================================
def time_call_ns(fn, *args, **kwargs):
    t0 = time.perf_counter_ns()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter_ns()
    return out, (t1 - t0)


def ns_to_us(arr) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64) / 1_000.0


# =========================================================
# Benchmark per sample
# =========================================================
def benchmark_one_sample(
    coeffs: np.ndarray,
    center: float,
    a: float,
    b: float,
    root_true: float,
    model,
    ckpt_args: Dict,
    scaler: Dict,
    repeats: int,
    warmup: int,
    max_newton_iter: int,
    tol: float,
) -> Dict[str, Dict]:
    feat = build_feature_one(coeffs, center, a, b, ckpt_args, scaler)

    # warmup
    for _ in range(warmup):
        _ = zero_init(a, b)
        _ = heuristic_init(a, b)
        _ = neural_predict_one(model, feat)
        _ = newton_refine_original(zero_init(a, b), a, b, tol=tol, max_iter=max_newton_iter)
        _ = newton_refine_original(heuristic_init(a, b), a, b, tol=tol, max_iter=max_newton_iter)
        _ = newton_refine_original(neural_predict_one(model, feat), a, b, tol=tol, max_iter=max_newton_iter)

    out = {}

    # zero direct
    times, preds = [], []
    for _ in range(repeats):
        pred, t_ns = time_call_ns(zero_init, a, b)
        times.append(t_ns)
        preds.append(pred)
    pred0 = float(preds[-1])
    out["zero_init_direct"] = {
        "pred": pred0,
        "times_ns": np.array(times, dtype=np.int64),
        "mae": abs(pred0 - root_true),
        "residual": abs(F_scalar(pred0, a, b)),
    }

    # zero + newton
    times_total, times_refine, iters, preds_ref, succ = [], [], [], [], []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        x0 = zero_init(a, b)
        (x_ref, k, ok), t_ref = time_call_ns(
            newton_refine_original, x0, a, b, tol=tol, max_iter=max_newton_iter
        )
        t1 = time.perf_counter_ns()
        times_total.append(t1 - t0)
        times_refine.append(t_ref)
        iters.append(k)
        preds_ref.append(x_ref)
        succ.append(int(ok))
    pred0n = float(preds_ref[-1])
    out["zero_init_plus_newton"] = {
        "pred": pred0n,
        "times_ns": np.array(times_total, dtype=np.int64),
        "refine_times_ns": np.array(times_refine, dtype=np.int64),
        "iters": np.array(iters, dtype=np.int64),
        "success": np.array(succ, dtype=np.int64),
        "mae": abs(pred0n - root_true),
        "residual": abs(F_scalar(pred0n, a, b)),
    }

    # heuristic direct
    times, preds = [], []
    for _ in range(repeats):
        pred, t_ns = time_call_ns(heuristic_init, a, b)
        times.append(t_ns)
        preds.append(pred)
    predh = float(preds[-1])
    out["heuristic_direct"] = {
        "pred": predh,
        "times_ns": np.array(times, dtype=np.int64),
        "mae": abs(predh - root_true),
        "residual": abs(F_scalar(predh, a, b)),
    }

    # heuristic + newton
    times_total, times_refine, iters, preds_ref, succ = [], [], [], [], []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        x0 = heuristic_init(a, b)
        (x_ref, k, ok), t_ref = time_call_ns(
            newton_refine_original, x0, a, b, tol=tol, max_iter=max_newton_iter
        )
        t1 = time.perf_counter_ns()
        times_total.append(t1 - t0)
        times_refine.append(t_ref)
        iters.append(k)
        preds_ref.append(x_ref)
        succ.append(int(ok))
    predhn = float(preds_ref[-1])
    out["heuristic_plus_newton"] = {
        "pred": predhn,
        "times_ns": np.array(times_total, dtype=np.int64),
        "refine_times_ns": np.array(times_refine, dtype=np.int64),
        "iters": np.array(iters, dtype=np.int64),
        "success": np.array(succ, dtype=np.int64),
        "mae": abs(predhn - root_true),
        "residual": abs(F_scalar(predhn, a, b)),
    }

    # neural direct
    times, preds = [], []
    for _ in range(repeats):
        pred, t_ns = time_call_ns(neural_predict_one, model, feat)
        times.append(t_ns)
        preds.append(pred)
    predm = float(preds[-1])
    out["neural_direct"] = {
        "pred": predm,
        "times_ns": np.array(times, dtype=np.int64),
        "mae": abs(predm - root_true),
        "residual": abs(F_scalar(predm, a, b)),
    }

    # neural + newton
    times_total, times_pred, times_refine, iters, preds_ref, succ = [], [], [], [], [], []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        pred, t_pred = time_call_ns(neural_predict_one, model, feat)
        (x_ref, k, ok), t_ref = time_call_ns(
            newton_refine_original, pred, a, b, tol=tol, max_iter=max_newton_iter
        )
        t1 = time.perf_counter_ns()
        times_total.append(t1 - t0)
        times_pred.append(t_pred)
        times_refine.append(t_ref)
        iters.append(k)
        preds_ref.append(x_ref)
        succ.append(int(ok))
    predmn = float(preds_ref[-1])
    out["neural_plus_newton"] = {
        "pred": predmn,
        "times_ns": np.array(times_total, dtype=np.int64),
        "pred_times_ns": np.array(times_pred, dtype=np.int64),
        "refine_times_ns": np.array(times_refine, dtype=np.int64),
        "iters": np.array(iters, dtype=np.int64),
        "success": np.array(succ, dtype=np.int64),
        "mae": abs(predmn - root_true),
        "residual": abs(F_scalar(predmn, a, b)),
    }

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_npz", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max_newton_iter", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load dataset
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

    n_total = len(root_true)
    n_use = min(args.num_samples, n_total)
    indices = np.arange(n_total)
    if n_use < n_total:
        indices = np.random.choice(indices, size=n_use, replace=False)
    indices = np.sort(indices)

    # load model
    ckpt, model = load_model_checkpoint_cpu(args.model)

    # per-sample logs
    per_sample_rows = []

    # global arrays
    G = {
        "zero_init_direct_times_ns": [],
        "zero_init_plus_newton_times_ns": [],
        "zero_init_plus_newton_iters": [],
        "zero_init_plus_newton_success": [],
        "heuristic_direct_times_ns": [],
        "heuristic_plus_newton_times_ns": [],
        "heuristic_plus_newton_iters": [],
        "heuristic_plus_newton_success": [],
        "neural_direct_times_ns": [],
        "neural_plus_newton_times_ns": [],
        "neural_plus_newton_pred_times_ns": [],
        "neural_plus_newton_refine_times_ns": [],
        "neural_plus_newton_iters": [],
        "neural_plus_newton_success": [],
        "zero_init_direct_mae": [],
        "zero_init_plus_newton_mae": [],
        "heuristic_direct_mae": [],
        "heuristic_plus_newton_mae": [],
        "neural_direct_mae": [],
        "neural_plus_newton_mae": [],
    }

    for idx in indices:
        res = benchmark_one_sample(
            coeffs=coeffs[idx],
            center=float(center[idx]),
            a=float(a[idx]),
            b=float(b[idx]),
            root_true=float(root_true[idx]),
            model=model,
            ckpt_args=ckpt["args"],
            scaler=ckpt["scaler"],
            repeats=args.repeats,
            warmup=args.warmup,
            max_newton_iter=args.max_newton_iter,
            tol=args.tol,
        )

        row = {
            "sample_id": int(idx),
            "a": float(a[idx]),
            "b": float(b[idx]),
            "root_true": float(root_true[idx]),
            "zero_init_direct_mean_us": float(np.mean(ns_to_us(res["zero_init_direct"]["times_ns"]))),
            "zero_init_plus_newton_mean_us": float(np.mean(ns_to_us(res["zero_init_plus_newton"]["times_ns"]))),
            "zero_init_plus_newton_iter_mean": float(np.mean(res["zero_init_plus_newton"]["iters"])),
            "heuristic_direct_mean_us": float(np.mean(ns_to_us(res["heuristic_direct"]["times_ns"]))),
            "heuristic_plus_newton_mean_us": float(np.mean(ns_to_us(res["heuristic_plus_newton"]["times_ns"]))),
            "heuristic_plus_newton_iter_mean": float(np.mean(res["heuristic_plus_newton"]["iters"])),
            "neural_direct_mean_us": float(np.mean(ns_to_us(res["neural_direct"]["times_ns"]))),
            "neural_plus_newton_mean_us": float(np.mean(ns_to_us(res["neural_plus_newton"]["times_ns"]))),
            "neural_plus_newton_pred_mean_us": float(np.mean(ns_to_us(res["neural_plus_newton"]["pred_times_ns"]))),
            "neural_plus_newton_refine_mean_us": float(np.mean(ns_to_us(res["neural_plus_newton"]["refine_times_ns"]))),
            "neural_plus_newton_iter_mean": float(np.mean(res["neural_plus_newton"]["iters"])),
            "zero_init_direct_mae": float(res["zero_init_direct"]["mae"]),
            "zero_init_plus_newton_mae": float(res["zero_init_plus_newton"]["mae"]),
            "heuristic_direct_mae": float(res["heuristic_direct"]["mae"]),
            "heuristic_plus_newton_mae": float(res["heuristic_plus_newton"]["mae"]),
            "neural_direct_mae": float(res["neural_direct"]["mae"]),
            "neural_plus_newton_mae": float(res["neural_plus_newton"]["mae"]),
        }
        per_sample_rows.append(row)

        G["zero_init_direct_times_ns"].extend(res["zero_init_direct"]["times_ns"].tolist())
        G["zero_init_plus_newton_times_ns"].extend(res["zero_init_plus_newton"]["times_ns"].tolist())
        G["zero_init_plus_newton_iters"].extend(res["zero_init_plus_newton"]["iters"].tolist())
        G["zero_init_plus_newton_success"].extend(res["zero_init_plus_newton"]["success"].tolist())

        G["heuristic_direct_times_ns"].extend(res["heuristic_direct"]["times_ns"].tolist())
        G["heuristic_plus_newton_times_ns"].extend(res["heuristic_plus_newton"]["times_ns"].tolist())
        G["heuristic_plus_newton_iters"].extend(res["heuristic_plus_newton"]["iters"].tolist())
        G["heuristic_plus_newton_success"].extend(res["heuristic_plus_newton"]["success"].tolist())

        G["neural_direct_times_ns"].extend(res["neural_direct"]["times_ns"].tolist())
        G["neural_plus_newton_times_ns"].extend(res["neural_plus_newton"]["times_ns"].tolist())
        G["neural_plus_newton_pred_times_ns"].extend(res["neural_plus_newton"]["pred_times_ns"].tolist())
        G["neural_plus_newton_refine_times_ns"].extend(res["neural_plus_newton"]["refine_times_ns"].tolist())
        G["neural_plus_newton_iters"].extend(res["neural_plus_newton"]["iters"].tolist())
        G["neural_plus_newton_success"].extend(res["neural_plus_newton"]["success"].tolist())

        G["zero_init_direct_mae"].append(res["zero_init_direct"]["mae"])
        G["zero_init_plus_newton_mae"].append(res["zero_init_plus_newton"]["mae"])
        G["heuristic_direct_mae"].append(res["heuristic_direct"]["mae"])
        G["heuristic_plus_newton_mae"].append(res["heuristic_plus_newton"]["mae"])
        G["neural_direct_mae"].append(res["neural_direct"]["mae"])
        G["neural_plus_newton_mae"].append(res["neural_plus_newton"]["mae"])

    # save per-sample csv
    per_sample_csv = out_dir / "per_sample_times.csv"
    with open(per_sample_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_sample_rows)

    summary_rows = []

    def add_summary(method_name: str, times_ns, mae_arr, iters_arr=None, success_arr=None):
        time_us = ns_to_us(np.array(times_ns, dtype=np.int64))
        row = {
            "method": method_name,
            "n_timed_calls": int(len(time_us)),
            "time_mean_us": float(np.mean(time_us)),
            "time_median_us": float(np.median(time_us)),
            "time_p90_us": float(np.percentile(time_us, 90)),
            "time_p95_us": float(np.percentile(time_us, 95)),
            "time_p99_us": float(np.percentile(time_us, 99)),
            "time_min_us": float(np.min(time_us)),
            "time_max_us": float(np.max(time_us)),
            "mae_mean": float(np.mean(mae_arr)),
            "mae_median": float(np.median(mae_arr)),
        }
        if iters_arr is not None:
            iters_arr = np.array(iters_arr, dtype=np.float64)
            row.update({
                "iter_mean": float(np.mean(iters_arr)),
                "iter_median": float(np.median(iters_arr)),
                "iter_p90": float(np.percentile(iters_arr, 90)),
                "iter_p95": float(np.percentile(iters_arr, 95)),
            })
        if success_arr is not None:
            success_arr = np.array(success_arr, dtype=np.float64)
            row["success_rate"] = float(np.mean(success_arr))
        summary_rows.append(row)

    add_summary("zero_init_direct", G["zero_init_direct_times_ns"], G["zero_init_direct_mae"])
    add_summary(
        "zero_init_plus_newton",
        G["zero_init_plus_newton_times_ns"],
        G["zero_init_plus_newton_mae"],
        G["zero_init_plus_newton_iters"],
        G["zero_init_plus_newton_success"],
    )
    add_summary("heuristic_direct", G["heuristic_direct_times_ns"], G["heuristic_direct_mae"])
    add_summary(
        "heuristic_plus_newton",
        G["heuristic_plus_newton_times_ns"],
        G["heuristic_plus_newton_mae"],
        G["heuristic_plus_newton_iters"],
        G["heuristic_plus_newton_success"],
    )
    add_summary("neural_direct", G["neural_direct_times_ns"], G["neural_direct_mae"])
    add_summary(
        "neural_plus_newton",
        G["neural_plus_newton_times_ns"],
        G["neural_plus_newton_mae"],
        G["neural_plus_newton_iters"],
        G["neural_plus_newton_success"],
    )
    add_summary("neural_plus_newton_pred_only", G["neural_plus_newton_pred_times_ns"], G["neural_direct_mae"])
    add_summary(
        "neural_plus_newton_refine_only",
        G["neural_plus_newton_refine_times_ns"],
        G["neural_plus_newton_mae"],
        G["neural_plus_newton_iters"],
        G["neural_plus_newton_success"],
    )

    summary_csv = out_dir / "benchmark_summary.csv"
    fieldnames = []
    seen = set()
    for row in summary_rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in summary_rows:
            full_row = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(full_row)

    report_txt = out_dir / "benchmark_report.txt"
    lines = []
    lines.append("=== CPU Per-Sample Benchmark Report ===")
    lines.append("")
    lines.append("[Args]")
    for k, v in vars(args).items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("[Summary]")
    for row in summary_rows:
        lines.append("-" * 80)
        for k, v in row.items():
            lines.append(f"{k}: {v}")

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[DONE] Saved to: {out_dir.resolve()}")
    print("  - benchmark_summary.csv")
    print("  - benchmark_report.txt")
    print("  - per_sample_times.csv")


if __name__ == "__main__":
    main()
