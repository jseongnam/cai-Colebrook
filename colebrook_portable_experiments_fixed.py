#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
colebrook_portable_experiments_fixed.py

설명
----
기존 colebrook_portable_experiments.py에서 direct prediction이 비정상적으로
깨지던 문제를 수정한 버전.

핵심 수정
--------
1) checkpoint의 args를 읽어 입력 feature 구성을 정확히 복원
2) checkpoint의 scaler(mean/std)를 적용
3) x0 key가 없으면 center를 자동으로 사용
4) dropout 포함 모델 구조를 정확히 복원

가정
----
- 식: x + 2 log10(a + b x) = 0
- NPZ key:
    coeffs, a, b, root, (x0 또는 center)
- checkpoint:
    train_colebrook_root.py 계열 best_model.pt
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


# =========================================================
# Utilities
# =========================================================
def percentile(x, q):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def save_csv(path: Path, rows: List[Dict]):
    if not rows:
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


# =========================================================
# Original Colebrook-like equation
# =========================================================
LN10 = math.log(10.0)


def colebrook_f(x, a, b):
    x = np.asarray(x, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = a + b * x
    out = np.full_like(x, np.nan, dtype=np.float64)
    mask = z > 0
    out[mask] = x[mask] + 2.0 * np.log10(z[mask])
    return out


def colebrook_df(x, a, b):
    x = np.asarray(x, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = a + b * x
    out = np.full_like(x, np.nan, dtype=np.float64)
    mask = z > 0
    out[mask] = 1.0 + 2.0 * b[mask] / (z[mask] * LN10)
    return out


def project_to_domain_np(x, a, b, eps=1e-10):
    left = -a / b + eps
    return np.maximum(x, left)


def newton_refine_original_vectorized(
    x0,
    a,
    b,
    tol=1e-12,
    max_iter=20,
    damping=1.0,
    deriv_eps=1e-14,
    max_step=5.0,
):
    x = project_to_domain_np(np.asarray(x0, dtype=np.float64).copy(), a, b, eps=1e-10)
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

        fx = colebrook_f(xa, aa, bb)
        dfx = colebrook_df(xa, aa, bb)

        small = np.abs(dfx) < deriv_eps
        dfx[small] = np.where(dfx[small] >= 0, deriv_eps, -deriv_eps)

        step = damping * fx / dfx
        step = np.clip(step, -max_step, max_step)
        x_new = xa - step
        x_new = project_to_domain_np(x_new, aa, bb, eps=1e-10)

        fx_new = np.abs(colebrook_f(x_new, aa, bb))
        worse = fx_new > np.abs(fx)
        if np.any(worse):
            x_half = xa[worse] - 0.5 * step[worse]
            x_half = project_to_domain_np(x_half, aa[worse], bb[worse], eps=1e-10)
            fx_half = np.abs(colebrook_f(x_half, aa[worse], bb[worse]))
            better_half = fx_half < fx_new[worse]
            x_new[worse] = np.where(better_half, x_half, x_new[worse])

        x[idx] = x_new
        iters_used[idx] = k

        fx_now = np.abs(colebrook_f(x_new, aa, bb))
        done = fx_now <= tol
        converged[idx[done]] = True
        active[idx[done]] = False

    return x, iters_used, converged


# =========================================================
# Model loading
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
        return self.net(x).squeeze(-1)


def infer_hidden_dims_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, List[int]]:
    linear_weights = []
    for k, v in state_dict.items():
        if k.endswith(".weight") and v.ndim == 2:
            linear_weights.append((k, v.shape))
    linear_weights = sorted(linear_weights, key=lambda kv: kv[0])

    if not linear_weights:
        raise RuntimeError("No linear layers found in state_dict")

    dims = []
    for _, shape in linear_weights:
        out_dim, in_dim = shape
        dims.append((in_dim, out_dim))

    input_dim = dims[0][0]
    hidden_dims = [out_dim for (_, out_dim) in dims[:-1]]
    return input_dim, hidden_dims


def load_model_checkpoint(path: str, device="cpu"):
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise RuntimeError("Unsupported checkpoint format")

    if isinstance(ckpt, dict) and "hidden_dims" in ckpt:
        hidden_dims = list(ckpt["hidden_dims"])
        input_dim = None
        for k, v in state_dict.items():
            if k.endswith(".weight") and v.ndim == 2:
                input_dim = int(v.shape[1])
                break
        if input_dim is None:
            input_dim, _ = infer_hidden_dims_from_state_dict(state_dict)
    else:
        input_dim, hidden_dims = infer_hidden_dims_from_state_dict(state_dict)

    dropout = 0.0
    if isinstance(ckpt, dict) and "args" in ckpt and isinstance(ckpt["args"], dict):
        dropout = float(ckpt["args"].get("dropout", 0.0))

    model = MLPRegressor(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return ckpt, model, {
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
    }


# =========================================================
# Data loading
# =========================================================
def load_npz(path: str):
    data = np.load(path, allow_pickle=True)
    available = list(data.keys())

    for k in ["coeffs", "a", "b", "root"]:
        if k not in data:
            raise KeyError(f"Missing key '{k}'; available: {available}")

    if "x0" in data:
        x0_arr = np.asarray(data["x0"], dtype=np.float64).reshape(-1)
        x0_source = "x0"
    elif "center" in data:
        x0_arr = np.asarray(data["center"], dtype=np.float64).reshape(-1)
        x0_source = "center"
    else:
        raise KeyError(f"Missing key 'x0' (or alias 'center'); available: {available}")

    out = {
        "coeffs": np.asarray(data["coeffs"], dtype=np.float64),
        "x0": x0_arr,
        "a": np.asarray(data["a"], dtype=np.float64).reshape(-1),
        "b": np.asarray(data["b"], dtype=np.float64).reshape(-1),
        "root": np.asarray(data["root"], dtype=np.float64).reshape(-1),
        "_x0_source": x0_source,
    }
    return out


# =========================================================
# Feature reconstruction from checkpoint args
# =========================================================
def build_features_from_checkpoint_args(
    coeffs: np.ndarray,
    x0: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    ckpt_args: Dict,
) -> np.ndarray:
    use_log_ab = bool(ckpt_args.get("use_log_ab", False))
    include_center = bool(ckpt_args.get("include_center", True))
    include_ab = bool(ckpt_args.get("include_ab", True))

    feats = [coeffs.astype(np.float32)]

    if include_center:
        feats.append(x0.reshape(-1, 1).astype(np.float32))

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


def predict_with_checkpoint(
    coeffs: np.ndarray,
    x0: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    ckpt: Dict,
    model,
    device="cpu",
    batch_size=4096,
) -> np.ndarray:
    if "args" not in ckpt or "scaler" not in ckpt:
        raise RuntimeError("Checkpoint must contain 'args' and 'scaler' for this fixed evaluator.")

    X = build_features_from_checkpoint_args(coeffs, x0, a, b, ckpt["args"])
    X = apply_scaler(X, ckpt["scaler"])

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).to(device)
            yb = model(xb).detach().cpu().numpy().reshape(-1)
            preds.append(yb)

    pred = np.concatenate(preds, axis=0)
    return pred.astype(np.float64)


# =========================================================
# Metrics
# =========================================================
def compute_metrics(pred, true_root, a, b):
    pred = np.asarray(pred, dtype=np.float64)
    true_root = np.asarray(true_root, dtype=np.float64)

    mae = float(np.mean(np.abs(pred - true_root)))
    rmse = float(np.sqrt(np.mean((pred - true_root) ** 2)))

    ss_res = float(np.sum((pred - true_root) ** 2))
    ss_tot = float(np.sum((true_root - np.mean(true_root)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    residual = np.abs(colebrook_f(pred, a, b))
    valid_mask = np.isfinite(residual)

    out = {
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2),
        "valid_ratio": float(np.mean(valid_mask)),
        "residual_mean": float(np.nanmean(residual)),
        "residual_median": float(np.nanmedian(residual)),
        "residual_p90": percentile(residual[np.isfinite(residual)], 90),
        "max_abs_error": float(np.max(np.abs(pred - true_root))),
    }
    return out


# =========================================================
# Baselines
# =========================================================
def baseline_zero_init(a):
    return np.zeros_like(a, dtype=np.float64)


def baseline_heuristic_init(a):
    return -2.0 * np.log10(np.clip(a, 1e-15, None))


# =========================================================
# Experiments
# =========================================================
def eval_basic(data, models, out_dir: Path, device="cpu", newton_iter=20, tol=1e-12):
    coeffs, x0, a, b, root = data["coeffs"], data["x0"], data["a"], data["b"], data["root"]
    rows = []

    zero = baseline_zero_init(a)
    rows.append({"name": "zero_init_direct", **compute_metrics(zero, root, a, b)})

    zref, ziter, zconv = newton_refine_original_vectorized(zero, a, b, max_iter=newton_iter, tol=tol)
    rows.append({
        "name": "zero_init_plus_newton",
        **compute_metrics(zref, root, a, b),
        "newton_iter_mean": float(np.mean(ziter)),
        "newton_iter_median": float(np.median(ziter)),
        "newton_iter_p90": percentile(ziter, 90),
        "newton_converged_ratio": float(np.mean(zconv)),
    })

    heur = baseline_heuristic_init(a)
    rows.append({"name": "heuristic_direct", **compute_metrics(heur, root, a, b)})

    href, hiter, hconv = newton_refine_original_vectorized(heur, a, b, max_iter=newton_iter, tol=tol)
    rows.append({
        "name": "heuristic_plus_newton",
        **compute_metrics(href, root, a, b),
        "newton_iter_mean": float(np.mean(hiter)),
        "newton_iter_median": float(np.median(hiter)),
        "newton_iter_p90": percentile(hiter, 90),
        "newton_converged_ratio": float(np.mean(hconv)),
    })

    for mode, pack in models.items():
        pred = predict_with_checkpoint(coeffs, x0, a, b, pack["ckpt"], pack["model"], device=device)
        rows.append({"name": f"{mode}_direct", **compute_metrics(pred, root, a, b)})

        pref, piter, pconv = newton_refine_original_vectorized(pred, a, b, max_iter=newton_iter, tol=tol)
        rows.append({
            "name": f"{mode}_plus_newton",
            **compute_metrics(pref, root, a, b),
            "newton_iter_mean": float(np.mean(piter)),
            "newton_iter_median": float(np.median(piter)),
            "newton_iter_p90": percentile(piter, 90),
            "newton_converged_ratio": float(np.mean(pconv)),
        })

    save_csv(out_dir / "basic_summary.csv", rows)


def eval_newton_budget(data, models, out_dir: Path, device="cpu", max_budget=10, tol=1e-12):
    coeffs, x0, a, b, root = data["coeffs"], data["x0"], data["a"], data["b"], data["root"]
    rows = []

    inits = {
        "zero_init": baseline_zero_init(a),
        "heuristic": baseline_heuristic_init(a),
    }
    for mode, pack in models.items():
        inits[mode] = predict_with_checkpoint(coeffs, x0, a, b, pack["ckpt"], pack["model"], device=device)

    for name, init in inits.items():
        for budget in range(max_budget + 1):
            if budget == 0:
                pred = init.copy()
                iters = np.zeros_like(root, dtype=np.int32)
                conv = np.isfinite(colebrook_f(pred, a, b)) & (np.abs(colebrook_f(pred, a, b)) <= tol)
            else:
                pred, iters, conv = newton_refine_original_vectorized(init, a, b, max_iter=budget, tol=tol)

            rows.append({
                "method": name,
                "newton_budget": budget,
                **compute_metrics(pred, root, a, b),
                "iter_mean": float(np.mean(iters)),
                "iter_median": float(np.median(iters)),
                "converged_ratio": float(np.mean(conv)),
            })

    save_csv(out_dir / "newton_budget.csv", rows)


def eval_noise(data, models, out_dir: Path, device="cpu"):
    coeffs, x0, a, b, root = data["coeffs"], data["x0"], data["a"], data["b"], data["root"]
    rows = []

    coeff_noise_list = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
    ab_noise_list = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
    coeff_std = np.std(coeffs, axis=0, keepdims=True) + 1e-12

    for mode, pack in models.items():
        for sigma in coeff_noise_list:
            noisy_coeffs = coeffs + np.random.randn(*coeffs.shape) * coeff_std * sigma
            pred = predict_with_checkpoint(noisy_coeffs, x0, a, b, pack["ckpt"], pack["model"], device=device)
            rows.append({
                "mode": mode,
                "noise_type": "coeff_gaussian",
                "noise_level": sigma,
                **compute_metrics(pred, root, a, b),
            })

        for sigma in ab_noise_list:
            na = np.clip(a * (1.0 + sigma * np.random.randn(*a.shape)), 1e-12, None)
            nb = np.clip(b * (1.0 + sigma * np.random.randn(*b.shape)), 1e-12, None)

            pred = predict_with_checkpoint(coeffs, x0, na, nb, pack["ckpt"], pack["model"], device=device)
            noisy_res = np.abs(colebrook_f(pred, na, nb))
            rows.append({
                "mode": mode,
                "noise_type": "ab_multiplicative",
                "noise_level": sigma,
                **compute_metrics(pred, root, a, b),
                "noisy_equation_residual_mean": float(np.nanmean(noisy_res)),
                "noisy_equation_residual_p90": percentile(noisy_res[np.isfinite(noisy_res)], 90),
            })

    save_csv(out_dir / "noise_robustness.csv", rows)


def make_ood_masks(data):
    a, b, root = data["a"], data["b"], data["root"]
    r_abs = np.abs(root)

    a_q80 = np.quantile(a, 0.8)
    b_q80 = np.quantile(b, 0.8)
    r_q80 = np.quantile(r_abs, 0.8)

    return {
        "ID_all": np.ones_like(a, dtype=bool),
        "OOD_large_a": a >= a_q80,
        "OOD_large_b": b >= b_q80,
        "OOD_large_root_abs": r_abs >= r_q80,
        "HARD_union": (a >= a_q80) | (b >= b_q80) | (r_abs >= r_q80),
        "EASY_intersection": (a < np.quantile(a, 0.5)) & (b < np.quantile(b, 0.5)) & (r_abs < np.quantile(r_abs, 0.5)),
    }


def eval_ood(data, models, out_dir: Path, device="cpu", newton_iter=20, tol=1e-12):
    coeffs, x0, a, b, root = data["coeffs"], data["x0"], data["a"], data["b"], data["root"]
    masks = make_ood_masks(data)
    rows = []

    pred_cache = {}
    for mode, pack in models.items():
        pred_cache[mode] = predict_with_checkpoint(coeffs, x0, a, b, pack["ckpt"], pack["model"], device=device)

    for split_name, mask in masks.items():
        if np.sum(mask) == 0:
            continue

        for init_name, init in {
            "zero_init": baseline_zero_init(a),
            "heuristic": baseline_heuristic_init(a),
        }.items():
            rows.append({
                "split": split_name,
                "method": f"{init_name}_direct",
                "n": int(np.sum(mask)),
                **compute_metrics(init[mask], root[mask], a[mask], b[mask]),
            })

            ref, iters, conv = newton_refine_original_vectorized(init[mask], a[mask], b[mask], max_iter=newton_iter, tol=tol)
            rows.append({
                "split": split_name,
                "method": f"{init_name}_plus_newton",
                "n": int(np.sum(mask)),
                **compute_metrics(ref, root[mask], a[mask], b[mask]),
                "iter_mean": float(np.mean(iters)),
                "iter_median": float(np.median(iters)),
                "converged_ratio": float(np.mean(conv)),
            })

        for mode, pred_all in pred_cache.items():
            rows.append({
                "split": split_name,
                "method": f"{mode}_direct",
                "n": int(np.sum(mask)),
                **compute_metrics(pred_all[mask], root[mask], a[mask], b[mask]),
            })

            ref, iters, conv = newton_refine_original_vectorized(pred_all[mask], a[mask], b[mask], max_iter=newton_iter, tol=tol)
            rows.append({
                "split": split_name,
                "method": f"{mode}_plus_newton",
                "n": int(np.sum(mask)),
                **compute_metrics(ref, root[mask], a[mask], b[mask]),
                "iter_mean": float(np.mean(iters)),
                "iter_median": float(np.median(iters)),
                "converged_ratio": float(np.mean(conv)),
            })

    save_csv(out_dir / "ood_analysis.csv", rows)


def eval_basin(data, models, out_dir: Path, device="cpu", n_cases=25, radius=1.0, grid_points=81, tol=1e-12, max_iter=20):
    a, b, root = data["a"], data["b"], data["root"]
    coeffs, x0 = data["coeffs"], data["x0"]

    idxs = np.linspace(0, len(root) - 1, min(n_cases, len(root))).astype(int)
    rows = []

    pred_cache = {}
    for mode, pack in models.items():
        pred_cache[mode] = predict_with_checkpoint(coeffs, x0, a, b, pack["ckpt"], pack["model"], device=device)

    for idx in idxs:
        true_r = root[idx]
        grid = np.linspace(true_r - radius, true_r + radius, grid_points)
        ai = np.full(grid_points, a[idx], dtype=np.float64)
        bi = np.full(grid_points, b[idx], dtype=np.float64)

        ref, iters, conv = newton_refine_original_vectorized(grid, ai, bi, max_iter=max_iter, tol=tol)
        conv_to_true = np.abs(ref - true_r) <= 1e-6

        row = {
            "sample_index": int(idx),
            "a": float(a[idx]),
            "b": float(b[idx]),
            "true_root": float(true_r),
            "basin_ratio_around_true_root": float(np.mean(conv_to_true)),
            "grid_left": float(grid[0]),
            "grid_right": float(grid[-1]),
        }

        for mode, preds in pred_cache.items():
            row[f"{mode}_pred"] = float(preds[idx])
            row[f"{mode}_abs_init_error"] = float(abs(preds[idx] - true_r))

        rows.append(row)

    save_csv(out_dir / "basin_analysis.csv", rows)


def time_callable(fn, repeats=10000):
    t0 = time.perf_counter_ns()
    for _ in range(repeats):
        fn()
    t1 = time.perf_counter_ns()
    return (t1 - t0) / repeats / 1000.0  # us


def eval_timing(data, models, out_dir: Path, device="cpu", repeats=20000):
    coeffs, x0, a, b = data["coeffs"], data["x0"], data["a"], data["b"]
    idx = 0
    ai = np.array([a[idx]], dtype=np.float64)
    bi = np.array([b[idx]], dtype=np.float64)

    rows = []

    zero = baseline_zero_init(ai)
    heur = baseline_heuristic_init(ai)

    rows.append({"method": "zero_init_direct", "time_mean_us": time_callable(lambda: baseline_zero_init(ai), repeats)})
    rows.append({"method": "heuristic_direct", "time_mean_us": time_callable(lambda: baseline_heuristic_init(ai), repeats)})
    rows.append({"method": "zero_init_plus_newton", "time_mean_us": time_callable(lambda: newton_refine_original_vectorized(zero, ai, bi, max_iter=20, tol=1e-12), repeats)})
    rows.append({"method": "heuristic_plus_newton", "time_mean_us": time_callable(lambda: newton_refine_original_vectorized(heur, ai, bi, max_iter=20, tol=1e-12), repeats)})

    for mode, pack in models.items():
        coeffs1 = coeffs[idx:idx+1]
        x01 = x0[idx:idx+1]
        a1 = a[idx:idx+1]
        b1 = b[idx:idx+1]

        def pred_only():
            _ = predict_with_checkpoint(coeffs1, x01, a1, b1, pack["ckpt"], pack["model"], device=device, batch_size=1)

        pred = predict_with_checkpoint(coeffs1, x01, a1, b1, pack["ckpt"], pack["model"], device=device, batch_size=1)

        def pred_plus_newton():
            _ = newton_refine_original_vectorized(pred, ai, bi, max_iter=20, tol=1e-12)

        rows.append({"method": f"{mode}_direct", "time_mean_us": time_callable(pred_only, repeats)})
        rows.append({"method": f"{mode}_plus_newton_refine_only", "time_mean_us": time_callable(pred_plus_newton, repeats)})

    save_csv(out_dir / "timing_microbench.csv", rows)


# =========================================================
# CLI
# =========================================================
def parse_model_args(items: List[str]) -> Dict[str, str]:
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--model must be mode=path, got {item}")
        mode, path = item.split("=", 1)
        out[mode] = path
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_npz", required=True)
    parser.add_argument("--model", nargs="+", required=True, help="e.g. coeffs_x0_ab=./runs/model.pt")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max_newton_iter", type=int, default=20)

    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--eval_basic", action="store_true")
    parser.add_argument("--eval_newton_budget", action="store_true")
    parser.add_argument("--eval_noise", action="store_true")
    parser.add_argument("--eval_ood", action="store_true")
    parser.add_argument("--eval_basin", action="store_true")
    parser.add_argument("--eval_timing", action="store_true")

    parser.add_argument("--newton_budget_max", type=int, default=10)
    parser.add_argument("--basin_cases", type=int, default=25)
    parser.add_argument("--basin_radius", type=float, default=1.0)
    parser.add_argument("--basin_grid_points", type=int, default=81)
    parser.add_argument("--timing_repeats", type=int, default=20000)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_npz(args.test_npz)
    model_paths = parse_model_args(args.model)

    models = {}
    for mode, path in model_paths.items():
        ckpt, model, meta = load_model_checkpoint(path, device=args.device)
        models[mode] = {"ckpt": ckpt, "model": model, "meta": meta, "path": path}

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "test_npz": args.test_npz,
            "models": model_paths,
            "device": args.device,
            "x0_source": data.get("_x0_source", "unknown"),
        }, f, ensure_ascii=False, indent=2)

    if args.run_all or args.eval_basic:
        print("[1] basic evaluation")
        eval_basic(data, models, out_dir, device=args.device, newton_iter=args.max_newton_iter, tol=args.tol)

    if args.run_all or args.eval_newton_budget:
        print("[2] newton budget")
        eval_newton_budget(data, models, out_dir, device=args.device, max_budget=args.newton_budget_max, tol=args.tol)

    if args.run_all or args.eval_noise:
        print("[3] noise robustness")
        eval_noise(data, models, out_dir, device=args.device)

    if args.run_all or args.eval_ood:
        print("[4] ood analysis")
        eval_ood(data, models, out_dir, device=args.device, newton_iter=args.max_newton_iter, tol=args.tol)

    if args.run_all or args.eval_basin:
        print("[5] basin analysis")
        eval_basin(
            data, models, out_dir, device=args.device,
            n_cases=args.basin_cases,
            radius=args.basin_radius,
            grid_points=args.basin_grid_points,
            tol=args.tol,
            max_iter=args.max_newton_iter,
        )

    if args.run_all or args.eval_timing:
        print("[6] timing")
        eval_timing(data, models, out_dir, device=args.device, repeats=args.timing_repeats)

    print(f"[DONE] saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
colebrook_portable_experiments_fixed.py

설명
----
기존 colebrook_portable_experiments.py에서 direct prediction이 비정상적으로
깨지던 문제를 수정한 버전.

핵심 수정
--------
1) checkpoint의 args를 읽어 입력 feature 구성을 정확히 복원
2) checkpoint의 scaler(mean/std)를 적용
3) x0 key가 없으면 center를 자동으로 사용
4) dropout 포함 모델 구조를 정확히 복원

가정
----
- 식: x + 2 log10(a + b x) = 0
- NPZ key:
    coeffs, a, b, root, (x0 또는 center)
- checkpoint:
    train_colebrook_root.py 계열 best_model.pt
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


# =========================================================
# Utilities
# =========================================================
def percentile(x, q):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def save_csv(path: Path, rows: List[Dict]):
    if not rows:
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


# =========================================================
# Original Colebrook-like equation
# =========================================================
LN10 = math.log(10.0)


def colebrook_f(x, a, b):
    x = np.asarray(x, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = a + b * x
    out = np.full_like(x, np.nan, dtype=np.float64)
    mask = z > 0
    out[mask] = x[mask] + 2.0 * np.log10(z[mask])
    return out


def colebrook_df(x, a, b):
    x = np.asarray(x, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = a + b * x
    out = np.full_like(x, np.nan, dtype=np.float64)
    mask = z > 0
    out[mask] = 1.0 + 2.0 * b[mask] / (z[mask] * LN10)
    return out


def project_to_domain_np(x, a, b, eps=1e-10):
    left = -a / b + eps
    return np.maximum(x, left)


def newton_refine_original_vectorized(
    x0,
    a,
    b,
    tol=1e-12,
    max_iter=20,
    damping=1.0,
    deriv_eps=1e-14,
    max_step=5.0,
):
    x = project_to_domain_np(np.asarray(x0, dtype=np.float64).copy(), a, b, eps=1e-10)
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

        fx = colebrook_f(xa, aa, bb)
        dfx = colebrook_df(xa, aa, bb)

        small = np.abs(dfx) < deriv_eps
        dfx[small] = np.where(dfx[small] >= 0, deriv_eps, -deriv_eps)

        step = damping * fx / dfx
        step = np.clip(step, -max_step, max_step)
        x_new = xa - step
        x_new = project_to_domain_np(x_new, aa, bb, eps=1e-10)

        fx_new = np.abs(colebrook_f(x_new, aa, bb))
        worse = fx_new > np.abs(fx)
        if np.any(worse):
            x_half = xa[worse] - 0.5 * step[worse]
            x_half = project_to_domain_np(x_half, aa[worse], bb[worse], eps=1e-10)
            fx_half = np.abs(colebrook_f(x_half, aa[worse], bb[worse]))
            better_half = fx_half < fx_new[worse]
            x_new[worse] = np.where(better_half, x_half, x_new[worse])

        x[idx] = x_new
        iters_used[idx] = k

        fx_now = np.abs(colebrook_f(x_new, aa, bb))
        done = fx_now <= tol
        converged[idx[done]] = True
        active[idx[done]] = False

    return x, iters_used, converged


# =========================================================
# Model loading
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
        return self.net(x).squeeze(-1)


def infer_hidden_dims_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, List[int]]:
    linear_weights = []
    for k, v in state_dict.items():
        if k.endswith(".weight") and v.ndim == 2:
            linear_weights.append((k, v.shape))
    linear_weights = sorted(linear_weights, key=lambda kv: kv[0])

    if not linear_weights:
        raise RuntimeError("No linear layers found in state_dict")

    dims = []
    for _, shape in linear_weights:
        out_dim, in_dim = shape
        dims.append((in_dim, out_dim))

    input_dim = dims[0][0]
    hidden_dims = [out_dim for (_, out_dim) in dims[:-1]]
    return input_dim, hidden_dims


def load_model_checkpoint(path: str, device="cpu"):
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise RuntimeError("Unsupported checkpoint format")

    if isinstance(ckpt, dict) and "hidden_dims" in ckpt:
        hidden_dims = list(ckpt["hidden_dims"])
        input_dim = None
        for k, v in state_dict.items():
            if k.endswith(".weight") and v.ndim == 2:
                input_dim = int(v.shape[1])
                break
        if input_dim is None:
            input_dim, _ = infer_hidden_dims_from_state_dict(state_dict)
    else:
        input_dim, hidden_dims = infer_hidden_dims_from_state_dict(state_dict)

    dropout = 0.0
    if isinstance(ckpt, dict) and "args" in ckpt and isinstance(ckpt["args"], dict):
        dropout = float(ckpt["args"].get("dropout", 0.0))

    model = MLPRegressor(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return ckpt, model, {
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
    }


# =========================================================
# Data loading
# =========================================================
def load_npz(path: str):
    data = np.load(path, allow_pickle=True)
    available = list(data.keys())

    for k in ["coeffs", "a", "b", "root"]:
        if k not in data:
            raise KeyError(f"Missing key '{k}'; available: {available}")

    if "x0" in data:
        x0_arr = np.asarray(data["x0"], dtype=np.float64).reshape(-1)
        x0_source = "x0"
    elif "center" in data:
        x0_arr = np.asarray(data["center"], dtype=np.float64).reshape(-1)
        x0_source = "center"
    else:
        raise KeyError(f"Missing key 'x0' (or alias 'center'); available: {available}")

    out = {
        "coeffs": np.asarray(data["coeffs"], dtype=np.float64),
        "x0": x0_arr,
        "a": np.asarray(data["a"], dtype=np.float64).reshape(-1),
        "b": np.asarray(data["b"], dtype=np.float64).reshape(-1),
        "root": np.asarray(data["root"], dtype=np.float64).reshape(-1),
        "_x0_source": x0_source,
    }
    return out


# =========================================================
# Feature reconstruction from checkpoint args
# =========================================================
def build_features_from_checkpoint_args(
    coeffs: np.ndarray,
    x0: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    ckpt_args: Dict,
) -> np.ndarray:
    use_log_ab = bool(ckpt_args.get("use_log_ab", False))
    include_center = bool(ckpt_args.get("include_center", True))
    include_ab = bool(ckpt_args.get("include_ab", True))

    feats = [coeffs.astype(np.float32)]

    if include_center:
        feats.append(x0.reshape(-1, 1).astype(np.float32))

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


def predict_with_checkpoint(
    coeffs: np.ndarray,
    x0: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    ckpt: Dict,
    model,
    device="cpu",
    batch_size=4096,
) -> np.ndarray:
    if "args" not in ckpt or "scaler" not in ckpt:
        raise RuntimeError("Checkpoint must contain 'args' and 'scaler' for this fixed evaluator.")

    X = build_features_from_checkpoint_args(coeffs, x0, a, b, ckpt["args"])
    X = apply_scaler(X, ckpt["scaler"])

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).to(device)
            yb = model(xb).detach().cpu().numpy().reshape(-1)
            preds.append(yb)

    pred = np.concatenate(preds, axis=0)
    return pred.astype(np.float64)


# =========================================================
# Metrics
# =========================================================
def compute_metrics(pred, true_root, a, b):
    pred = np.asarray(pred, dtype=np.float64)
    true_root = np.asarray(true_root, dtype=np.float64)

    mae = float(np.mean(np.abs(pred - true_root)))
    rmse = float(np.sqrt(np.mean((pred - true_root) ** 2)))

    ss_res = float(np.sum((pred - true_root) ** 2))
    ss_tot = float(np.sum((true_root - np.mean(true_root)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    residual = np.abs(colebrook_f(pred, a, b))
    valid_mask = np.isfinite(residual)

    out = {
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2),
        "valid_ratio": float(np.mean(valid_mask)),
        "residual_mean": float(np.nanmean(residual)),
        "residual_median": float(np.nanmedian(residual)),
        "residual_p90": percentile(residual[np.isfinite(residual)], 90),
        "max_abs_error": float(np.max(np.abs(pred - true_root))),
    }
    return out


# =========================================================
# Baselines
# =========================================================
def baseline_zero_init(a):
    return np.zeros_like(a, dtype=np.float64)


def baseline_heuristic_init(a):
    return -2.0 * np.log10(np.clip(a, 1e-15, None))


# =========================================================
# Experiments
# =========================================================
def eval_basic(data, models, out_dir: Path, device="cpu", newton_iter=20, tol=1e-12):
    coeffs, x0, a, b, root = data["coeffs"], data["x0"], data["a"], data["b"], data["root"]
    rows = []

    zero = baseline_zero_init(a)
    rows.append({"name": "zero_init_direct", **compute_metrics(zero, root, a, b)})

    zref, ziter, zconv = newton_refine_original_vectorized(zero, a, b, max_iter=newton_iter, tol=tol)
    rows.append({
        "name": "zero_init_plus_newton",
        **compute_metrics(zref, root, a, b),
        "newton_iter_mean": float(np.mean(ziter)),
        "newton_iter_median": float(np.median(ziter)),
        "newton_iter_p90": percentile(ziter, 90),
        "newton_converged_ratio": float(np.mean(zconv)),
    })

    heur = baseline_heuristic_init(a)
    rows.append({"name": "heuristic_direct", **compute_metrics(heur, root, a, b)})

    href, hiter, hconv = newton_refine_original_vectorized(heur, a, b, max_iter=newton_iter, tol=tol)
    rows.append({
        "name": "heuristic_plus_newton",
        **compute_metrics(href, root, a, b),
        "newton_iter_mean": float(np.mean(hiter)),
        "newton_iter_median": float(np.median(hiter)),
        "newton_iter_p90": percentile(hiter, 90),
        "newton_converged_ratio": float(np.mean(hconv)),
    })

    for mode, pack in models.items():
        pred = predict_with_checkpoint(coeffs, x0, a, b, pack["ckpt"], pack["model"], device=device)
        rows.append({"name": f"{mode}_direct", **compute_metrics(pred, root, a, b)})

        pref, piter, pconv = newton_refine_original_vectorized(pred, a, b, max_iter=newton_iter, tol=tol)
        rows.append({
            "name": f"{mode}_plus_newton",
            **compute_metrics(pref, root, a, b),
            "newton_iter_mean": float(np.mean(piter)),
            "newton_iter_median": float(np.median(piter)),
            "newton_iter_p90": percentile(piter, 90),
            "newton_converged_ratio": float(np.mean(pconv)),
        })

    save_csv(out_dir / "basic_summary.csv", rows)


def eval_newton_budget(data, models, out_dir: Path, device="cpu", max_budget=10, tol=1e-12):
    coeffs, x0, a, b, root = data["coeffs"], data["x0"], data["a"], data["b"], data["root"]
    rows = []

    inits = {
        "zero_init": baseline_zero_init(a),
        "heuristic": baseline_heuristic_init(a),
    }
    for mode, pack in models.items():
        inits[mode] = predict_with_checkpoint(coeffs, x0, a, b, pack["ckpt"], pack["model"], device=device)

    for name, init in inits.items():
        for budget in range(max_budget + 1):
            if budget == 0:
                pred = init.copy()
                iters = np.zeros_like(root, dtype=np.int32)
                conv = np.isfinite(colebrook_f(pred, a, b)) & (np.abs(colebrook_f(pred, a, b)) <= tol)
            else:
                pred, iters, conv = newton_refine_original_vectorized(init, a, b, max_iter=budget, tol=tol)

            rows.append({
                "method": name,
                "newton_budget": budget,
                **compute_metrics(pred, root, a, b),
                "iter_mean": float(np.mean(iters)),
                "iter_median": float(np.median(iters)),
                "converged_ratio": float(np.mean(conv)),
            })

    save_csv(out_dir / "newton_budget.csv", rows)


def eval_noise(data, models, out_dir: Path, device="cpu"):
    coeffs, x0, a, b, root = data["coeffs"], data["x0"], data["a"], data["b"], data["root"]
    rows = []

    coeff_noise_list = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
    ab_noise_list = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
    coeff_std = np.std(coeffs, axis=0, keepdims=True) + 1e-12

    for mode, pack in models.items():
        for sigma in coeff_noise_list:
            noisy_coeffs = coeffs + np.random.randn(*coeffs.shape) * coeff_std * sigma
            pred = predict_with_checkpoint(noisy_coeffs, x0, a, b, pack["ckpt"], pack["model"], device=device)
            rows.append({
                "mode": mode,
                "noise_type": "coeff_gaussian",
                "noise_level": sigma,
                **compute_metrics(pred, root, a, b),
            })

        for sigma in ab_noise_list:
            na = np.clip(a * (1.0 + sigma * np.random.randn(*a.shape)), 1e-12, None)
            nb = np.clip(b * (1.0 + sigma * np.random.randn(*b.shape)), 1e-12, None)

            pred = predict_with_checkpoint(coeffs, x0, na, nb, pack["ckpt"], pack["model"], device=device)
            noisy_res = np.abs(colebrook_f(pred, na, nb))
            rows.append({
                "mode": mode,
                "noise_type": "ab_multiplicative",
                "noise_level": sigma,
                **compute_metrics(pred, root, a, b),
                "noisy_equation_residual_mean": float(np.nanmean(noisy_res)),
                "noisy_equation_residual_p90": percentile(noisy_res[np.isfinite(noisy_res)], 90),
            })

    save_csv(out_dir / "noise_robustness.csv", rows)


def make_ood_masks(data):
    a, b, root = data["a"], data["b"], data["root"]
    r_abs = np.abs(root)

    a_q80 = np.quantile(a, 0.8)
    b_q80 = np.quantile(b, 0.8)
    r_q80 = np.quantile(r_abs, 0.8)

    return {
        "ID_all": np.ones_like(a, dtype=bool),
        "OOD_large_a": a >= a_q80,
        "OOD_large_b": b >= b_q80,
        "OOD_large_root_abs": r_abs >= r_q80,
        "HARD_union": (a >= a_q80) | (b >= b_q80) | (r_abs >= r_q80),
        "EASY_intersection": (a < np.quantile(a, 0.5)) & (b < np.quantile(b, 0.5)) & (r_abs < np.quantile(r_abs, 0.5)),
    }


def eval_ood(data, models, out_dir: Path, device="cpu", newton_iter=20, tol=1e-12):
    coeffs, x0, a, b, root = data["coeffs"], data["x0"], data["a"], data["b"], data["root"]
    masks = make_ood_masks(data)
    rows = []

    pred_cache = {}
    for mode, pack in models.items():
        pred_cache[mode] = predict_with_checkpoint(coeffs, x0, a, b, pack["ckpt"], pack["model"], device=device)

    for split_name, mask in masks.items():
        if np.sum(mask) == 0:
            continue

        for init_name, init in {
            "zero_init": baseline_zero_init(a),
            "heuristic": baseline_heuristic_init(a),
        }.items():
            rows.append({
                "split": split_name,
                "method": f"{init_name}_direct",
                "n": int(np.sum(mask)),
                **compute_metrics(init[mask], root[mask], a[mask], b[mask]),
            })

            ref, iters, conv = newton_refine_original_vectorized(init[mask], a[mask], b[mask], max_iter=newton_iter, tol=tol)
            rows.append({
                "split": split_name,
                "method": f"{init_name}_plus_newton",
                "n": int(np.sum(mask)),
                **compute_metrics(ref, root[mask], a[mask], b[mask]),
                "iter_mean": float(np.mean(iters)),
                "iter_median": float(np.median(iters)),
                "converged_ratio": float(np.mean(conv)),
            })

        for mode, pred_all in pred_cache.items():
            rows.append({
                "split": split_name,
                "method": f"{mode}_direct",
                "n": int(np.sum(mask)),
                **compute_metrics(pred_all[mask], root[mask], a[mask], b[mask]),
            })

            ref, iters, conv = newton_refine_original_vectorized(pred_all[mask], a[mask], b[mask], max_iter=newton_iter, tol=tol)
            rows.append({
                "split": split_name,
                "method": f"{mode}_plus_newton",
                "n": int(np.sum(mask)),
                **compute_metrics(ref, root[mask], a[mask], b[mask]),
                "iter_mean": float(np.mean(iters)),
                "iter_median": float(np.median(iters)),
                "converged_ratio": float(np.mean(conv)),
            })

    save_csv(out_dir / "ood_analysis.csv", rows)


def eval_basin(data, models, out_dir: Path, device="cpu", n_cases=25, radius=1.0, grid_points=81, tol=1e-12, max_iter=20):
    a, b, root = data["a"], data["b"], data["root"]
    coeffs, x0 = data["coeffs"], data["x0"]

    idxs = np.linspace(0, len(root) - 1, min(n_cases, len(root))).astype(int)
    rows = []

    pred_cache = {}
    for mode, pack in models.items():
        pred_cache[mode] = predict_with_checkpoint(coeffs, x0, a, b, pack["ckpt"], pack["model"], device=device)

    for idx in idxs:
        true_r = root[idx]
        grid = np.linspace(true_r - radius, true_r + radius, grid_points)
        ai = np.full(grid_points, a[idx], dtype=np.float64)
        bi = np.full(grid_points, b[idx], dtype=np.float64)

        ref, iters, conv = newton_refine_original_vectorized(grid, ai, bi, max_iter=max_iter, tol=tol)
        conv_to_true = np.abs(ref - true_r) <= 1e-6

        row = {
            "sample_index": int(idx),
            "a": float(a[idx]),
            "b": float(b[idx]),
            "true_root": float(true_r),
            "basin_ratio_around_true_root": float(np.mean(conv_to_true)),
            "grid_left": float(grid[0]),
            "grid_right": float(grid[-1]),
        }

        for mode, preds in pred_cache.items():
            row[f"{mode}_pred"] = float(preds[idx])
            row[f"{mode}_abs_init_error"] = float(abs(preds[idx] - true_r))

        rows.append(row)

    save_csv(out_dir / "basin_analysis.csv", rows)


def time_callable(fn, repeats=10000):
    t0 = time.perf_counter_ns()
    for _ in range(repeats):
        fn()
    t1 = time.perf_counter_ns()
    return (t1 - t0) / repeats / 1000.0  # us


def eval_timing(data, models, out_dir: Path, device="cpu", repeats=20000):
    coeffs, x0, a, b = data["coeffs"], data["x0"], data["a"], data["b"]
    idx = 0
    ai = np.array([a[idx]], dtype=np.float64)
    bi = np.array([b[idx]], dtype=np.float64)

    rows = []

    zero = baseline_zero_init(ai)
    heur = baseline_heuristic_init(ai)

    rows.append({"method": "zero_init_direct", "time_mean_us": time_callable(lambda: baseline_zero_init(ai), repeats)})
    rows.append({"method": "heuristic_direct", "time_mean_us": time_callable(lambda: baseline_heuristic_init(ai), repeats)})
    rows.append({"method": "zero_init_plus_newton", "time_mean_us": time_callable(lambda: newton_refine_original_vectorized(zero, ai, bi, max_iter=20, tol=1e-12), repeats)})
    rows.append({"method": "heuristic_plus_newton", "time_mean_us": time_callable(lambda: newton_refine_original_vectorized(heur, ai, bi, max_iter=20, tol=1e-12), repeats)})

    for mode, pack in models.items():
        coeffs1 = coeffs[idx:idx+1]
        x01 = x0[idx:idx+1]
        a1 = a[idx:idx+1]
        b1 = b[idx:idx+1]

        def pred_only():
            _ = predict_with_checkpoint(coeffs1, x01, a1, b1, pack["ckpt"], pack["model"], device=device, batch_size=1)

        pred = predict_with_checkpoint(coeffs1, x01, a1, b1, pack["ckpt"], pack["model"], device=device, batch_size=1)

        def pred_plus_newton():
            _ = newton_refine_original_vectorized(pred, ai, bi, max_iter=20, tol=1e-12)

        rows.append({"method": f"{mode}_direct", "time_mean_us": time_callable(pred_only, repeats)})
        rows.append({"method": f"{mode}_plus_newton_refine_only", "time_mean_us": time_callable(pred_plus_newton, repeats)})

    save_csv(out_dir / "timing_microbench.csv", rows)


# =========================================================
# CLI
# =========================================================
def parse_model_args(items: List[str]) -> Dict[str, str]:
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--model must be mode=path, got {item}")
        mode, path = item.split("=", 1)
        out[mode] = path
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_npz", required=True)
    parser.add_argument("--model", nargs="+", required=True, help="e.g. coeffs_x0_ab=./runs/model.pt")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max_newton_iter", type=int, default=20)

    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--eval_basic", action="store_true")
    parser.add_argument("--eval_newton_budget", action="store_true")
    parser.add_argument("--eval_noise", action="store_true")
    parser.add_argument("--eval_ood", action="store_true")
    parser.add_argument("--eval_basin", action="store_true")
    parser.add_argument("--eval_timing", action="store_true")

    parser.add_argument("--newton_budget_max", type=int, default=10)
    parser.add_argument("--basin_cases", type=int, default=25)
    parser.add_argument("--basin_radius", type=float, default=1.0)
    parser.add_argument("--basin_grid_points", type=int, default=81)
    parser.add_argument("--timing_repeats", type=int, default=20000)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_npz(args.test_npz)
    model_paths = parse_model_args(args.model)

    models = {}
    for mode, path in model_paths.items():
        ckpt, model, meta = load_model_checkpoint(path, device=args.device)
        models[mode] = {"ckpt": ckpt, "model": model, "meta": meta, "path": path}

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "test_npz": args.test_npz,
            "models": model_paths,
            "device": args.device,
            "x0_source": data.get("_x0_source", "unknown"),
        }, f, ensure_ascii=False, indent=2)

    if args.run_all or args.eval_basic:
        print("[1] basic evaluation")
        eval_basic(data, models, out_dir, device=args.device, newton_iter=args.max_newton_iter, tol=args.tol)

    if args.run_all or args.eval_newton_budget:
        print("[2] newton budget")
        eval_newton_budget(data, models, out_dir, device=args.device, max_budget=args.newton_budget_max, tol=args.tol)

    if args.run_all or args.eval_noise:
        print("[3] noise robustness")
        eval_noise(data, models, out_dir, device=args.device)

    if args.run_all or args.eval_ood:
        print("[4] ood analysis")
        eval_ood(data, models, out_dir, device=args.device, newton_iter=args.max_newton_iter, tol=args.tol)

    if args.run_all or args.eval_basin:
        print("[5] basin analysis")
        eval_basin(
            data, models, out_dir, device=args.device,
            n_cases=args.basin_cases,
            radius=args.basin_radius,
            grid_points=args.basin_grid_points,
            tol=args.tol,
            max_iter=args.max_newton_iter,
        )

    if args.run_all or args.eval_timing:
        print("[6] timing")
        eval_timing(data, models, out_dir, device=args.device, repeats=args.timing_repeats)

    print(f"[DONE] saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
