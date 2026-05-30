#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
measure_baseline_cpu_ms_per_sample.py

Purpose
-------
Classical / nonlearned baseline initializers for the coupled two-branch Colebrook system:
  1) heuristic
  2) cond_haaland
  3) cond_swamee_jain
  4) cond_serghides

This script measures CPU-only runtime in ms/sample.

Important
---------
- No PyTorch.
- No CUDA.
- No neural model.
- No Taylor coefficients are used.
- Degree is not required for the baseline itself.
- Use one representative held-out test NPZ, e.g. degree-25 test set.

Measured values
---------------
For each initializer:

1. init_cpu_ms_per_sample
   Time to construct the initial guess z0 = (Q1, x1, x2).

2. newton_cpu_ms_per_sample
   Time to apply coupled Newton refinement starting from z0.

3. total_cpu_ms_per_sample
   init time + Newton time.

4. direct metrics
   Accuracy/residual of z0 before Newton.

5. plus_newton metrics
   Accuracy/residual after Newton refinement.

Input NPZ required keys
-----------------------
target, Q_total, D1, D2, eps1, eps2, L1, L2, rho, mu, g

Recommended paper usage
-----------------------
Use:
  --max_parallel 1
  --cpu_threads 1
  --repeats 3 or 5

Output
------
out_dir/
  baseline_cpu_all_raw.csv
  baseline_cpu_paper_table.csv
  baseline_cpu_paper_table.md
  baseline_cpu_paper_table.tex
"""

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np


PI = math.pi
LN10 = math.log(10.0)


# =========================================================
# Utility
# =========================================================
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


def percentile(x, q):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


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


def fmt_float(x, digits=4):
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
        vals = [str(row.get(c, "")) for c in columns]
        lines.append("| " + " | ".join(vals) + " |")
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
# Data
# =========================================================
def load_npz(path: str):
    data = np.load(path, allow_pickle=True)

    required = [
        "target",
        "Q_total",
        "D1",
        "D2",
        "eps1",
        "eps2",
        "L1",
        "L2",
        "rho",
        "mu",
        "g",
    ]

    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(
            f"Missing keys in {path}: {missing}\n"
            f"Available keys: {data.files}"
        )

    return {k: np.asarray(data[k], dtype=np.float64) for k in required}


# =========================================================
# Colebrook explicit formulas
# =========================================================
def reynolds_from_q(Q, D, rho, mu):
    return 4.0 * rho * Q / (PI * mu * D)


def x_haaland_from_Q(Q, D, eps, rho, mu):
    Re = reynolds_from_q(Q, D, rho, mu)
    rr = eps / D

    Re = np.maximum(Re, 1e-12)
    inside = (rr / 3.7) ** 1.11 + 6.9 / Re
    inside = np.maximum(inside, 1e-300)

    x = -1.8 * np.log10(inside)
    return np.maximum(x, 1e-6).astype(np.float64)


def x_swamee_jain_from_Q(Q, D, eps, rho, mu):
    Re = reynolds_from_q(Q, D, rho, mu)
    rr = eps / D

    Re = np.maximum(Re, 1e-12)
    inside = rr / 3.7 + 5.74 / (Re ** 0.9)
    inside = np.maximum(inside, 1e-300)

    # f = 0.25 / log10(inside)^2
    # x = 1/sqrt(f) = -2 log10(inside), since inside < 1.
    x = -2.0 * np.log10(inside)
    return np.maximum(x, 1e-6).astype(np.float64)


def x_serghides_from_Q(Q, D, eps, rho, mu):
    Re = reynolds_from_q(Q, D, rho, mu)
    rr = eps / D

    Re = np.maximum(Re, 1e-12)
    base = rr / 3.7

    A = -2.0 * np.log10(np.maximum(base + 12.0 / Re, 1e-300))
    B = -2.0 * np.log10(np.maximum(base + 2.51 * A / Re, 1e-300))
    C = -2.0 * np.log10(np.maximum(base + 2.51 * B / Re, 1e-300))

    denom = C - 2.0 * B + A
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)

    x = A - ((B - A) ** 2) / denom
    return np.maximum(x, 1e-6).astype(np.float64)


def x_explicit_from_Q(method, Q, D, eps, rho, mu):
    if method == "cond_haaland":
        return x_haaland_from_Q(Q, D, eps, rho, mu)
    if method == "cond_swamee_jain":
        return x_swamee_jain_from_Q(Q, D, eps, rho, mu)
    if method == "cond_serghides":
        return x_serghides_from_Q(Q, D, eps, rho, mu)
    raise ValueError(f"Unknown explicit method: {method}")


# =========================================================
# Heuristic branch Colebrook solver
# =========================================================
def colebrook_F_x(x, Q, D, eps, rho, mu):
    Re = 4.0 * rho * Q / (PI * mu * D)
    rr = eps / D
    inside = rr / 3.7 + 2.51 * x / Re

    if inside <= 0 or not math.isfinite(inside):
        return float("nan")

    return x + 2.0 * math.log10(inside)


def colebrook_dFdx_x(x, Q, D, eps, rho, mu):
    Re = 4.0 * rho * Q / (PI * mu * D)
    rr = eps / D
    inside = rr / 3.7 + 2.51 * x / Re

    if inside <= 0 or not math.isfinite(inside):
        return float("nan")

    return 1.0 + 2.0 * (2.51 / Re) / (LN10 * inside)


def solve_x_from_Q_scalar(Q, D, eps, rho, mu, tol=1e-12, max_iter=30):
    x = float(x_haaland_from_Q(
        np.array([Q], dtype=np.float64),
        np.array([D], dtype=np.float64),
        np.array([eps], dtype=np.float64),
        np.array([rho], dtype=np.float64),
        np.array([mu], dtype=np.float64),
    )[0])

    if not math.isfinite(x) or x <= 0:
        x = 7.0

    for _ in range(max_iter):
        f = colebrook_F_x(x, Q, D, eps, rho, mu)
        df = colebrook_dFdx_x(x, Q, D, eps, rho, mu)

        if not math.isfinite(f) or not math.isfinite(df) or abs(df) < 1e-15:
            break

        step = f / df
        step = max(min(step, 5.0), -5.0)

        x_new = x - step
        if not math.isfinite(x_new) or x_new <= 0:
            x_new = 0.5 * x

        x = x_new

        if abs(f) <= tol:
            break

    return max(float(x), 1e-6)


def solve_x_from_Q_array(Q, D, eps, rho, mu, tol=1e-12, max_iter=30):
    Q = np.asarray(Q, dtype=np.float64)
    out = np.zeros_like(Q, dtype=np.float64)

    for i in range(len(Q)):
        out[i] = solve_x_from_Q_scalar(
            float(Q[i]),
            float(D[i]),
            float(eps[i]),
            float(rho[i]),
            float(mu[i]),
            tol=tol,
            max_iter=max_iter,
        )

    return out


# =========================================================
# Initializers
# =========================================================
def build_heuristic_initializer(data: Dict[str, np.ndarray]):
    QT = data["Q_total"]
    Q1 = 0.5 * QT
    Q2 = QT - Q1

    x1 = solve_x_from_Q_array(Q1, data["D1"], data["eps1"], data["rho"], data["mu"])
    x2 = solve_x_from_Q_array(Q2, data["D2"], data["eps2"], data["rho"], data["mu"])

    return np.stack([Q1, x1, x2], axis=1).astype(np.float64)


def build_conductance_explicit_initializer(data: Dict[str, np.ndarray], method: str):
    QT = data["Q_total"]
    Q_half = 0.5 * QT

    x1_half = x_explicit_from_Q(method, Q_half, data["D1"], data["eps1"], data["rho"], data["mu"])
    x2_half = x_explicit_from_Q(method, Q_half, data["D2"], data["eps2"], data["rho"], data["mu"])

    C1 = np.sqrt(
        data["g"] * (PI ** 2) * (data["D1"] ** 5) * (x1_half ** 2)
        / np.maximum(8.0 * data["L1"], 1e-300)
    )
    C2 = np.sqrt(
        data["g"] * (PI ** 2) * (data["D2"] ** 5) * (x2_half ** 2)
        / np.maximum(8.0 * data["L2"], 1e-300)
    )

    denom = np.maximum(C1 + C2, 1e-300)
    Q1 = QT * C1 / denom

    min_q = np.maximum(QT * 1e-6, 1e-12)
    Q1 = np.clip(Q1, min_q, QT - min_q)
    Q2 = QT - Q1

    x1 = x_explicit_from_Q(method, Q1, data["D1"], data["eps1"], data["rho"], data["mu"])
    x2 = x_explicit_from_Q(method, Q2, data["D2"], data["eps2"], data["rho"], data["mu"])

    return np.stack([Q1, x1, x2], axis=1).astype(np.float64)


def build_initializer(data: Dict[str, np.ndarray], name: str):
    if name == "heuristic":
        return build_heuristic_initializer(data)

    if name in ["cond_haaland", "cond_swamee_jain", "cond_serghides"]:
        return build_conductance_explicit_initializer(data, name)

    raise ValueError(f"Unknown initializer: {name}")


def initializer_display_name(name: str):
    mapping = {
        "heuristic": "Heuristic",
        "cond_haaland": "Cond.-Haaland",
        "cond_swamee_jain": "Cond.-Swamee-Jain",
        "cond_serghides": "Cond.-Serghides",
    }
    return mapping.get(name, name)


# =========================================================
# Coupled system and Newton
# =========================================================
def system_F(z, params):
    z = np.asarray(z, dtype=np.float64)

    Q1 = z[..., 0]
    x1 = z[..., 1]
    x2 = z[..., 2]

    QT = params["Q_total"]
    D1 = params["D1"]
    D2 = params["D2"]
    eps1 = params["eps1"]
    eps2 = params["eps2"]
    L1 = params["L1"]
    L2 = params["L2"]
    rho = params["rho"]
    mu = params["mu"]
    g = params["g"]

    Q2 = QT - Q1

    Re1 = 4.0 * rho * Q1 / (PI * mu * D1)
    Re2 = 4.0 * rho * Q2 / (PI * mu * D2)

    rr1 = eps1 / D1
    rr2 = eps2 / D2

    z1 = rr1 / 3.7 + 2.51 * x1 / Re1
    z2 = rr2 / 3.7 + 2.51 * x2 / Re2

    F1 = np.full_like(Q1, np.nan, dtype=np.float64)
    F2 = np.full_like(Q2, np.nan, dtype=np.float64)

    m1 = (Q1 > 0) & (Re1 > 0) & (z1 > 0) & np.isfinite(z1)
    m2 = (Q2 > 0) & (Re2 > 0) & (z2 > 0) & np.isfinite(z2)

    F1[m1] = x1[m1] + 2.0 * np.log10(z1[m1])
    F2[m2] = x2[m2] + 2.0 * np.log10(z2[m2])

    H1 = 8.0 * L1 * (Q1 ** 2) / (g * (PI ** 2) * (D1 ** 5) * (x1 ** 2))
    H2 = 8.0 * L2 * (Q2 ** 2) / (g * (PI ** 2) * (D2 ** 5) * (x2 ** 2))
    F3 = H1 - H2

    return np.stack([F1, F2, F3], axis=-1)


def params_for_index(data, i):
    return {
        k: float(data[k][i])
        for k in ["Q_total", "D1", "D2", "eps1", "eps2", "L1", "L2", "rho", "mu", "g"]
    }


def system_F_single(z, p):
    zz = np.asarray(z, dtype=np.float64).reshape(1, 3)
    pp = {k: np.asarray([v], dtype=np.float64) for k, v in p.items()}
    return system_F(zz, pp)[0]


def numerical_jacobian_single(z, p, eps=1e-6):
    z = np.asarray(z, dtype=np.float64)
    J = np.zeros((3, 3), dtype=np.float64)

    for j in range(3):
        step = eps * max(1.0, abs(z[j]))

        zp = z.copy()
        zm = z.copy()

        zp[j] += step
        zm[j] -= step

        fp = system_F_single(zp, p)
        fm = system_F_single(zm, p)

        J[:, j] = (fp - fm) / (2.0 * step)

    f0 = system_F_single(z, p)
    return J, f0


def project_feasible(z, p):
    z = np.asarray(z, dtype=np.float64).copy()

    QT = float(p["Q_total"])
    min_q = max(1e-10, QT * 1e-6)

    z[0] = np.clip(z[0], min_q, QT - min_q)
    z[1] = max(z[1], 1e-6)
    z[2] = max(z[2], 1e-6)

    return z


def newton_system_single(z0, p, tol=1e-12, max_iter=20, damping=1.0):
    z = project_feasible(z0, p)

    converged = False
    used_iter = 0

    for k in range(1, max_iter + 1):
        J, f = numerical_jacobian_single(z, p)

        if not np.all(np.isfinite(J)) or not np.all(np.isfinite(f)):
            break

        try:
            step = np.linalg.solve(J, f)
        except np.linalg.LinAlgError:
            break

        if not np.all(np.isfinite(step)):
            break

        step = np.clip(step, -5.0, 5.0)

        z_new = project_feasible(z - damping * step, p)
        f_new = system_F_single(z_new, p)

        if not np.all(np.isfinite(f_new)):
            break

        # Damping safeguard
        if np.linalg.norm(f_new, ord=2) > np.linalg.norm(f, ord=2):
            z_half = project_feasible(z - 0.5 * damping * step, p)
            f_half = system_F_single(z_half, p)

            if np.all(np.isfinite(f_half)) and np.linalg.norm(f_half, ord=2) < np.linalg.norm(f_new, ord=2):
                z_new = z_half
                f_new = f_half

        z = z_new
        used_iter = k

        if np.linalg.norm(f_new, ord=np.inf) <= tol:
            converged = True
            break

    return z, used_iter, converged


def refine_batch(z_init, data, tol=1e-12, max_iter=20):
    z_init = np.asarray(z_init, dtype=np.float64)

    n = len(z_init)
    out = np.zeros_like(z_init, dtype=np.float64)
    iters = np.zeros(n, dtype=np.int32)
    conv = np.zeros(n, dtype=bool)

    for i in range(n):
        p = params_for_index(data, i)
        zf, it, ok = newton_system_single(
            z_init[i],
            p,
            tol=tol,
            max_iter=max_iter,
        )
        out[i] = zf
        iters[i] = it
        conv[i] = ok

    return out, iters, conv


# =========================================================
# Metrics
# =========================================================
def regression_metrics(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)

    err = pred - true

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mae_Q1": float(np.mean(np.abs(err[:, 0]))),
        "mae_x1": float(np.mean(np.abs(err[:, 1]))),
        "mae_x2": float(np.mean(np.abs(err[:, 2]))),
        "max_abs_error": float(np.max(np.abs(err))),
    }


def residual_metrics(pred, data):
    params = {
        k: np.asarray(data[k], dtype=np.float64)
        for k in ["Q_total", "D1", "D2", "eps1", "eps2", "L1", "L2", "rho", "mu", "g"]
    }

    F = system_F(np.asarray(pred, dtype=np.float64), params)

    valid = np.all(np.isfinite(F), axis=1)
    norms_inf = np.max(np.abs(F), axis=1)
    finite = norms_inf[np.isfinite(norms_inf)]

    if finite.size == 0:
        return {
            "valid_ratio": float(np.mean(valid)),
            "residual_mean": float("inf"),
            "residual_median": float("inf"),
            "residual_p90": float("inf"),
        }

    return {
        "valid_ratio": float(np.mean(valid)),
        "residual_mean": float(np.mean(finite)),
        "residual_median": float(np.median(finite)),
        "residual_p90": percentile(finite, 90),
    }


def combined_metrics(pred, true, data):
    m = regression_metrics(pred, true)
    m.update(residual_metrics(pred, data))
    return m


# =========================================================
# Timing
# =========================================================
def time_call_ms_per_sample(fn, n_samples: int, repeats: int, warmup: int = 0):
    for _ in range(warmup):
        _ = fn()

    times_ms_per_sample = []
    last = None

    for _ in range(repeats):
        t0 = time.perf_counter()
        last = fn()
        t1 = time.perf_counter()

        times_ms_per_sample.append((t1 - t0) * 1000.0 / n_samples)

    return {
        "ms_per_sample_mean": float(np.mean(times_ms_per_sample)),
        "ms_per_sample_std": float(np.std(times_ms_per_sample)),
        "ms_per_sample_min": float(np.min(times_ms_per_sample)),
        "ms_per_sample_max": float(np.max(times_ms_per_sample)),
        "last": last,
    }


# =========================================================
# Evaluation
# =========================================================
def evaluate_initializer(
    data,
    initializer: str,
    repeats: int,
    warmup: int,
    tol: float,
    max_newton_iter: int,
):
    true = np.asarray(data["target"], dtype=np.float64)
    n_samples = len(true)

    def init_fn():
        return build_initializer(data, initializer)

    init_time = time_call_ms_per_sample(
        init_fn,
        n_samples=n_samples,
        repeats=repeats,
        warmup=warmup,
    )
    z0 = init_time["last"]

    direct_metrics = combined_metrics(z0, true, data)

    def newton_fn():
        return refine_batch(
            z0,
            data,
            tol=tol,
            max_iter=max_newton_iter,
        )

    newton_time = time_call_ms_per_sample(
        newton_fn,
        n_samples=n_samples,
        repeats=repeats,
        warmup=0,
    )
    z_newton, iters, conv = newton_time["last"]

    plus_metrics = combined_metrics(z_newton, true, data)
    plus_metrics["newton_iter_mean"] = float(np.mean(iters))
    plus_metrics["newton_iter_median"] = float(np.median(iters))
    plus_metrics["newton_iter_p90"] = float(np.percentile(iters, 90))
    plus_metrics["newton_converged_ratio"] = float(np.mean(conv))

    total_ms_per_sample_mean = (
        init_time["ms_per_sample_mean"]
        + newton_time["ms_per_sample_mean"]
    )

    total_ms_per_sample_std = math.sqrt(
        init_time["ms_per_sample_std"] ** 2
        + newton_time["ms_per_sample_std"] ** 2
    )

    row = {
        "initializer": initializer,
        "initializer_display": initializer_display_name(initializer),
        "n_samples": n_samples,
        "repeats": repeats,
        "warmup": warmup,
        "tol": tol,
        "max_newton_iter": max_newton_iter,

        # Direct metrics
        "direct_mae": direct_metrics["mae"],
        "direct_rmse": direct_metrics["rmse"],
        "direct_r2": direct_metrics["r2"],
        "direct_valid_ratio": direct_metrics["valid_ratio"],
        "direct_residual_mean": direct_metrics["residual_mean"],
        "direct_residual_median": direct_metrics["residual_median"],
        "direct_residual_p90": direct_metrics["residual_p90"],

        # Newton metrics
        "plus_newton_mae": plus_metrics["mae"],
        "plus_newton_rmse": plus_metrics["rmse"],
        "plus_newton_r2": plus_metrics["r2"],
        "plus_newton_valid_ratio": plus_metrics["valid_ratio"],
        "plus_newton_residual_mean": plus_metrics["residual_mean"],
        "plus_newton_residual_median": plus_metrics["residual_median"],
        "plus_newton_residual_p90": plus_metrics["residual_p90"],
        "plus_newton_newton_iter_mean": plus_metrics["newton_iter_mean"],
        "plus_newton_newton_iter_median": plus_metrics["newton_iter_median"],
        "plus_newton_newton_iter_p90": plus_metrics["newton_iter_p90"],
        "plus_newton_converged_ratio": plus_metrics["newton_converged_ratio"],

        # CPU timing, already normalized by n_samples
        "init_cpu_ms_per_sample_mean": init_time["ms_per_sample_mean"],
        "init_cpu_ms_per_sample_std": init_time["ms_per_sample_std"],
        "init_cpu_ms_per_sample_min": init_time["ms_per_sample_min"],
        "init_cpu_ms_per_sample_max": init_time["ms_per_sample_max"],

        "newton_cpu_ms_per_sample_mean": newton_time["ms_per_sample_mean"],
        "newton_cpu_ms_per_sample_std": newton_time["ms_per_sample_std"],
        "newton_cpu_ms_per_sample_min": newton_time["ms_per_sample_min"],
        "newton_cpu_ms_per_sample_max": newton_time["ms_per_sample_max"],

        "total_cpu_ms_per_sample_mean": total_ms_per_sample_mean,
        "total_cpu_ms_per_sample_std": total_ms_per_sample_std,
    }

    return row


def make_paper_rows(rows):
    out = []

    for r in rows:
        out.append({
            "Initializer": r["initializer_display"],
            "N Samples": r["n_samples"],
            "Direct RMSE": fmt_sci(r["direct_rmse"], 3),
            "Direct Residual Mean": fmt_sci(r["direct_residual_mean"], 3),
            "Final RMSE": fmt_sci(r["plus_newton_rmse"], 3),
            "Final Residual Mean": fmt_sci(r["plus_newton_residual_mean"], 3),
            "Init CPU ms/sample": fmt_ms(r["init_cpu_ms_per_sample_mean"], 5),
            "Newton CPU ms/sample": fmt_ms(r["newton_cpu_ms_per_sample_mean"], 5),
            "Total CPU ms/sample": fmt_ms(r["total_cpu_ms_per_sample_mean"], 5),
            "Iter. Mean": fmt_float(r["plus_newton_newton_iter_mean"], 4),
            "Conv. Ratio": fmt_ratio(r["plus_newton_converged_ratio"], 5),
        })

    return out


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_npz",
        type=str,
        required=True,
        help="Representative held-out test NPZ. Example: multi_colebrook_data_deg25/parallel2_colebrook_deg25_test.npz",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--initializers",
        nargs="+",
        default=["heuristic", "cond_haaland", "cond_swamee_jain", "cond_serghides"],
        choices=["heuristic", "cond_haaland", "cond_swamee_jain", "cond_serghides"],
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timing repeats. Use 3 or 5 for paper reporting.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup repeats for initializer construction only. Warmup is not included in timing.",
    )
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max_newton_iter", type=int, default=20)
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=1,
        help="For reproducible CPU timing, 1 is recommended.",
    )

    args = parser.parse_args()

    # CPU-only reproducibility
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.cpu_threads)

    test_npz = Path(args.test_npz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not test_npz.exists():
        raise FileNotFoundError(f"test_npz not found: {test_npz}")

    print("[INFO] CPU-only classical baseline timing")
    print(f"[INFO] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    print(f"[INFO] test_npz={test_npz}")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] initializers={args.initializers}")
    print(f"[INFO] repeats={args.repeats}")
    print(f"[INFO] warmup={args.warmup}")
    print(f"[INFO] cpu_threads={args.cpu_threads}")
    print(f"[INFO] tol={args.tol}")
    print(f"[INFO] max_newton_iter={args.max_newton_iter}")

    data = load_npz(str(test_npz))
    n_samples = len(data["target"])
    print(f"[INFO] n_samples={n_samples}")

    all_rows = []

    for initializer in args.initializers:
        print("\n" + "=" * 80)
        print(f"[RUN] initializer={initializer_display_name(initializer)}")

        row = evaluate_initializer(
            data=data,
            initializer=initializer,
            repeats=args.repeats,
            warmup=args.warmup,
            tol=args.tol,
            max_newton_iter=args.max_newton_iter,
        )

        all_rows.append(row)

        print(
            f"[RESULT] {initializer_display_name(initializer)} | "
            f"direct_rmse={row['direct_rmse']:.3e} | "
            f"final_rmse={row['plus_newton_rmse']:.3e} | "
            f"init_ms/sample={row['init_cpu_ms_per_sample_mean']:.5f} | "
            f"newton_ms/sample={row['newton_cpu_ms_per_sample_mean']:.5f} | "
            f"total_ms/sample={row['total_cpu_ms_per_sample_mean']:.5f} | "
            f"iter={row['plus_newton_newton_iter_mean']:.4f} | "
            f"conv={row['plus_newton_converged_ratio']:.5f}"
        )

    save_csv(out_dir / "baseline_cpu_all_raw.csv", all_rows)
    save_json(out_dir / "baseline_cpu_all_raw.json", all_rows)

    paper_rows = make_paper_rows(all_rows)
    paper_columns = [
        "Initializer",
        "N Samples",
        "Direct RMSE",
        "Direct Residual Mean",
        "Final RMSE",
        "Final Residual Mean",
        "Init CPU ms/sample",
        "Newton CPU ms/sample",
        "Total CPU ms/sample",
        "Iter. Mean",
        "Conv. Ratio",
    ]

    save_csv(out_dir / "baseline_cpu_paper_table.csv", paper_rows)

    (out_dir / "baseline_cpu_paper_table.md").write_text(
        markdown_table(paper_rows, paper_columns),
        encoding="utf-8",
    )

    (out_dir / "baseline_cpu_paper_table.tex").write_text(
        latex_table(
            paper_rows,
            paper_columns,
            caption=(
                "CPU-only runtime of nonlearned baseline initializers with Newton refinement. "
                "All times are reported in milliseconds per test sample and include only CPU execution."
            ),
            label="tab:baseline_cpu_ms_per_sample",
        ),
        encoding="utf-8",
    )

    print("\n[DONE]")
    print("Saved:")
    for name in [
        "baseline_cpu_all_raw.csv",
        "baseline_cpu_all_raw.json",
        "baseline_cpu_paper_table.csv",
        "baseline_cpu_paper_table.md",
        "baseline_cpu_paper_table.tex",
    ]:
        print(" -", out_dir / name)


if __name__ == "__main__":
    main()