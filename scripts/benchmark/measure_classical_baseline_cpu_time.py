#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
measure_classical_baseline_cpu_time.py

목적
----
Coupled two-branch Colebrook pipe-flow dataset에서
학습 모델 없이 classical/nonlearned initializer들의 CPU runtime을 측정한다.

측정 대상
---------
1. heuristic
   - Q1 = Q_total / 2
   - 각 branch의 x1, x2는 해당 Q에서 Colebrook 방정식을 CPU Newton으로 풂
   - 이후 coupled Newton refinement 적용

2. cond_haaland
   - Q_total/2에서 Haaland explicit x를 계산
   - branch conductance로 Q1 split 계산
   - split Q1, Q2에서 explicit x1, x2 재계산
   - 이후 coupled Newton refinement 적용

3. cond_swamee_jain
   - 위와 동일하되 Swamee-Jain explicit formula 사용

4. cond_serghides
   - 위와 동일하되 Serghides explicit formula 사용

출력
----
out_dir/
  classical_cpu_time_all_raw.csv
  classical_cpu_time_paper_table.csv
  classical_cpu_time_paper_table.md
  classical_cpu_time_paper_table.tex
  classical_cpu_time_by_initializer.csv
  classical_cpu_time_by_initializer.md
  classical_cpu_time_by_initializer.tex

논문에서 쓰기 좋은 열
---------------------
Degree
Initializer
Direct RMSE
Direct Residual Mean
Final RMSE
Final Residual Mean
Init ms/sample
Newton ms/sample
Total ms/sample
Iter. Mean
Converged Ratio

주의
----
- publishable CPU timing은 --max_parallel 1 권장.
- --max_parallel > 1은 빠르게 여러 degree를 돌리는 용도지만, CPU contention으로 timing이 왜곡될 수 있음.
"""

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple

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


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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


def percentile(x, q):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


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

    out = {k: np.asarray(data[k], dtype=np.float64) for k in required}
    return out


def dataset_path_for_degree(data_root: str, degree: int):
    return (
        Path(data_root)
        / f"multi_colebrook_data_deg{degree}"
        / f"parallel2_colebrook_deg{degree}_test.npz"
    )


# =========================================================
# Colebrook explicit formulas
# =========================================================
def reynolds_from_q(Q, D, rho, mu):
    Q = np.asarray(Q, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    return 4.0 * rho * Q / (PI * mu * D)


def x_haaland_from_Q(Q, D, eps, rho, mu):
    Re = reynolds_from_q(Q, D, rho, mu)
    rr = eps / D

    Re = np.maximum(Re, 1e-12)
    inside = (rr / 3.7) ** 1.11 + 6.9 / Re
    inside = np.maximum(inside, 1e-300)

    x = -1.8 * np.log10(inside)
    x = np.maximum(x, 1e-6)
    return x.astype(np.float64)


def x_swamee_jain_from_Q(Q, D, eps, rho, mu):
    Re = reynolds_from_q(Q, D, rho, mu)
    rr = eps / D

    Re = np.maximum(Re, 1e-12)
    inside = rr / 3.7 + 5.74 / (Re ** 0.9)
    inside = np.maximum(inside, 1e-300)

    # f = 0.25 / log10(inside)^2
    # x = 1/sqrt(f) = 2 * |log10(inside)|
    x = -2.0 * np.log10(inside)
    x = np.maximum(x, 1e-6)
    return x.astype(np.float64)


def x_serghides_from_Q(Q, D, eps, rho, mu):
    Re = reynolds_from_q(Q, D, rho, mu)
    rr = eps / D

    Re = np.maximum(Re, 1e-12)
    base = rr / 3.7

    A_in = np.maximum(base + 12.0 / Re, 1e-300)
    A = -2.0 * np.log10(A_in)

    B_in = np.maximum(base + 2.51 * A / Re, 1e-300)
    B = -2.0 * np.log10(B_in)

    C_in = np.maximum(base + 2.51 * B / Re, 1e-300)
    C = -2.0 * np.log10(C_in)

    denom = C - 2.0 * B + A
    denom = np.where(np.abs(denom) < 1e-12, np.sign(denom) * 1e-12 + 1e-12, denom)

    x = A - ((B - A) ** 2) / denom
    x = np.maximum(x, 1e-6)
    return x.astype(np.float64)


def x_explicit_from_Q(method, Q, D, eps, rho, mu):
    if method == "cond_haaland":
        return x_haaland_from_Q(Q, D, eps, rho, mu)
    if method == "cond_swamee_jain":
        return x_swamee_jain_from_Q(Q, D, eps, rho, mu)
    if method == "cond_serghides":
        return x_serghides_from_Q(Q, D, eps, rho, mu)
    raise ValueError(f"Unknown explicit method: {method}")


# =========================================================
# Heuristic branch Colebrook solve
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
    """
    Given Q and pipe parameters, solve x + 2 log10(rr/3.7 + 2.51 x/Re) = 0.
    This is used only for the heuristic baseline.
    """
    # Haaland gives a good starting point for x
    x = float(x_haaland_from_Q(
        np.array([Q]), np.array([D]), np.array([eps]), np.array([rho]), np.array([mu])
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

    x1 = solve_x_from_Q_array(
        Q1,
        data["D1"],
        data["eps1"],
        data["rho"],
        data["mu"],
    )
    x2 = solve_x_from_Q_array(
        Q2,
        data["D2"],
        data["eps2"],
        data["rho"],
        data["mu"],
    )

    return np.stack([Q1, x1, x2], axis=1).astype(np.float64)


def build_conductance_explicit_initializer(data: Dict[str, np.ndarray], method: str):
    """
    Conductance-based explicit initializer.

    Step 1:
      evaluate explicit x1, x2 at Q_total/2.

    Step 2:
      compute approximate branch conductance from
        H_i = 8 L_i Q_i^2 / (g pi^2 D_i^5 x_i^2)
      so
        Q_i = C_i sqrt(H),
        C_i = sqrt(g pi^2 D_i^5 x_i^2 / (8 L_i)).

    Step 3:
      split total flow by C1/(C1+C2).

    Step 4:
      recompute explicit x1, x2 at final Q1,Q2.
    """
    QT = data["Q_total"]
    Q_half = 0.5 * QT

    x1_half = x_explicit_from_Q(
        method,
        Q_half,
        data["D1"],
        data["eps1"],
        data["rho"],
        data["mu"],
    )
    x2_half = x_explicit_from_Q(
        method,
        Q_half,
        data["D2"],
        data["eps2"],
        data["rho"],
        data["mu"],
    )

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

    x1 = x_explicit_from_Q(
        method,
        Q1,
        data["D1"],
        data["eps1"],
        data["rho"],
        data["mu"],
    )
    x2 = x_explicit_from_Q(
        method,
        Q2,
        data["D2"],
        data["eps2"],
        data["rho"],
        data["mu"],
    )

    return np.stack([Q1, x1, x2], axis=1).astype(np.float64)


def build_initializer(data: Dict[str, np.ndarray], name: str):
    if name == "heuristic":
        return build_heuristic_initializer(data)
    if name in ["cond_haaland", "cond_swamee_jain", "cond_serghides"]:
        return build_conductance_explicit_initializer(data, name)
    raise ValueError(f"Unknown initializer: {name}")


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
        for k in [
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

        # one-step damping safeguard
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
        for k in [
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
# Timing helpers
# =========================================================
def time_call(fn, repeats=1):
    times = []
    last = None

    for _ in range(repeats):
        t0 = time.perf_counter()
        last = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return {
        "sec_mean": float(np.mean(times)),
        "sec_std": float(np.std(times)),
        "sec_min": float(np.min(times)),
        "sec_max": float(np.max(times)),
        "last": last,
    }


# =========================================================
# Degree evaluation
# =========================================================
def evaluate_initializer_for_degree(
    degree: int,
    data: Dict[str, np.ndarray],
    initializer: str,
    repeats: int,
    tol: float,
    max_newton_iter: int,
):
    true = np.asarray(data["target"], dtype=np.float64)
    n = len(true)

    def build_init():
        return build_initializer(data, initializer)

    init_t = time_call(build_init, repeats=repeats)
    z0 = init_t["last"]

    direct_metrics = combined_metrics(z0, true, data)

    def run_newton():
        return refine_batch(
            z0,
            data,
            tol=tol,
            max_iter=max_newton_iter,
        )

    newton_t = time_call(run_newton, repeats=repeats)
    refined, iters, conv = newton_t["last"]

    plus_metrics = combined_metrics(refined, true, data)
    plus_metrics["newton_iter_mean"] = float(np.mean(iters))
    plus_metrics["newton_iter_median"] = float(np.median(iters))
    plus_metrics["newton_iter_p90"] = float(np.percentile(iters, 90))
    plus_metrics["newton_converged_ratio"] = float(np.mean(conv))

    total_ms_per_sample = (
        init_t["sec_mean"] + newton_t["sec_mean"]
    ) * 1000.0 / n

    row = {
        "degree": degree,
        "initializer": initializer,
        "n_samples": n,

        "direct_mae": direct_metrics["mae"],
        "direct_rmse": direct_metrics["rmse"],
        "direct_r2": direct_metrics["r2"],
        "direct_valid_ratio": direct_metrics["valid_ratio"],
        "direct_residual_mean": direct_metrics["residual_mean"],
        "direct_residual_median": direct_metrics["residual_median"],
        "direct_residual_p90": direct_metrics["residual_p90"],

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

        "init_cpu_sec_mean": init_t["sec_mean"],
        "init_cpu_sec_std": init_t["sec_std"],
        "newton_cpu_sec_mean": newton_t["sec_mean"],
        "newton_cpu_sec_std": newton_t["sec_std"],

        "init_cpu_ms_per_sample": init_t["sec_mean"] * 1000.0 / n,
        "newton_cpu_ms_per_sample": newton_t["sec_mean"] * 1000.0 / n,
        "total_cpu_ms_per_sample": total_ms_per_sample,

        "repeats": repeats,
        "tol": tol,
        "max_newton_iter": max_newton_iter,
    }

    return row


def evaluate_degree_worker(args_tuple):
    degree, data_root, initializers, repeats, tol, max_newton_iter = args_tuple

    npz_path = dataset_path_for_degree(data_root, degree)
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing test NPZ: {npz_path}")

    print(f"[LOAD] degree={degree} path={npz_path}", flush=True)
    data = load_npz(str(npz_path))

    rows = []
    for initializer in initializers:
        print(f"[EVAL] degree={degree} initializer={initializer}", flush=True)
        row = evaluate_initializer_for_degree(
            degree=degree,
            data=data,
            initializer=initializer,
            repeats=repeats,
            tol=tol,
            max_newton_iter=max_newton_iter,
        )
        rows.append(row)

        print(
            f"[DONE] degree={degree} init={initializer} "
            f"direct_rmse={row['direct_rmse']:.3e} "
            f"plus_rmse={row['plus_newton_rmse']:.3e} "
            f"iter={row['plus_newton_newton_iter_mean']:.4f} "
            f"total_ms={row['total_cpu_ms_per_sample']:.5f}",
            flush=True,
        )

    return rows


# =========================================================
# Aggregation and paper tables
# =========================================================
def initializer_display_name(name):
    mapping = {
        "heuristic": "Heuristic + Newton",
        "cond_haaland": "Cond.-Haaland + Newton",
        "cond_swamee_jain": "Cond.-Swamee-Jain + Newton",
        "cond_serghides": "Cond.-Serghides + Newton",
    }
    return mapping.get(name, name)


def make_paper_rows(rows):
    out = []

    for r in rows:
        out.append({
            "Degree": r["degree"],
            "Initializer": initializer_display_name(r["initializer"]),
            "Direct RMSE": fmt_sci(r["direct_rmse"], 3),
            "Direct Residual Mean": fmt_sci(r["direct_residual_mean"], 3),
            "Final RMSE": fmt_sci(r["plus_newton_rmse"], 3),
            "Final Residual Mean": fmt_sci(r["plus_newton_residual_mean"], 3),
            "Init CPU ms/sample": fmt_ms(r["init_cpu_ms_per_sample"], 5),
            "Newton CPU ms/sample": fmt_ms(r["newton_cpu_ms_per_sample"], 5),
            "Total CPU ms/sample": fmt_ms(r["total_cpu_ms_per_sample"], 5),
            "Iter. Mean": fmt_float(r["plus_newton_newton_iter_mean"], 4),
            "Conv. Ratio": fmt_ratio(r["plus_newton_converged_ratio"], 5),
        })

    return out


def aggregate_by_initializer(rows):
    grouped = {}
    for r in rows:
        grouped.setdefault(r["initializer"], []).append(r)

    out = []

    metric_keys = [
        "direct_rmse",
        "direct_residual_mean",
        "plus_newton_rmse",
        "plus_newton_residual_mean",
        "init_cpu_ms_per_sample",
        "newton_cpu_ms_per_sample",
        "total_cpu_ms_per_sample",
        "plus_newton_newton_iter_mean",
        "plus_newton_converged_ratio",
    ]

    for initializer, sub in sorted(grouped.items()):
        row = {
            "initializer": initializer,
            "n_degrees": len(sub),
        }

        for key in metric_keys:
            vals = [safe_float(x.get(key)) for x in sub]
            vals = [v for v in vals if math.isfinite(v)]

            if vals:
                row[f"{key}_mean"] = float(np.mean(vals))
                row[f"{key}_std"] = float(np.std(vals))
                row[f"{key}_min"] = float(np.min(vals))
                row[f"{key}_max"] = float(np.max(vals))
            else:
                row[f"{key}_mean"] = float("nan")
                row[f"{key}_std"] = float("nan")
                row[f"{key}_min"] = float("nan")
                row[f"{key}_max"] = float("nan")

        out.append(row)

    return out


def make_initializer_average_paper_rows(summary):
    out = []

    for r in summary:
        out.append({
            "Initializer": initializer_display_name(r["initializer"]),
            "N Degrees": r["n_degrees"],
            "Avg Direct RMSE": fmt_sci(r["direct_rmse_mean"], 3),
            "Avg Direct Residual": fmt_sci(r["direct_residual_mean_mean"], 3),
            "Avg Final RMSE": fmt_sci(r["plus_newton_rmse_mean"], 3),
            "Avg Final Residual": fmt_sci(r["plus_newton_residual_mean_mean"], 3),
            "Avg Init CPU ms/sample": fmt_ms(r["init_cpu_ms_per_sample_mean"], 5),
            "Avg Newton CPU ms/sample": fmt_ms(r["newton_cpu_ms_per_sample_mean"], 5),
            "Avg Total CPU ms/sample": fmt_ms(r["total_cpu_ms_per_sample_mean"], 5),
            "Avg Iter.": fmt_float(r["plus_newton_newton_iter_mean_mean"], 4),
            "Avg Conv.": fmt_ratio(r["plus_newton_converged_ratio_mean"], 5),
        })

    return out


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/project/dataset/math_03_14",
        help="Root containing multi_colebrook_data_deg{N}/parallel2_colebrook_deg{N}_test.npz",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/root/project/dataset/math_03_14/classical_cpu_time_results",
    )
    parser.add_argument(
        "--degrees",
        nargs="+",
        type=int,
        default=[10, 15, 20, 25, 30, 35],
    )
    parser.add_argument(
        "--initializers",
        nargs="+",
        default=[
            "heuristic",
            "cond_haaland",
            "cond_swamee_jain",
            "cond_serghides",
        ],
        choices=[
            "heuristic",
            "cond_haaland",
            "cond_swamee_jain",
            "cond_serghides",
        ],
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Timing repeats. For publishable CPU total time, use 3 or more if runtime allows.",
    )
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max_newton_iter", type=int, default=20)
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=1,
        help="Parallel degree jobs. For accurate CPU timing, keep 1. For quick exploratory run, use >1.",
    )
    parser.add_argument(
        "--cpu_threads_per_job",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads_per_job))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.cpu_threads_per_job))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.cpu_threads_per_job))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(args.cpu_threads_per_job))

    print("[INFO] data_root:", args.data_root)
    print("[INFO] out_dir:", args.out_dir)
    print("[INFO] degrees:", args.degrees)
    print("[INFO] initializers:", args.initializers)
    print("[INFO] repeats:", args.repeats)
    print("[INFO] max_parallel:", args.max_parallel)
    print("[INFO] tol:", args.tol)
    print("[INFO] max_newton_iter:", args.max_newton_iter)

    all_rows = []

    tasks = [
        (
            degree,
            args.data_root,
            args.initializers,
            args.repeats,
            args.tol,
            args.max_newton_iter,
        )
        for degree in args.degrees
    ]

    if args.max_parallel <= 1:
        for task in tasks:
            rows = evaluate_degree_worker(task)
            all_rows.extend(rows)
    else:
        print(
            "[WARN] max_parallel > 1 can distort CPU timing because degree jobs share CPU resources. "
            "Use this only for exploratory timing.",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=args.max_parallel) as ex:
            futures = [ex.submit(evaluate_degree_worker, t) for t in tasks]
            for fut in as_completed(futures):
                rows = fut.result()
                all_rows.extend(rows)

    all_rows = sorted(
        all_rows,
        key=lambda r: (
            int(r["degree"]),
            str(r["initializer"]),
        ),
    )

    # Raw output
    save_csv(out_dir / "classical_cpu_time_all_raw.csv", all_rows)
    save_json(out_dir / "classical_cpu_time_all_raw.json", all_rows)

    # Paper full table
    paper_rows = make_paper_rows(all_rows)
    paper_columns = [
        "Degree",
        "Initializer",
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

    save_csv(out_dir / "classical_cpu_time_paper_table.csv", paper_rows)
    (out_dir / "classical_cpu_time_paper_table.md").write_text(
        markdown_table(paper_rows, paper_columns),
        encoding="utf-8",
    )
    (out_dir / "classical_cpu_time_paper_table.tex").write_text(
        latex_table(
            paper_rows,
            paper_columns,
            caption=(
                "CPU runtime of heuristic and explicit nonlearned initializers "
                "with Newton refinement under the same stopping criterion."
            ),
            label="tab:classical_cpu_runtime",
        ),
        encoding="utf-8",
    )

    # Average by initializer
    summary = aggregate_by_initializer(all_rows)
    save_csv(out_dir / "classical_cpu_time_by_initializer_raw.csv", summary)

    avg_rows = make_initializer_average_paper_rows(summary)
    avg_columns = [
        "Initializer",
        "N Degrees",
        "Avg Direct RMSE",
        "Avg Direct Residual",
        "Avg Final RMSE",
        "Avg Final Residual",
        "Avg Init CPU ms/sample",
        "Avg Newton CPU ms/sample",
        "Avg Total CPU ms/sample",
        "Avg Iter.",
        "Avg Conv.",
    ]

    save_csv(out_dir / "classical_cpu_time_by_initializer.csv", avg_rows)
    (out_dir / "classical_cpu_time_by_initializer.md").write_text(
        markdown_table(avg_rows, avg_columns),
        encoding="utf-8",
    )
    (out_dir / "classical_cpu_time_by_initializer.tex").write_text(
        latex_table(
            avg_rows,
            avg_columns,
            caption=(
                "Initializer-wise average CPU runtime across Taylor degrees. "
                "The heuristic and explicit baselines are evaluated without neural inference."
            ),
            label="tab:classical_cpu_runtime_average",
        ),
        encoding="utf-8",
    )

    print("\n[DONE]")
    print("Saved:")
    for name in [
        "classical_cpu_time_all_raw.csv",
        "classical_cpu_time_paper_table.csv",
        "classical_cpu_time_paper_table.md",
        "classical_cpu_time_paper_table.tex",
        "classical_cpu_time_by_initializer_raw.csv",
        "classical_cpu_time_by_initializer.csv",
        "classical_cpu_time_by_initializer.md",
        "classical_cpu_time_by_initializer.tex",
    ]:
        print(" -", out_dir / name)


if __name__ == "__main__":
    main()