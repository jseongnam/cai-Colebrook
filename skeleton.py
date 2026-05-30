#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_npz_colebrook_1d_parallel2.py

NPZ-based evaluation code for:
  (A) 1D Colebrook--White residual
      x + 2 log10(eps_rel/3.7 + 2.51 x / Re) = 0

  (B) Two-branch parallel pipe-flow system composed of branch-wise
      Colebrook--White residuals and a head-loss coupling constraint:
      F1(Q1, x1) = Colebrook(x1, Re(Q1,D1), eps1/D1) = 0
      F2(Q1, x2) = Colebrook(x2, Re(QT-Q1,D2), eps2/D2) = 0
      F3(Q1, x1, x2) = h1(Q1,x1) - h2(QT-Q1,x2) = 0

This code is designed for your degree-25 NPZ datasets.
It supports:
  - explicit Colebrook baseline initializers: Haaland, Swamee--Jain, Serghides
  - baseline + Newton refinement
  - simple NPZ-feature-based learned direct vs learned correction ablation
  - initial error / residual reduction analysis
  - iteration correlation analysis
  - derivative/Jacobian condition-bin analysis
  - correction magnitude analysis

Expected multidimensional NPZ keys from your generator:
  coeffs:   (N, 3, degree+1)
  center:   (N, 3)
  target:   (N, 3), columns [Q1*, x1*, x2*]
  Q_total, D1, D2, eps1, eps2, L1, L2, rho, mu, g

Flexible 1D NPZ keys:
  Preferred:
    coeffs, center, target, Re, eps_rel
  Also accepted:
    coeffs, center, target, a, b
    where a = eps_rel/3.7 and b = 2.51/Re.

Usage examples
--------------
1D explicit-baseline evaluation:
  python evaluate_npz_colebrook_1d_parallel2.py --test path/to/colebrook1d_test.npz --out out_1d

Two-branch parallel pipe-flow evaluation:
  python evaluate_npz_colebrook_1d_parallel2.py --test path/to/parallel2_colebrook_deg25_test.npz --out out_multi

Train simple MLP direct-vs-correction ablation using NPZ train/val/test:
  python evaluate_npz_colebrook_1d_parallel2.py \
      --train parallel2_colebrook_deg25_train.npz \
      --val parallel2_colebrook_deg25_val.npz \
      --test parallel2_colebrook_deg25_test.npz \
      --out out_multi \
      --run_torch_ablation \
      --epochs 80

Notes
-----
- The "multi" problem is not called a "multidimensional Colebrook equation" here.
  It is a two-branch parallel pipe-flow system composed of branch-wise
  Colebrook--White residuals and a head-loss coupling constraint.
- The PyTorch ablation is intentionally simple and architecture-neutral.
  It is meant to support the paper's direct-regression vs correction-learning
  analysis, not to replace your main MLP/LSTM/GRU/Transformer experiments.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:
    pearsonr = None
    spearmanr = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


PI = math.pi
LN10 = math.log(10.0)
EPS = 1e-12


# ============================================================
# Basic hydraulic / Colebrook functions
# ============================================================

def as1d(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def re_from_Q(Q, rho, mu, D):
    Q = np.asarray(Q, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    return 4.0 * rho * Q / (PI * mu * D)


def colebrook_residual_x(x, Re, eps_rel):
    """
    Standard x-form Colebrook--White residual:
      F(x; Re, eps_rel) = x + 2 log10(eps_rel/3.7 + 2.51 x / Re)
    where x = 1/sqrt(lambda).
    """
    x = np.asarray(x, dtype=np.float64)
    Re = np.asarray(Re, dtype=np.float64)
    eps_rel = np.asarray(eps_rel, dtype=np.float64)
    z = eps_rel / 3.7 + 2.51 * x / Re

    out = np.full(np.broadcast_shapes(x.shape, Re.shape, eps_rel.shape), np.nan, dtype=np.float64)
    xb, Reb, eb = np.broadcast_arrays(x, Re, eps_rel)
    mask = (Reb > 0.0) & (z > 0.0) & np.isfinite(xb) & np.isfinite(Reb) & np.isfinite(eb)
    out[mask] = xb[mask] + 2.0 * np.log10(z[mask])
    return out


def colebrook_derivative_x(x, Re, eps_rel):
    x = np.asarray(x, dtype=np.float64)
    Re = np.asarray(Re, dtype=np.float64)
    eps_rel = np.asarray(eps_rel, dtype=np.float64)
    z = eps_rel / 3.7 + 2.51 * x / Re

    out = np.full(np.broadcast_shapes(x.shape, Re.shape, eps_rel.shape), np.nan, dtype=np.float64)
    xb, Reb, eb = np.broadcast_arrays(x, Re, eps_rel)
    mask = (Reb > 0.0) & (z > 0.0) & np.isfinite(xb) & np.isfinite(Reb) & np.isfinite(eb)
    out[mask] = 1.0 + 2.0 * ((2.51 / Reb[mask]) / (z[mask] * LN10))
    return out


def safe_clip_x(x, x_min=1.0, x_max=30.0):
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, 7.0)
    return np.clip(x, x_min, x_max)


def safe_clip_Q1(Q1, QT):
    Q1 = np.asarray(Q1, dtype=np.float64)
    QT = np.asarray(QT, dtype=np.float64)
    lo = np.maximum(1e-10, 1e-6 * QT)
    hi = np.maximum(lo + 1e-10, QT - np.maximum(1e-10, 1e-6 * QT))
    return np.clip(Q1, lo, hi)


def head_loss(Q, x, L, D, g):
    Q = np.asarray(Q, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    return 8.0 * L * (Q ** 2) / (g * (PI ** 2) * (D ** 5) * (x ** 2))


# ============================================================
# Explicit Colebrook approximations returning x0 = 1/sqrt(lambda)
# ============================================================

def init_haaland_x(Re, eps_rel):
    """
    Haaland (1983):
      1/sqrt(f) = -1.8 log10( (eps/D/3.7)^1.11 + 6.9/Re )
    """
    Re = np.asarray(Re, dtype=np.float64)
    eps_rel = np.asarray(eps_rel, dtype=np.float64)
    term = (eps_rel / 3.7) ** 1.11 + 6.9 / Re
    x0 = -1.8 * np.log10(term)
    return safe_clip_x(x0)


def init_swamee_jain_x(Re, eps_rel):
    """
    Swamee--Jain (1976):
      f = 0.25 / [log10(eps/D/3.7 + 5.74/Re^0.9)]^2
      x = 1/sqrt(f) = -2 log10(...)
    """
    Re = np.asarray(Re, dtype=np.float64)
    eps_rel = np.asarray(eps_rel, dtype=np.float64)
    term = eps_rel / 3.7 + 5.74 / (Re ** 0.9)
    x0 = -2.0 * np.log10(term)
    return safe_clip_x(x0)


def init_serghides_x(Re, eps_rel):
    """
    Serghides multi-log approximation in x = 1/sqrt(lambda).
    """
    Re = np.asarray(Re, dtype=np.float64)
    eps_rel = np.asarray(eps_rel, dtype=np.float64)
    A = -2.0 * np.log10(eps_rel / 3.7 + 12.0 / Re)
    B = -2.0 * np.log10(eps_rel / 3.7 + 2.51 * A / Re)
    C = -2.0 * np.log10(eps_rel / 3.7 + 2.51 * B / Re)
    denom = C - 2.0 * B + A
    denom = np.where(np.abs(denom) < 1e-12, np.sign(denom + 1e-12) * 1e-12, denom)
    x0 = A - ((B - A) ** 2) / denom
    return safe_clip_x(x0)


def init_fixed_x(Re, eps_rel):
    Re = np.asarray(Re, dtype=np.float64)
    return np.ones_like(Re, dtype=np.float64)


def init_scale_ab_x(Re, eps_rel):
    """
    In x + 2 log10(a + b x), a = eps_rel/3.7, b = 2.51/Re.
    a/b = eps_rel * Re / (3.7*2.51).
    This is a scale heuristic, not a friction-factor approximation.
    """
    Re = np.asarray(Re, dtype=np.float64)
    eps_rel = np.asarray(eps_rel, dtype=np.float64)
    a = eps_rel / 3.7
    b = 2.51 / Re
    return safe_clip_x(a / np.maximum(b, EPS))


X_INITIALIZERS: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "Fixed x0=1": init_fixed_x,
    "Scale a/b": init_scale_ab_x,
    "Haaland": init_haaland_x,
    "Swamee-Jain": init_swamee_jain_x,
    "Serghides": init_serghides_x,
}


# ============================================================
# 1D Newton
# ============================================================

@dataclass
class Newton1DResult:
    x_final: np.ndarray
    residual_abs: np.ndarray
    n_iter: np.ndarray
    converged: np.ndarray


def batch_newton_1d(
    x0,
    Re,
    eps_rel,
    tol=1e-12,
    max_iter=30,
    step_clip=5.0,
    x_min=1.0,
    x_max=30.0,
) -> Newton1DResult:
    x = safe_clip_x(as1d(x0), x_min, x_max)
    Re = as1d(Re)
    eps_rel = as1d(eps_rel)
    n = len(x)

    n_iter = np.zeros(n, dtype=int)
    converged = np.zeros(n, dtype=bool)

    for _ in range(max_iter):
        f = colebrook_residual_x(x, Re, eps_rel)
        df = colebrook_derivative_x(x, Re, eps_rel)

        now = np.isfinite(f) & (np.abs(f) < tol)
        converged |= now
        active = ~converged
        if not np.any(active):
            break

        safe_df = np.where(np.abs(df) < 1e-14, np.sign(df + 1e-14) * 1e-14, df)
        step = np.zeros_like(x)
        step[active] = f[active] / safe_df[active]
        step = np.clip(step, -step_clip, step_clip)

        x[active] = x[active] - step[active]
        x = safe_clip_x(x, x_min, x_max)
        n_iter[active] += 1

    residual_abs = np.abs(colebrook_residual_x(x, Re, eps_rel))
    converged = np.isfinite(residual_abs) & (residual_abs < tol)
    return Newton1DResult(x, residual_abs, n_iter, converged)


# ============================================================
# Parallel two-branch pipe-flow system
# ============================================================

def parallel2_residual_vector(z, params):
    """
    z = [Q1, x1, x2]
    params contains:
      QT, D1, D2, eps1, eps2, L1, L2, rho, mu, g
    Returns vector [F1, F2, F3].
    """
    Q1, x1, x2 = float(z[0]), float(z[1]), float(z[2])
    QT = float(params["QT"])
    D1 = float(params["D1"])
    D2 = float(params["D2"])
    eps1 = float(params["eps1"])
    eps2 = float(params["eps2"])
    L1 = float(params["L1"])
    L2 = float(params["L2"])
    rho = float(params["rho"])
    mu = float(params["mu"])
    g = float(params["g"])

    Q1 = float(safe_clip_Q1(np.array([Q1]), np.array([QT]))[0])
    Q2 = QT - Q1
    x1 = float(safe_clip_x(np.array([x1]))[0])
    x2 = float(safe_clip_x(np.array([x2]))[0])

    Re1 = re_from_Q(np.array([Q1]), np.array([rho]), np.array([mu]), np.array([D1]))[0]
    Re2 = re_from_Q(np.array([Q2]), np.array([rho]), np.array([mu]), np.array([D2]))[0]
    rr1 = eps1 / D1
    rr2 = eps2 / D2

    F1 = colebrook_residual_x(np.array([x1]), np.array([Re1]), np.array([rr1]))[0]
    F2 = colebrook_residual_x(np.array([x2]), np.array([Re2]), np.array([rr2]))[0]
    h1 = head_loss(np.array([Q1]), np.array([x1]), np.array([L1]), np.array([D1]), np.array([g]))[0]
    h2 = head_loss(np.array([Q2]), np.array([x2]), np.array([L2]), np.array([D2]), np.array([g]))[0]
    F3 = h1 - h2

    return np.array([F1, F2, F3], dtype=np.float64)


def parallel2_residual_norm(z, params, mode="max"):
    r = parallel2_residual_vector(z, params)
    if not np.all(np.isfinite(r)):
        return np.inf
    if mode == "l2":
        return float(np.linalg.norm(r))
    return float(np.max(np.abs(r)))


def finite_difference_jacobian(func, z, eps=1e-6):
    z = np.asarray(z, dtype=np.float64)
    f0 = func(z)
    m = len(f0)
    n = len(z)
    J = np.zeros((m, n), dtype=np.float64)

    for j in range(n):
        step = eps * max(1.0, abs(z[j]))
        zp = z.copy()
        zm = z.copy()
        zp[j] += step
        zm[j] -= step
        fp = func(zp)
        fm = func(zm)
        if np.all(np.isfinite(fp)) and np.all(np.isfinite(fm)):
            J[:, j] = (fp - fm) / (2.0 * step)
        else:
            J[:, j] = np.nan
    return J


@dataclass
class NewtonMultiResult:
    z_final: np.ndarray
    residual_norm: np.ndarray
    n_iter: np.ndarray
    converged: np.ndarray
    jac_cond0: np.ndarray


def project_parallel2_z(z, params):
    z = np.asarray(z, dtype=np.float64).copy()
    QT = float(params["QT"])
    z[0] = float(safe_clip_Q1(np.array([z[0]]), np.array([QT]))[0])
    z[1] = float(safe_clip_x(np.array([z[1]]))[0])
    z[2] = float(safe_clip_x(np.array([z[2]]))[0])
    return z


def newton_parallel2_single(
    z0,
    params,
    tol=1e-10,
    max_iter=40,
    step_clip_Q_frac=0.25,
    step_clip_x=5.0,
    jac_eps=1e-6,
):
    z = project_parallel2_z(z0, params)
    QT = float(params["QT"])
    converged = False
    n_iter = 0

    def F(zz):
        return parallel2_residual_vector(project_parallel2_z(zz, params), params)

    J0 = finite_difference_jacobian(F, z, eps=jac_eps)
    try:
        cond0 = float(np.linalg.cond(J0)) if np.all(np.isfinite(J0)) else np.inf
    except Exception:
        cond0 = np.inf

    for k in range(max_iter):
        r = F(z)
        rn = float(np.max(np.abs(r))) if np.all(np.isfinite(r)) else np.inf
        if rn < tol:
            converged = True
            break

        J = finite_difference_jacobian(F, z, eps=jac_eps)
        if not np.all(np.isfinite(J)):
            break

        # Solve J step = F, with fallback to least squares.
        try:
            step = np.linalg.solve(J, r)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(J, r, rcond=None)[0]

        if not np.all(np.isfinite(step)):
            break

        # Step clipping. Q1 step is clipped relative to total flow.
        step[0] = np.clip(step[0], -step_clip_Q_frac * QT, step_clip_Q_frac * QT)
        step[1] = np.clip(step[1], -step_clip_x, step_clip_x)
        step[2] = np.clip(step[2], -step_clip_x, step_clip_x)

        # Half-step backtracking if needed.
        old_rn = rn
        accepted = False
        alpha = 1.0
        for _bt in range(12):
            cand = project_parallel2_z(z - alpha * step, params)
            cand_rn = parallel2_residual_norm(cand, params, mode="max")
            if np.isfinite(cand_rn) and cand_rn <= old_rn:
                z = cand
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            z = project_parallel2_z(z - 0.1 * step, params)

        n_iter += 1

    rn = parallel2_residual_norm(z, params, mode="max")
    converged = np.isfinite(rn) and (rn < tol)
    return z, rn, n_iter, converged, cond0


def batch_newton_parallel2(z0_batch, data, tol=1e-10, max_iter=40) -> NewtonMultiResult:
    z0_batch = np.asarray(z0_batch, dtype=np.float64)
    n = z0_batch.shape[0]
    z_final = np.zeros_like(z0_batch, dtype=np.float64)
    residual = np.zeros(n, dtype=np.float64)
    iters = np.zeros(n, dtype=int)
    conv = np.zeros(n, dtype=bool)
    cond0 = np.zeros(n, dtype=np.float64)

    for i in range(n):
        params = get_parallel2_params(data, i)
        z, rn, ni, ok, c0 = newton_parallel2_single(z0_batch[i], params, tol=tol, max_iter=max_iter)
        z_final[i] = z
        residual[i] = rn
        iters[i] = ni
        conv[i] = ok
        cond0[i] = c0

    return NewtonMultiResult(z_final, residual, iters, conv, cond0)


def get_parallel2_params(data: Dict[str, np.ndarray], i: int) -> Dict[str, float]:
    return {
        "QT": float(data["Q_total"][i]),
        "D1": float(data["D1"][i]),
        "D2": float(data["D2"][i]),
        "eps1": float(data["eps1"][i]),
        "eps2": float(data["eps2"][i]),
        "L1": float(data["L1"][i]),
        "L2": float(data["L2"][i]),
        "rho": float(data["rho"][i]),
        "mu": float(data["mu"][i]),
        "g": float(data["g"][i]),
    }


def parallel2_baseline_z(data: Dict[str, np.ndarray], method="Serghides", split="conductance"):
    """
    Build nonlearned initializers for the two-branch parallel pipe-flow system.

    split:
      equal:
        Q1 = 0.5 QT
      conductance:
        Q1/QT approx K1/(K1+K2), K_i = D_i^(5/2)/sqrt(L_i)
        This follows h ~ L Q^2/(D^5 x^2) and assumes comparable x.
      diameter:
        Q1/QT approx D1/(D1+D2)

    x1, x2 are initialized by explicit Colebrook approximations using the
    corresponding branch flow and relative roughness.
    """
    QT = as1d(data["Q_total"])
    D1 = as1d(data["D1"])
    D2 = as1d(data["D2"])
    eps1 = as1d(data["eps1"])
    eps2 = as1d(data["eps2"])
    L1 = as1d(data["L1"])
    L2 = as1d(data["L2"])
    rho = as1d(data["rho"])
    mu = as1d(data["mu"])

    if split == "equal":
        r = np.full_like(QT, 0.5)
    elif split == "diameter":
        r = D1 / np.maximum(D1 + D2, EPS)
    else:
        K1 = (D1 ** 2.5) / np.sqrt(np.maximum(L1, EPS))
        K2 = (D2 ** 2.5) / np.sqrt(np.maximum(L2, EPS))
        r = K1 / np.maximum(K1 + K2, EPS)

    Q1 = safe_clip_Q1(QT * r, QT)
    Q2 = QT - Q1

    Re1 = re_from_Q(Q1, rho, mu, D1)
    Re2 = re_from_Q(Q2, rho, mu, D2)
    rr1 = eps1 / D1
    rr2 = eps2 / D2

    init_fn = X_INITIALIZERS[method]
    x1 = init_fn(Re1, rr1)
    x2 = init_fn(Re2, rr2)

    return np.stack([Q1, x1, x2], axis=1)


def direct_parallel2_residuals(z_batch, data):
    z_batch = np.asarray(z_batch, dtype=np.float64)
    out = np.zeros(z_batch.shape[0], dtype=np.float64)
    for i in range(z_batch.shape[0]):
        out[i] = parallel2_residual_norm(z_batch[i], get_parallel2_params(data, i), mode="max")
    return out


def jacobian_condition_parallel2(z_batch, data):
    z_batch = np.asarray(z_batch, dtype=np.float64)
    out = np.zeros(z_batch.shape[0], dtype=np.float64)
    for i in range(z_batch.shape[0]):
        params = get_parallel2_params(data, i)
        z = project_parallel2_z(z_batch[i], params)

        def F(zz):
            return parallel2_residual_vector(project_parallel2_z(zz, params), params)

        J = finite_difference_jacobian(F, z)
        try:
            out[i] = float(np.linalg.cond(J)) if np.all(np.isfinite(J)) else np.inf
        except Exception:
            out[i] = np.inf
    return out


# ============================================================
# NPZ loading and type detection
# ============================================================

def load_npz(path: Path) -> Dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def detect_problem_type(data: Dict[str, np.ndarray]) -> str:
    keys = set(data.keys())
    if {"Q_total", "D1", "D2", "eps1", "eps2", "L1", "L2", "rho", "mu", "g", "target"}.issubset(keys):
        target = np.asarray(data["target"])
        if target.ndim == 2 and target.shape[1] == 3:
            return "parallel2"
    if "target" in keys:
        return "1d"
    raise ValueError("Could not detect problem type from NPZ keys.")


def extract_1d_Re_eps_target(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.asarray(data["target"], dtype=np.float64)
    if target.ndim == 2:
        target = target[:, 0]
    target = as1d(target)

    if "Re" in data and "eps_rel" in data:
        return as1d(data["Re"]), as1d(data["eps_rel"]), target

    if "re" in data and "rel_rough" in data:
        return as1d(data["re"]), as1d(data["rel_rough"]), target

    if "a" in data and "b" in data:
        a = as1d(data["a"])
        b = as1d(data["b"])
        eps_rel = 3.7 * a
        Re = 2.51 / np.maximum(b, EPS)
        return Re, eps_rel, target

    # Try to infer from Q/D/eps/rho/mu if a single-pipe dataset is stored that way.
    if {"Q", "D", "eps", "rho", "mu"}.issubset(set(data.keys())):
        Q = as1d(data["Q"])
        D = as1d(data["D"])
        eps = as1d(data["eps"])
        rho = as1d(data["rho"])
        mu = as1d(data["mu"])
        return re_from_Q(Q, rho, mu, D), eps / D, target

    raise ValueError(
        "1D NPZ must contain either (Re, eps_rel), (a,b), or (Q,D,eps,rho,mu), plus target."
    )


def make_features(data: Dict[str, np.ndarray], problem_type: str) -> np.ndarray:
    """
    Build numeric feature matrix from NPZ.

    For degree-25 Taylor data:
      1D:
        coeffs may be (N,26) or (N,1,26).
      parallel2:
        coeffs is (N,3,26).

    We also append center and physical/global parameters when available.
    """
    feats: List[np.ndarray] = []

    if "coeffs" in data:
        coeffs = np.asarray(data["coeffs"], dtype=np.float64)
        feats.append(coeffs.reshape(coeffs.shape[0], -1))

    if "center" in data:
        center = np.asarray(data["center"], dtype=np.float64)
        feats.append(center.reshape(center.shape[0], -1))

    if problem_type == "1d":
        try:
            Re, eps_rel, _ = extract_1d_Re_eps_target(data)
            feats.append(np.stack([np.log10(np.maximum(Re, EPS)), eps_rel], axis=1))
        except Exception:
            pass
    else:
        global_keys = ["Q_total", "D1", "D2", "eps1", "eps2", "L1", "L2", "rho", "mu", "g"]
        globals_ = [as1d(data[k]) for k in global_keys if k in data]
        if globals_:
            feats.append(np.stack(globals_, axis=1))

    if not feats:
        raise ValueError("No usable feature keys found. Expected coeffs and/or center and global parameters.")

    X = np.concatenate(feats, axis=1)
    X = np.where(np.isfinite(X), X, 0.0)
    # Signed log transform for stability.
    X = np.sign(X) * np.log1p(np.abs(X))
    return X.astype(np.float32)


# ============================================================
# Metrics
# ============================================================

def regression_metrics(pred, true, residual=None) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    err = pred - true

    if err.ndim == 1:
        abs_err = np.abs(err)
        sq = err ** 2
    else:
        abs_err = np.linalg.norm(err, axis=1)
        sq = np.sum(err ** 2, axis=1)

    out = {
        "MAE": float(np.mean(abs_err)),
        "RMSE": float(np.sqrt(np.mean(sq))),
        "MaxAbsError": float(np.max(abs_err)),
    }

    # R2 over flattened components.
    ef = err.reshape(-1)
    tf = true.reshape(-1)
    ss_res = float(np.sum(ef ** 2))
    ss_tot = float(np.sum((tf - np.mean(tf)) ** 2))
    out["R2"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    if residual is not None:
        residual = np.asarray(residual, dtype=np.float64)
        out.update({
            "ResidualMean": float(np.mean(residual)),
            "ResidualMedian": float(np.median(residual)),
            "ResidualP90": float(np.percentile(residual, 90)),
        })
    return out


def refinement_metrics_1d(res: Newton1DResult, true) -> Dict[str, float]:
    true = as1d(true)
    err = res.x_final - true
    return {
        "FinalRMSE": float(np.sqrt(np.mean(err ** 2))),
        "FinalMAE": float(np.mean(np.abs(err))),
        "IterMean": float(np.mean(res.n_iter)),
        "IterMedian": float(np.median(res.n_iter)),
        "IterP90": float(np.percentile(res.n_iter, 90)),
        "ConvergedRatio": float(np.mean(res.converged)),
        "FinalResidualMean": float(np.mean(res.residual_abs)),
        "FinalResidualP90": float(np.percentile(res.residual_abs, 90)),
    }


def refinement_metrics_multi(res: NewtonMultiResult, true) -> Dict[str, float]:
    true = np.asarray(true, dtype=np.float64)
    err = res.z_final - true
    err_norm = np.linalg.norm(err, axis=1)
    return {
        "FinalRMSE": float(np.sqrt(np.mean(np.sum(err ** 2, axis=1)))),
        "FinalMAE": float(np.mean(err_norm)),
        "IterMean": float(np.mean(res.n_iter)),
        "IterMedian": float(np.median(res.n_iter)),
        "IterP90": float(np.percentile(res.n_iter, 90)),
        "ConvergedRatio": float(np.mean(res.converged)),
        "FinalResidualMean": float(np.mean(res.residual_norm)),
        "FinalResidualP90": float(np.percentile(res.residual_norm, 90)),
        "JacCond0Median": float(np.median(res.jac_cond0[np.isfinite(res.jac_cond0)])) if np.any(np.isfinite(res.jac_cond0)) else np.inf,
    }


def corr_value(a, b):
    if pearsonr is None:
        return np.nan
    try:
        return float(pearsonr(a, b)[0])
    except Exception:
        return np.nan


def spear_value(a, b):
    if spearmanr is None:
        return np.nan
    try:
        return float(spearmanr(a, b)[0])
    except Exception:
        return np.nan


# ============================================================
# Nonlearned baseline evaluation
# ============================================================

def evaluate_1d_nonlearned(data: Dict[str, np.ndarray], tol: float, max_iter: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    Re, eps_rel, x_true = extract_1d_Re_eps_target(data)
    direct_rows = []
    newton_rows = []

    for name, init_fn in X_INITIALIZERS.items():
        x0 = init_fn(Re, eps_rel)
        res0 = np.abs(colebrook_residual_x(x0, Re, eps_rel))
        direct_rows.append({
            "Method": name,
            "Type": "Nonlearned explicit/heuristic initializer",
            **regression_metrics(x0, x_true, res0),
        })

        nr = batch_newton_1d(x0, Re, eps_rel, tol=tol, max_iter=max_iter)
        newton_rows.append({
            "Method": f"{name} + Newton",
            "Type": "Nonlearned explicit/heuristic initializer",
            **refinement_metrics_1d(nr, x_true),
        })

    return pd.DataFrame(direct_rows), pd.DataFrame(newton_rows)


def evaluate_parallel2_nonlearned(data: Dict[str, np.ndarray], tol: float, max_iter: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    z_true = np.asarray(data["target"], dtype=np.float64)
    direct_rows = []
    newton_rows = []

    for split in ["equal", "diameter", "conductance"]:
        for x_method in ["Haaland", "Swamee-Jain", "Serghides"]:
            method = f"{split} split + {x_method} branch x"
            z0 = parallel2_baseline_z(data, method=x_method, split=split)
            res0 = direct_parallel2_residuals(z0, data)
            direct_rows.append({
                "Method": method,
                "Type": "Two-branch pipe-flow nonlearned initializer",
                **regression_metrics(z0, z_true, res0),
            })

            nr = batch_newton_parallel2(z0, data, tol=tol, max_iter=max_iter)
            newton_rows.append({
                "Method": f"{method} + Newton",
                "Type": "Two-branch pipe-flow nonlearned initializer",
                **refinement_metrics_multi(nr, z_true),
            })

    return pd.DataFrame(direct_rows), pd.DataFrame(newton_rows)


# ============================================================
# Torch direct-vs-correction ablation
# ============================================================

class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)
        return self

    def transform(self, X):
        return ((np.asarray(X, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class TorchMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 128, 64), dropout=0.05):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_torch_mlp(
    X_train, y_train, X_val=None, y_val=None,
    epochs=80, batch_size=1024, lr=1e-3, weight_decay=1e-5,
    device=None, seed=42,
):
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Install torch or run without --run_torch_ablation.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)

    model = TorchMLP(X_train.shape[1], y_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()

    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    best_state = None
    best_val = float("inf")
    patience = 15
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                xv = torch.from_numpy(np.asarray(X_val, dtype=np.float32)).to(device)
                yv = torch.from_numpy(np.asarray(y_val, dtype=np.float32)).to(device)
                val = float(loss_fn(model(xv), yv).detach().cpu().item())
            if val < best_val:
                best_val = val
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, device


def predict_torch(model, X, device):
    model.eval()
    outs = []
    bs = 4096
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(np.asarray(X[i:i+bs], dtype=np.float32)).to(device)
            outs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def choose_best_parallel2_baseline(data: Dict[str, np.ndarray]) -> Tuple[str, np.ndarray]:
    """
    Choose a strong default baseline for correction ablation.
    Preference: conductance split + Serghides branch x.
    """
    return "conductance split + Serghides branch x", parallel2_baseline_z(data, method="Serghides", split="conductance")


def choose_best_1d_baseline(data: Dict[str, np.ndarray]) -> Tuple[str, np.ndarray]:
    Re, eps_rel, _ = extract_1d_Re_eps_target(data)
    return "Serghides", init_serghides_x(Re, eps_rel)


def build_logit_parallel2_target(z, data):
    """
    Convert z=[Q1,x1,x2] into [logit(Q1/QT), x1, x2].
    Useful for scale-stable multi-output learning.
    """
    z = np.asarray(z, dtype=np.float64)
    QT = as1d(data["Q_total"])
    r = np.clip(z[:, 0] / np.maximum(QT, EPS), 1e-6, 1 - 1e-6)
    ell = np.log(r / (1.0 - r))
    return np.stack([ell, z[:, 1], z[:, 2]], axis=1)


def decode_logit_parallel2(y, data):
    y = np.asarray(y, dtype=np.float64)
    QT = as1d(data["Q_total"])
    r = 1.0 / (1.0 + np.exp(-y[:, 0]))
    Q1 = safe_clip_Q1(QT * r, QT)
    x1 = safe_clip_x(y[:, 1])
    x2 = safe_clip_x(y[:, 2])
    return np.stack([Q1, x1, x2], axis=1)


def run_torch_ablation(
    train_data: Dict[str, np.ndarray],
    val_data: Optional[Dict[str, np.ndarray]],
    test_data: Dict[str, np.ndarray],
    problem_type: str,
    tol: float,
    max_iter: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Trains two simple MLPs:
      - direct regression
      - baseline-aware correction learning

    For 1D:
      direct target: x*
      correction target: x* - x_base

    For parallel2:
      direct target in stable coordinates:
        [logit(Q1/QT), x1, x2]
      correction target:
        stable(target) - stable(base)
      decoded correction:
        stable(base) + predicted_delta
    """
    Xtr_raw = make_features(train_data, problem_type)
    Xte_raw = make_features(test_data, problem_type)
    Xval_raw = make_features(val_data, problem_type) if val_data is not None else None

    sx = Standardizer().fit(Xtr_raw)
    Xtr = sx.transform(Xtr_raw)
    Xte = sx.transform(Xte_raw)
    Xval = sx.transform(Xval_raw) if Xval_raw is not None else None

    if problem_type == "1d":
        _, _, ytr_true = extract_1d_Re_eps_target(train_data)
        _, _, yte_true = extract_1d_Re_eps_target(test_data)
        _, _, yval_true = extract_1d_Re_eps_target(val_data) if val_data is not None else (None, None, None)

        _, base_tr = choose_best_1d_baseline(train_data)
        base_name, base_te = choose_best_1d_baseline(test_data)
        _, base_val = choose_best_1d_baseline(val_data) if val_data is not None else (None, None)

        ytr_direct = ytr_true.reshape(-1, 1)
        yte_eval_true = yte_true
        yval_direct = yval_true.reshape(-1, 1) if val_data is not None else None

        ytr_corr = (ytr_true - base_tr).reshape(-1, 1)
        yval_corr = (yval_true - base_val).reshape(-1, 1) if val_data is not None else None

        y_scaler_direct = Standardizer().fit(ytr_direct)
        y_scaler_corr = Standardizer().fit(ytr_corr)

        direct_model, device = train_torch_mlp(
            Xtr, y_scaler_direct.transform(ytr_direct),
            Xval, y_scaler_direct.transform(yval_direct) if yval_direct is not None else None,
            epochs=epochs, batch_size=batch_size, seed=seed,
        )
        corr_model, device = train_torch_mlp(
            Xtr, y_scaler_corr.transform(ytr_corr),
            Xval, y_scaler_corr.transform(yval_corr) if yval_corr is not None else None,
            epochs=epochs, batch_size=batch_size, seed=seed + 1,
        )

        direct_pred_std = predict_torch(direct_model, Xte, device)
        direct_pred = y_scaler_direct.transform(np.zeros_like(direct_pred_std))  # placeholder to get shape
        direct_pred = direct_pred_std * y_scaler_direct.std + y_scaler_direct.mean
        direct_x = safe_clip_x(direct_pred.reshape(-1))

        corr_pred_std = predict_torch(corr_model, Xte, device)
        corr_delta = corr_pred_std * y_scaler_corr.std + y_scaler_corr.mean
        corr_x = safe_clip_x(base_te + corr_delta.reshape(-1))

        Re, eps_rel, x_true = extract_1d_Re_eps_target(test_data)
        rows_direct = []
        rows_newton = []

        for method, x0 in [
            ("Torch MLP direct regression", direct_x),
            (f"Torch MLP correction over {base_name}", corr_x),
        ]:
            r0 = np.abs(colebrook_residual_x(x0, Re, eps_rel))
            rows_direct.append({"Method": method, "Type": "Torch ablation", **regression_metrics(x0, x_true, r0)})
            nr = batch_newton_1d(x0, Re, eps_rel, tol=tol, max_iter=max_iter)
            rows_newton.append({"Method": f"{method} + Newton", "Type": "Torch ablation", **refinement_metrics_1d(nr, x_true)})

        analysis_df = build_analysis_table_1d(test_data, base_te, corr_x, direct_x, tol, max_iter)
        return pd.DataFrame(rows_direct), pd.DataFrame(rows_newton), analysis_df

    # parallel2
    ytr_true_z = np.asarray(train_data["target"], dtype=np.float64)
    yte_true_z = np.asarray(test_data["target"], dtype=np.float64)
    yval_true_z = np.asarray(val_data["target"], dtype=np.float64) if val_data is not None else None

    _, base_tr_z = choose_best_parallel2_baseline(train_data)
    base_name, base_te_z = choose_best_parallel2_baseline(test_data)
    _, base_val_z = choose_best_parallel2_baseline(val_data) if val_data is not None else (None, None)

    ytr_stable = build_logit_parallel2_target(ytr_true_z, train_data)
    yte_eval_true = yte_true_z
    yval_stable = build_logit_parallel2_target(yval_true_z, val_data) if val_data is not None else None

    base_tr_stable = build_logit_parallel2_target(base_tr_z, train_data)
    base_te_stable = build_logit_parallel2_target(base_te_z, test_data)
    base_val_stable = build_logit_parallel2_target(base_val_z, val_data) if val_data is not None else None

    ytr_direct = ytr_stable
    yval_direct = yval_stable
    ytr_corr = ytr_stable - base_tr_stable
    yval_corr = yval_stable - base_val_stable if val_data is not None else None

    y_scaler_direct = Standardizer().fit(ytr_direct)
    y_scaler_corr = Standardizer().fit(ytr_corr)

    direct_model, device = train_torch_mlp(
        Xtr, y_scaler_direct.transform(ytr_direct),
        Xval, y_scaler_direct.transform(yval_direct) if yval_direct is not None else None,
        epochs=epochs, batch_size=batch_size, seed=seed,
    )
    corr_model, device = train_torch_mlp(
        Xtr, y_scaler_corr.transform(ytr_corr),
        Xval, y_scaler_corr.transform(yval_corr) if yval_corr is not None else None,
        epochs=epochs, batch_size=batch_size, seed=seed + 1,
    )

    direct_pred_std = predict_torch(direct_model, Xte, device)
    direct_pred_stable = direct_pred_std * y_scaler_direct.std + y_scaler_direct.mean
    direct_z = decode_logit_parallel2(direct_pred_stable, test_data)

    corr_pred_std = predict_torch(corr_model, Xte, device)
    corr_delta = corr_pred_std * y_scaler_corr.std + y_scaler_corr.mean
    corr_stable = base_te_stable + corr_delta
    corr_z = decode_logit_parallel2(corr_stable, test_data)

    rows_direct = []
    rows_newton = []
    for method, z0 in [
        ("Torch MLP direct regression", direct_z),
        (f"Torch MLP correction over {base_name}", corr_z),
    ]:
        r0 = direct_parallel2_residuals(z0, test_data)
        rows_direct.append({"Method": method, "Type": "Torch ablation", **regression_metrics(z0, yte_true_z, r0)})
        nr = batch_newton_parallel2(z0, test_data, tol=tol, max_iter=max_iter)
        rows_newton.append({"Method": f"{method} + Newton", "Type": "Torch ablation", **refinement_metrics_multi(nr, yte_true_z)})

    analysis_df = build_analysis_table_parallel2(test_data, base_te_z, corr_z, direct_z, tol, max_iter)
    return pd.DataFrame(rows_direct), pd.DataFrame(rows_newton), analysis_df


# ============================================================
# Analysis tables: weakness 4 support
# ============================================================

def build_analysis_table_1d(data, x_base, x_corr, x_direct, tol, max_iter) -> pd.DataFrame:
    Re, eps_rel, x_true = extract_1d_Re_eps_target(data)
    base_err = np.abs(x_base - x_true)
    corr_err = np.abs(x_corr - x_true)
    direct_err = np.abs(x_direct - x_true)

    base_res = np.abs(colebrook_residual_x(x_base, Re, eps_rel))
    corr_res = np.abs(colebrook_residual_x(x_corr, Re, eps_rel))
    direct_res = np.abs(colebrook_residual_x(x_direct, Re, eps_rel))

    nr_base = batch_newton_1d(x_base, Re, eps_rel, tol=tol, max_iter=max_iter)
    nr_corr = batch_newton_1d(x_corr, Re, eps_rel, tol=tol, max_iter=max_iter)
    nr_direct = batch_newton_1d(x_direct, Re, eps_rel, tol=tol, max_iter=max_iter)

    rows = []

    def add(name, x0, err, res, nr):
        rows.append({
            "Analysis": name,
            "InitErrorMean": float(np.mean(err)),
            "InitResidualMean": float(np.mean(res)),
            "IterMean": float(np.mean(nr.n_iter)),
            "ConvergedRatio": float(np.mean(nr.converged)),
            "Pearson_InitError_Iter": corr_value(err, nr.n_iter),
            "Spearman_InitError_Iter": spear_value(err, nr.n_iter),
            "Pearson_InitResidual_Iter": corr_value(res, nr.n_iter),
            "Spearman_InitResidual_Iter": spear_value(res, nr.n_iter),
            "DerivativeAbsMedian": float(np.median(np.abs(colebrook_derivative_x(x0, Re, eps_rel)))),
        })

    add("Base explicit initializer", x_base, base_err, base_res, nr_base)
    add("Direct regression initializer", x_direct, direct_err, direct_res, nr_direct)
    add("Correction initializer", x_corr, corr_err, corr_res, nr_corr)

    rows.append({
        "Analysis": "Correction improvement over base",
        "InitErrorMean": float(np.mean((base_err - corr_err) / np.maximum(base_err, EPS))),
        "InitResidualMean": float(np.mean((base_res - corr_res) / np.maximum(base_res, EPS))),
        "IterMean": float(np.mean(nr_base.n_iter - nr_corr.n_iter)),
        "ConvergedRatio": float(np.mean(corr_err < base_err)),
        "Pearson_InitError_Iter": np.nan,
        "Spearman_InitError_Iter": np.nan,
        "Pearson_InitResidual_Iter": np.nan,
        "Spearman_InitResidual_Iter": np.nan,
        "DerivativeAbsMedian": np.nan,
    })

    return pd.DataFrame(rows)


def build_analysis_table_parallel2(data, z_base, z_corr, z_direct, tol, max_iter) -> pd.DataFrame:
    z_true = np.asarray(data["target"], dtype=np.float64)

    def norm_err(z):
        return np.linalg.norm(z - z_true, axis=1)

    base_err = norm_err(z_base)
    corr_err = norm_err(z_corr)
    direct_err = norm_err(z_direct)

    base_res = direct_parallel2_residuals(z_base, data)
    corr_res = direct_parallel2_residuals(z_corr, data)
    direct_res = direct_parallel2_residuals(z_direct, data)

    nr_base = batch_newton_parallel2(z_base, data, tol=tol, max_iter=max_iter)
    nr_corr = batch_newton_parallel2(z_corr, data, tol=tol, max_iter=max_iter)
    nr_direct = batch_newton_parallel2(z_direct, data, tol=tol, max_iter=max_iter)

    rows = []

    def add(name, z0, err, res, nr):
        cond = jacobian_condition_parallel2(z0, data)
        finite_cond = cond[np.isfinite(cond)]
        rows.append({
            "Analysis": name,
            "InitErrorMean": float(np.mean(err)),
            "InitResidualMean": float(np.mean(res)),
            "IterMean": float(np.mean(nr.n_iter)),
            "ConvergedRatio": float(np.mean(nr.converged)),
            "Pearson_InitError_Iter": corr_value(err, nr.n_iter),
            "Spearman_InitError_Iter": spear_value(err, nr.n_iter),
            "Pearson_InitResidual_Iter": corr_value(res, nr.n_iter),
            "Spearman_InitResidual_Iter": spear_value(res, nr.n_iter),
            "JacCondMedian": float(np.median(finite_cond)) if finite_cond.size else np.inf,
        })

    add("Base explicit/coupling initializer", z_base, base_err, base_res, nr_base)
    add("Direct regression initializer", z_direct, direct_err, direct_res, nr_direct)
    add("Correction initializer", z_corr, corr_err, corr_res, nr_corr)

    rows.append({
        "Analysis": "Correction improvement over base",
        "InitErrorMean": float(np.mean((base_err - corr_err) / np.maximum(base_err, EPS))),
        "InitResidualMean": float(np.mean((base_res - corr_res) / np.maximum(base_res, EPS))),
        "IterMean": float(np.mean(nr_base.n_iter - nr_corr.n_iter)),
        "ConvergedRatio": float(np.mean(corr_err < base_err)),
        "Pearson_InitError_Iter": np.nan,
        "Spearman_InitError_Iter": np.nan,
        "Pearson_InitResidual_Iter": np.nan,
        "Spearman_InitResidual_Iter": np.nan,
        "JacCondMedian": np.nan,
    })

    return pd.DataFrame(rows)


def derivative_or_condition_bins_1d(data, x0, tol, max_iter, n_bins=4) -> pd.DataFrame:
    Re, eps_rel, x_true = extract_1d_Re_eps_target(data)
    deriv_abs = np.abs(colebrook_derivative_x(x0, Re, eps_rel))
    init_res = np.abs(colebrook_residual_x(x0, Re, eps_rel))
    init_err = np.abs(x0 - x_true)
    nr = batch_newton_1d(x0, Re, eps_rel, tol=tol, max_iter=max_iter)

    df = pd.DataFrame({
        "condition_proxy": deriv_abs,
        "init_residual": init_res,
        "init_error": init_err,
        "iters": nr.n_iter,
        "converged": nr.converged,
    })
    df = df[np.isfinite(df["condition_proxy"])]
    df["condition_bin"] = pd.qcut(df["condition_proxy"], q=n_bins, duplicates="drop")

    return df.groupby("condition_bin").agg(
        Count=("iters", "size"),
        ConditionProxyMean=("condition_proxy", "mean"),
        InitErrorMean=("init_error", "mean"),
        InitResidualMean=("init_residual", "mean"),
        IterMean=("iters", "mean"),
        IterP90=("iters", lambda s: np.percentile(s, 90)),
        ConvergedRatio=("converged", "mean"),
    ).reset_index()


def condition_bins_parallel2(data, z0, tol, max_iter, n_bins=4) -> pd.DataFrame:
    cond = jacobian_condition_parallel2(z0, data)
    init_res = direct_parallel2_residuals(z0, data)
    init_err = np.linalg.norm(z0 - np.asarray(data["target"], dtype=np.float64), axis=1)
    nr = batch_newton_parallel2(z0, data, tol=tol, max_iter=max_iter)

    df = pd.DataFrame({
        "condition_proxy": cond,
        "init_residual": init_res,
        "init_error": init_err,
        "iters": nr.n_iter,
        "converged": nr.converged,
    })
    df = df[np.isfinite(df["condition_proxy"])]
    df["condition_bin"] = pd.qcut(df["condition_proxy"], q=n_bins, duplicates="drop")

    return df.groupby("condition_bin").agg(
        Count=("iters", "size"),
        ConditionProxyMean=("condition_proxy", "mean"),
        InitErrorMean=("init_error", "mean"),
        InitResidualMean=("init_residual", "mean"),
        IterMean=("iters", "mean"),
        IterP90=("iters", lambda s: np.percentile(s, 90)),
        ConvergedRatio=("converged", "mean"),
    ).reset_index()


# ============================================================
# Main
# ============================================================

def save_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)
    print(f"[saved] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=str, required=True, help="Test NPZ path.")
    ap.add_argument("--train", type=str, default=None, help="Optional train NPZ path for torch ablation.")
    ap.add_argument("--val", type=str, default=None, help="Optional val NPZ path for torch ablation.")
    ap.add_argument("--out", type=str, default="results_npz_eval")
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--multi_tol", type=float, default=1e-10)
    ap.add_argument("--max_iter", type=int, default=40)
    ap.add_argument("--run_torch_ablation", action="store_true")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_data = load_npz(Path(args.test))
    problem_type = detect_problem_type(test_data)
    print(f"[detected problem] {problem_type}")

    meta = {
        "test": args.test,
        "train": args.train,
        "val": args.val,
        "problem_type": problem_type,
        "tol": args.tol,
        "multi_tol": args.multi_tol,
        "max_iter": args.max_iter,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if problem_type == "1d":
        direct_df, newton_df = evaluate_1d_nonlearned(test_data, tol=args.tol, max_iter=args.max_iter)
        save_csv(direct_df.sort_values("RMSE"), out_dir / "direct_nonlearned_1d.csv")
        save_csv(newton_df.sort_values("IterMean"), out_dir / "newton_nonlearned_1d.csv")

        _, x_base = choose_best_1d_baseline(test_data)
        bins = derivative_or_condition_bins_1d(test_data, x_base, tol=args.tol, max_iter=args.max_iter)
        save_csv(bins, out_dir / "condition_bins_1d_best_explicit.csv")

    else:
        direct_df, newton_df = evaluate_parallel2_nonlearned(test_data, tol=args.multi_tol, max_iter=args.max_iter)
        save_csv(direct_df.sort_values("RMSE"), out_dir / "direct_nonlearned_parallel2.csv")
        save_csv(newton_df.sort_values("IterMean"), out_dir / "newton_nonlearned_parallel2.csv")

        _, z_base = choose_best_parallel2_baseline(test_data)
        bins = condition_bins_parallel2(test_data, z_base, tol=args.multi_tol, max_iter=args.max_iter)
        save_csv(bins, out_dir / "condition_bins_parallel2_best_explicit.csv")

    if args.run_torch_ablation:
        if args.train is None:
            raise ValueError("--run_torch_ablation requires --train. --val is recommended but optional.")
        train_data = load_npz(Path(args.train))
        val_data = load_npz(Path(args.val)) if args.val else None
        train_type = detect_problem_type(train_data)
        if train_type != problem_type:
            raise ValueError(f"Train problem type {train_type} differs from test type {problem_type}.")
        if val_data is not None and detect_problem_type(val_data) != problem_type:
            raise ValueError("Val problem type differs from test type.")

        ab_direct, ab_newton, analysis = run_torch_ablation(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            problem_type=problem_type,
            tol=args.tol if problem_type == "1d" else args.multi_tol,
            max_iter=args.max_iter,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )

        save_csv(ab_direct.sort_values("RMSE"), out_dir / f"ablation_direct_stage_{problem_type}.csv")
        save_csv(ab_newton.sort_values("IterMean"), out_dir / f"ablation_newton_stage_{problem_type}.csv")
        save_csv(analysis, out_dir / f"weakness4_analysis_{problem_type}.csv")

    print("[DONE]")


if __name__ == "__main__":
    main()