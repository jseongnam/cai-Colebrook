#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================
# Numeric utilities
# ============================================================

def clean_np(x: np.ndarray, clip: Optional[float] = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    if clip is not None:
        x = np.clip(x, -clip, clip)
    return x.astype(np.float32)


def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    b_safe = np.where(np.abs(b) < eps, eps, b)
    out = a / b_safe
    return clean_np(out)


def safe_clip_ratio(r: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    r = clean_np(r)
    return np.clip(r, eps, 1.0 - eps).astype(np.float32)


def signed_log1p(x: np.ndarray) -> np.ndarray:
    x = clean_np(x, clip=1e12)
    return clean_np(np.sign(x) * np.log1p(np.abs(x)), clip=50.0)


def standardize_fit(x: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    x = clean_np(x)
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    mean = clean_np(mean)
    std = clean_np(std)
    std = np.where((std < eps) | (~np.isfinite(std)), 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return clean_np((clean_np(x) - mean) / std, clip=100.0)


def first_existing_key(data: np.lib.npyio.NpzFile, candidates: Sequence[str]) -> Optional[str]:
    keys = set(data.files)
    for k in candidates:
        if k in keys:
            return k
    return None


def require_key(data: np.lib.npyio.NpzFile, candidates: Sequence[str], name: str) -> str:
    k = first_existing_key(data, candidates)
    if k is None:
        raise KeyError(f"Cannot find {name}. Tried={candidates}. Actual={data.files}")
    return k


def to_numpy_float32(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == object:
        arr = arr.astype(np.float64)
    return clean_np(arr)


def ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x[:, None]
    return x


def rmse_np(pred: np.ndarray, true: np.ndarray) -> float:
    pred = clean_np(pred)
    true = clean_np(true)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae_np(pred: np.ndarray, true: np.ndarray) -> float:
    pred = clean_np(pred)
    true = clean_np(true)
    return float(np.mean(np.abs(pred - true)))


def r2_np(pred: np.ndarray, true: np.ndarray) -> float:
    pred = clean_np(pred)
    true = clean_np(true)
    ss_res = float(np.sum((pred - true) ** 2))
    ss_tot = float(np.sum((true - np.mean(true, axis=0, keepdims=True)) ** 2))
    if ss_tot <= 1e-30:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def residual_stats(values: np.ndarray) -> Dict[str, float]:
    values = clean_np(values)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


# ============================================================
# Residual / Newton
# ============================================================

def colebrook_residual_x(x: float, Q: float, D: float, eps: float, rho: float, mu: float) -> float:
    Q = max(float(Q), 1e-12)
    D = max(float(D), 1e-12)
    eps = max(float(eps), 0.0)
    rho = max(float(rho), 1e-12)
    mu = max(float(mu), 1e-12)
    x = max(float(x), 1e-8)

    Re = 4.0 * rho * Q / (math.pi * mu * D)
    Re = max(Re, 1e-12)
    arg = eps / (3.7 * D) + 2.51 * x / Re
    arg = max(arg, 1e-30)
    return float(x + 2.0 * math.log10(arg))


def head_loss(Q: float, x: float, D: float, L: float, g: float) -> float:
    Q = max(float(Q), 1e-12)
    x = max(float(x), 1e-8)
    D = max(float(D), 1e-12)
    L = max(float(L), 1e-12)
    g = max(float(g), 1e-12)
    lam = 1.0 / (x * x)
    v = 4.0 * Q / (math.pi * D * D)
    return float(lam * (L / D) * (v * v) / (2.0 * g))


def coupled_residual(z: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    QT = max(float(params["QT"]), 1e-9)
    Q1 = min(max(float(z[0]), 1e-9), QT - 1e-9)
    Q2 = max(QT - Q1, 1e-9)
    x1 = max(float(z[1]), 1e-8)
    x2 = max(float(z[2]), 1e-8)

    D1, D2 = float(params["D1"]), float(params["D2"])
    eps1, eps2 = float(params["eps1"]), float(params["eps2"])
    L1, L2 = float(params["L1"]), float(params["L2"])
    rho, mu, g = float(params["rho"]), float(params["mu"]), float(params["g"])

    f1 = colebrook_residual_x(x1, Q1, D1, eps1, rho, mu)
    f2 = colebrook_residual_x(x2, Q2, D2, eps2, rho, mu)
    h1 = head_loss(Q1, x1, D1, L1, g)
    h2 = head_loss(Q2, x2, D2, L2, g)
    return clean_np(np.asarray([f1, f2, h1 - h2], dtype=np.float64)).astype(np.float64)


def residual_norm(z: np.ndarray, params: Dict[str, float]) -> float:
    return float(np.linalg.norm(coupled_residual(z, params), ord=2))


def project_z(z: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    out = clean_np(z).astype(np.float64)
    QT = max(float(params["QT"]), 1e-9)
    out[0] = min(max(out[0], 1e-9), QT - 1e-9)
    out[1] = max(out[1], 1e-8)
    out[2] = max(out[2], 1e-8)
    return out


def numerical_jacobian(z: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    z = project_z(z, params)
    J = np.zeros((3, 3), dtype=np.float64)
    for j in range(3):
        step = 1e-5 * max(1.0, abs(float(z[j])))
        zp = z.copy()
        zm = z.copy()
        zp[j] += step
        zm[j] -= step
        zp = project_z(zp, params)
        zm = project_z(zm, params)
        J[:, j] = (coupled_residual(zp, params) - coupled_residual(zm, params)) / (2.0 * step)
    J = np.nan_to_num(J, nan=0.0, posinf=1e6, neginf=-1e6)
    return J


def newton_refine(z0: np.ndarray, params: Dict[str, float], tol: float, max_iter: int):
    z = project_z(z0, params)
    r = residual_norm(z, params)
    if r < tol:
        return z, r, 0, True

    for it in range(1, max_iter + 1):
        F = coupled_residual(z, params)
        J = numerical_jacobian(z, params)
        try:
            step = np.linalg.solve(J, F)
        except Exception:
            step = np.linalg.lstsq(J, F, rcond=None)[0]
        step = np.nan_to_num(step, nan=0.0, posinf=1.0, neginf=-1.0)

        best_z = z
        best_r = r
        alpha = 1.0
        for _ in range(12):
            cand = project_z(z - alpha * step, params)
            cand_r = residual_norm(cand, params)
            if np.isfinite(cand_r) and cand_r < best_r:
                best_z, best_r = cand, cand_r
                break
            alpha *= 0.5

        z, r = best_z, best_r
        if r < tol:
            return z, r, it, True

    return z, r, max_iter, False


# ============================================================
# Baseline
# ============================================================

PARAM_ALIASES = {
    "QT": ["QT", "Q_T", "q_total", "total_flow", "Q_total"],
    "D1": ["D1", "d1", "diam1", "diameter1", "D_1"],
    "D2": ["D2", "d2", "diam2", "diameter2", "D_2"],
    "eps1": ["eps1", "epsilon1", "rough1", "roughness1", "e1", "eps_1"],
    "eps2": ["eps2", "epsilon2", "rough2", "roughness2", "e2", "eps_2"],
    "L1": ["L1", "l1", "length1", "pipe_length1", "L_1"],
    "L2": ["L2", "l2", "length2", "pipe_length2", "L_2"],
    "rho": ["rho", "density"],
    "mu": ["mu", "viscosity", "dynamic_viscosity"],
    "g": ["g", "gravity"],
}


def load_param_arrays(data: np.lib.npyio.NpzFile) -> Dict[str, np.ndarray]:
    out = {}
    for c, aliases in PARAM_ALIASES.items():
        k = first_existing_key(data, aliases)
        if k is None:
            raise KeyError(f"Cannot find param {c}. actual keys={data.files}")
        out[c] = clean_np(data[k]).reshape(-1)
    return out


def haaland_lambda(Re, rr):
    Re = np.maximum(clean_np(Re), 1e-12)
    rr = np.maximum(clean_np(rr), 0.0)
    inv = -1.8 * np.log10((rr / 3.7) ** 1.11 + 6.9 / Re)
    inv = np.maximum(clean_np(inv), 1e-8)
    return clean_np(1.0 / (inv ** 2))


def compute_explicit_x(Q, D, eps, rho, mu):
    Re = 4.0 * rho * np.maximum(Q, 1e-12) / (np.pi * np.maximum(mu, 1e-12) * np.maximum(D, 1e-12))
    rr = eps / np.maximum(D, 1e-12)
    lam = haaland_lambda(Re, rr)
    return clean_np(1.0 / np.sqrt(np.maximum(lam, 1e-30)))


def compute_baseline_from_params(params: Dict[str, np.ndarray], mode: str = "heuristic") -> np.ndarray:
    QT = clean_np(params["QT"])
    D1, D2 = clean_np(params["D1"]), clean_np(params["D2"])
    eps1, eps2 = clean_np(params["eps1"]), clean_np(params["eps2"])
    L1, L2 = clean_np(params["L1"]), clean_np(params["L2"])
    rho, mu = clean_np(params["rho"]), clean_np(params["mu"])

    if mode == "conductance":
        c1 = (np.maximum(D1, 1e-12) ** 5) / np.maximum(L1, 1e-12)
        c2 = (np.maximum(D2, 1e-12) ** 5) / np.maximum(L2, 1e-12)
        r = safe_clip_ratio(c1 / np.maximum(c1 + c2, 1e-12))
        Q1 = QT * r
    else:
        Q1 = 0.5 * QT

    Q2 = QT - Q1
    x1 = compute_explicit_x(Q1, D1, eps1, rho, mu)
    x2 = compute_explicit_x(Q2, D2, eps2, rho, mu)
    return clean_np(np.stack([Q1, x1, x2], axis=1))


def load_coeffs(data, coeff_key=None, degree=25):
    if coeff_key is None:
        coeff_key = require_key(data, ["coeffs", "taylor_coeffs", "taylor", "X_seq", "x_seq"], "coeffs")
    c = clean_np(data[coeff_key])
    if c.ndim == 3:
        if c.shape[1] == 3:
            c = c[:, :, : degree + 1]
            c = np.transpose(c, (0, 2, 1))
        elif c.shape[2] == 3:
            c = c[:, : degree + 1, :]
    elif c.ndim == 2:
        c = c[:, : degree + 1, None]
    else:
        raise ValueError(f"unsupported coeff shape: {c.shape}")
    return signed_log1p(c)


def load_targets(data, target_key=None):
    if target_key is None:
        target_key = first_existing_key(data, ["z_true", "targets", "target", "y", "solution", "z_star"])
    if target_key:
        y = clean_np(data[target_key])
        if y.ndim == 2 and y.shape[1] >= 3:
            return clean_np(y[:, :3])
    q1 = first_existing_key(data, ["Q1", "Q1_true", "q1_true", "Q1_star"])
    x1 = first_existing_key(data, ["x1", "x1_true", "x1_star"])
    x2 = first_existing_key(data, ["x2", "x2_true", "x2_star"])
    if q1 and x1 and x2:
        return clean_np(np.stack([data[q1], data[x1], data[x2]], axis=1))
    raise KeyError(f"Cannot find target keys. actual={data.files}")


def load_baseline(data, params, baseline_key=None, baseline_mode="heuristic"):
    if baseline_key is None:
        baseline_key = first_existing_key(data, ["z_base", "baseline", "z0", "init", "heuristic_z0"])
    if baseline_key:
        z = clean_np(data[baseline_key])
        if z.ndim == 2 and z.shape[1] >= 3:
            return clean_np(z[:, :3])
    return compute_baseline_from_params(params, mode=baseline_mode)


def build_global_features(params, z_base):
    keys = ["QT", "D1", "D2", "eps1", "eps2", "L1", "L2", "rho", "mu", "g"]
    phys = np.concatenate([ensure_2d(params[k]) for k in keys], axis=1)
    phys = signed_log1p(np.maximum(phys, 0.0))
    zb = signed_log1p(z_base)
    return clean_np(np.concatenate([phys, zb], axis=1), clip=100.0)


class SplitData:
    def __init__(self, x_seq, x_glob, z_base, z_true, target_raw, params):
        self.x_seq = x_seq
        self.x_glob = x_glob
        self.z_base = z_base
        self.z_true = z_true
        self.target_raw = target_raw
        self.params = params


def filter_finite_split(split: SplitData, name: str) -> SplitData:
    arrays = [
        split.x_seq.reshape(split.x_seq.shape[0], -1),
        split.x_glob,
        split.z_base,
        split.z_true,
        split.target_raw,
    ]
    mask = np.ones(split.z_true.shape[0], dtype=bool)
    for a in arrays:
        mask &= np.all(np.isfinite(a), axis=1)

    # target 폭주 방지. no-logit delta_r는 원칙적으로 -1~1 근처여야 함.
    mask &= np.abs(split.target_raw[:, 0]) < 2.0
    mask &= np.abs(split.target_raw[:, 1]) < 1e3
    mask &= np.abs(split.target_raw[:, 2]) < 1e3

    kept = int(mask.sum())
    total = int(mask.shape[0])
    print(f"[FILTER] {name}: kept {kept}/{total} finite samples")

    if kept == 0:
        raise RuntimeError(f"No finite samples left in {name}")

    params2 = {k: v[mask] for k, v in split.params.items()}
    return SplitData(
        x_seq=split.x_seq[mask],
        x_glob=split.x_glob[mask],
        z_base=split.z_base[mask],
        z_true=split.z_true[mask],
        target_raw=split.target_raw[mask],
        params=params2,
    )


def load_split(path, degree, coeff_key, target_key, baseline_key, baseline_mode):
    data = np.load(path, allow_pickle=True)
    params = load_param_arrays(data)
    coeffs = load_coeffs(data, coeff_key, degree)
    z_true = load_targets(data, target_key)
    z_base = load_baseline(data, params, baseline_key, baseline_mode)

    QT = np.maximum(clean_np(params["QT"]), 1e-9)
    r_true = safe_clip_ratio(safe_divide(z_true[:, 0], QT))
    r_base = safe_clip_ratio(safe_divide(z_base[:, 0], QT))

    target_raw = clean_np(np.stack([
        r_true - r_base,
        z_true[:, 1] - z_base[:, 1],
        z_true[:, 2] - z_base[:, 2],
    ], axis=1), clip=1e4)

    x_glob = build_global_features(params, z_base)

    split = SplitData(coeffs, x_glob, z_base, z_true, target_raw, params)
    return filter_finite_split(split, str(path))


# ============================================================
# Dataset / Models
# ============================================================

class CoupledDataset(Dataset):
    def __init__(self, split, seq_mean, seq_std, glob_mean, glob_std, y_mean, y_std):
        self.x_seq = standardize_apply(split.x_seq, seq_mean, seq_std)
        self.x_glob = standardize_apply(split.x_glob, glob_mean, glob_std)
        self.y = standardize_apply(split.target_raw, y_mean, y_std)

        print("[DATASET] x_seq finite:", np.isfinite(self.x_seq).mean())
        print("[DATASET] x_glob finite:", np.isfinite(self.x_glob).mean())
        print("[DATASET] y finite:", np.isfinite(self.y).mean())
        print("[DATASET] y min/max:", np.min(self.y), np.max(self.y))

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x_seq[idx]).float(),
            torch.from_numpy(self.x_glob[idx]).float(),
            torch.from_numpy(self.y[idx]).float(),
        )


class MLPBackbone(nn.Module):
    def __init__(self, seq_len, seq_dim, glob_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len * seq_dim + glob_dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 3),
        )

    def forward(self, x_seq, x_glob):
        return self.net(torch.cat([x_seq.reshape(x_seq.shape[0], -1), x_glob], dim=-1))


class LSTMBackbone(nn.Module):
    def __init__(self, seq_dim, glob_dim):
        super().__init__()
        self.rnn = nn.LSTM(seq_dim, 128, batch_first=True)
        self.head = nn.Sequential(nn.Linear(128 + glob_dim, 128), nn.GELU(), nn.Linear(128, 3))

    def forward(self, x_seq, x_glob):
        _, (h, _) = self.rnn(x_seq)
        return self.head(torch.cat([h[-1], x_glob], dim=-1))


class GRUBackbone(nn.Module):
    def __init__(self, seq_dim, glob_dim):
        super().__init__()
        self.rnn = nn.GRU(seq_dim, 96, batch_first=True)
        self.head = nn.Sequential(nn.Linear(96 + glob_dim, 128), nn.GELU(), nn.Linear(128, 3))

    def forward(self, x_seq, x_glob):
        _, h = self.rnn(x_seq)
        return self.head(torch.cat([h[-1], x_glob], dim=-1))


class TransformerBackbone(nn.Module):
    def __init__(self, seq_len, seq_dim, glob_dim):
        super().__init__()
        d_model = 96
        self.d_model = d_model
        self.proj = nn.Linear(seq_dim, d_model)

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

        # 넉넉하게 잡음: 실제 seq_len + CLS보다 작아서 터지는 문제 방지
        # degree 25면 coeff 길이 26, CLS 추가 후 27이 필요함.
        self.pos = nn.Parameter(torch.zeros(1, seq_len + 2, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=192,
            batch_first=True,
            activation="gelu",
            dropout=0.0,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(d_model + glob_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x_seq, x_glob):
        b = x_seq.shape[0]
        x = self.proj(x_seq)

        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # 혹시 데이터 길이가 예상보다 길어져도 자동 확장
        if x.shape[1] > self.pos.shape[1]:
            extra_len = x.shape[1] - self.pos.shape[1]
            extra = torch.zeros(
                1,
                extra_len,
                self.d_model,
                device=self.pos.device,
                dtype=self.pos.dtype,
            )
            nn.init.normal_(extra, std=0.02)
            self.pos = nn.Parameter(torch.cat([self.pos, extra], dim=1))

        x = x + self.pos[:, :x.shape[1], :]

        h = self.enc(x)[:, 0]
        return self.head(torch.cat([h, x_glob], dim=-1))

def build_model(name, seq_len, seq_dim, glob_dim):
    if name == "mlp":
        return MLPBackbone(seq_len, seq_dim, glob_dim)
    if name == "lstm":
        return LSTMBackbone(seq_dim, glob_dim)
    if name == "gru":
        return GRUBackbone(seq_dim, glob_dim)
    if name == "transformer":
        return TransformerBackbone(seq_len, seq_dim, glob_dim)
    raise ValueError(name)


# ============================================================
# Train / Eval
# ============================================================

def train_epoch(model, loader, optimizer, loss_fn, device, grad_clip):
    model.train()
    total, n = 0.0, 0
    for x_seq, x_glob, y in loader:
        x_seq, x_glob, y = x_seq.to(device), x_glob.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_seq, x_glob)
        loss = loss_fn(pred, y)

        if not torch.isfinite(loss):
            print("[ERROR] non-finite loss")
            print("x_seq finite:", torch.isfinite(x_seq).float().mean().item())
            print("x_glob finite:", torch.isfinite(x_glob).float().mean().item())
            print("y finite:", torch.isfinite(y).float().mean().item())
            print("pred finite:", torch.isfinite(pred).float().mean().item())
            print("y min max:", y.min().item(), y.max().item())
            print("pred min max:", pred.min().item(), pred.max().item())
            raise RuntimeError("Non-finite loss detected")

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = y.shape[0]
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def eval_loss(model, loader, loss_fn, device):
    model.eval()
    total, n = 0.0, 0
    for x_seq, x_glob, y in loader:
        x_seq, x_glob, y = x_seq.to(device), x_glob.to(device), y.to(device)
        pred = model(x_seq, x_glob)
        loss = loss_fn(pred, y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss detected")
        bs = y.shape[0]
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def predict_raw(model, split, seq_mean, seq_std, glob_mean, glob_std, y_mean, y_std, batch_size, device):
    model.eval()
    xs = standardize_apply(split.x_seq, seq_mean, seq_std)
    xg = standardize_apply(split.x_glob, glob_mean, glob_std)
    outs = []
    for s in range(0, xs.shape[0], batch_size):
        e = min(s + batch_size, xs.shape[0])
        pred = model(torch.from_numpy(xs[s:e]).float().to(device),
                     torch.from_numpy(xg[s:e]).float().to(device))
        pred = pred.cpu().numpy()
        outs.append(clean_np(pred * y_std + y_mean))
    return np.concatenate(outs, axis=0)


def decode_nologit(pred_delta, z_base, params):
    QT = np.maximum(clean_np(params["QT"]), 1e-9)
    r_base = safe_clip_ratio(safe_divide(z_base[:, 0], QT))
    r_hat = safe_clip_ratio(r_base + pred_delta[:, 0])
    Q1 = QT * r_hat
    x1 = np.maximum(z_base[:, 1] + pred_delta[:, 1], 1e-8)
    x2 = np.maximum(z_base[:, 2] + pred_delta[:, 2], 1e-8)
    return clean_np(np.stack([Q1, x1, x2], axis=1))


def params_for_i(params, i):
    return {k: float(v[i]) for k, v in params.items()}


def evaluate_direct(z_pred, split):
    residuals = []
    valid = []
    for i in range(z_pred.shape[0]):
        p = params_for_i(split.params, i)
        z = project_z(z_pred[i], p)
        rn = residual_norm(z, p)
        residuals.append(rn)
        valid.append(np.isfinite(rn))
    rs = residual_stats(np.asarray(residuals))
    return {
        "direct_mae": mae_np(z_pred, split.z_true),
        "direct_rmse": rmse_np(z_pred, split.z_true),
        "direct_r2": r2_np(z_pred, split.z_true),
        "direct_valid_ratio": float(np.mean(valid)),
        "direct_residual_mean": rs["mean"],
        "direct_residual_median": rs["median"],
        "direct_residual_p90": rs["p90"],
        "direct_residual_max": rs["max"],
        "max_abs_error": float(np.max(np.abs(clean_np(z_pred) - clean_np(split.z_true)))),
    }


def evaluate_newton(z_init, split, tol, max_iter):
    n = z_init.shape[0]
    z_ref = np.zeros_like(z_init, dtype=np.float64)
    residuals, iters, convs = [], [], []
    t0 = time.perf_counter()
    for i in range(n):
        z, r, it, ok = newton_refine(z_init[i], params_for_i(split.params, i), tol, max_iter)
        z_ref[i] = z
        residuals.append(r)
        iters.append(it)
        convs.append(ok)
    t1 = time.perf_counter()
    rs = residual_stats(np.asarray(residuals))
    return {
        "plus_newton_mae": mae_np(z_ref, split.z_true),
        "plus_newton_rmse": rmse_np(z_ref, split.z_true),
        "plus_newton_r2": r2_np(z_ref, split.z_true),
        "plus_newton_residual_mean": rs["mean"],
        "plus_newton_residual_median": rs["median"],
        "plus_newton_residual_p90": rs["p90"],
        "plus_newton_iter_mean": float(np.mean(iters)),
        "plus_newton_iter_median": float(np.median(iters)),
        "plus_newton_iter_p90": float(np.percentile(iters, 90)),
        "plus_newton_converged_ratio": float(np.mean(convs)),
        "plus_newton_ms_per_sample": float((t1 - t0) * 1000.0 / max(n, 1)),
    }


def save_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--val-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--degree", type=int, default=25)
    ap.add_argument("--model", choices=["mlp", "lstm", "gru", "transformer"], default="lstm")
    ap.add_argument("--coeff-key", default=None)
    ap.add_argument("--target-key", default=None)
    ap.add_argument("--baseline-key", default=None)
    ap.add_argument("--baseline-mode", choices=["heuristic", "conductance"], default="heuristic")
    ap.add_argument("--explicit-formula", default="haaland")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--eval-batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trial-id", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--newton-tol", type=float, default=1e-12)
    ap.add_argument("--newton-max-iter", type=int, default=20)
    ap.add_argument("--amp", action="store_true")  # compatibility only, not used
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    train = load_split(args.train_npz, args.degree, args.coeff_key, args.target_key, args.baseline_key, args.baseline_mode)
    val = load_split(args.val_npz, args.degree, args.coeff_key, args.target_key, args.baseline_key, args.baseline_mode)
    test = load_split(args.test_npz, args.degree, args.coeff_key, args.target_key, args.baseline_key, args.baseline_mode)

    seq_mean, seq_std = standardize_fit(train.x_seq)
    glob_mean, glob_std = standardize_fit(train.x_glob)
    y_mean, y_std = standardize_fit(train.target_raw)

    print("[SCALER] y_mean:", y_mean)
    print("[SCALER] y_std:", y_std)
    print("[SCALER] target min/max:", train.target_raw.min(axis=0), train.target_raw.max(axis=0))

    train_ds = CoupledDataset(train, seq_mean, seq_std, glob_mean, glob_std, y_mean, y_std)
    val_ds = CoupledDataset(val, seq_mean, seq_std, glob_mean, glob_std, y_mean, y_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    seq_len, seq_dim = train.x_seq.shape[1], train.x_seq.shape[2]
    glob_dim = train.x_glob.shape[1]
    model = build_model(args.model, seq_len, seq_dim, glob_dim).to(device)

    if args.lr is None:
        args.lr = {"mlp": 5e-4, "lstm": 1e-3, "gru": 5e-4, "transformer": 5e-4}[args.model]
    if args.weight_decay is None:
        args.weight_decay = 1e-5 if args.model == "gru" else 1e-4

    loss_fn = nn.SmoothL1Loss(beta=1.0) if args.model == "mlp" else nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_epoch = -1
    best_path = out_dir / f"best_nologit_deg{args.degree}_{args.model}.pt"
    hist = []

    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, opt, loss_fn, device, args.grad_clip)
        va = eval_loss(model, val_loader, loss_fn, device)

        hist.append({"epoch": epoch, "train_loss": tr, "val_loss": va, "lr": opt.param_groups[0]["lr"]})

        if va < best_val:
            best_val = va
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "seq_mean": seq_mean,
                "seq_std": seq_std,
                "glob_mean": glob_mean,
                "glob_std": glob_std,
                "y_mean": y_mean,
                "y_std": y_std,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "args": vars(args),
            }, best_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"[epoch {epoch:04d}] train={tr:.6e} val={va:.6e} best={best_val:.6e}@{best_epoch}")

        if epoch - best_epoch >= args.patience:
            print(f"[EARLY STOP] epoch={epoch}, best_epoch={best_epoch}")
            break

    train_time = time.perf_counter() - t0
    save_csv(out_dir / "train_history.csv", hist)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    pred_delta = predict_raw(model, test, seq_mean, seq_std, glob_mean, glob_std, y_mean, y_std,
                             args.eval_batch_size, device)
    z_direct = decode_nologit(pred_delta, test.z_base, test.params)

    result = {
        "experiment": "no_logit_correction_ablation",
        "degree": args.degree,
        "model": args.model.upper() if args.model != "transformer" else "Transformer",
        "trial": args.trial_id,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val),
        "train_time_sec": float(train_time),
        "target_definition": "[r_star-r_base, x1_star-x1_base, x2_star-x2_base]",
        "decode_definition": "r_hat=clip(r_base+delta_r), Q1_hat=QT*r_hat",
    }
    result.update(evaluate_direct(z_direct, test))
    result.update(evaluate_newton(z_direct, test, args.newton_tol, args.newton_max_iter))

    json_path = out_dir / f"trial_{args.trial_id:03d}_{args.model}_nologit.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    save_csv(out_dir / f"trial_{args.trial_id:03d}_{args.model}_nologit_compact.csv", [{
        "Degree": args.degree,
        "Model": result["model"],
        "Direct RMSE": result["direct_rmse"],
        "Direct Residual Mean": result["direct_residual_mean"],
        "Newton RMSE": result["plus_newton_rmse"],
        "Newton Residual Mean": result["plus_newton_residual_mean"],
        "Newton Iter.": result["plus_newton_iter_mean"],
        "Conv. Ratio": result["plus_newton_converged_ratio"],
    }])

    print("[DONE]", json_path)
    for k in ["direct_rmse", "direct_residual_mean", "plus_newton_rmse",
              "plus_newton_residual_mean", "plus_newton_iter_mean",
              "plus_newton_converged_ratio"]:
        print(k, result[k])


if __name__ == "__main__":
    main()