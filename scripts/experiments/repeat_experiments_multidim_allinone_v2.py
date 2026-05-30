#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import random
import statistics
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PI = math.pi
LN10 = math.log(10.0)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sanitize_array(x: np.ndarray, clip_value: float = 1e12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)
    x = np.clip(x, -clip_value, clip_value)
    return x

def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))

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
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def mean_std(values):
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0, "min": vals[0], "max": vals[0]}
    return {
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals),
        "min": min(vals),
        "max": max(vals),
    }

class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        X = sanitize_array(X)
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0
        self.mean[~np.isfinite(self.mean)] = 0.0
        self.std[~np.isfinite(self.std)] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = sanitize_array(X)
        Xs = (X - self.mean) / self.std
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        Xs = np.clip(Xs, -1e6, 1e6)
        return Xs.astype(np.float32)

    def save(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @staticmethod
    def from_saved(saved: Dict):
        s = Standardizer()
        s.mean = np.array(saved["mean"], dtype=np.float64)
        s.std = np.array(saved["std"], dtype=np.float64)
        return s

# Physics
def re_from_Q(Q, rho, mu, D):
    return 4.0 * rho * Q / (PI * mu * D)

def colebrook_single_x_eq(x, Re, rel_rough):
    x = np.asarray(x, dtype=np.float64)
    Re = np.asarray(Re, dtype=np.float64)
    rel_rough = np.asarray(rel_rough, dtype=np.float64)
    z = rel_rough / 3.7 + 2.51 * x / Re
    out = np.full_like(x, np.nan, dtype=np.float64)
    mask = (Re > 0) & (z > 0)
    out[mask] = x[mask] + 2.0 * np.log10(z[mask])
    return out

def colebrook_single_x_df(x, Re, rel_rough):
    x = np.asarray(x, dtype=np.float64)
    Re = np.asarray(Re, dtype=np.float64)
    rel_rough = np.asarray(rel_rough, dtype=np.float64)
    z = rel_rough / 3.7 + 2.51 * x / Re
    out = np.full_like(x, np.nan, dtype=np.float64)
    mask = (Re > 0) & (z > 0)
    out[mask] = 1.0 + 2.0 * ((2.51 / Re[mask]) / (z[mask] * LN10))
    return out

def solve_x_from_Q(Q, D, eps, rho, mu, x_init=7.0, tol=1e-13, max_iter=50):
    Re = re_from_Q(np.array([Q]), np.array([rho]), np.array([mu]), np.array([D]))[0]
    rr = eps / D
    x = float(x_init)
    for _ in range(max_iter):
        fx = colebrook_single_x_eq(np.array([x]), np.array([Re]), np.array([rr]))[0]
        dfx = colebrook_single_x_df(np.array([x]), np.array([Re]), np.array([rr]))[0]
        if (not np.isfinite(fx)) or (not np.isfinite(dfx)) or abs(dfx) < 1e-15:
            break
        x_new = float(np.clip(x - fx / dfx, 1e-3, 1e3))
        if abs(x_new - x) < tol and abs(fx) < tol:
            x = x_new
            break
        x = x_new
    return float(x)

def head_loss(Q, x, L, D, g):
    return 8.0 * L * (Q ** 2) / (g * (PI ** 2) * (D ** 5) * (x ** 2))

def system_F(z, params):
    Q1, x1, x2 = z[..., 0], z[..., 1], z[..., 2]
    QT = params["Q_total"]; D1 = params["D1"]; D2 = params["D2"]; eps1 = params["eps1"]; eps2 = params["eps2"]
    L1 = params["L1"]; L2 = params["L2"]; rho = params["rho"]; mu = params["mu"]; g = params["g"]
    Q2 = QT - Q1
    Re1 = re_from_Q(Q1, rho, mu, D1); Re2 = re_from_Q(Q2, rho, mu, D2)
    rr1 = eps1 / D1; rr2 = eps2 / D2
    F1 = colebrook_single_x_eq(x1, Re1, rr1)
    F2 = colebrook_single_x_eq(x2, Re2, rr2)
    F3 = head_loss(Q1, x1, L1, D1, g) - head_loss(Q2, x2, L2, D2, g)
    return np.stack([F1, F2, F3], axis=-1)

def numerical_jacobian_single(z, p, eps=1e-6):
    z = np.asarray(z, dtype=np.float64)
    J = np.zeros((3, 3), dtype=np.float64)
    f0 = system_F(z[None, :], p)[0]
    for j in range(3):
        zp = z.copy(); zm = z.copy()
        step = eps * max(1.0, abs(z[j]))
        zp[j] += step; zm[j] -= step
        fp = system_F(zp[None, :], p)[0]; fm = system_F(zm[None, :], p)[0]
        J[:, j] = (fp - fm) / (2.0 * step)
    return J, f0

def project_feasible(z, p):
    z = np.asarray(z, dtype=np.float64).copy()
    QT = float(p["Q_total"])
    z[0] = np.clip(z[0], max(1e-8, QT * 1e-5), QT - max(1e-8, QT * 1e-5))
    z[1] = max(z[1], 1e-3); z[2] = max(z[2], 1e-3)
    return z

def newton_system_single(z0, p, tol=1e-12, max_iter=20, damping=1.0):
    z = project_feasible(z0, p)
    converged = False; used_iter = 0
    for k in range(1, max_iter + 1):
        J, f = numerical_jacobian_single(z, p)
        if not np.all(np.isfinite(J)) or not np.all(np.isfinite(f)):
            break
        try:
            step = np.linalg.solve(J, f)
        except np.linalg.LinAlgError:
            break
        step = np.clip(step, -5.0, 5.0)
        z_new = project_feasible(z - damping * step, p)
        f_new = system_F(z_new[None, :], p)[0]
        if np.linalg.norm(f_new, ord=2) > np.linalg.norm(f, ord=2):
            z_half = project_feasible(z - 0.5 * damping * step, p)
            f_half = system_F(z_half[None, :], p)[0]
            if np.linalg.norm(f_half, ord=2) < np.linalg.norm(f_new, ord=2):
                z_new, f_new = z_half, f_half
        z = z_new; used_iter = k
        if np.linalg.norm(f_new, ord=np.inf) <= tol:
            converged = True
            break
    return z, used_iter, converged

def refine_batch(z_init, data, tol=1e-12, max_iter=20):
    n = len(z_init)
    out = np.zeros_like(z_init, dtype=np.float64)
    iters = np.zeros(n, dtype=np.int32)
    conv = np.zeros(n, dtype=bool)
    for i in range(n):
        p = {k: float(np.asarray(data[k])[i]) for k in ["Q_total","D1","D2","eps1","eps2","L1","L2","rho","mu","g"]}
        zf, it, ok = newton_system_single(z_init[i], p, tol=tol, max_iter=max_iter)
        out[i] = zf; iters[i] = it; conv[i] = ok
    return out, iters, conv

# Data / targets
def safe_logit(r):
    r = np.clip(r, 1e-6, 1.0 - 1e-6)
    return np.log(r / (1.0 - r))

def compute_delta_targets(y, z0, q_total):
    r_true = np.clip(y[:, 0] / q_total, 1e-6, 1.0 - 1e-6)
    r0 = np.clip(z0[:, 0] / q_total, 1e-6, 1.0 - 1e-6)
    dlogit_r = (safe_logit(r_true) - safe_logit(r0)).reshape(-1, 1)
    dx1 = (y[:, 1] - z0[:, 1]).reshape(-1, 1)
    dx2 = (y[:, 2] - z0[:, 2]).reshape(-1, 1)
    return np.concatenate([dlogit_r, dx1, dx2], axis=1)

def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    required = ["coeffs","center","target","Q_total","D1","D2","eps1","eps2","L1","L2","rho","mu","g"]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {npz_path}. Available: {list(data.keys())}")
    return {k: np.asarray(data[k]) for k in required}

def build_inputs_and_baseline(data: Dict[str, np.ndarray], use_log_features: bool = True):
    coeffs = sanitize_array(np.asarray(data["coeffs"], dtype=np.float64), clip_value=1e30)
    center = sanitize_array(np.asarray(data["center"], dtype=np.float64), clip_value=1e12)
    y = sanitize_array(np.asarray(data["target"], dtype=np.float64), clip_value=1e12)
    coeffs = signed_log1p(coeffs)
    seq_x = np.concatenate([coeffs, center[..., None]], axis=2)
    seq_x = sanitize_array(seq_x, 1e12)

    globals_raw = [np.asarray(data[k], dtype=np.float64).reshape(-1,1) for k in ["Q_total","D1","D2","eps1","eps2","L1","L2","rho","mu","g"]]
    globals_raw = [sanitize_array(g, 1e12) for g in globals_raw]
    if use_log_features:
        globals_proc = []
        for i, arr in enumerate(globals_raw):
            globals_proc.append(np.log(np.clip(arr, 1e-12, None)) if i < 9 else arr)
        glob_x = np.concatenate(globals_proc, axis=1)
    else:
        glob_x = np.concatenate(globals_raw, axis=1)
    glob_x = sanitize_array(glob_x, 1e12)

    QT = np.asarray(data["Q_total"], dtype=np.float64)
    D1 = np.asarray(data["D1"], dtype=np.float64); D2 = np.asarray(data["D2"], dtype=np.float64)
    eps1 = np.asarray(data["eps1"], dtype=np.float64); eps2 = np.asarray(data["eps2"], dtype=np.float64)
    rho = np.asarray(data["rho"], dtype=np.float64); mu = np.asarray(data["mu"], dtype=np.float64)

    n = len(QT)
    z0 = np.zeros((n, 3), dtype=np.float64); z0[:, 0] = QT / 2.0
    for i in range(n):
        qh = QT[i] / 2.0
        z0[i,1] = solve_x_from_Q(qh, D1[i], eps1[i], rho[i], mu[i])
        z0[i,2] = solve_x_from_Q(qh, D2[i], eps2[i], rho[i], mu[i])

    delta_target = compute_delta_targets(y, z0, QT)
    return seq_x, glob_x, y, z0, delta_target

class HybridDataset(Dataset):
    def __init__(self, seq_x, glob_x, y, z0, delta_target, raw_data):
        self.seq_x = torch.from_numpy(seq_x.astype(np.float32))
        self.glob_x = torch.from_numpy(glob_x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.z0 = torch.from_numpy(z0.astype(np.float32))
        self.delta_target = torch.from_numpy(delta_target.astype(np.float32))
        self.raw = {}
        for k in ["Q_total","D1","D2","eps1","eps2","L1","L2","rho","mu","g"]:
            self.raw[k] = torch.from_numpy(np.asarray(raw_data[k]).astype(np.float32).reshape(-1,1))

    def __len__(self): return self.y.shape[0]

    def __getitem__(self, idx):
        item = {"seq_x": self.seq_x[idx], "glob_x": self.glob_x[idx], "y": self.y[idx], "z0": self.z0[idx], "delta_target": self.delta_target[idx]}
        for k, v in self.raw.items():
            item[k] = v[idx]
        return item

def standardize_datasets(train_ds, val_ds, test_ds):
    seq_scaler = Standardizer(); glob_scaler = Standardizer(); delta_scaler = Standardizer()
    seq_scaler.fit(train_ds.seq_x.numpy().reshape(-1, train_ds.seq_x.shape[-1]))
    train_ds.seq_x = torch.from_numpy(seq_scaler.transform(train_ds.seq_x.numpy().reshape(-1, train_ds.seq_x.shape[-1])).reshape(train_ds.seq_x.shape))
    val_ds.seq_x = torch.from_numpy(seq_scaler.transform(val_ds.seq_x.numpy().reshape(-1, val_ds.seq_x.shape[-1])).reshape(val_ds.seq_x.shape))
    test_ds.seq_x = torch.from_numpy(seq_scaler.transform(test_ds.seq_x.numpy().reshape(-1, test_ds.seq_x.shape[-1])).reshape(test_ds.seq_x.shape))
    if train_ds.glob_x.shape[1] > 0:
        glob_scaler.fit(train_ds.glob_x.numpy())
        train_ds.glob_x = torch.from_numpy(glob_scaler.transform(train_ds.glob_x.numpy()))
        val_ds.glob_x = torch.from_numpy(glob_scaler.transform(val_ds.glob_x.numpy()))
        test_ds.glob_x = torch.from_numpy(glob_scaler.transform(test_ds.glob_x.numpy()))
    else:
        glob_scaler.mean = np.zeros((1,0), dtype=np.float64); glob_scaler.std = np.ones((1,0), dtype=np.float64)
    delta_scaler.fit(train_ds.delta_target.numpy())
    train_ds.delta_target = torch.from_numpy(delta_scaler.transform(train_ds.delta_target.numpy()))
    val_ds.delta_target = torch.from_numpy(delta_scaler.transform(val_ds.delta_target.numpy()))
    test_ds.delta_target = torch.from_numpy(delta_scaler.transform(test_ds.delta_target.numpy()))
    return seq_scaler, glob_scaler, delta_scaler

# Models
class MLPBackbone(nn.Module):
    def __init__(self, seq_dim, seq_len, glob_dim, hidden_dims=(256,256,128), dropout=0.1):
        super().__init__()
        in_dim = seq_dim * seq_len + glob_dim
        layers = []; prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.feat = nn.Sequential(*layers); self.out_dim = prev

    def forward(self, seq_x, glob_x):
        return self.feat(torch.cat([seq_x.flatten(1), glob_x], dim=1))

class LSTMBackbone(nn.Module):
    def __init__(self, seq_dim, glob_dim, hidden_size=128, num_layers=2, dropout=0.1, head_hidden=128, head_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(seq_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fusion = self._build_head(hidden_size + glob_dim, head_hidden, head_hidden, dropout, head_layers)
        self.out_dim = head_hidden

    @staticmethod
    def _build_head(in_dim, hidden_dim, out_dim, dropout, head_layers):
        layers = []; prev = in_dim; cur = hidden_dim
        for _ in range(max(head_layers - 1, 0)):
            layers += [nn.Linear(prev, cur), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = cur; cur = max(cur // 2, 32)
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        _, (hn, _) = self.lstm(seq_x)
        return self.fusion(torch.cat([hn[-1], glob_x], dim=1))

class GRUBackbone(nn.Module):
    def __init__(self, seq_dim, glob_dim, hidden_size=128, num_layers=2, dropout=0.1, head_hidden=128, head_layers=2):
        super().__init__()
        self.gru = nn.GRU(seq_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fusion = LSTMBackbone._build_head(hidden_size + glob_dim, head_hidden, head_hidden, dropout, head_layers)
        self.out_dim = head_hidden

    def forward(self, seq_x, glob_x):
        _, hn = self.gru(seq_x)
        return self.fusion(torch.cat([hn[-1], glob_x], dim=1))

class TransformerBackbone(nn.Module):
    def __init__(self, seq_dim, seq_len, glob_dim, d_model=96, nhead=4, num_layers=2, dropout=0.1, ff_dim=192, head_hidden=128, head_layers=2, use_cls_token=True):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.input_proj = nn.Linear(seq_dim, d_model)
        total_len = seq_len + (1 if use_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, total_len, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if use_cls_token else None
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fusion = LSTMBackbone._build_head(d_model + glob_dim, head_hidden, head_hidden, dropout, head_layers)
        self.out_dim = head_hidden

    def forward(self, seq_x, glob_x):
        bsz = seq_x.size(0)
        x = self.input_proj(seq_x)
        if self.use_cls_token:
            x = torch.cat([self.cls_token.expand(bsz, -1, -1), x], dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        h = self.norm(self.encoder(x))
        pooled = h[:, 0, :] if self.use_cls_token else h.mean(dim=1)
        return self.fusion(torch.cat([pooled, glob_x], dim=1))

class HybridCorrectionModel(nn.Module):
    def __init__(self, model_name, seq_dim, seq_len, glob_dim, hp):
        super().__init__()
        if model_name == "mlp":
            self.backbone = MLPBackbone(seq_dim, seq_len, glob_dim, hidden_dims=tuple(hp["hidden_dims"]), dropout=hp["dropout"])
        elif model_name == "lstm":
            self.backbone = LSTMBackbone(seq_dim, glob_dim, hidden_size=hp["hidden_size"], num_layers=hp["num_layers"], dropout=hp["dropout"], head_hidden=hp["head_hidden"], head_layers=hp["head_layers"])
        elif model_name == "gru":
            self.backbone = GRUBackbone(seq_dim, glob_dim, hidden_size=hp["hidden_size"], num_layers=hp["num_layers"], dropout=hp["dropout"], head_hidden=hp["head_hidden"], head_layers=hp["head_layers"])
        elif model_name == "transformer":
            self.backbone = TransformerBackbone(seq_dim, seq_len, glob_dim, d_model=hp["d_model"], nhead=hp["nhead"], num_layers=hp["num_layers"], dropout=hp["dropout"], ff_dim=hp["ff_dim"], head_hidden=hp["head_hidden"], head_layers=hp["head_layers"], use_cls_token=hp["use_cls_token"])
        else:
            raise ValueError(model_name)
        self.delta_head = nn.Linear(self.backbone.out_dim, 3)

    def forward_delta(self, seq_x, glob_x):
        feat = self.backbone(seq_x, glob_x)
        return self.delta_head(feat)

    def decode(self, delta_norm, z0, q_total, delta_scaler_t):
        mean = delta_scaler_t["mean"].to(delta_norm.device)
        std = delta_scaler_t["std"].to(delta_norm.device)
        delta_real = delta_norm * std + mean
        q0 = z0[:, 0]; x10 = z0[:, 1]; x20 = z0[:, 2]
        r0 = torch.clamp(q0 / q_total.squeeze(1), 1e-6, 1.0 - 1e-6)
        logit_r0 = torch.log(r0 / (1.0 - r0))
        logit_r = logit_r0 + delta_real[:, 0]
        r = torch.sigmoid(logit_r)
        q1 = r * q_total.squeeze(1)
        x1 = torch.clamp(x10 + delta_real[:, 1], min=1e-3)
        x2 = torch.clamp(x20 + delta_real[:, 2], min=1e-3)
        pred = torch.stack([q1, x1, x2], dim=1)
        return pred, delta_real

    def forward(self, seq_x, glob_x, z0, q_total, delta_scaler_t):
        delta_norm = self.forward_delta(seq_x, glob_x)
        pred, delta_real = self.decode(delta_norm, z0, q_total, delta_scaler_t)
        return pred, delta_norm, delta_real

# Train/eval wrappers
def build_hp_from_args(args):
    return {
        "model": args.model,
        "use_log_features": args.use_log_features,
        "optimizer": args.optimizer,
        "loss_name": args.loss_name,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dims": args.hidden_dims,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "head_hidden": args.head_hidden,
        "head_layers": args.head_layers,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "ff_dim": args.ff_dim,
        "use_cls_token": args.use_cls_token,
    }

def delta_supervised_loss(delta_pred_norm, delta_target_norm, loss_name="smoothl1"):
    if loss_name == "mse":
        return torch.mean((delta_pred_norm - delta_target_norm) ** 2)
    return torch.nn.functional.smooth_l1_loss(delta_pred_norm, delta_target_norm, beta=0.1)

def vector_metrics(pred, true):
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "mae": mae, "rmse": rmse, "r2": r2,
        "mae_Q1": float(np.mean(np.abs(err[:, 0]))),
        "mae_x1": float(np.mean(np.abs(err[:, 1]))),
        "mae_x2": float(np.mean(np.abs(err[:, 2]))),
        "max_abs_error": float(np.max(np.abs(err))),
    }

def residual_metrics(pred, data):
    params = {k: np.asarray(data[k], dtype=np.float64) for k in ["Q_total","D1","D2","eps1","eps2","L1","L2","rho","mu","g"]}
    F = system_F(pred.astype(np.float64), params)
    norms_inf = np.max(np.abs(F), axis=1)
    valid = np.all(np.isfinite(F), axis=1)
    return {
        "valid_ratio": float(np.mean(valid)),
        "residual_mean": float(np.nanmean(norms_inf)),
        "residual_median": float(np.nanmedian(norms_inf)),
        "residual_p90": percentile(norms_inf[np.isfinite(norms_inf)], 90),
    }

def run_eval(model, loader, loss_name, device, delta_scaler_t):
    model.eval()
    preds = []; trues = []; total_loss = 0.0; total_n = 0
    with torch.no_grad():
        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            pred, delta_norm, delta_real = model(batch["seq_x"], batch["glob_x"], batch["z0"], batch["Q_total"], delta_scaler_t)
            loss = delta_supervised_loss(delta_norm, batch["delta_target"], loss_name=loss_name)
            bs = pred.shape[0]
            total_loss += float(loss.detach().cpu().item()) * bs; total_n += bs
            preds.append(pred.detach().cpu().numpy()); trues.append(batch["y"].detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0); true = np.concatenate(trues, axis=0)
    m = vector_metrics(pred, true); m["loss"] = total_loss / max(total_n, 1)
    return m, pred, true

def train_once(args, seed: int, save_dir: Path):
    set_seed(seed); device = torch.device(args.device); save_dir.mkdir(parents=True, exist_ok=True)
    train_raw = load_npz(args.train_npz); val_raw = load_npz(args.val_npz); test_raw = load_npz(args.test_npz)
    tr_seq, tr_glob, tr_y, tr_z0, tr_delta = build_inputs_and_baseline(train_raw, use_log_features=args.use_log_features)
    va_seq, va_glob, va_y, va_z0, va_delta = build_inputs_and_baseline(val_raw, use_log_features=args.use_log_features)
    te_seq, te_glob, te_y, te_z0, te_delta = build_inputs_and_baseline(test_raw, use_log_features=args.use_log_features)

    train_ds = HybridDataset(tr_seq, tr_glob, tr_y, tr_z0, tr_delta, train_raw)
    val_ds = HybridDataset(va_seq, va_glob, va_y, va_z0, va_delta, val_raw)
    test_ds = HybridDataset(te_seq, te_glob, te_y, te_z0, te_delta, test_raw)

    seq_scaler, glob_scaler, delta_scaler = standardize_datasets(train_ds, val_ds, test_ds)
    delta_scaler_t = {
        "mean": torch.tensor(delta_scaler.mean.astype(np.float32), device=device),
        "std": torch.tensor(delta_scaler.std.astype(np.float32), device=device),
    }

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    hp = build_hp_from_args(args)
    model = HybridCorrectionModel(args.model, train_ds.seq_x.shape[2], train_ds.seq_x.shape[1], train_ds.glob_x.shape[1], hp).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) if args.optimizer == "adamw" else torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_rmse = float("inf"); best_epoch = -1; best_state = None; wait = 0
    for epoch in range(1, args.epochs + 1):
        model.train(); train_loss_sum = 0.0; train_n = 0
        for batch in train_loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            pred, delta_norm, delta_real = model(batch["seq_x"], batch["glob_x"], batch["z0"], batch["Q_total"], delta_scaler_t)
            loss = delta_supervised_loss(delta_norm, batch["delta_target"], loss_name=args.loss_name)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            bs = pred.shape[0]; train_loss_sum += float(loss.detach().cpu().item()) * bs; train_n += bs

        train_loss = train_loss_sum / max(train_n, 1)
        val_metrics, _, _ = run_eval(model, val_loader, args.loss_name, device, delta_scaler_t)
        print(f"[Epoch {epoch:04d}] train_loss={train_loss:.8f} val_loss={val_metrics['loss']:.8f} val_rmse={val_metrics['rmse']:.8f} val_mae={val_metrics['mae']:.8f} val_r2={val_metrics['r2']:.8f}")
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]; best_epoch = epoch; best_state = deepcopy(model.state_dict()); wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[Early stopping] patience={args.patience}")
                break

    if best_state is None:
        raise RuntimeError("Training failed.")
    model.load_state_dict(best_state)
    ckpt = {
        "model_state_dict": best_state,
        "seq_scaler": seq_scaler.save(),
        "glob_scaler": glob_scaler.save(),
        "delta_scaler": delta_scaler.save(),
        "hp": hp,
        "seq_dim": int(train_ds.seq_x.shape[2]),
        "seq_len": int(train_ds.seq_x.shape[1]),
        "glob_dim": int(train_ds.glob_x.shape[1]),
        "best_val_rmse": float(best_val_rmse),
        "best_epoch": int(best_epoch),
    }
    torch.save(ckpt, save_dir / "best_model.pt")
    return save_dir / "best_model.pt"

def load_model_checkpoint(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    hp = ckpt["hp"]
    model = HybridCorrectionModel(hp["model"], ckpt["seq_dim"], ckpt["seq_len"], ckpt["glob_dim"], hp)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device); model.eval()
    seq_scaler = {"mean": np.array(ckpt["seq_scaler"]["mean"], dtype=np.float64), "std": np.array(ckpt["seq_scaler"]["std"], dtype=np.float64)}
    glob_scaler = {"mean": np.array(ckpt["glob_scaler"]["mean"], dtype=np.float64), "std": np.array(ckpt["glob_scaler"]["std"], dtype=np.float64)}
    delta_scaler = {"mean": np.array(ckpt["delta_scaler"]["mean"], dtype=np.float64), "std": np.array(ckpt["delta_scaler"]["std"], dtype=np.float64)}
    return ckpt, model, seq_scaler, glob_scaler, delta_scaler

def evaluate_once(args, model_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_npz(args.test_npz)
    ckpt, model, seq_scaler, glob_scaler, delta_scaler = load_model_checkpoint(str(model_path), device=args.device)
    hp = ckpt["hp"]; use_log_features = bool(hp.get("use_log_features", False))
    seq_x, glob_x, y_true, z0, _ = build_inputs_and_baseline(data, use_log_features=use_log_features)
    seq_shape = seq_x.shape
    seq_scaler_obj = Standardizer.from_saved({"mean": seq_scaler["mean"].tolist(), "std": seq_scaler["std"].tolist()})
    glob_scaler_obj = Standardizer.from_saved({"mean": glob_scaler["mean"].tolist(), "std": glob_scaler["std"].tolist()})
    seq_x = seq_scaler_obj.transform(seq_x.reshape(-1, seq_x.shape[-1])).reshape(seq_shape)
    glob_x = glob_scaler_obj.transform(glob_x)
    delta_scaler_t = {"mean": torch.tensor(delta_scaler["mean"].astype(np.float32), device=args.device), "std": torch.tensor(delta_scaler["std"].astype(np.float32), device=args.device)}

    preds = []
    with torch.no_grad():
        bs = 4096
        for i in range(0, len(seq_x), bs):
            s = torch.from_numpy(seq_x[i:i+bs]).to(args.device)
            g = torch.from_numpy(glob_x[i:i+bs]).to(args.device)
            z = torch.from_numpy(z0[i:i+bs].astype(np.float32)).to(args.device)
            qt = torch.from_numpy(np.asarray(data["Q_total"][i:i+bs]).astype(np.float32).reshape(-1,1)).to(args.device)
            pred, delta_norm, delta_real = model(s, g, z, qt, delta_scaler_t)
            preds.append(pred.cpu().numpy())
    pred_direct = np.concatenate(preds, axis=0)
    heur_pred = z0.copy()

    rows = []
    hd = vector_metrics(heur_pred, y_true); hd.update(residual_metrics(heur_pred, data)); hd["name"] = "heuristic_direct"; rows.append(hd)
    href, hit, hconv = refine_batch(heur_pred.astype(np.float64), data, tol=args.tol, max_iter=args.max_newton_iter)
    hr = vector_metrics(href, y_true.astype(np.float64)); hr.update(residual_metrics(href, data)); hr["name"] = "heuristic_plus_newton"; hr["newton_iter_mean"] = float(np.mean(hit)); hr["newton_iter_median"] = float(np.median(hit)); hr["newton_iter_p90"] = float(np.percentile(hit, 90)); hr["newton_converged_ratio"] = float(np.mean(hconv)); rows.append(hr)
    nd = vector_metrics(pred_direct, y_true); nd.update(residual_metrics(pred_direct, data)); nd["name"] = "neural_direct"; rows.append(nd)
    nref, nit, nconv = refine_batch(pred_direct.astype(np.float64), data, tol=args.tol, max_iter=args.max_newton_iter)
    nr = vector_metrics(nref, y_true.astype(np.float64)); nr.update(residual_metrics(nref, data)); nr["name"] = "neural_plus_newton"; nr["newton_iter_mean"] = float(np.mean(nit)); nr["newton_iter_median"] = float(np.median(nit)); nr["newton_iter_p90"] = float(np.percentile(nit, 90)); nr["newton_converged_ratio"] = float(np.mean(nconv)); rows.append(nr)
    save_csv(out_dir / "summary_metrics.csv", rows)
    return rows

def repeat_experiments(args):
    output_root = Path(args.output_root); output_root.mkdir(parents=True, exist_ok=True)
    all_run_rows = []
    for i in range(args.num_runs):
        seed = args.seed_start + i
        print(f"\n[RUN {i+1}/{args.num_runs}] seed={seed}")
        run_dir = output_root / f"run_{seed:03d}"
        train_dir = run_dir / "train"; eval_dir = run_dir / "eval"
        model_path = train_once(args, seed=seed, save_dir=train_dir)
        summary_rows = evaluate_once(args, model_path=model_path, out_dir=eval_dir)
        row = {"seed": seed, "train_dir": str(train_dir), "eval_dir": str(eval_dir)}
        keep_metrics = ["mae","rmse","r2","valid_ratio","residual_mean","residual_median","residual_p90","max_abs_error","newton_iter_mean","newton_iter_median","newton_iter_p90","newton_converged_ratio"]
        for name in ["heuristic_direct","heuristic_plus_newton","neural_direct","neural_plus_newton"]:
            matched = [r for r in summary_rows if r["name"] == name]
            if not matched:
                continue
            sr = matched[0]
            for m in keep_metrics:
                if m in sr:
                    row[f"{name}__{m}"] = sr[m]
        all_run_rows.append(row)

    save_csv(output_root / "all_runs_raw.csv", all_run_rows)

    grouped_summary = {}
    for prefix in ["heuristic_direct","heuristic_plus_newton","neural_direct","neural_plus_newton"]:
        records = []
        for row in all_run_rows:
            rec = {}
            for k, v in row.items():
                if k.startswith(prefix + "__"):
                    rec[k[len(prefix)+2:]] = v
            if rec:
                records.append(rec)
        if not records:
            continue
        grouped_summary[prefix] = {}
        for metric in ["mae","rmse","r2","valid_ratio","residual_mean","residual_median","residual_p90","max_abs_error","newton_iter_mean","newton_iter_median","newton_iter_p90","newton_converged_ratio"]:
            vals = [to_float(r.get(metric, math.nan)) for r in records]
            grouped_summary[prefix][metric] = mean_std(vals)

    (output_root / "summary_stats.json").write_text(json.dumps(grouped_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    paper_rows = []
    for group_name, metrics in grouped_summary.items():
        row = {"group": group_name}
        for metric_name, stat in metrics.items():
            row[f"{metric_name}_mean"] = stat["mean"]
            row[f"{metric_name}_std"] = stat["std"]
            row[f"{metric_name}_min"] = stat["min"]
            row[f"{metric_name}_max"] = stat["max"]
        paper_rows.append(row)
    save_csv(output_root / "summary_stats_for_paper.csv", paper_rows)

    print("\n[DONE]")
    print("raw:", output_root / "all_runs_raw.csv")
    print("summary:", output_root / "summary_stats.json")
    print("paper:", output_root / "summary_stats_for_paper.csv")

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","eval","repeat"], required=True)
    parser.add_argument("--model", choices=["mlp","lstm","gru","transformer"], default="transformer")
    parser.add_argument("--train_npz"); parser.add_argument("--val_npz"); parser.add_argument("--test_npz")
    parser.add_argument("--save_dir"); parser.add_argument("--model_path"); parser.add_argument("--out_dir"); parser.add_argument("--output_root")
    parser.add_argument("--use_log_features", action="store_true")
    parser.add_argument("--optimizer", choices=["adam","adamw"], default="adamw")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--loss_name", choices=["smoothl1","mse"], default="smoothl1")
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[256,256,128])
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--head_hidden", type=int, default=128)
    parser.add_argument("--head_layers", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=128)
    parser.add_argument("--use_cls_token", action="store_true")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max_newton_iter", type=int, default=20)
    parser.add_argument("--num_runs", type=int, default=50)
    parser.add_argument("--seed_start", type=int, default=1)
    return parser

def main():
    parser = build_parser(); args = parser.parse_args()
    if args.mode == "train":
        if not (args.train_npz and args.val_npz and args.test_npz and args.save_dir):
            raise ValueError("--train_npz --val_npz --test_npz --save_dir 필요")
        train_once(args, seed=args.seed, save_dir=Path(args.save_dir))
    elif args.mode == "eval":
        if not (args.test_npz and args.model_path and args.out_dir):
            raise ValueError("--test_npz --model_path --out_dir 필요")
        evaluate_once(args, model_path=Path(args.model_path), out_dir=Path(args.out_dir))
    elif args.mode == "repeat":
        if not (args.train_npz and args.val_npz and args.test_npz and args.output_root):
            raise ValueError("--train_npz --val_npz --test_npz --output_root 필요")
        repeat_experiments(args)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import random
import statistics
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PI = math.pi
LN10 = math.log(10.0)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sanitize_array(x: np.ndarray, clip_value: float = 1e12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)
    x = np.clip(x, -clip_value, clip_value)
    return x

def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))

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
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def mean_std(values):
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0, "min": vals[0], "max": vals[0]}
    return {
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals),
        "min": min(vals),
        "max": max(vals),
    }

class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        X = sanitize_array(X)
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0
        self.mean[~np.isfinite(self.mean)] = 0.0
        self.std[~np.isfinite(self.std)] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = sanitize_array(X)
        Xs = (X - self.mean) / self.std
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        Xs = np.clip(Xs, -1e6, 1e6)
        return Xs.astype(np.float32)

    def save(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @staticmethod
    def from_saved(saved: Dict):
        s = Standardizer()
        s.mean = np.array(saved["mean"], dtype=np.float64)
        s.std = np.array(saved["std"], dtype=np.float64)
        return s

# Physics
def re_from_Q(Q, rho, mu, D):
    return 4.0 * rho * Q / (PI * mu * D)

def colebrook_single_x_eq(x, Re, rel_rough):
    x = np.asarray(x, dtype=np.float64)
    Re = np.asarray(Re, dtype=np.float64)
    rel_rough = np.asarray(rel_rough, dtype=np.float64)
    z = rel_rough / 3.7 + 2.51 * x / Re
    out = np.full_like(x, np.nan, dtype=np.float64)
    mask = (Re > 0) & (z > 0)
    out[mask] = x[mask] + 2.0 * np.log10(z[mask])
    return out

def colebrook_single_x_df(x, Re, rel_rough):
    x = np.asarray(x, dtype=np.float64)
    Re = np.asarray(Re, dtype=np.float64)
    rel_rough = np.asarray(rel_rough, dtype=np.float64)
    z = rel_rough / 3.7 + 2.51 * x / Re
    out = np.full_like(x, np.nan, dtype=np.float64)
    mask = (Re > 0) & (z > 0)
    out[mask] = 1.0 + 2.0 * ((2.51 / Re[mask]) / (z[mask] * LN10))
    return out

def solve_x_from_Q(Q, D, eps, rho, mu, x_init=7.0, tol=1e-13, max_iter=50):
    Re = re_from_Q(np.array([Q]), np.array([rho]), np.array([mu]), np.array([D]))[0]
    rr = eps / D
    x = float(x_init)
    for _ in range(max_iter):
        fx = colebrook_single_x_eq(np.array([x]), np.array([Re]), np.array([rr]))[0]
        dfx = colebrook_single_x_df(np.array([x]), np.array([Re]), np.array([rr]))[0]
        if (not np.isfinite(fx)) or (not np.isfinite(dfx)) or abs(dfx) < 1e-15:
            break
        x_new = float(np.clip(x - fx / dfx, 1e-3, 1e3))
        if abs(x_new - x) < tol and abs(fx) < tol:
            x = x_new
            break
        x = x_new
    return float(x)

def head_loss(Q, x, L, D, g):
    return 8.0 * L * (Q ** 2) / (g * (PI ** 2) * (D ** 5) * (x ** 2))

def system_F(z, params):
    Q1, x1, x2 = z[..., 0], z[..., 1], z[..., 2]
    QT = params["Q_total"]; D1 = params["D1"]; D2 = params["D2"]; eps1 = params["eps1"]; eps2 = params["eps2"]
    L1 = params["L1"]; L2 = params["L2"]; rho = params["rho"]; mu = params["mu"]; g = params["g"]
    Q2 = QT - Q1
    Re1 = re_from_Q(Q1, rho, mu, D1); Re2 = re_from_Q(Q2, rho, mu, D2)
    rr1 = eps1 / D1; rr2 = eps2 / D2
    F1 = colebrook_single_x_eq(x1, Re1, rr1)
    F2 = colebrook_single_x_eq(x2, Re2, rr2)
    F3 = head_loss(Q1, x1, L1, D1, g) - head_loss(Q2, x2, L2, D2, g)
    return np.stack([F1, F2, F3], axis=-1)

def numerical_jacobian_single(z, p, eps=1e-6):
    z = np.asarray(z, dtype=np.float64)
    J = np.zeros((3, 3), dtype=np.float64)
    f0 = system_F(z[None, :], p)[0]
    for j in range(3):
        zp = z.copy(); zm = z.copy()
        step = eps * max(1.0, abs(z[j]))
        zp[j] += step; zm[j] -= step
        fp = system_F(zp[None, :], p)[0]; fm = system_F(zm[None, :], p)[0]
        J[:, j] = (fp - fm) / (2.0 * step)
    return J, f0

def project_feasible(z, p):
    z = np.asarray(z, dtype=np.float64).copy()
    QT = float(p["Q_total"])
    z[0] = np.clip(z[0], max(1e-8, QT * 1e-5), QT - max(1e-8, QT * 1e-5))
    z[1] = max(z[1], 1e-3); z[2] = max(z[2], 1e-3)
    return z

def newton_system_single(z0, p, tol=1e-12, max_iter=20, damping=1.0):
    z = project_feasible(z0, p)
    converged = False; used_iter = 0
    for k in range(1, max_iter + 1):
        J, f = numerical_jacobian_single(z, p)
        if not np.all(np.isfinite(J)) or not np.all(np.isfinite(f)):
            break
        try:
            step = np.linalg.solve(J, f)
        except np.linalg.LinAlgError:
            break
        step = np.clip(step, -5.0, 5.0)
        z_new = project_feasible(z - damping * step, p)
        f_new = system_F(z_new[None, :], p)[0]
        if np.linalg.norm(f_new, ord=2) > np.linalg.norm(f, ord=2):
            z_half = project_feasible(z - 0.5 * damping * step, p)
            f_half = system_F(z_half[None, :], p)[0]
            if np.linalg.norm(f_half, ord=2) < np.linalg.norm(f_new, ord=2):
                z_new, f_new = z_half, f_half
        z = z_new; used_iter = k
        if np.linalg.norm(f_new, ord=np.inf) <= tol:
            converged = True
            break
    return z, used_iter, converged

def refine_batch(z_init, data, tol=1e-12, max_iter=20):
    n = len(z_init)
    out = np.zeros_like(z_init, dtype=np.float64)
    iters = np.zeros(n, dtype=np.int32)
    conv = np.zeros(n, dtype=bool)
    for i in range(n):
        p = {k: float(np.asarray(data[k])[i]) for k in ["Q_total","D1","D2","eps1","eps2","L1","L2","rho","mu","g"]}
        zf, it, ok = newton_system_single(z_init[i], p, tol=tol, max_iter=max_iter)
        out[i] = zf; iters[i] = it; conv[i] = ok
    return out, iters, conv

# Data / targets
def safe_logit(r):
    r = np.clip(r, 1e-6, 1.0 - 1e-6)
    return np.log(r / (1.0 - r))

def compute_delta_targets(y, z0, q_total):
    r_true = np.clip(y[:, 0] / q_total, 1e-6, 1.0 - 1e-6)
    r0 = np.clip(z0[:, 0] / q_total, 1e-6, 1.0 - 1e-6)
    dlogit_r = (safe_logit(r_true) - safe_logit(r0)).reshape(-1, 1)
    dx1 = (y[:, 1] - z0[:, 1]).reshape(-1, 1)
    dx2 = (y[:, 2] - z0[:, 2]).reshape(-1, 1)
    return np.concatenate([dlogit_r, dx1, dx2], axis=1)

def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    required = ["coeffs","center","target","Q_total","D1","D2","eps1","eps2","L1","L2","rho","mu","g"]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {npz_path}. Available: {list(data.keys())}")
    return {k: np.asarray(data[k]) for k in required}

def build_inputs_and_baseline(data: Dict[str, np.ndarray], use_log_features: bool = True):
    coeffs = sanitize_array(np.asarray(data["coeffs"], dtype=np.float64), clip_value=1e30)
    center = sanitize_array(np.asarray(data["center"], dtype=np.float64), clip_value=1e12)
    y = sanitize_array(np.asarray(data["target"], dtype=np.float64), clip_value=1e12)
    coeffs = signed_log1p(coeffs)
    seq_x = np.concatenate([coeffs, center[..., None]], axis=2)
    seq_x = sanitize_array(seq_x, 1e12)

    globals_raw = [np.asarray(data[k], dtype=np.float64).reshape(-1,1) for k in ["Q_total","D1","D2","eps1","eps2","L1","L2","rho","mu","g"]]
    globals_raw = [sanitize_array(g, 1e12) for g in globals_raw]
    if use_log_features:
        globals_proc = []
        for i, arr in enumerate(globals_raw):
            globals_proc.append(np.log(np.clip(arr, 1e-12, None)) if i < 9 else arr)
        glob_x = np.concatenate(globals_proc, axis=1)
    else:
        glob_x = np.concatenate(globals_raw, axis=1)
    glob_x = sanitize_array(glob_x, 1e12)

    QT = np.asarray(data["Q_total"], dtype=np.float64)
    D1 = np.asarray(data["D1"], dtype=np.float64); D2 = np.asarray(data["D2"], dtype=np.float64)
    eps1 = np.asarray(data["eps1"], dtype=np.float64); eps2 = np.asarray(data["eps2"], dtype=np.float64)
    rho = np.asarray(data["rho"], dtype=np.float64); mu = np.asarray(data["mu"], dtype=np.float64)

    n = len(QT)
    z0 = np.zeros((n, 3), dtype=np.float64); z0[:, 0] = QT / 2.0
    for i in range(n):
        qh = QT[i] / 2.0
        z0[i,1] = solve_x_from_Q(qh, D1[i], eps1[i], rho[i], mu[i])
        z0[i,2] = solve_x_from_Q(qh, D2[i], eps2[i], rho[i], mu[i])

    delta_target = compute_delta_targets(y, z0, QT)
    return seq_x, glob_x, y, z0, delta_target

class HybridDataset(Dataset):
    def __init__(self, seq_x, glob_x, y, z0, delta_target, raw_data):
        self.seq_x = torch.from_numpy(seq_x.astype(np.float32))
        self.glob_x = torch.from_numpy(glob_x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.z0 = torch.from_numpy(z0.astype(np.float32))
        self.delta_target = torch.from_numpy(delta_target.astype(np.float32))
        self.raw = {}
        for k in ["Q_total","D1","D2","eps1","eps2","L1","L2","rho","mu","g"]:
            self.raw[k] = torch.from_numpy(np.asarray(raw_data[k]).astype(np.float32).reshape(-1,1))

    def __len__(self): return self.y.shape[0]

    def __getitem__(self, idx):
        item = {"seq_x": self.seq_x[idx], "glob_x": self.glob_x[idx], "y": self.y[idx], "z0": self.z0[idx], "delta_target": self.delta_target[idx]}
        for k, v in self.raw.items():
            item[k] = v[idx]
        return item

def standardize_datasets(train_ds, val_ds, test_ds):
    seq_scaler = Standardizer(); glob_scaler = Standardizer(); delta_scaler = Standardizer()
    seq_scaler.fit(train_ds.seq_x.numpy().reshape(-1, train_ds.seq_x.shape[-1]))
    train_ds.seq_x = torch.from_numpy(seq_scaler.transform(train_ds.seq_x.numpy().reshape(-1, train_ds.seq_x.shape[-1])).reshape(train_ds.seq_x.shape))
    val_ds.seq_x = torch.from_numpy(seq_scaler.transform(val_ds.seq_x.numpy().reshape(-1, val_ds.seq_x.shape[-1])).reshape(val_ds.seq_x.shape))
    test_ds.seq_x = torch.from_numpy(seq_scaler.transform(test_ds.seq_x.numpy().reshape(-1, test_ds.seq_x.shape[-1])).reshape(test_ds.seq_x.shape))
    if train_ds.glob_x.shape[1] > 0:
        glob_scaler.fit(train_ds.glob_x.numpy())
        train_ds.glob_x = torch.from_numpy(glob_scaler.transform(train_ds.glob_x.numpy()))
        val_ds.glob_x = torch.from_numpy(glob_scaler.transform(val_ds.glob_x.numpy()))
        test_ds.glob_x = torch.from_numpy(glob_scaler.transform(test_ds.glob_x.numpy()))
    else:
        glob_scaler.mean = np.zeros((1,0), dtype=np.float64); glob_scaler.std = np.ones((1,0), dtype=np.float64)
    delta_scaler.fit(train_ds.delta_target.numpy())
    train_ds.delta_target = torch.from_numpy(delta_scaler.transform(train_ds.delta_target.numpy()))
    val_ds.delta_target = torch.from_numpy(delta_scaler.transform(val_ds.delta_target.numpy()))
    test_ds.delta_target = torch.from_numpy(delta_scaler.transform(test_ds.delta_target.numpy()))
    return seq_scaler, glob_scaler, delta_scaler

# Models
class MLPBackbone(nn.Module):
    def __init__(self, seq_dim, seq_len, glob_dim, hidden_dims=(256,256,128), dropout=0.1):
        super().__init__()
        in_dim = seq_dim * seq_len + glob_dim
        layers = []; prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.feat = nn.Sequential(*layers); self.out_dim = prev

    def forward(self, seq_x, glob_x):
        return self.feat(torch.cat([seq_x.flatten(1), glob_x], dim=1))

class LSTMBackbone(nn.Module):
    def __init__(self, seq_dim, glob_dim, hidden_size=128, num_layers=2, dropout=0.1, head_hidden=128, head_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(seq_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fusion = self._build_head(hidden_size + glob_dim, head_hidden, head_hidden, dropout, head_layers)
        self.out_dim = head_hidden

    @staticmethod
    def _build_head(in_dim, hidden_dim, out_dim, dropout, head_layers):
        layers = []; prev = in_dim; cur = hidden_dim
        for _ in range(max(head_layers - 1, 0)):
            layers += [nn.Linear(prev, cur), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = cur; cur = max(cur // 2, 32)
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        _, (hn, _) = self.lstm(seq_x)
        return self.fusion(torch.cat([hn[-1], glob_x], dim=1))

class GRUBackbone(nn.Module):
    def __init__(self, seq_dim, glob_dim, hidden_size=128, num_layers=2, dropout=0.1, head_hidden=128, head_layers=2):
        super().__init__()
        self.gru = nn.GRU(seq_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fusion = LSTMBackbone._build_head(hidden_size + glob_dim, head_hidden, head_hidden, dropout, head_layers)
        self.out_dim = head_hidden

    def forward(self, seq_x, glob_x):
        _, hn = self.gru(seq_x)
        return self.fusion(torch.cat([hn[-1], glob_x], dim=1))

class TransformerBackbone(nn.Module):
    def __init__(self, seq_dim, seq_len, glob_dim, d_model=96, nhead=4, num_layers=2, dropout=0.1, ff_dim=192, head_hidden=128, head_layers=2, use_cls_token=True):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.input_proj = nn.Linear(seq_dim, d_model)
        total_len = seq_len + (1 if use_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, total_len, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if use_cls_token else None
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fusion = LSTMBackbone._build_head(d_model + glob_dim, head_hidden, head_hidden, dropout, head_layers)
        self.out_dim = head_hidden

    def forward(self, seq_x, glob_x):
        bsz = seq_x.size(0)
        x = self.input_proj(seq_x)
        if self.use_cls_token:
            x = torch.cat([self.cls_token.expand(bsz, -1, -1), x], dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        h = self.norm(self.encoder(x))
        pooled = h[:, 0, :] if self.use_cls_token else h.mean(dim=1)
        return self.fusion(torch.cat([pooled, glob_x], dim=1))

class HybridCorrectionModel(nn.Module):
    def __init__(self, model_name, seq_dim, seq_len, glob_dim, hp):
        super().__init__()
        if model_name == "mlp":
            self.backbone = MLPBackbone(seq_dim, seq_len, glob_dim, hidden_dims=tuple(hp["hidden_dims"]), dropout=hp["dropout"])
        elif model_name == "lstm":
            self.backbone = LSTMBackbone(seq_dim, glob_dim, hidden_size=hp["hidden_size"], num_layers=hp["num_layers"], dropout=hp["dropout"], head_hidden=hp["head_hidden"], head_layers=hp["head_layers"])
        elif model_name == "gru":
            self.backbone = GRUBackbone(seq_dim, glob_dim, hidden_size=hp["hidden_size"], num_layers=hp["num_layers"], dropout=hp["dropout"], head_hidden=hp["head_hidden"], head_layers=hp["head_layers"])
        elif model_name == "transformer":
            self.backbone = TransformerBackbone(seq_dim, seq_len, glob_dim, d_model=hp["d_model"], nhead=hp["nhead"], num_layers=hp["num_layers"], dropout=hp["dropout"], ff_dim=hp["ff_dim"], head_hidden=hp["head_hidden"], head_layers=hp["head_layers"], use_cls_token=hp["use_cls_token"])
        else:
            raise ValueError(model_name)
        self.delta_head = nn.Linear(self.backbone.out_dim, 3)

    def forward_delta(self, seq_x, glob_x):
        feat = self.backbone(seq_x, glob_x)
        return self.delta_head(feat)

    def decode(self, delta_norm, z0, q_total, delta_scaler_t):
        mean = delta_scaler_t["mean"].to(delta_norm.device)
        std = delta_scaler_t["std"].to(delta_norm.device)
        delta_real = delta_norm * std + mean
        q0 = z0[:, 0]; x10 = z0[:, 1]; x20 = z0[:, 2]
        r0 = torch.clamp(q0 / q_total.squeeze(1), 1e-6, 1.0 - 1e-6)
        logit_r0 = torch.log(r0 / (1.0 - r0))
        logit_r = logit_r0 + delta_real[:, 0]
        r = torch.sigmoid(logit_r)
        q1 = r * q_total.squeeze(1)
        x1 = torch.clamp(x10 + delta_real[:, 1], min=1e-3)
        x2 = torch.clamp(x20 + delta_real[:, 2], min=1e-3)
        pred = torch.stack([q1, x1, x2], dim=1)
        return pred, delta_real

    def forward(self, seq_x, glob_x, z0, q_total, delta_scaler_t):
        delta_norm = self.forward_delta(seq_x, glob_x)
        pred, delta_real = self.decode(delta_norm, z0, q_total, delta_scaler_t)
        return pred, delta_norm, delta_real

# Train/eval wrappers
def build_hp_from_args(args):
    return {
        "model": args.model,
        "use_log_features": args.use_log_features,
        "optimizer": args.optimizer,
        "loss_name": args.loss_name,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dims": args.hidden_dims,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "head_hidden": args.head_hidden,
        "head_layers": args.head_layers,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "ff_dim": args.ff_dim,
        "use_cls_token": args.use_cls_token,
    }

def delta_supervised_loss(delta_pred_norm, delta_target_norm, loss_name="smoothl1"):
    if loss_name == "mse":
        return torch.mean((delta_pred_norm - delta_target_norm) ** 2)
    return torch.nn.functional.smooth_l1_loss(delta_pred_norm, delta_target_norm, beta=0.1)

def vector_metrics(pred, true):
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "mae": mae, "rmse": rmse, "r2": r2,
        "mae_Q1": float(np.mean(np.abs(err[:, 0]))),
        "mae_x1": float(np.mean(np.abs(err[:, 1]))),
        "mae_x2": float(np.mean(np.abs(err[:, 2]))),
        "max_abs_error": float(np.max(np.abs(err))),
    }

def residual_metrics(pred, data):
    params = {k: np.asarray(data[k], dtype=np.float64) for k in ["Q_total","D1","D2","eps1","eps2","L1","L2","rho","mu","g"]}
    F = system_F(pred.astype(np.float64), params)
    norms_inf = np.max(np.abs(F), axis=1)
    valid = np.all(np.isfinite(F), axis=1)
    return {
        "valid_ratio": float(np.mean(valid)),
        "residual_mean": float(np.nanmean(norms_inf)),
        "residual_median": float(np.nanmedian(norms_inf)),
        "residual_p90": percentile(norms_inf[np.isfinite(norms_inf)], 90),
    }

def run_eval(model, loader, loss_name, device, delta_scaler_t):
    model.eval()
    preds = []; trues = []; total_loss = 0.0; total_n = 0
    with torch.no_grad():
        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            pred, delta_norm, delta_real = model(batch["seq_x"], batch["glob_x"], batch["z0"], batch["Q_total"], delta_scaler_t)
            loss = delta_supervised_loss(delta_norm, batch["delta_target"], loss_name=loss_name)
            bs = pred.shape[0]
            total_loss += float(loss.detach().cpu().item()) * bs; total_n += bs
            preds.append(pred.detach().cpu().numpy()); trues.append(batch["y"].detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0); true = np.concatenate(trues, axis=0)
    m = vector_metrics(pred, true); m["loss"] = total_loss / max(total_n, 1)
    return m, pred, true

def train_once(args, seed: int, save_dir: Path):
    set_seed(seed); device = torch.device(args.device); save_dir.mkdir(parents=True, exist_ok=True)
    train_raw = load_npz(args.train_npz); val_raw = load_npz(args.val_npz); test_raw = load_npz(args.test_npz)
    tr_seq, tr_glob, tr_y, tr_z0, tr_delta = build_inputs_and_baseline(train_raw, use_log_features=args.use_log_features)
    va_seq, va_glob, va_y, va_z0, va_delta = build_inputs_and_baseline(val_raw, use_log_features=args.use_log_features)
    te_seq, te_glob, te_y, te_z0, te_delta = build_inputs_and_baseline(test_raw, use_log_features=args.use_log_features)

    train_ds = HybridDataset(tr_seq, tr_glob, tr_y, tr_z0, tr_delta, train_raw)
    val_ds = HybridDataset(va_seq, va_glob, va_y, va_z0, va_delta, val_raw)
    test_ds = HybridDataset(te_seq, te_glob, te_y, te_z0, te_delta, test_raw)

    seq_scaler, glob_scaler, delta_scaler = standardize_datasets(train_ds, val_ds, test_ds)
    delta_scaler_t = {
        "mean": torch.tensor(delta_scaler.mean.astype(np.float32), device=device),
        "std": torch.tensor(delta_scaler.std.astype(np.float32), device=device),
    }

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    hp = build_hp_from_args(args)
    model = HybridCorrectionModel(args.model, train_ds.seq_x.shape[2], train_ds.seq_x.shape[1], train_ds.glob_x.shape[1], hp).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) if args.optimizer == "adamw" else torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_rmse = float("inf"); best_epoch = -1; best_state = None; wait = 0
    for epoch in range(1, args.epochs + 1):
        model.train(); train_loss_sum = 0.0; train_n = 0
        for batch in train_loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            pred, delta_norm, delta_real = model(batch["seq_x"], batch["glob_x"], batch["z0"], batch["Q_total"], delta_scaler_t)
            loss = delta_supervised_loss(delta_norm, batch["delta_target"], loss_name=args.loss_name)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            bs = pred.shape[0]; train_loss_sum += float(loss.detach().cpu().item()) * bs; train_n += bs

        train_loss = train_loss_sum / max(train_n, 1)
        val_metrics, _, _ = run_eval(model, val_loader, args.loss_name, device, delta_scaler_t)
        print(f"[Epoch {epoch:04d}] train_loss={train_loss:.8f} val_loss={val_metrics['loss']:.8f} val_rmse={val_metrics['rmse']:.8f} val_mae={val_metrics['mae']:.8f} val_r2={val_metrics['r2']:.8f}")
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]; best_epoch = epoch; best_state = deepcopy(model.state_dict()); wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[Early stopping] patience={args.patience}")
                break

    if best_state is None:
        raise RuntimeError("Training failed.")
    model.load_state_dict(best_state)
    ckpt = {
        "model_state_dict": best_state,
        "seq_scaler": seq_scaler.save(),
        "glob_scaler": glob_scaler.save(),
        "delta_scaler": delta_scaler.save(),
        "hp": hp,
        "seq_dim": int(train_ds.seq_x.shape[2]),
        "seq_len": int(train_ds.seq_x.shape[1]),
        "glob_dim": int(train_ds.glob_x.shape[1]),
        "best_val_rmse": float(best_val_rmse),
        "best_epoch": int(best_epoch),
    }
    torch.save(ckpt, save_dir / "best_model.pt")
    return save_dir / "best_model.pt"

def load_model_checkpoint(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    hp = ckpt["hp"]
    model = HybridCorrectionModel(hp["model"], ckpt["seq_dim"], ckpt["seq_len"], ckpt["glob_dim"], hp)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device); model.eval()
    seq_scaler = {"mean": np.array(ckpt["seq_scaler"]["mean"], dtype=np.float64), "std": np.array(ckpt["seq_scaler"]["std"], dtype=np.float64)}
    glob_scaler = {"mean": np.array(ckpt["glob_scaler"]["mean"], dtype=np.float64), "std": np.array(ckpt["glob_scaler"]["std"], dtype=np.float64)}
    delta_scaler = {"mean": np.array(ckpt["delta_scaler"]["mean"], dtype=np.float64), "std": np.array(ckpt["delta_scaler"]["std"], dtype=np.float64)}
    return ckpt, model, seq_scaler, glob_scaler, delta_scaler

def evaluate_once(args, model_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_npz(args.test_npz)
    ckpt, model, seq_scaler, glob_scaler, delta_scaler = load_model_checkpoint(str(model_path), device=args.device)
    hp = ckpt["hp"]; use_log_features = bool(hp.get("use_log_features", False))
    seq_x, glob_x, y_true, z0, _ = build_inputs_and_baseline(data, use_log_features=use_log_features)
    seq_shape = seq_x.shape
    seq_scaler_obj = Standardizer.from_saved({"mean": seq_scaler["mean"].tolist(), "std": seq_scaler["std"].tolist()})
    glob_scaler_obj = Standardizer.from_saved({"mean": glob_scaler["mean"].tolist(), "std": glob_scaler["std"].tolist()})
    seq_x = seq_scaler_obj.transform(seq_x.reshape(-1, seq_x.shape[-1])).reshape(seq_shape)
    glob_x = glob_scaler_obj.transform(glob_x)
    delta_scaler_t = {"mean": torch.tensor(delta_scaler["mean"].astype(np.float32), device=args.device), "std": torch.tensor(delta_scaler["std"].astype(np.float32), device=args.device)}

    preds = []
    with torch.no_grad():
        bs = 4096
        for i in range(0, len(seq_x), bs):
            s = torch.from_numpy(seq_x[i:i+bs]).to(args.device)
            g = torch.from_numpy(glob_x[i:i+bs]).to(args.device)
            z = torch.from_numpy(z0[i:i+bs].astype(np.float32)).to(args.device)
            qt = torch.from_numpy(np.asarray(data["Q_total"][i:i+bs]).astype(np.float32).reshape(-1,1)).to(args.device)
            pred, delta_norm, delta_real = model(s, g, z, qt, delta_scaler_t)
            preds.append(pred.cpu().numpy())
    pred_direct = np.concatenate(preds, axis=0)
    heur_pred = z0.copy()

    rows = []
    hd = vector_metrics(heur_pred, y_true); hd.update(residual_metrics(heur_pred, data)); hd["name"] = "heuristic_direct"; rows.append(hd)
    href, hit, hconv = refine_batch(heur_pred.astype(np.float64), data, tol=args.tol, max_iter=args.max_newton_iter)
    hr = vector_metrics(href, y_true.astype(np.float64)); hr.update(residual_metrics(href, data)); hr["name"] = "heuristic_plus_newton"; hr["newton_iter_mean"] = float(np.mean(hit)); hr["newton_iter_median"] = float(np.median(hit)); hr["newton_iter_p90"] = float(np.percentile(hit, 90)); hr["newton_converged_ratio"] = float(np.mean(hconv)); rows.append(hr)
    nd = vector_metrics(pred_direct, y_true); nd.update(residual_metrics(pred_direct, data)); nd["name"] = "neural_direct"; rows.append(nd)
    nref, nit, nconv = refine_batch(pred_direct.astype(np.float64), data, tol=args.tol, max_iter=args.max_newton_iter)
    nr = vector_metrics(nref, y_true.astype(np.float64)); nr.update(residual_metrics(nref, data)); nr["name"] = "neural_plus_newton"; nr["newton_iter_mean"] = float(np.mean(nit)); nr["newton_iter_median"] = float(np.median(nit)); nr["newton_iter_p90"] = float(np.percentile(nit, 90)); nr["newton_converged_ratio"] = float(np.mean(nconv)); rows.append(nr)
    save_csv(out_dir / "summary_metrics.csv", rows)
    return rows

def repeat_experiments(args):
    output_root = Path(args.output_root); output_root.mkdir(parents=True, exist_ok=True)
    all_run_rows = []
    for i in range(args.num_runs):
        seed = args.seed_start + i
        print(f"\n[RUN {i+1}/{args.num_runs}] seed={seed}")
        run_dir = output_root / f"run_{seed:03d}"
        train_dir = run_dir / "train"; eval_dir = run_dir / "eval"
        model_path = train_once(args, seed=seed, save_dir=train_dir)
        summary_rows = evaluate_once(args, model_path=model_path, out_dir=eval_dir)
        row = {"seed": seed, "train_dir": str(train_dir), "eval_dir": str(eval_dir)}
        keep_metrics = ["mae","rmse","r2","valid_ratio","residual_mean","residual_median","residual_p90","max_abs_error","newton_iter_mean","newton_iter_median","newton_iter_p90","newton_converged_ratio"]
        for name in ["heuristic_direct","heuristic_plus_newton","neural_direct","neural_plus_newton"]:
            matched = [r for r in summary_rows if r["name"] == name]
            if not matched:
                continue
            sr = matched[0]
            for m in keep_metrics:
                if m in sr:
                    row[f"{name}__{m}"] = sr[m]
        all_run_rows.append(row)

    save_csv(output_root / "all_runs_raw.csv", all_run_rows)

    grouped_summary = {}
    for prefix in ["heuristic_direct","heuristic_plus_newton","neural_direct","neural_plus_newton"]:
        records = []
        for row in all_run_rows:
            rec = {}
            for k, v in row.items():
                if k.startswith(prefix + "__"):
                    rec[k[len(prefix)+2:]] = v
            if rec:
                records.append(rec)
        if not records:
            continue
        grouped_summary[prefix] = {}
        for metric in ["mae","rmse","r2","valid_ratio","residual_mean","residual_median","residual_p90","max_abs_error","newton_iter_mean","newton_iter_median","newton_iter_p90","newton_converged_ratio"]:
            vals = [to_float(r.get(metric, math.nan)) for r in records]
            grouped_summary[prefix][metric] = mean_std(vals)

    (output_root / "summary_stats.json").write_text(json.dumps(grouped_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    paper_rows = []
    for group_name, metrics in grouped_summary.items():
        row = {"group": group_name}
        for metric_name, stat in metrics.items():
            row[f"{metric_name}_mean"] = stat["mean"]
            row[f"{metric_name}_std"] = stat["std"]
            row[f"{metric_name}_min"] = stat["min"]
            row[f"{metric_name}_max"] = stat["max"]
        paper_rows.append(row)
    save_csv(output_root / "summary_stats_for_paper.csv", paper_rows)

    print("\n[DONE]")
    print("raw:", output_root / "all_runs_raw.csv")
    print("summary:", output_root / "summary_stats.json")
    print("paper:", output_root / "summary_stats_for_paper.csv")

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","eval","repeat"], required=True)
    parser.add_argument("--model", choices=["mlp","lstm","gru","transformer"], default="transformer")
    parser.add_argument("--train_npz"); parser.add_argument("--val_npz"); parser.add_argument("--test_npz")
    parser.add_argument("--save_dir"); parser.add_argument("--model_path"); parser.add_argument("--out_dir"); parser.add_argument("--output_root")
    parser.add_argument("--use_log_features", action="store_true")
    parser.add_argument("--optimizer", choices=["adam","adamw"], default="adamw")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--loss_name", choices=["smoothl1","mse"], default="smoothl1")
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[256,256,128])
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--head_hidden", type=int, default=128)
    parser.add_argument("--head_layers", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=128)
    parser.add_argument("--use_cls_token", action="store_true")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max_newton_iter", type=int, default=20)
    parser.add_argument("--num_runs", type=int, default=50)
    parser.add_argument("--seed_start", type=int, default=1)
    return parser

def main():
    parser = build_parser(); args = parser.parse_args()
    if args.mode == "train":
        if not (args.train_npz and args.val_npz and args.test_npz and args.save_dir):
            raise ValueError("--train_npz --val_npz --test_npz --save_dir 필요")
        train_once(args, seed=args.seed, save_dir=Path(args.save_dir))
    elif args.mode == "eval":
        if not (args.test_npz and args.model_path and args.out_dir):
            raise ValueError("--test_npz --model_path --out_dir 필요")
        evaluate_once(args, model_path=Path(args.model_path), out_dir=Path(args.out_dir))
    elif args.mode == "repeat":
        if not (args.train_npz and args.val_npz and args.test_npz and args.output_root):
            raise ValueError("--train_npz --val_npz --test_npz --output_root 필요")
        repeat_experiments(args)

if __name__ == "__main__":
    main()
