#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import itertools
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PI = math.pi
LN10 = math.log(10.0)


# =========================================================
# Utility
# =========================================================
def save_csv(path, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def percentile(x, q):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def sanitize_array(x, clip_value=1e12):
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)
    x = np.clip(x, -clip_value, clip_value)
    return x


def fit_standard_scaler(X):
    X = sanitize_array(X, 1e12)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    mean[~np.isfinite(mean)] = 0.0
    std[~np.isfinite(std)] = 1.0
    std[std < 1e-8] = 1.0
    return {"mean": mean.astype(np.float64), "std": std.astype(np.float64)}


def apply_scaler(X, scaler, clip_out=1e6):
    X = sanitize_array(X, 1e12)
    Xs = (X - scaler["mean"]) / scaler["std"]
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = np.clip(Xs, -clip_out, clip_out)
    return Xs.astype(np.float32)


def filter_valid_rows(seq_x, glob_x, y):
    seq_ok = np.all(np.isfinite(seq_x.reshape(seq_x.shape[0], -1)), axis=1)
    glob_ok = np.all(np.isfinite(glob_x), axis=1)
    y_ok = np.all(np.isfinite(y), axis=1)
    mask = seq_ok & glob_ok & y_ok
    return seq_x[mask], glob_x[mask], y[mask]


# =========================================================
# Data
# =========================================================
def load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    required = [
        "coeffs", "center", "target",
        "Q_total", "D1", "D2", "eps1", "eps2",
        "L1", "L2", "rho", "mu", "g"
    ]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {npz_path}. Available: {list(data.keys())}")
    return {k: np.asarray(data[k]) for k in required}


def build_inputs(data, use_log_features=True):
    coeffs = np.asarray(data["coeffs"], dtype=np.float64)
    center = np.asarray(data["center"], dtype=np.float64)
    y = np.asarray(data["target"], dtype=np.float64)

    coeffs = sanitize_array(coeffs, 1e30)
    center = sanitize_array(center, 1e12)
    y = sanitize_array(y, 1e12)

    coeffs = signed_log1p(coeffs)
    seq_x = np.concatenate([coeffs, center[..., None]], axis=2)
    seq_x = sanitize_array(seq_x, 1e12)

    globals_raw = [
        np.asarray(data["Q_total"], dtype=np.float64).reshape(-1, 1),
        np.asarray(data["D1"], dtype=np.float64).reshape(-1, 1),
        np.asarray(data["D2"], dtype=np.float64).reshape(-1, 1),
        np.asarray(data["eps1"], dtype=np.float64).reshape(-1, 1),
        np.asarray(data["eps2"], dtype=np.float64).reshape(-1, 1),
        np.asarray(data["L1"], dtype=np.float64).reshape(-1, 1),
        np.asarray(data["L2"], dtype=np.float64).reshape(-1, 1),
        np.asarray(data["rho"], dtype=np.float64).reshape(-1, 1),
        np.asarray(data["mu"], dtype=np.float64).reshape(-1, 1),
        np.asarray(data["g"], dtype=np.float64).reshape(-1, 1),
    ]
    globals_raw = [sanitize_array(x, 1e12) for x in globals_raw]

    if use_log_features:
        globals_proc = []
        for i, arr in enumerate(globals_raw):
            if i < 9:
                globals_proc.append(np.log(np.clip(arr, 1e-12, None)))
            else:
                globals_proc.append(arr)
        glob_x = np.concatenate(globals_proc, axis=1)
    else:
        glob_x = np.concatenate(globals_raw, axis=1)

    glob_x = sanitize_array(glob_x, 1e12)
    return seq_x, glob_x, y


class MultiInputDataset(Dataset):
    def __init__(self, seq_x, glob_x, y):
        self.seq_x = torch.from_numpy(seq_x.astype(np.float32))
        self.glob_x = torch.from_numpy(glob_x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.seq_x[idx], self.glob_x[idx], self.y[idx]


# =========================================================
# Models
# =========================================================
class MLPModel(nn.Module):
    def __init__(self, seq_dim, seq_len, glob_dim, hidden_dims=(256, 256, 128), dropout=0.1, out_dim=3):
        super().__init__()
        in_dim = seq_dim * seq_len + glob_dim
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        x = torch.cat([seq_x.flatten(1), glob_x], dim=1)
        return self.net(x)


class LSTMModel(nn.Module):
    def __init__(self, seq_dim, glob_dim, hidden_size=128, num_layers=2, dropout=0.1, out_dim=3, head_hidden=128, head_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = self._build_head(hidden_size + glob_dim, head_hidden, out_dim, dropout, head_layers)

    @staticmethod
    def _build_head(in_dim, hidden_dim, out_dim, dropout, head_layers):
        layers = []
        prev = in_dim
        cur = hidden_dim
        for _ in range(head_layers - 1):
            layers += [nn.Linear(prev, cur), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = cur
            cur = max(cur // 2, 32)
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        _, (hn, _) = self.lstm(seq_x)
        h_last = hn[-1]
        return self.head(torch.cat([h_last, glob_x], dim=1))


class GRUModel(nn.Module):
    def __init__(self, seq_dim, glob_dim, hidden_size=128, num_layers=2, dropout=0.1, out_dim=3, head_hidden=128, head_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=seq_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = self._build_head(hidden_size + glob_dim, head_hidden, out_dim, dropout, head_layers)

    @staticmethod
    def _build_head(in_dim, hidden_dim, out_dim, dropout, head_layers):
        layers = []
        prev = in_dim
        cur = hidden_dim
        for _ in range(head_layers - 1):
            layers += [nn.Linear(prev, cur), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = cur
            cur = max(cur // 2, 32)
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        _, hn = self.gru(seq_x)
        h_last = hn[-1]
        return self.head(torch.cat([h_last, glob_x], dim=1))


class TransformerModel(nn.Module):
    def __init__(self, seq_dim, seq_len, glob_dim,
                 d_model=96, nhead=4, num_layers=2, dropout=0.1,
                 out_dim=3, ff_dim=192, head_hidden=128,
                 head_layers=2, use_cls_token=True):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.input_proj = nn.Linear(seq_dim, d_model)

        total_len = seq_len + (1 if use_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, total_len, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if use_cls_token else None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = self._build_head(d_model + glob_dim, head_hidden, out_dim, dropout, head_layers)

    @staticmethod
    def _build_head(in_dim, hidden_dim, out_dim, dropout, head_layers):
        layers = []
        prev = in_dim
        cur = hidden_dim
        for _ in range(head_layers - 1):
            layers += [nn.Linear(prev, cur), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = cur
            cur = max(cur // 2, 32)
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        bsz = seq_x.size(0)
        x = self.input_proj(seq_x)

        if self.use_cls_token:
            cls = self.cls_token.expand(bsz, -1, -1)
            x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed[:, :x.size(1), :]
        h = self.norm(self.encoder(x))

        pooled = h[:, 0, :] if self.use_cls_token else h.mean(dim=1)
        return self.head(torch.cat([pooled, glob_x], dim=1))


def build_model(model_name, seq_dim, seq_len, glob_dim, hp):
    if model_name == "mlp":
        return MLPModel(seq_dim, seq_len, glob_dim, tuple(hp["hidden_dims"]), hp["dropout"], 3)
    if model_name == "lstm":
        return LSTMModel(seq_dim, glob_dim, hp["hidden_size"], hp["num_layers"], hp["dropout"], 3, hp["head_hidden"], hp["head_layers"])
    if model_name == "gru":
        return GRUModel(seq_dim, glob_dim, hp["hidden_size"], hp["num_layers"], hp["dropout"], 3, hp["head_hidden"], hp["head_layers"])
    if model_name == "transformer":
        return TransformerModel(seq_dim, seq_len, glob_dim, hp["d_model"], hp["nhead"], hp["num_layers"], hp["dropout"], 3, hp["ff_dim"], hp["head_hidden"], hp["head_layers"], hp["use_cls_token"])
    raise ValueError(model_name)


# =========================================================
# Train / Eval metrics
# =========================================================
def regression_metrics(pred, true):
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


def build_optimizer(model, name, lr, weight_decay):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(name)


def run_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.SmoothL1Loss(beta=0.1)
    total_loss = 0.0
    n = 0
    for seq_x, glob_x, yb in loader:
        seq_x, glob_x, yb = seq_x.to(device), glob_x.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(seq_x, glob_x)
        loss = criterion(pred, yb)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        bs = seq_x.shape[0]
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_direct(model, loader, device):
    model.eval()
    criterion = nn.SmoothL1Loss(beta=0.1)
    total_loss = 0.0
    n = 0
    preds, trues = [], []
    for seq_x, glob_x, yb in loader:
        seq_x, glob_x, yb = seq_x.to(device), glob_x.to(device), yb.to(device)
        pred = model(seq_x, glob_x)
        loss = criterion(pred, yb)
        bs = seq_x.shape[0]
        if torch.isfinite(loss):
            total_loss += loss.item() * bs
            n += bs
        preds.append(pred.cpu().numpy())
        trues.append(yb.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    m = regression_metrics(preds, trues)
    m["loss"] = total_loss / max(n, 1)
    return m, preds, trues


# =========================================================
# Newton refinement
# =========================================================
def system_F(z, params):
    Q1, x1, x2 = z[..., 0], z[..., 1], z[..., 2]

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

    m1 = (Re1 > 0) & (z1 > 0)
    m2 = (Re2 > 0) & (z2 > 0)

    F1[m1] = x1[m1] + 2.0 * np.log10(z1[m1])
    F2[m2] = x2[m2] + 2.0 * np.log10(z2[m2])

    H1 = 8.0 * L1 * (Q1 ** 2) / (g * (PI ** 2) * (D1 ** 5) * (x1 ** 2))
    H2 = 8.0 * L2 * (Q2 ** 2) / (g * (PI ** 2) * (D2 ** 5) * (x2 ** 2))
    F3 = H1 - H2

    return np.stack([F1, F2, F3], axis=-1)


def numerical_jacobian_single(z, p, eps=1e-6):
    z = np.asarray(z, dtype=np.float64)
    J = np.zeros((3, 3), dtype=np.float64)
    f0 = system_F(z[None, :], p)[0]
    for j in range(3):
        zp = z.copy()
        zm = z.copy()
        step = eps * max(1.0, abs(z[j]))
        zp[j] += step
        zm[j] -= step
        fp = system_F(zp[None, :], p)[0]
        fm = system_F(zm[None, :], p)[0]
        J[:, j] = (fp - fm) / (2.0 * step)
    return J, f0


def project_feasible(z, p):
    z = np.asarray(z, dtype=np.float64).copy()
    QT = float(p["Q_total"])
    z[0] = np.clip(z[0], max(1e-8, QT * 1e-5), QT - max(1e-8, QT * 1e-5))
    z[1] = max(z[1], 1e-3)
    z[2] = max(z[2], 1e-3)
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

        step = np.clip(step, -5.0, 5.0)
        z_new = project_feasible(z - damping * step, p)
        f_new = system_F(z_new[None, :], p)[0]

        if np.linalg.norm(f_new, ord=2) > np.linalg.norm(f, ord=2):
            z_half = project_feasible(z - 0.5 * damping * step, p)
            f_half = system_F(z_half[None, :], p)[0]
            if np.linalg.norm(f_half, ord=2) < np.linalg.norm(f_new, ord=2):
                z_new = z_half
                f_new = f_half

        z = z_new
        used_iter = k
        if np.linalg.norm(f_new, ord=np.inf) <= tol:
            converged = True
            break

    return z, used_iter, converged


def residual_metrics(pred, data):
    params = {k: np.asarray(data[k], dtype=np.float64) for k in ["Q_total", "D1", "D2", "eps1", "eps2", "L1", "L2", "rho", "mu", "g"]}
    F = system_F(pred.astype(np.float64), params)
    norms_inf = np.max(np.abs(F), axis=1)
    valid = np.all(np.isfinite(F), axis=1)
    return {
        "valid_ratio": float(np.mean(valid)),
        "residual_mean": float(np.nanmean(norms_inf)),
        "residual_median": float(np.nanmedian(norms_inf)),
        "residual_p90": percentile(norms_inf[np.isfinite(norms_inf)], 90),
    }


def refine_batch(z_init, data, tol=1e-12, max_iter=20):
    n = len(z_init)
    out = np.zeros_like(z_init, dtype=np.float64)
    iters = np.zeros(n, dtype=np.int32)
    conv = np.zeros(n, dtype=bool)

    for i in range(n):
        p = {k: float(np.asarray(data[k])[i]) for k in ["Q_total", "D1", "D2", "eps1", "eps2", "L1", "L2", "rho", "mu", "g"]}
        zf, it, ok = newton_system_single(z_init[i], p, tol=tol, max_iter=max_iter)
        out[i] = zf
        iters[i] = it
        conv[i] = ok

    return out, iters, conv


# =========================================================
# Grid
# =========================================================
def grid_product(grid):
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in itertools.product(*vals):
        yield {k: v for k, v in zip(keys, combo)}


def build_model_grids(selected_models):
    grids = []

    if "mlp" in selected_models:
        grids += list(grid_product({
            "model": ["mlp"],
            "use_log_features": [True],
            "optimizer": ["adamw"],
            "weight_decay": [1e-4],
            "dropout": [0.05, 0.1],
            "hidden_dims": [[256, 256, 128], [256, 128, 64]],
            "lr": [5e-4, 1e-3],
        }))

    if "lstm" in selected_models:
        grids += list(grid_product({
            "model": ["lstm"],
            "use_log_features": [True],
            "optimizer": ["adamw"],
            "weight_decay": [1e-4],
            "dropout": [0.05, 0.1],
            "hidden_size": [96, 128],
            "num_layers": [1, 2],
            "head_hidden": [128],
            "head_layers": [2],
            "lr": [5e-4, 1e-3],
        }))

    if "gru" in selected_models:
        grids += list(grid_product({
            "model": ["gru"],
            "use_log_features": [True],
            "optimizer": ["adamw"],
            "weight_decay": [1e-4],
            "dropout": [0.05, 0.1],
            "hidden_size": [96, 128],
            "num_layers": [1, 2],
            "head_hidden": [128],
            "head_layers": [2],
            "lr": [5e-4, 1e-3],
        }))

    if "transformer" in selected_models:
        grids += list(grid_product({
            "model": ["transformer"],
            "use_log_features": [True],
            "optimizer": ["adamw"],
            "weight_decay": [1e-4],
            "dropout": [0.05, 0.1],
            "d_model": [64, 96],
            "nhead": [4],
            "num_layers": [1, 2],
            "ff_dim": [128, 192],
            "head_hidden": [128],
            "head_layers": [2],
            "use_cls_token": [False, True],
            "lr": [5e-4],
        }))

    normalized = []
    for hp in grids:
        base = {
            "model": hp["model"],
            "use_log_features": hp.get("use_log_features", True),
            "optimizer": hp.get("optimizer", "adamw"),
            "weight_decay": hp.get("weight_decay", 1e-4),
            "dropout": hp.get("dropout", 0.1),
            "lr": hp.get("lr", 5e-4),

            "hidden_dims": hp.get("hidden_dims", [256, 256, 128]),

            "hidden_size": hp.get("hidden_size", 128),
            "num_layers": hp.get("num_layers", 2),
            "head_hidden": hp.get("head_hidden", 128),
            "head_layers": hp.get("head_layers", 2),

            "d_model": hp.get("d_model", 96),
            "nhead": hp.get("nhead", 4),
            "ff_dim": hp.get("ff_dim", 192),
            "use_cls_token": hp.get("use_cls_token", False),
        }
        normalized.append(base)

    return normalized


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz", required=True)
    ap.add_argument("--val_npz", required=True)
    ap.add_argument("--test_npz", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--models", nargs="+", default=["transformer", "lstm", "gru", "mlp"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=256)

    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--max_newton_iter", type=int, default=20)

    ap.add_argument(
        "--rank_metric",
        default="plus_newton_r2",
        choices=[
            "plus_newton_r2",
            "plus_newton_rmse",
            "plus_newton_mae",
            "plus_newton_converged_ratio",
            "direct_r2",
            "direct_rmse",
            "direct_mae",
        ],
    )
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data = load_npz(args.train_npz)
    val_data = load_npz(args.val_npz)
    test_data = load_npz(args.test_npz)

    grids = build_model_grids(args.models)

    trial_rows = []
    best_rank_value = None
    best_rank_row = None
    best_ckpt = None

    for trial_idx, hp in enumerate(grids, start=1):
        trial_name = f"trial_{trial_idx:03d}_{hp['model']}"
        print(f"\n========== {trial_name} ==========")
        print(json.dumps(hp, ensure_ascii=False))

        start_time = time.time()

        tr_seq, tr_glob, tr_y = build_inputs(train_data, hp["use_log_features"])
        va_seq, va_glob, va_y = build_inputs(val_data, hp["use_log_features"])
        te_seq, te_glob, te_y = build_inputs(test_data, hp["use_log_features"])

        tr_seq, tr_glob, tr_y = filter_valid_rows(tr_seq, tr_glob, tr_y)
        va_seq, va_glob, va_y = filter_valid_rows(va_seq, va_glob, va_y)
        te_seq, te_glob, te_y = filter_valid_rows(te_seq, te_glob, te_y)

        seq_scaler = fit_standard_scaler(tr_seq.reshape(-1, tr_seq.shape[-1]))
        glob_scaler = fit_standard_scaler(tr_glob)

        tr_seq = apply_scaler(tr_seq.reshape(-1, tr_seq.shape[-1]), seq_scaler).reshape(tr_seq.shape)
        va_seq = apply_scaler(va_seq.reshape(-1, va_seq.shape[-1]), seq_scaler).reshape(va_seq.shape)
        te_seq = apply_scaler(te_seq.reshape(-1, te_seq.shape[-1]), seq_scaler).reshape(te_seq.shape)

        tr_glob = apply_scaler(tr_glob, glob_scaler)
        va_glob = apply_scaler(va_glob, glob_scaler)
        te_glob = apply_scaler(te_glob, glob_scaler)

        train_loader = DataLoader(MultiInputDataset(tr_seq, tr_glob, tr_y), batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(MultiInputDataset(va_seq, va_glob, va_y), batch_size=args.batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(MultiInputDataset(te_seq, te_glob, te_y), batch_size=args.batch_size, shuffle=False, drop_last=False)

        model = build_model(hp["model"], tr_seq.shape[2], tr_seq.shape[1], tr_glob.shape[1], hp).to(args.device)
        optimizer = build_optimizer(model, hp["optimizer"], hp["lr"], hp["weight_decay"])

        best_val_rmse = float("inf")
        best_state = None
        best_epoch = -1
        wait = 0

        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, train_loader, optimizer, args.device)
            val_metrics, _, _ = evaluate_direct(model, val_loader, args.device)

            print(
                f"[{trial_name}] "
                f"Epoch {epoch:03d} "
                f"train_loss={train_loss:.6f} "
                f"val_rmse={val_metrics['rmse']:.6f} "
                f"val_r2={val_metrics['r2']:.6f}"
            )

            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                best_epoch = epoch
                wait = 0
                best_state = deepcopy(model.state_dict())
            else:
                wait += 1
                if wait >= args.patience:
                    break

        if best_state is None:
            print(f"[WARN] {trial_name} skipped")
            continue

        model.load_state_dict(best_state)

        direct_metrics, pred_direct, y_true = evaluate_direct(model, test_loader, args.device)
        direct_metrics.update(residual_metrics(pred_direct.astype(np.float64), test_data))

        refined, iters, conv = refine_batch(pred_direct.astype(np.float64), test_data, tol=args.tol, max_iter=args.max_newton_iter)
        plus_metrics = regression_metrics(refined, y_true.astype(np.float64))
        plus_metrics.update(residual_metrics(refined, test_data))
        plus_metrics["newton_iter_mean"] = float(np.mean(iters))
        plus_metrics["newton_iter_median"] = float(np.median(iters))
        plus_metrics["newton_iter_p90"] = float(np.percentile(iters, 90))
        plus_metrics["newton_converged_ratio"] = float(np.mean(conv))

        elapsed = time.time() - start_time

        row = {
            "trial_id": trial_idx,
            "trial_name": trial_name,
            "model": hp["model"],
            "best_epoch": best_epoch,
            "elapsed_sec": elapsed,

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

            "hp_use_log_features": hp["use_log_features"],
            "hp_optimizer": hp["optimizer"],
            "hp_weight_decay": hp["weight_decay"],
            "hp_dropout": hp["dropout"],
            "hp_lr": hp["lr"],
            "hp_hidden_dims": json.dumps(hp["hidden_dims"]),
            "hp_hidden_size": hp["hidden_size"],
            "hp_num_layers": hp["num_layers"],
            "hp_head_hidden": hp["head_hidden"],
            "hp_head_layers": hp["head_layers"],
            "hp_d_model": hp["d_model"],
            "hp_nhead": hp["nhead"],
            "hp_ff_dim": hp["ff_dim"],
            "hp_use_cls_token": hp["use_cls_token"],
        }

        trial_rows.append(row)

        metric_value = row[args.rank_metric]
        if best_rank_value is None:
            better = True
        else:
            if args.rank_metric in ["plus_newton_rmse", "plus_newton_mae", "direct_rmse", "direct_mae"]:
                better = metric_value < best_rank_value
            else:
                better = metric_value > best_rank_value

        if better:
            best_rank_value = metric_value
            best_rank_row = dict(row)
            best_ckpt = {
                "state_dict": deepcopy(model.state_dict()),
                "hp": deepcopy(hp),
                "seq_scaler": seq_scaler,
                "glob_scaler": glob_scaler,
                "meta": {
                    "seq_dim": int(tr_seq.shape[2]),
                    "seq_len": int(tr_seq.shape[1]),
                    "glob_dim": int(tr_glob.shape[1]),
                    "rank_metric": args.rank_metric,
                    "rank_value": metric_value,
                }
            }

        with open(out_dir / f"{trial_name}.json", "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2)

    if not trial_rows:
        raise RuntimeError("No successful trials were completed.")

    reverse = args.rank_metric not in ["plus_newton_rmse", "plus_newton_mae", "direct_rmse", "direct_mae"]
    trial_rows_sorted = sorted(trial_rows, key=lambda r: r[args.rank_metric], reverse=reverse)

    save_csv(out_dir / "all_trials.csv", trial_rows_sorted)

    with open(out_dir / "best_result.json", "w", encoding="utf-8") as f:
        json.dump(best_rank_row, f, ensure_ascii=False, indent=2)

    if best_ckpt is not None:
        torch.save(best_ckpt, out_dir / "best_model_by_grid.pt")

    print("\n================ FINAL RANKING ================")
    for row in trial_rows_sorted[:10]:
        print({
            "trial_id": row["trial_id"],
            "model": row["model"],
            args.rank_metric: row[args.rank_metric],
            "plus_newton_rmse": row["plus_newton_rmse"],
            "plus_newton_r2": row["plus_newton_r2"],
            "plus_newton_converged_ratio": row["plus_newton_converged_ratio"],
            "iter_mean": row["plus_newton_newton_iter_mean"],
        })

    print("\n[DONE]")
    print(out_dir / "all_trials.csv")
    print(out_dir / "best_result.json")
    print(out_dir / "best_model_by_grid.pt")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import itertools
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PI = math.pi
LN10 = math.log(10.0)


# =========================================================
# Utility
# =========================================================
def save_csv(path, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def percentile(x, q):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def sanitize_array(x, clip_value=1e12):
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)
    x = np.clip(x, -clip_value, clip_value)
    return x


def fit_standard_scaler(X):
    X = sanitize_array(X, 1e12)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    mean[~np.isfinite(mean)] = 0.0
    std[~np.isfinite(std)] = 1.0
    std[std < 1e-8] = 1.0
    return {"mean": mean.astype(np.float64), "std": std.astype(np.float64)}


def apply_scaler(X, scaler, clip_out=1e6):
    X = sanitize_array(X, 1e12)
    Xs = (X - scaler["mean"]) / scaler["std"]
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = np.clip(Xs, -clip_out, clip_out)
    return Xs.astype(np.float32)


def filter_valid_rows(seq_x, glob_x, y):
    seq_ok = np.all(np.isfinite(seq_x.reshape(seq_x.shape[0], -1)), axis=1)
    glob_ok = np.all(np.isfinite(glob_x), axis=1)
    y_ok = np.all(np.isfinite(y), axis=1)
    mask = seq_ok & glob_ok & y_ok
    return seq_x[mask], glob_x[mask], y[mask]


# =========================================================
# Data
# =========================================================
def load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    required = [
        "coeffs", "center", "target",
        "Q_total", "D1", "D2", "eps1", "eps2",
        "L1", "L2", "rho", "mu", "g"
    ]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {npz_path}. Available: {list(data.keys())}")
    return {k: np.asarray(data[k]) for k in required}


def build_inputs(data, use_log_features=True):
    coeffs = np.asarray(data["coeffs"], dtype=np.float64)
    center = np.asarray(data["center"], dtype=np.float64)
    y = np.asarray(data["target"], dtype=np.float64)

    coeffs = sanitize_array(coeffs, 1e30)
    center = sanitize_array(center, 1e12)
    y = sanitize_array(y, 1e12)

    coeffs = signed_log1p(coeffs)
    seq_x = np.concatenate([coeffs, center[..., None]], axis=2)
    seq_x = sanitize_array(seq_x, 1e12)

    globals_raw = [
        np.asarray(data["Q_total"], dtype=np.float64).reshape(-1, 1),
        np.asarray(data["D1"], dtype=np.float64).reshape(-1, 1),
        np.asarray(data["D2"], dtype=np.float64).reshape(-1, 1),
        np.asarray(data["eps1"], dtype=np.float64).reshape(-1, 1),
        np.asarray(data["eps2"], dtype=np.float64).reshape(-1, 1),
        np.asarray(data["L1"], dtype=np.float64).reshape(-1, 1),
        np.asarray(data["L2"], dtype=np.float64).reshape(-1, 1),
        np.asarray(data["rho"], dtype=np.float64).reshape(-1, 1),
        np.asarray(data["mu"], dtype=np.float64).reshape(-1, 1),
        np.asarray(data["g"], dtype=np.float64).reshape(-1, 1),
    ]
    globals_raw = [sanitize_array(x, 1e12) for x in globals_raw]

    if use_log_features:
        globals_proc = []
        for i, arr in enumerate(globals_raw):
            if i < 9:
                globals_proc.append(np.log(np.clip(arr, 1e-12, None)))
            else:
                globals_proc.append(arr)
        glob_x = np.concatenate(globals_proc, axis=1)
    else:
        glob_x = np.concatenate(globals_raw, axis=1)

    glob_x = sanitize_array(glob_x, 1e12)
    return seq_x, glob_x, y


class MultiInputDataset(Dataset):
    def __init__(self, seq_x, glob_x, y):
        self.seq_x = torch.from_numpy(seq_x.astype(np.float32))
        self.glob_x = torch.from_numpy(glob_x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.seq_x[idx], self.glob_x[idx], self.y[idx]


# =========================================================
# Models
# =========================================================
class MLPModel(nn.Module):
    def __init__(self, seq_dim, seq_len, glob_dim, hidden_dims=(256, 256, 128), dropout=0.1, out_dim=3):
        super().__init__()
        in_dim = seq_dim * seq_len + glob_dim
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        x = torch.cat([seq_x.flatten(1), glob_x], dim=1)
        return self.net(x)


class LSTMModel(nn.Module):
    def __init__(self, seq_dim, glob_dim, hidden_size=128, num_layers=2, dropout=0.1, out_dim=3, head_hidden=128, head_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = self._build_head(hidden_size + glob_dim, head_hidden, out_dim, dropout, head_layers)

    @staticmethod
    def _build_head(in_dim, hidden_dim, out_dim, dropout, head_layers):
        layers = []
        prev = in_dim
        cur = hidden_dim
        for _ in range(head_layers - 1):
            layers += [nn.Linear(prev, cur), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = cur
            cur = max(cur // 2, 32)
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        _, (hn, _) = self.lstm(seq_x)
        h_last = hn[-1]
        return self.head(torch.cat([h_last, glob_x], dim=1))


class GRUModel(nn.Module):
    def __init__(self, seq_dim, glob_dim, hidden_size=128, num_layers=2, dropout=0.1, out_dim=3, head_hidden=128, head_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=seq_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = self._build_head(hidden_size + glob_dim, head_hidden, out_dim, dropout, head_layers)

    @staticmethod
    def _build_head(in_dim, hidden_dim, out_dim, dropout, head_layers):
        layers = []
        prev = in_dim
        cur = hidden_dim
        for _ in range(head_layers - 1):
            layers += [nn.Linear(prev, cur), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = cur
            cur = max(cur // 2, 32)
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        _, hn = self.gru(seq_x)
        h_last = hn[-1]
        return self.head(torch.cat([h_last, glob_x], dim=1))


class TransformerModel(nn.Module):
    def __init__(self, seq_dim, seq_len, glob_dim,
                 d_model=96, nhead=4, num_layers=2, dropout=0.1,
                 out_dim=3, ff_dim=192, head_hidden=128,
                 head_layers=2, use_cls_token=True):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.input_proj = nn.Linear(seq_dim, d_model)

        total_len = seq_len + (1 if use_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, total_len, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if use_cls_token else None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = self._build_head(d_model + glob_dim, head_hidden, out_dim, dropout, head_layers)

    @staticmethod
    def _build_head(in_dim, hidden_dim, out_dim, dropout, head_layers):
        layers = []
        prev = in_dim
        cur = hidden_dim
        for _ in range(head_layers - 1):
            layers += [nn.Linear(prev, cur), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = cur
            cur = max(cur // 2, 32)
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        bsz = seq_x.size(0)
        x = self.input_proj(seq_x)

        if self.use_cls_token:
            cls = self.cls_token.expand(bsz, -1, -1)
            x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed[:, :x.size(1), :]
        h = self.norm(self.encoder(x))

        pooled = h[:, 0, :] if self.use_cls_token else h.mean(dim=1)
        return self.head(torch.cat([pooled, glob_x], dim=1))


def build_model(model_name, seq_dim, seq_len, glob_dim, hp):
    if model_name == "mlp":
        return MLPModel(seq_dim, seq_len, glob_dim, tuple(hp["hidden_dims"]), hp["dropout"], 3)
    if model_name == "lstm":
        return LSTMModel(seq_dim, glob_dim, hp["hidden_size"], hp["num_layers"], hp["dropout"], 3, hp["head_hidden"], hp["head_layers"])
    if model_name == "gru":
        return GRUModel(seq_dim, glob_dim, hp["hidden_size"], hp["num_layers"], hp["dropout"], 3, hp["head_hidden"], hp["head_layers"])
    if model_name == "transformer":
        return TransformerModel(seq_dim, seq_len, glob_dim, hp["d_model"], hp["nhead"], hp["num_layers"], hp["dropout"], 3, hp["ff_dim"], hp["head_hidden"], hp["head_layers"], hp["use_cls_token"])
    raise ValueError(model_name)


# =========================================================
# Train / Eval metrics
# =========================================================
def regression_metrics(pred, true):
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


def build_optimizer(model, name, lr, weight_decay):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(name)


def run_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.SmoothL1Loss(beta=0.1)
    total_loss = 0.0
    n = 0
    for seq_x, glob_x, yb in loader:
        seq_x, glob_x, yb = seq_x.to(device), glob_x.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(seq_x, glob_x)
        loss = criterion(pred, yb)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        bs = seq_x.shape[0]
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_direct(model, loader, device):
    model.eval()
    criterion = nn.SmoothL1Loss(beta=0.1)
    total_loss = 0.0
    n = 0
    preds, trues = [], []
    for seq_x, glob_x, yb in loader:
        seq_x, glob_x, yb = seq_x.to(device), glob_x.to(device), yb.to(device)
        pred = model(seq_x, glob_x)
        loss = criterion(pred, yb)
        bs = seq_x.shape[0]
        if torch.isfinite(loss):
            total_loss += loss.item() * bs
            n += bs
        preds.append(pred.cpu().numpy())
        trues.append(yb.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    m = regression_metrics(preds, trues)
    m["loss"] = total_loss / max(n, 1)
    return m, preds, trues


# =========================================================
# Newton refinement
# =========================================================
def system_F(z, params):
    Q1, x1, x2 = z[..., 0], z[..., 1], z[..., 2]

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

    m1 = (Re1 > 0) & (z1 > 0)
    m2 = (Re2 > 0) & (z2 > 0)

    F1[m1] = x1[m1] + 2.0 * np.log10(z1[m1])
    F2[m2] = x2[m2] + 2.0 * np.log10(z2[m2])

    H1 = 8.0 * L1 * (Q1 ** 2) / (g * (PI ** 2) * (D1 ** 5) * (x1 ** 2))
    H2 = 8.0 * L2 * (Q2 ** 2) / (g * (PI ** 2) * (D2 ** 5) * (x2 ** 2))
    F3 = H1 - H2

    return np.stack([F1, F2, F3], axis=-1)


def numerical_jacobian_single(z, p, eps=1e-6):
    z = np.asarray(z, dtype=np.float64)
    J = np.zeros((3, 3), dtype=np.float64)
    f0 = system_F(z[None, :], p)[0]
    for j in range(3):
        zp = z.copy()
        zm = z.copy()
        step = eps * max(1.0, abs(z[j]))
        zp[j] += step
        zm[j] -= step
        fp = system_F(zp[None, :], p)[0]
        fm = system_F(zm[None, :], p)[0]
        J[:, j] = (fp - fm) / (2.0 * step)
    return J, f0


def project_feasible(z, p):
    z = np.asarray(z, dtype=np.float64).copy()
    QT = float(p["Q_total"])
    z[0] = np.clip(z[0], max(1e-8, QT * 1e-5), QT - max(1e-8, QT * 1e-5))
    z[1] = max(z[1], 1e-3)
    z[2] = max(z[2], 1e-3)
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

        step = np.clip(step, -5.0, 5.0)
        z_new = project_feasible(z - damping * step, p)
        f_new = system_F(z_new[None, :], p)[0]

        if np.linalg.norm(f_new, ord=2) > np.linalg.norm(f, ord=2):
            z_half = project_feasible(z - 0.5 * damping * step, p)
            f_half = system_F(z_half[None, :], p)[0]
            if np.linalg.norm(f_half, ord=2) < np.linalg.norm(f_new, ord=2):
                z_new = z_half
                f_new = f_half

        z = z_new
        used_iter = k
        if np.linalg.norm(f_new, ord=np.inf) <= tol:
            converged = True
            break

    return z, used_iter, converged


def residual_metrics(pred, data):
    params = {k: np.asarray(data[k], dtype=np.float64) for k in ["Q_total", "D1", "D2", "eps1", "eps2", "L1", "L2", "rho", "mu", "g"]}
    F = system_F(pred.astype(np.float64), params)
    norms_inf = np.max(np.abs(F), axis=1)
    valid = np.all(np.isfinite(F), axis=1)
    return {
        "valid_ratio": float(np.mean(valid)),
        "residual_mean": float(np.nanmean(norms_inf)),
        "residual_median": float(np.nanmedian(norms_inf)),
        "residual_p90": percentile(norms_inf[np.isfinite(norms_inf)], 90),
    }


def refine_batch(z_init, data, tol=1e-12, max_iter=20):
    n = len(z_init)
    out = np.zeros_like(z_init, dtype=np.float64)
    iters = np.zeros(n, dtype=np.int32)
    conv = np.zeros(n, dtype=bool)

    for i in range(n):
        p = {k: float(np.asarray(data[k])[i]) for k in ["Q_total", "D1", "D2", "eps1", "eps2", "L1", "L2", "rho", "mu", "g"]}
        zf, it, ok = newton_system_single(z_init[i], p, tol=tol, max_iter=max_iter)
        out[i] = zf
        iters[i] = it
        conv[i] = ok

    return out, iters, conv


# =========================================================
# Grid
# =========================================================
def grid_product(grid):
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in itertools.product(*vals):
        yield {k: v for k, v in zip(keys, combo)}


def build_model_grids(selected_models):
    grids = []

    if "mlp" in selected_models:
        grids += list(grid_product({
            "model": ["mlp"],
            "use_log_features": [True],
            "optimizer": ["adamw"],
            "weight_decay": [1e-4],
            "dropout": [0.05, 0.1],
            "hidden_dims": [[256, 256, 128], [256, 128, 64]],
            "lr": [5e-4, 1e-3],
        }))

    if "lstm" in selected_models:
        grids += list(grid_product({
            "model": ["lstm"],
            "use_log_features": [True],
            "optimizer": ["adamw"],
            "weight_decay": [1e-4],
            "dropout": [0.05, 0.1],
            "hidden_size": [96, 128],
            "num_layers": [1, 2],
            "head_hidden": [128],
            "head_layers": [2],
            "lr": [5e-4, 1e-3],
        }))

    if "gru" in selected_models:
        grids += list(grid_product({
            "model": ["gru"],
            "use_log_features": [True],
            "optimizer": ["adamw"],
            "weight_decay": [1e-4],
            "dropout": [0.05, 0.1],
            "hidden_size": [96, 128],
            "num_layers": [1, 2],
            "head_hidden": [128],
            "head_layers": [2],
            "lr": [5e-4, 1e-3],
        }))

    if "transformer" in selected_models:
        grids += list(grid_product({
            "model": ["transformer"],
            "use_log_features": [True],
            "optimizer": ["adamw"],
            "weight_decay": [1e-4],
            "dropout": [0.05, 0.1],
            "d_model": [64, 96],
            "nhead": [4],
            "num_layers": [1, 2],
            "ff_dim": [128, 192],
            "head_hidden": [128],
            "head_layers": [2],
            "use_cls_token": [False, True],
            "lr": [5e-4],
        }))

    normalized = []
    for hp in grids:
        base = {
            "model": hp["model"],
            "use_log_features": hp.get("use_log_features", True),
            "optimizer": hp.get("optimizer", "adamw"),
            "weight_decay": hp.get("weight_decay", 1e-4),
            "dropout": hp.get("dropout", 0.1),
            "lr": hp.get("lr", 5e-4),

            "hidden_dims": hp.get("hidden_dims", [256, 256, 128]),

            "hidden_size": hp.get("hidden_size", 128),
            "num_layers": hp.get("num_layers", 2),
            "head_hidden": hp.get("head_hidden", 128),
            "head_layers": hp.get("head_layers", 2),

            "d_model": hp.get("d_model", 96),
            "nhead": hp.get("nhead", 4),
            "ff_dim": hp.get("ff_dim", 192),
            "use_cls_token": hp.get("use_cls_token", False),
        }
        normalized.append(base)

    return normalized


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz", required=True)
    ap.add_argument("--val_npz", required=True)
    ap.add_argument("--test_npz", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--models", nargs="+", default=["transformer", "lstm", "gru", "mlp"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=256)

    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--max_newton_iter", type=int, default=20)

    ap.add_argument(
        "--rank_metric",
        default="plus_newton_r2",
        choices=[
            "plus_newton_r2",
            "plus_newton_rmse",
            "plus_newton_mae",
            "plus_newton_converged_ratio",
            "direct_r2",
            "direct_rmse",
            "direct_mae",
        ],
    )
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data = load_npz(args.train_npz)
    val_data = load_npz(args.val_npz)
    test_data = load_npz(args.test_npz)

    grids = build_model_grids(args.models)

    trial_rows = []
    best_rank_value = None
    best_rank_row = None
    best_ckpt = None

    for trial_idx, hp in enumerate(grids, start=1):
        trial_name = f"trial_{trial_idx:03d}_{hp['model']}"
        print(f"\n========== {trial_name} ==========")
        print(json.dumps(hp, ensure_ascii=False))

        start_time = time.time()

        tr_seq, tr_glob, tr_y = build_inputs(train_data, hp["use_log_features"])
        va_seq, va_glob, va_y = build_inputs(val_data, hp["use_log_features"])
        te_seq, te_glob, te_y = build_inputs(test_data, hp["use_log_features"])

        tr_seq, tr_glob, tr_y = filter_valid_rows(tr_seq, tr_glob, tr_y)
        va_seq, va_glob, va_y = filter_valid_rows(va_seq, va_glob, va_y)
        te_seq, te_glob, te_y = filter_valid_rows(te_seq, te_glob, te_y)

        seq_scaler = fit_standard_scaler(tr_seq.reshape(-1, tr_seq.shape[-1]))
        glob_scaler = fit_standard_scaler(tr_glob)

        tr_seq = apply_scaler(tr_seq.reshape(-1, tr_seq.shape[-1]), seq_scaler).reshape(tr_seq.shape)
        va_seq = apply_scaler(va_seq.reshape(-1, va_seq.shape[-1]), seq_scaler).reshape(va_seq.shape)
        te_seq = apply_scaler(te_seq.reshape(-1, te_seq.shape[-1]), seq_scaler).reshape(te_seq.shape)

        tr_glob = apply_scaler(tr_glob, glob_scaler)
        va_glob = apply_scaler(va_glob, glob_scaler)
        te_glob = apply_scaler(te_glob, glob_scaler)

        train_loader = DataLoader(MultiInputDataset(tr_seq, tr_glob, tr_y), batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(MultiInputDataset(va_seq, va_glob, va_y), batch_size=args.batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(MultiInputDataset(te_seq, te_glob, te_y), batch_size=args.batch_size, shuffle=False, drop_last=False)

        model = build_model(hp["model"], tr_seq.shape[2], tr_seq.shape[1], tr_glob.shape[1], hp).to(args.device)
        optimizer = build_optimizer(model, hp["optimizer"], hp["lr"], hp["weight_decay"])

        best_val_rmse = float("inf")
        best_state = None
        best_epoch = -1
        wait = 0

        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, train_loader, optimizer, args.device)
            val_metrics, _, _ = evaluate_direct(model, val_loader, args.device)

            print(
                f"[{trial_name}] "
                f"Epoch {epoch:03d} "
                f"train_loss={train_loss:.6f} "
                f"val_rmse={val_metrics['rmse']:.6f} "
                f"val_r2={val_metrics['r2']:.6f}"
            )

            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                best_epoch = epoch
                wait = 0
                best_state = deepcopy(model.state_dict())
            else:
                wait += 1
                if wait >= args.patience:
                    break

        if best_state is None:
            print(f"[WARN] {trial_name} skipped")
            continue

        model.load_state_dict(best_state)

        direct_metrics, pred_direct, y_true = evaluate_direct(model, test_loader, args.device)
        direct_metrics.update(residual_metrics(pred_direct.astype(np.float64), test_data))

        refined, iters, conv = refine_batch(pred_direct.astype(np.float64), test_data, tol=args.tol, max_iter=args.max_newton_iter)
        plus_metrics = regression_metrics(refined, y_true.astype(np.float64))
        plus_metrics.update(residual_metrics(refined, test_data))
        plus_metrics["newton_iter_mean"] = float(np.mean(iters))
        plus_metrics["newton_iter_median"] = float(np.median(iters))
        plus_metrics["newton_iter_p90"] = float(np.percentile(iters, 90))
        plus_metrics["newton_converged_ratio"] = float(np.mean(conv))

        elapsed = time.time() - start_time

        row = {
            "trial_id": trial_idx,
            "trial_name": trial_name,
            "model": hp["model"],
            "best_epoch": best_epoch,
            "elapsed_sec": elapsed,

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

            "hp_use_log_features": hp["use_log_features"],
            "hp_optimizer": hp["optimizer"],
            "hp_weight_decay": hp["weight_decay"],
            "hp_dropout": hp["dropout"],
            "hp_lr": hp["lr"],
            "hp_hidden_dims": json.dumps(hp["hidden_dims"]),
            "hp_hidden_size": hp["hidden_size"],
            "hp_num_layers": hp["num_layers"],
            "hp_head_hidden": hp["head_hidden"],
            "hp_head_layers": hp["head_layers"],
            "hp_d_model": hp["d_model"],
            "hp_nhead": hp["nhead"],
            "hp_ff_dim": hp["ff_dim"],
            "hp_use_cls_token": hp["use_cls_token"],
        }

        trial_rows.append(row)

        metric_value = row[args.rank_metric]
        if best_rank_value is None:
            better = True
        else:
            if args.rank_metric in ["plus_newton_rmse", "plus_newton_mae", "direct_rmse", "direct_mae"]:
                better = metric_value < best_rank_value
            else:
                better = metric_value > best_rank_value

        if better:
            best_rank_value = metric_value
            best_rank_row = dict(row)
            best_ckpt = {
                "state_dict": deepcopy(model.state_dict()),
                "hp": deepcopy(hp),
                "seq_scaler": seq_scaler,
                "glob_scaler": glob_scaler,
                "meta": {
                    "seq_dim": int(tr_seq.shape[2]),
                    "seq_len": int(tr_seq.shape[1]),
                    "glob_dim": int(tr_glob.shape[1]),
                    "rank_metric": args.rank_metric,
                    "rank_value": metric_value,
                }
            }

        with open(out_dir / f"{trial_name}.json", "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2)

    if not trial_rows:
        raise RuntimeError("No successful trials were completed.")

    reverse = args.rank_metric not in ["plus_newton_rmse", "plus_newton_mae", "direct_rmse", "direct_mae"]
    trial_rows_sorted = sorted(trial_rows, key=lambda r: r[args.rank_metric], reverse=reverse)

    save_csv(out_dir / "all_trials.csv", trial_rows_sorted)

    with open(out_dir / "best_result.json", "w", encoding="utf-8") as f:
        json.dump(best_rank_row, f, ensure_ascii=False, indent=2)

    if best_ckpt is not None:
        torch.save(best_ckpt, out_dir / "best_model_by_grid.pt")

    print("\n================ FINAL RANKING ================")
    for row in trial_rows_sorted[:10]:
        print({
            "trial_id": row["trial_id"],
            "model": row["model"],
            args.rank_metric: row[args.rank_metric],
            "plus_newton_rmse": row["plus_newton_rmse"],
            "plus_newton_r2": row["plus_newton_r2"],
            "plus_newton_converged_ratio": row["plus_newton_converged_ratio"],
            "iter_mean": row["plus_newton_newton_iter_mean"],
        })

    print("\n[DONE]")
    print(out_dir / "all_trials.csv")
    print(out_dir / "best_result.json")
    print(out_dir / "best_model_by_grid.pt")


if __name__ == "__main__":
    main()