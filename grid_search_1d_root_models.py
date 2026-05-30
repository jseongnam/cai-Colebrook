#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import itertools
import json
import math
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Utility
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def sanitize_array(x: np.ndarray, clip_value: float = 1e12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)
    x = np.clip(x, -clip_value, clip_value)
    return x


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def percentile(x, q):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def colebrook_like_F_np(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x + 2.0 * np.log10(a + b * x)


# =========================================================
# Standardizer
# =========================================================
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

    def save(self) -> Dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


# =========================================================
# Dataset
# =========================================================
class NPZRootDataset1D(Dataset):
    def __init__(
        self,
        npz_path: str,
        use_log_ab: bool = False,
        include_center: bool = True,
        include_ab: bool = True,
        target_key: str = "root",
    ):
        data = np.load(npz_path)

        coeffs = sanitize_array(data["coeffs"].astype(np.float32))
        center = sanitize_array(data["center"].astype(np.float32)).reshape(-1, 1)
        a = sanitize_array(data["a"].astype(np.float32)).reshape(-1, 1)
        b = sanitize_array(data["b"].astype(np.float32)).reshape(-1, 1)
        y = sanitize_array(data[target_key].astype(np.float32)).reshape(-1, 1)

        # sequence branch: coeffs only
        self.seq_x = coeffs[..., None].astype(np.float32)  # (N, L, 1)

        feats = []
        if include_center:
            feats.append(center)
        if include_ab:
            if use_log_ab:
                feats.append(np.log(np.clip(a, 1e-12, None)))
                feats.append(np.log(np.clip(b, 1e-12, None)))
            else:
                feats.append(a)
                feats.append(b)

        self.glob_x = np.concatenate(feats, axis=1).astype(np.float32) if feats else np.zeros((coeffs.shape[0], 0), dtype=np.float32)
        self.y = y.astype(np.float32)
        self.a = a.astype(np.float32)
        self.b = b.astype(np.float32)

        self.seq_x = torch.from_numpy(self.seq_x)
        self.glob_x = torch.from_numpy(self.glob_x)
        self.y = torch.from_numpy(self.y)
        self.a = torch.from_numpy(self.a)
        self.b = torch.from_numpy(self.b)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.seq_x[idx], self.glob_x[idx], self.y[idx], self.a[idx], self.b[idx]


def standardize_datasets(train_ds, val_ds, test_ds):
    seq_scaler = Standardizer()
    glob_scaler = Standardizer()

    train_seq_flat = train_ds.seq_x.numpy().reshape(-1, train_ds.seq_x.shape[-1])
    seq_scaler.fit(train_seq_flat)

    train_ds.seq_x = torch.from_numpy(seq_scaler.transform(train_ds.seq_x.numpy().reshape(-1, train_ds.seq_x.shape[-1])).reshape(train_ds.seq_x.shape))
    val_ds.seq_x = torch.from_numpy(seq_scaler.transform(val_ds.seq_x.numpy().reshape(-1, val_ds.seq_x.shape[-1])).reshape(val_ds.seq_x.shape))
    test_ds.seq_x = torch.from_numpy(seq_scaler.transform(test_ds.seq_x.numpy().reshape(-1, test_ds.seq_x.shape[-1])).reshape(test_ds.seq_x.shape))

    if train_ds.glob_x.shape[1] > 0:
        glob_scaler.fit(train_ds.glob_x.numpy())
        train_ds.glob_x = torch.from_numpy(glob_scaler.transform(train_ds.glob_x.numpy()))
        val_ds.glob_x = torch.from_numpy(glob_scaler.transform(val_ds.glob_x.numpy()))
        test_ds.glob_x = torch.from_numpy(glob_scaler.transform(test_ds.glob_x.numpy()))
    else:
        glob_scaler.mean = np.zeros((1, 0), dtype=np.float64)
        glob_scaler.std = np.ones((1, 0), dtype=np.float64)

    return seq_scaler, glob_scaler


# =========================================================
# Models
# =========================================================
class MLPRegressor(nn.Module):
    def __init__(self, seq_dim: int, seq_len: int, glob_dim: int, hidden_dims: List[int], dropout: float = 0.0):
        super().__init__()
        in_dim = seq_dim * seq_len + glob_dim
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        return self.net(torch.cat([seq_x.flatten(1), glob_x], dim=1))


class LSTMRegressor(nn.Module):
    def __init__(self, seq_dim: int, glob_dim: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1, head_hidden: int = 128, head_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(seq_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = self._build_head(hidden_size + glob_dim, head_hidden, 1, dropout, head_layers)

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
            cur = max(cur // 2, 16)
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        _, (hn, _) = self.lstm(seq_x)
        return self.head(torch.cat([hn[-1], glob_x], dim=1))


class GRURegressor(nn.Module):
    def __init__(self, seq_dim: int, glob_dim: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1, head_hidden: int = 128, head_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(seq_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = self._build_head(hidden_size + glob_dim, head_hidden, 1, dropout, head_layers)

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
            cur = max(cur // 2, 16)
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        _, hn = self.gru(seq_x)
        return self.head(torch.cat([hn[-1], glob_x], dim=1))


class TransformerRegressor(nn.Module):
    def __init__(self, seq_dim: int, seq_len: int, glob_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1, ff_dim: int = 128, head_hidden: int = 128, head_layers: int = 2, use_cls_token: bool = True):
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
        self.head = self._build_head(d_model + glob_dim, head_hidden, 1, dropout, head_layers)

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
            cur = max(cur // 2, 16)
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        bsz = seq_x.size(0)
        x = self.input_proj(seq_x)
        if self.use_cls_token:
            x = torch.cat([self.cls_token.expand(bsz, -1, -1), x], dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        h = self.norm(self.encoder(x))
        pooled = h[:, 0, :] if self.use_cls_token else h.mean(dim=1)
        return self.head(torch.cat([pooled, glob_x], dim=1))


def build_model(model_name: str, seq_dim: int, seq_len: int, glob_dim: int, hp: Dict):
    if model_name == "mlp":
        return MLPRegressor(seq_dim, seq_len, glob_dim, hp["hidden_dims"], hp["dropout"])
    if model_name == "lstm":
        return LSTMRegressor(seq_dim, glob_dim, hp["hidden_size"], hp["num_layers"], hp["dropout"], hp["head_hidden"], hp["head_layers"])
    if model_name == "gru":
        return GRURegressor(seq_dim, glob_dim, hp["hidden_size"], hp["num_layers"], hp["dropout"], hp["head_hidden"], hp["head_layers"])
    if model_name == "transformer":
        return TransformerRegressor(seq_dim, seq_len, glob_dim, hp["d_model"], hp["nhead"], hp["num_layers"], hp["dropout"], hp["ff_dim"], hp["head_hidden"], hp["head_layers"], hp["use_cls_token"])
    raise ValueError(model_name)


# =========================================================
# Train / Eval
# =========================================================
def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    count = 0
    ys, preds, as_, bs_ = [], [], [], []

    for seq_x, glob_x, y, a, b in loader:
        seq_x = seq_x.to(device)
        glob_x = glob_x.to(device)
        y = y.to(device)
        a = a.to(device)
        b = b.to(device)

        with torch.set_grad_enabled(train):
            pred = model(seq_x, glob_x)
            loss = criterion(pred, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        total_loss += loss.item() * seq_x.size(0)
        count += seq_x.size(0)

        ys.append(y.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())
        as_.append(a.detach().cpu().numpy())
        bs_.append(b.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0).reshape(-1)
    y_pred = np.concatenate(preds, axis=0).reshape(-1)
    a_np = np.concatenate(as_, axis=0).reshape(-1)
    b_np = np.concatenate(bs_, axis=0).reshape(-1)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(r2_score_np(y_true, y_pred))

    inside = a_np + b_np * y_pred
    valid_mask = inside > 0
    if np.any(valid_mask):
        residual = np.abs(colebrook_like_F_np(y_pred[valid_mask], a_np[valid_mask], b_np[valid_mask]))
        residual_mean = float(np.mean(residual))
        residual_median = float(np.median(residual))
        residual_p90 = float(np.percentile(residual, 90))
        valid_ratio = float(np.mean(valid_mask))
    else:
        residual_mean = float("inf")
        residual_median = float("inf")
        residual_p90 = float("inf")
        valid_ratio = 0.0

    return {
        "loss": total_loss / max(count, 1),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "residual_mean": residual_mean,
        "residual_median": residual_median,
        "residual_p90": residual_p90,
        "valid_ratio": valid_ratio,
    }


def build_optimizer(model, name, lr, weight_decay):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(name)


def newton_refine_batch(x0, a, b, tol=1e-12, max_iter=20):
    x = x0.astype(np.float64).copy()
    iters = np.zeros_like(x, dtype=np.int32)
    converged = np.zeros_like(x, dtype=bool)

    for i in range(len(x)):
        xi = float(x[i]); ai = float(a[i]); bi = float(b[i])
        for it in range(1, max_iter + 1):
            inside = ai + bi * xi
            if inside <= 0:
                break
            fx = xi + 2.0 * np.log10(inside)
            dfx = 1.0 + 2.0 * bi / (np.log(10.0) * inside)
            if not np.isfinite(fx) or not np.isfinite(dfx) or abs(dfx) < 1e-15:
                break
            step = np.clip(fx / dfx, -5.0, 5.0)
            x_new = xi - step
            if ai + bi * x_new <= 0:
                x_half = xi - 0.5 * step
                if ai + bi * x_half > 0:
                    x_new = x_half
                else:
                    break
            xi = x_new
            iters[i] = it
            inside_new = ai + bi * xi
            if inside_new > 0:
                f_new = xi + 2.0 * np.log10(inside_new)
                if abs(f_new) <= tol:
                    converged[i] = True
                    break
        x[i] = xi
    return x, iters, converged


def summarize_method(name, pred, true, a, b, iters=None, conv=None):
    pred = pred.reshape(-1)
    true = true.reshape(-1)

    mae = float(np.mean(np.abs(true - pred)))
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    r2 = float(r2_score_np(true, pred))
    max_abs_error = float(np.max(np.abs(true - pred)))

    inside = a + b * pred
    valid_mask = inside > 0
    if np.any(valid_mask):
        residual = np.abs(colebrook_like_F_np(pred[valid_mask], a[valid_mask], b[valid_mask]))
        residual_mean = float(np.mean(residual))
        residual_median = float(np.median(residual))
        residual_p90 = float(np.percentile(residual, 90))
        valid_ratio = float(np.mean(valid_mask))
    else:
        residual_mean = float("inf")
        residual_median = float("inf")
        residual_p90 = float("inf")
        valid_ratio = 0.0

    row = {
        "name": name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "valid_ratio": valid_ratio,
        "residual_mean": residual_mean,
        "residual_median": residual_median,
        "residual_p90": residual_p90,
        "max_abs_error": max_abs_error,
    }

    if iters is not None:
        row["newton_iter_mean"] = float(np.mean(iters))
        row["newton_iter_median"] = float(np.median(iters))
        row["newton_iter_p90"] = float(np.percentile(iters, 90))

    if conv is not None:
        row["newton_converged_ratio"] = float(np.mean(conv))

    return row


def grid_product(grid: Dict[str, list]):
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in __import__("itertools").product(*vals):
        yield {k: v for k, v in zip(keys, combo)}


def build_search_space(selected_models):
    configs = []

    if "mlp" in selected_models:
        grid = {
            "model": ["mlp"],
            "use_log_ab": [False, True],
            "optimizer": ["adamw"],
            "criterion": ["mse", "smoothl1"],
            "dropout": [0.0, 0.1],
            "lr": [1e-3, 5e-4],
            "weight_decay": [1e-5, 1e-4],
            "hidden_dims": [[256, 256, 128], [128, 128], [256, 128, 64]],
            "hidden_size": [128],
            "num_layers": [2],
            "head_hidden": [128],
            "head_layers": [2],
            "d_model": [64],
            "nhead": [4],
            "ff_dim": [128],
            "use_cls_token": [False],
        }
        configs.extend(list(grid_product(grid)))

    if "lstm" in selected_models:
        grid = {
            "model": ["lstm"],
            "use_log_ab": [False, True],
            "optimizer": ["adamw"],
            "criterion": ["mse", "smoothl1"],
            "dropout": [0.0, 0.1],
            "lr": [1e-3, 5e-4],
            "weight_decay": [1e-5, 1e-4],
            "hidden_dims": [[256, 256, 128]],
            "hidden_size": [64, 128],
            "num_layers": [1, 2],
            "head_hidden": [64, 128],
            "head_layers": [1, 2],
            "d_model": [64],
            "nhead": [4],
            "ff_dim": [128],
            "use_cls_token": [False],
        }
        configs.extend(list(grid_product(grid)))

    if "gru" in selected_models:
        grid = {
            "model": ["gru"],
            "use_log_ab": [False, True],
            "optimizer": ["adamw"],
            "criterion": ["mse", "smoothl1"],
            "dropout": [0.0, 0.1],
            "lr": [1e-3, 5e-4],
            "weight_decay": [1e-5, 1e-4],
            "hidden_dims": [[256, 256, 128]],
            "hidden_size": [64, 128],
            "num_layers": [1, 2],
            "head_hidden": [64, 128],
            "head_layers": [1, 2],
            "d_model": [64],
            "nhead": [4],
            "ff_dim": [128],
            "use_cls_token": [False],
        }
        configs.extend(list(grid_product(grid)))

    if "transformer" in selected_models:
        grid = {
            "model": ["transformer"],
            "use_log_ab": [False, True],
            "optimizer": ["adamw"],
            "criterion": ["mse", "smoothl1"],
            "dropout": [0.0, 0.1],
            "lr": [1e-3, 5e-4],
            "weight_decay": [1e-5, 1e-4],
            "hidden_dims": [[256, 256, 128]],
            "hidden_size": [128],
            "num_layers": [1, 2],
            "head_hidden": [64, 128],
            "head_layers": [1, 2],
            "d_model": [32, 64],
            "nhead": [2, 4],
            "ff_dim": [64, 128],
            "use_cls_token": [False, True],
        }
        configs.extend(list(grid_product(grid)))

    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_npz", type=str, required=True)
    parser.add_argument("--val_npz", type=str, required=True)
    parser.add_argument("--test_npz", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--models", nargs="+", default=["mlp", "lstm", "gru", "transformer"])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--include_center", action="store_true", default=True)
    parser.add_argument("--include_ab", action="store_true", default=True)
    parser.add_argument("--no_include_center", action="store_true")
    parser.add_argument("--no_include_ab", action="store_true")

    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max_newton_iter", type=int, default=20)

    parser.add_argument(
        "--rank_metric",
        type=str,
        default="plus_newton_r2",
        choices=[
            "direct_r2",
            "direct_rmse",
            "direct_mae",
            "plus_newton_r2",
            "plus_newton_rmse",
            "plus_newton_mae",
            "plus_newton_converged_ratio",
        ],
    )

    args = parser.parse_args()

    if args.no_include_center:
        args.include_center = False
    if args.no_include_ab:
        args.include_ab = False

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device(args.device)

    search_space = build_search_space(args.models)

    all_rows = []
    best_row = None
    best_metric = None
    best_ckpt = None

    for trial_id, hp in enumerate(search_space, start=1):
        trial_name = f"trial_{trial_id:03d}_{hp['model']}"
        print(f"\\n========== {trial_name} ==========")
        print(json.dumps(hp, ensure_ascii=False))

        start_time = time.time()

        train_ds = NPZRootDataset1D(args.train_npz, hp["use_log_ab"], args.include_center, args.include_ab)
        val_ds = NPZRootDataset1D(args.val_npz, hp["use_log_ab"], args.include_center, args.include_ab)
        test_ds = NPZRootDataset1D(args.test_npz, hp["use_log_ab"], args.include_center, args.include_ab)
        seq_scaler, glob_scaler = standardize_datasets(train_ds, val_ds, test_ds)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        seq_dim = train_ds.seq_x.shape[2]
        seq_len = train_ds.seq_x.shape[1]
        glob_dim = train_ds.glob_x.shape[1]

        model = build_model(hp["model"], seq_dim, seq_len, glob_dim, hp).to(device)
        optimizer = build_optimizer(model, hp["optimizer"], hp["lr"], hp["weight_decay"])
        criterion = nn.MSELoss() if hp["criterion"] == "mse" else nn.SmoothL1Loss(beta=0.1)

        best_val_rmse = float("inf")
        best_state = None
        best_epoch = -1
        wait = 0

        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
            val_metrics = run_epoch(model, val_loader, optimizer, criterion, device, train=False)

            print(
                f"[{trial_name}] "
                f"epoch={epoch:03d} "
                f"train_rmse={train_metrics['rmse']:.6f} "
                f"val_rmse={val_metrics['rmse']:.6f} "
                f"val_r2={val_metrics['r2']:.6f}"
            )

            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                best_state = deepcopy(model.state_dict())
                best_epoch = epoch
                wait = 0
            else:
                wait += 1
                if wait >= args.patience:
                    break

        if best_state is None:
            continue

        model.load_state_dict(best_state)

        # direct test eval
        direct_metrics = run_epoch(model, test_loader, optimizer=None, criterion=criterion, device=device, train=False)

        # raw predictions for Newton refinement
        ys, preds, as_, bs_ = [], [], [], []
        model.eval()
        with torch.no_grad():
            for seq_x, glob_x, y, a, b in test_loader:
                seq_x = seq_x.to(device)
                glob_x = glob_x.to(device)
                pred = model(seq_x, glob_x)
                ys.append(y.numpy())
                preds.append(pred.cpu().numpy())
                as_.append(a.numpy())
                bs_.append(b.numpy())

        y_true = np.concatenate(ys, axis=0).reshape(-1)
        pred_direct = np.concatenate(preds, axis=0).reshape(-1)
        a = np.concatenate(as_, axis=0).reshape(-1)
        b = np.concatenate(bs_, axis=0).reshape(-1)

        pred_refined, iters, conv = newton_refine_batch(pred_direct, a, b, tol=args.tol, max_iter=args.max_newton_iter)

        plus_row = summarize_method("neural_plus_newton", pred_refined, y_true, a, b, iters, conv)
        direct_row = summarize_method("neural_direct", pred_direct, y_true, a, b)

        elapsed = time.time() - start_time

        row = {
            "trial_id": trial_id,
            "trial_name": trial_name,
            "model": hp["model"],
            "best_epoch": best_epoch,
            "elapsed_sec": elapsed,

            "direct_mae": direct_row["mae"],
            "direct_rmse": direct_row["rmse"],
            "direct_r2": direct_row["r2"],
            "direct_valid_ratio": direct_row["valid_ratio"],
            "direct_residual_mean": direct_row["residual_mean"],
            "direct_residual_median": direct_row["residual_median"],
            "direct_residual_p90": direct_row["residual_p90"],

            "plus_newton_mae": plus_row["mae"],
            "plus_newton_rmse": plus_row["rmse"],
            "plus_newton_r2": plus_row["r2"],
            "plus_newton_valid_ratio": plus_row["valid_ratio"],
            "plus_newton_residual_mean": plus_row["residual_mean"],
            "plus_newton_residual_median": plus_row["residual_median"],
            "plus_newton_residual_p90": plus_row["residual_p90"],
            "plus_newton_newton_iter_mean": plus_row["newton_iter_mean"],
            "plus_newton_newton_iter_median": plus_row["newton_iter_median"],
            "plus_newton_newton_iter_p90": plus_row["newton_iter_p90"],
            "plus_newton_converged_ratio": plus_row["newton_converged_ratio"],

            "hp_use_log_ab": hp["use_log_ab"],
            "hp_optimizer": hp["optimizer"],
            "hp_criterion": hp["criterion"],
            "hp_dropout": hp["dropout"],
            "hp_lr": hp["lr"],
            "hp_weight_decay": hp["weight_decay"],
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

        all_rows.append(row)

        cur_metric = row[args.rank_metric]
        if best_metric is None:
            better = True
        else:
            if args.rank_metric in ["direct_rmse", "direct_mae", "plus_newton_rmse", "plus_newton_mae"]:
                better = cur_metric < best_metric
            else:
                better = cur_metric > best_metric

        if better:
            best_metric = cur_metric
            best_row = dict(row)
            best_ckpt = {
                "model_state_dict": deepcopy(model.state_dict()),
                "seq_scaler": seq_scaler.save(),
                "glob_scaler": glob_scaler.save(),
                "args": {
                    "model": hp["model"],
                    "hidden_dims": hp["hidden_dims"],
                    "hidden_size": hp["hidden_size"],
                    "num_layers": hp["num_layers"],
                    "head_hidden": hp["head_hidden"],
                    "head_layers": hp["head_layers"],
                    "d_model": hp["d_model"],
                    "nhead": hp["nhead"],
                    "ff_dim": hp["ff_dim"],
                    "use_cls_token": hp["use_cls_token"],
                    "dropout": hp["dropout"],
                    "use_log_ab": hp["use_log_ab"],
                    "include_center": args.include_center,
                    "include_ab": args.include_ab,
                },
                "seq_dim": seq_dim,
                "seq_len": seq_len,
                "glob_dim": glob_dim,
                "best_val_rmse": best_val_rmse,
                "best_epoch": best_epoch,
            }

        with open(out_dir / f"{trial_name}.json", "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2)

    if not all_rows:
        raise RuntimeError("No successful trials.")

    reverse = args.rank_metric not in ["direct_rmse", "direct_mae", "plus_newton_rmse", "plus_newton_mae"]
    all_rows_sorted = sorted(all_rows, key=lambda r: r[args.rank_metric], reverse=reverse)

    save_csv(out_dir / "all_trials.csv", all_rows_sorted)

    with open(out_dir / "best_result.json", "w", encoding="utf-8") as f:
        json.dump(best_row, f, ensure_ascii=False, indent=2)

    if best_ckpt is not None:
        torch.save(best_ckpt, out_dir / "best_model_by_grid.pt")

    print("\\n================ FINAL RANKING ================")
    for row in all_rows_sorted[:10]:
        print({
            "trial_id": row["trial_id"],
            "model": row["model"],
            args.rank_metric: row[args.rank_metric],
            "plus_newton_rmse": row["plus_newton_rmse"],
            "plus_newton_r2": row["plus_newton_r2"],
            "plus_newton_converged_ratio": row["plus_newton_converged_ratio"],
        })

    print("\\n[DONE]")
    print(out_dir / "all_trials.csv")
    print(out_dir / "best_result.json")
    print(out_dir / "best_model_by_grid.pt")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import itertools
import json
import math
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Utility
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def sanitize_array(x: np.ndarray, clip_value: float = 1e12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)
    x = np.clip(x, -clip_value, clip_value)
    return x


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def percentile(x, q):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def colebrook_like_F_np(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x + 2.0 * np.log10(a + b * x)


# =========================================================
# Standardizer
# =========================================================
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

    def save(self) -> Dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


# =========================================================
# Dataset
# =========================================================
class NPZRootDataset1D(Dataset):
    def __init__(
        self,
        npz_path: str,
        use_log_ab: bool = False,
        include_center: bool = True,
        include_ab: bool = True,
        target_key: str = "root",
    ):
        data = np.load(npz_path)

        coeffs = sanitize_array(data["coeffs"].astype(np.float32))
        center = sanitize_array(data["center"].astype(np.float32)).reshape(-1, 1)
        a = sanitize_array(data["a"].astype(np.float32)).reshape(-1, 1)
        b = sanitize_array(data["b"].astype(np.float32)).reshape(-1, 1)
        y = sanitize_array(data[target_key].astype(np.float32)).reshape(-1, 1)

        # sequence branch: coeffs only
        self.seq_x = coeffs[..., None].astype(np.float32)  # (N, L, 1)

        feats = []
        if include_center:
            feats.append(center)
        if include_ab:
            if use_log_ab:
                feats.append(np.log(np.clip(a, 1e-12, None)))
                feats.append(np.log(np.clip(b, 1e-12, None)))
            else:
                feats.append(a)
                feats.append(b)

        self.glob_x = np.concatenate(feats, axis=1).astype(np.float32) if feats else np.zeros((coeffs.shape[0], 0), dtype=np.float32)
        self.y = y.astype(np.float32)
        self.a = a.astype(np.float32)
        self.b = b.astype(np.float32)

        self.seq_x = torch.from_numpy(self.seq_x)
        self.glob_x = torch.from_numpy(self.glob_x)
        self.y = torch.from_numpy(self.y)
        self.a = torch.from_numpy(self.a)
        self.b = torch.from_numpy(self.b)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.seq_x[idx], self.glob_x[idx], self.y[idx], self.a[idx], self.b[idx]


def standardize_datasets(train_ds, val_ds, test_ds):
    seq_scaler = Standardizer()
    glob_scaler = Standardizer()

    train_seq_flat = train_ds.seq_x.numpy().reshape(-1, train_ds.seq_x.shape[-1])
    seq_scaler.fit(train_seq_flat)

    train_ds.seq_x = torch.from_numpy(seq_scaler.transform(train_ds.seq_x.numpy().reshape(-1, train_ds.seq_x.shape[-1])).reshape(train_ds.seq_x.shape))
    val_ds.seq_x = torch.from_numpy(seq_scaler.transform(val_ds.seq_x.numpy().reshape(-1, val_ds.seq_x.shape[-1])).reshape(val_ds.seq_x.shape))
    test_ds.seq_x = torch.from_numpy(seq_scaler.transform(test_ds.seq_x.numpy().reshape(-1, test_ds.seq_x.shape[-1])).reshape(test_ds.seq_x.shape))

    if train_ds.glob_x.shape[1] > 0:
        glob_scaler.fit(train_ds.glob_x.numpy())
        train_ds.glob_x = torch.from_numpy(glob_scaler.transform(train_ds.glob_x.numpy()))
        val_ds.glob_x = torch.from_numpy(glob_scaler.transform(val_ds.glob_x.numpy()))
        test_ds.glob_x = torch.from_numpy(glob_scaler.transform(test_ds.glob_x.numpy()))
    else:
        glob_scaler.mean = np.zeros((1, 0), dtype=np.float64)
        glob_scaler.std = np.ones((1, 0), dtype=np.float64)

    return seq_scaler, glob_scaler


# =========================================================
# Models
# =========================================================
class MLPRegressor(nn.Module):
    def __init__(self, seq_dim: int, seq_len: int, glob_dim: int, hidden_dims: List[int], dropout: float = 0.0):
        super().__init__()
        in_dim = seq_dim * seq_len + glob_dim
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        return self.net(torch.cat([seq_x.flatten(1), glob_x], dim=1))


class LSTMRegressor(nn.Module):
    def __init__(self, seq_dim: int, glob_dim: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1, head_hidden: int = 128, head_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(seq_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = self._build_head(hidden_size + glob_dim, head_hidden, 1, dropout, head_layers)

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
            cur = max(cur // 2, 16)
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        _, (hn, _) = self.lstm(seq_x)
        return self.head(torch.cat([hn[-1], glob_x], dim=1))


class GRURegressor(nn.Module):
    def __init__(self, seq_dim: int, glob_dim: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1, head_hidden: int = 128, head_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(seq_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = self._build_head(hidden_size + glob_dim, head_hidden, 1, dropout, head_layers)

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
            cur = max(cur // 2, 16)
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        _, hn = self.gru(seq_x)
        return self.head(torch.cat([hn[-1], glob_x], dim=1))


class TransformerRegressor(nn.Module):
    def __init__(self, seq_dim: int, seq_len: int, glob_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1, ff_dim: int = 128, head_hidden: int = 128, head_layers: int = 2, use_cls_token: bool = True):
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
        self.head = self._build_head(d_model + glob_dim, head_hidden, 1, dropout, head_layers)

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
            cur = max(cur // 2, 16)
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        bsz = seq_x.size(0)
        x = self.input_proj(seq_x)
        if self.use_cls_token:
            x = torch.cat([self.cls_token.expand(bsz, -1, -1), x], dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        h = self.norm(self.encoder(x))
        pooled = h[:, 0, :] if self.use_cls_token else h.mean(dim=1)
        return self.head(torch.cat([pooled, glob_x], dim=1))


def build_model(model_name: str, seq_dim: int, seq_len: int, glob_dim: int, hp: Dict):
    if model_name == "mlp":
        return MLPRegressor(seq_dim, seq_len, glob_dim, hp["hidden_dims"], hp["dropout"])
    if model_name == "lstm":
        return LSTMRegressor(seq_dim, glob_dim, hp["hidden_size"], hp["num_layers"], hp["dropout"], hp["head_hidden"], hp["head_layers"])
    if model_name == "gru":
        return GRURegressor(seq_dim, glob_dim, hp["hidden_size"], hp["num_layers"], hp["dropout"], hp["head_hidden"], hp["head_layers"])
    if model_name == "transformer":
        return TransformerRegressor(seq_dim, seq_len, glob_dim, hp["d_model"], hp["nhead"], hp["num_layers"], hp["dropout"], hp["ff_dim"], hp["head_hidden"], hp["head_layers"], hp["use_cls_token"])
    raise ValueError(model_name)


# =========================================================
# Train / Eval
# =========================================================
def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    count = 0
    ys, preds, as_, bs_ = [], [], [], []

    for seq_x, glob_x, y, a, b in loader:
        seq_x = seq_x.to(device)
        glob_x = glob_x.to(device)
        y = y.to(device)
        a = a.to(device)
        b = b.to(device)

        with torch.set_grad_enabled(train):
            pred = model(seq_x, glob_x)
            loss = criterion(pred, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        total_loss += loss.item() * seq_x.size(0)
        count += seq_x.size(0)

        ys.append(y.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())
        as_.append(a.detach().cpu().numpy())
        bs_.append(b.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0).reshape(-1)
    y_pred = np.concatenate(preds, axis=0).reshape(-1)
    a_np = np.concatenate(as_, axis=0).reshape(-1)
    b_np = np.concatenate(bs_, axis=0).reshape(-1)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(r2_score_np(y_true, y_pred))

    inside = a_np + b_np * y_pred
    valid_mask = inside > 0
    if np.any(valid_mask):
        residual = np.abs(colebrook_like_F_np(y_pred[valid_mask], a_np[valid_mask], b_np[valid_mask]))
        residual_mean = float(np.mean(residual))
        residual_median = float(np.median(residual))
        residual_p90 = float(np.percentile(residual, 90))
        valid_ratio = float(np.mean(valid_mask))
    else:
        residual_mean = float("inf")
        residual_median = float("inf")
        residual_p90 = float("inf")
        valid_ratio = 0.0

    return {
        "loss": total_loss / max(count, 1),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "residual_mean": residual_mean,
        "residual_median": residual_median,
        "residual_p90": residual_p90,
        "valid_ratio": valid_ratio,
    }


def build_optimizer(model, name, lr, weight_decay):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(name)


def newton_refine_batch(x0, a, b, tol=1e-12, max_iter=20):
    x = x0.astype(np.float64).copy()
    iters = np.zeros_like(x, dtype=np.int32)
    converged = np.zeros_like(x, dtype=bool)

    for i in range(len(x)):
        xi = float(x[i]); ai = float(a[i]); bi = float(b[i])
        for it in range(1, max_iter + 1):
            inside = ai + bi * xi
            if inside <= 0:
                break
            fx = xi + 2.0 * np.log10(inside)
            dfx = 1.0 + 2.0 * bi / (np.log(10.0) * inside)
            if not np.isfinite(fx) or not np.isfinite(dfx) or abs(dfx) < 1e-15:
                break
            step = np.clip(fx / dfx, -5.0, 5.0)
            x_new = xi - step
            if ai + bi * x_new <= 0:
                x_half = xi - 0.5 * step
                if ai + bi * x_half > 0:
                    x_new = x_half
                else:
                    break
            xi = x_new
            iters[i] = it
            inside_new = ai + bi * xi
            if inside_new > 0:
                f_new = xi + 2.0 * np.log10(inside_new)
                if abs(f_new) <= tol:
                    converged[i] = True
                    break
        x[i] = xi
    return x, iters, converged


def summarize_method(name, pred, true, a, b, iters=None, conv=None):
    pred = pred.reshape(-1)
    true = true.reshape(-1)

    mae = float(np.mean(np.abs(true - pred)))
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    r2 = float(r2_score_np(true, pred))
    max_abs_error = float(np.max(np.abs(true - pred)))

    inside = a + b * pred
    valid_mask = inside > 0
    if np.any(valid_mask):
        residual = np.abs(colebrook_like_F_np(pred[valid_mask], a[valid_mask], b[valid_mask]))
        residual_mean = float(np.mean(residual))
        residual_median = float(np.median(residual))
        residual_p90 = float(np.percentile(residual, 90))
        valid_ratio = float(np.mean(valid_mask))
    else:
        residual_mean = float("inf")
        residual_median = float("inf")
        residual_p90 = float("inf")
        valid_ratio = 0.0

    row = {
        "name": name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "valid_ratio": valid_ratio,
        "residual_mean": residual_mean,
        "residual_median": residual_median,
        "residual_p90": residual_p90,
        "max_abs_error": max_abs_error,
    }

    if iters is not None:
        row["newton_iter_mean"] = float(np.mean(iters))
        row["newton_iter_median"] = float(np.median(iters))
        row["newton_iter_p90"] = float(np.percentile(iters, 90))

    if conv is not None:
        row["newton_converged_ratio"] = float(np.mean(conv))

    return row


def grid_product(grid: Dict[str, list]):
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in __import__("itertools").product(*vals):
        yield {k: v for k, v in zip(keys, combo)}


def build_search_space(selected_models):
    """
    Grid search를 돌지 않고, 이미 찾은 best trial config만 학습한다.

    사용되는 best config:

    MLP:
      trial_056_mlp
      direct_rmse = 0.0018606808735057712
      plus_newton_iter_mean = 1.9948

    LSTM:
      trial_319_lstm
      direct_rmse = 0.12030239403247833
      plus_newton_iter_mean = 2.6016

    GRU:
      trial_108_gru
      direct_rmse = 0.012654785998165607
      plus_newton_iter_mean = 2.1182

    Transformer:
      trial_537_transformer
      direct_rmse = 0.03279717639088631
      plus_newton_iter_mean = 2.2996

    주의:
    - 제공된 CSV에 hp_use_log_ab 컬럼이 없으므로 use_log_ab=False로 둔다.
    - 원본 trial json에서 hp_use_log_ab=True였으면 해당 모델 config의 use_log_ab만 True로 바꾸면 된다.
    """

    selected_models = [m.lower() for m in selected_models]
    configs = []

    best_configs = {
        "mlp": {
            "model": "mlp",
            "use_log_ab": False,
            "optimizer": "adamw",
            "criterion": "mse",
            "dropout": 0.0,
            "lr": 5e-4,
            "weight_decay": 1e-5,
            "hidden_dims": [128, 128],

            # unused for MLP, but required by downstream saving code
            "hidden_size": 128,
            "num_layers": 2,
            "head_hidden": 128,
            "head_layers": 2,
            "d_model": 64,
            "nhead": 4,
            "ff_dim": 128,
            "use_cls_token": False,

            "source_trial": "trial_056_mlp",
        },

        "lstm": {
            "model": "lstm",
            "use_log_ab": False,
            "optimizer": "adamw",
            "criterion": "mse",
            "dropout": 0.0,
            "lr": 5e-4,
            "weight_decay": 1e-4,
            "hidden_dims": [256, 256, 128],

            "hidden_size": 128,
            "num_layers": 2,
            "head_hidden": 128,
            "head_layers": 1,
            "d_model": 64,
            "nhead": 4,
            "ff_dim": 128,
            "use_cls_token": False,

            "source_trial": "trial_319_lstm",
        },

        "gru": {
            "model": "gru",
            "use_log_ab": False,
            "optimizer": "adamw",
            "criterion": "mse",
            "dropout": 0.1,
            "lr": 5e-4,
            "weight_decay": 1e-5,
            "hidden_dims": [256, 256, 128],

            "hidden_size": 128,
            "num_layers": 1,
            "head_hidden": 128,
            "head_layers": 2,
            "d_model": 64,
            "nhead": 4,
            "ff_dim": 128,
            "use_cls_token": False,

            "source_trial": "trial_108_gru",
        },

        "transformer": {
            "model": "transformer",
            "use_log_ab": False,
            "optimizer": "adamw",
            "criterion": "mse",
            "dropout": 0.1,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "hidden_dims": [256, 256, 128],

            "hidden_size": 128,
            "num_layers": 1,
            "head_hidden": 64,
            "head_layers": 2,
            "d_model": 64,
            "nhead": 2,
            "ff_dim": 64,
            "use_cls_token": False,

            "source_trial": "trial_537_transformer",
        },
    }

    for model_name in ["mlp", "lstm", "gru", "transformer"]:
        if model_name in selected_models:
            configs.append(best_configs[model_name])

    if not configs:
        raise ValueError(
            f"No valid selected models. Got selected_models={selected_models}. "
            "Valid models: mlp, lstm, gru, transformer"
        )

    return configs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_npz", type=str, required=True)
    parser.add_argument("--val_npz", type=str, required=True)
    parser.add_argument("--test_npz", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--models", nargs="+", default=["mlp", "lstm", "gru", "transformer"])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--include_center", action="store_true", default=True)
    parser.add_argument("--include_ab", action="store_true", default=True)
    parser.add_argument("--no_include_center", action="store_true")
    parser.add_argument("--no_include_ab", action="store_true")

    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max_newton_iter", type=int, default=20)

    parser.add_argument(
        "--rank_metric",
        type=str,
        default="plus_newton_r2",
        choices=[
            "direct_r2",
            "direct_rmse",
            "direct_mae",
            "plus_newton_r2",
            "plus_newton_rmse",
            "plus_newton_mae",
            "plus_newton_converged_ratio",
        ],
    )

    args = parser.parse_args()

    if args.no_include_center:
        args.include_center = False
    if args.no_include_ab:
        args.include_ab = False

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device(args.device)

    search_space = build_search_space(args.models)

    all_rows = []
    best_row = None
    best_metric = None
    best_ckpt = None

    for trial_id, hp in enumerate(search_space, start=1):
        trial_name = f"trial_{trial_id:03d}_{hp['model']}"
        print(f"\\n========== {trial_name} ==========")
        print(json.dumps(hp, ensure_ascii=False))

        start_time = time.time()

        train_ds = NPZRootDataset1D(args.train_npz, hp["use_log_ab"], args.include_center, args.include_ab)
        val_ds = NPZRootDataset1D(args.val_npz, hp["use_log_ab"], args.include_center, args.include_ab)
        test_ds = NPZRootDataset1D(args.test_npz, hp["use_log_ab"], args.include_center, args.include_ab)
        seq_scaler, glob_scaler = standardize_datasets(train_ds, val_ds, test_ds)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        seq_dim = train_ds.seq_x.shape[2]
        seq_len = train_ds.seq_x.shape[1]
        glob_dim = train_ds.glob_x.shape[1]

        model = build_model(hp["model"], seq_dim, seq_len, glob_dim, hp).to(device)
        optimizer = build_optimizer(model, hp["optimizer"], hp["lr"], hp["weight_decay"])
        criterion = nn.MSELoss() if hp["criterion"] == "mse" else nn.SmoothL1Loss(beta=0.1)

        best_val_rmse = float("inf")
        best_state = None
        best_epoch = -1
        wait = 0

        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
            val_metrics = run_epoch(model, val_loader, optimizer, criterion, device, train=False)

            print(
                f"[{trial_name}] "
                f"epoch={epoch:03d} "
                f"train_rmse={train_metrics['rmse']:.6f} "
                f"val_rmse={val_metrics['rmse']:.6f} "
                f"val_r2={val_metrics['r2']:.6f}"
            )

            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                best_state = deepcopy(model.state_dict())
                best_epoch = epoch
                wait = 0
            else:
                wait += 1
                if wait >= args.patience:
                    break

        if best_state is None:
            continue

        model.load_state_dict(best_state)

        # direct test eval
        direct_metrics = run_epoch(model, test_loader, optimizer=None, criterion=criterion, device=device, train=False)

        # raw predictions for Newton refinement
        ys, preds, as_, bs_ = [], [], [], []
        model.eval()
        with torch.no_grad():
            for seq_x, glob_x, y, a, b in test_loader:
                seq_x = seq_x.to(device)
                glob_x = glob_x.to(device)
                pred = model(seq_x, glob_x)
                ys.append(y.numpy())
                preds.append(pred.cpu().numpy())
                as_.append(a.numpy())
                bs_.append(b.numpy())

        y_true = np.concatenate(ys, axis=0).reshape(-1)
        pred_direct = np.concatenate(preds, axis=0).reshape(-1)
        a = np.concatenate(as_, axis=0).reshape(-1)
        b = np.concatenate(bs_, axis=0).reshape(-1)

        pred_refined, iters, conv = newton_refine_batch(pred_direct, a, b, tol=args.tol, max_iter=args.max_newton_iter)

        plus_row = summarize_method("neural_plus_newton", pred_refined, y_true, a, b, iters, conv)
        direct_row = summarize_method("neural_direct", pred_direct, y_true, a, b)

        elapsed = time.time() - start_time

        row = {
            "trial_id": trial_id,
            "trial_name": trial_name,
            "model": hp["model"],
            "best_epoch": best_epoch,
            "elapsed_sec": elapsed,

            "direct_mae": direct_row["mae"],
            "direct_rmse": direct_row["rmse"],
            "direct_r2": direct_row["r2"],
            "direct_valid_ratio": direct_row["valid_ratio"],
            "direct_residual_mean": direct_row["residual_mean"],
            "direct_residual_median": direct_row["residual_median"],
            "direct_residual_p90": direct_row["residual_p90"],

            "plus_newton_mae": plus_row["mae"],
            "plus_newton_rmse": plus_row["rmse"],
            "plus_newton_r2": plus_row["r2"],
            "plus_newton_valid_ratio": plus_row["valid_ratio"],
            "plus_newton_residual_mean": plus_row["residual_mean"],
            "plus_newton_residual_median": plus_row["residual_median"],
            "plus_newton_residual_p90": plus_row["residual_p90"],
            "plus_newton_newton_iter_mean": plus_row["newton_iter_mean"],
            "plus_newton_newton_iter_median": plus_row["newton_iter_median"],
            "plus_newton_newton_iter_p90": plus_row["newton_iter_p90"],
            "plus_newton_converged_ratio": plus_row["newton_converged_ratio"],

            "hp_use_log_ab": hp["use_log_ab"],
            "hp_optimizer": hp["optimizer"],
            "hp_criterion": hp["criterion"],
            "hp_dropout": hp["dropout"],
            "hp_lr": hp["lr"],
            "hp_weight_decay": hp["weight_decay"],
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

        all_rows.append(row)

        cur_metric = row[args.rank_metric]
        if best_metric is None:
            better = True
        else:
            if args.rank_metric in ["direct_rmse", "direct_mae", "plus_newton_rmse", "plus_newton_mae"]:
                better = cur_metric < best_metric
            else:
                better = cur_metric > best_metric

        if better:
            best_metric = cur_metric
            best_row = dict(row)
            best_ckpt = {
                "model_state_dict": deepcopy(model.state_dict()),
                "seq_scaler": seq_scaler.save(),
                "glob_scaler": glob_scaler.save(),
                "args": {
                    "model": hp["model"],
                    "hidden_dims": hp["hidden_dims"],
                    "hidden_size": hp["hidden_size"],
                    "num_layers": hp["num_layers"],
                    "head_hidden": hp["head_hidden"],
                    "head_layers": hp["head_layers"],
                    "d_model": hp["d_model"],
                    "nhead": hp["nhead"],
                    "ff_dim": hp["ff_dim"],
                    "use_cls_token": hp["use_cls_token"],
                    "dropout": hp["dropout"],
                    "use_log_ab": hp["use_log_ab"],
                    "include_center": args.include_center,
                    "include_ab": args.include_ab,
                },
                "seq_dim": seq_dim,
                "seq_len": seq_len,
                "glob_dim": glob_dim,
                "best_val_rmse": best_val_rmse,
                "best_epoch": best_epoch,
            }

        with open(out_dir / f"{trial_name}.json", "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2)

    if not all_rows:
        raise RuntimeError("No successful trials.")

    reverse = args.rank_metric not in ["direct_rmse", "direct_mae", "plus_newton_rmse", "plus_newton_mae"]
    all_rows_sorted = sorted(all_rows, key=lambda r: r[args.rank_metric], reverse=reverse)

    save_csv(out_dir / "all_trials.csv", all_rows_sorted)

    with open(out_dir / "best_result.json", "w", encoding="utf-8") as f:
        json.dump(best_row, f, ensure_ascii=False, indent=2)

    if best_ckpt is not None:
        torch.save(best_ckpt, out_dir / "best_model_by_grid.pt")

    print("\\n================ FINAL RANKING ================")
    for row in all_rows_sorted[:10]:
        print({
            "trial_id": row["trial_id"],
            "model": row["model"],
            args.rank_metric: row[args.rank_metric],
            "plus_newton_rmse": row["plus_newton_rmse"],
            "plus_newton_r2": row["plus_newton_r2"],
            "plus_newton_converged_ratio": row["plus_newton_converged_ratio"],
        })

    print("\\n[DONE]")
    print(out_dir / "all_trials.csv")
    print(out_dir / "best_result.json")
    print(out_dir / "best_model_by_grid.pt")


if __name__ == "__main__":
    main()
