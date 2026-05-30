#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_correction_effect_1d.py

목적:
- 1D Colebrook-type root prediction 실험에서
  correction success / error reduction / residual reduction /
  Newton basin-proximity proxy 지표를 계산한다.

지원하는 분석:
1) baseline direct error/residual
2) model direct error/residual
3) correction success rate:
   |x_model - root| < |x_base - root|
4) error ratio:
   |x_model-root| / (|x_base-root| + eps)
5) residual ratio:
   |F(x_model)| / (|F(x_base)| + eps)
6) Newton iteration reduction:
   mean_iter_base - mean_iter_model
7) iteration reduction rate:
   1 - mean_iter_model / mean_iter_base
8) FastConv@1, FastConv@2, FastConv@3
9) Newton convergence ratio

주의:
- 모델 재훈련 없음.
- checkpoint의 model_state_dict, scaler, args를 사용해서 test prediction만 수행.
- checkpoint는 기존 train 코드에서 저장한 best_model_by_grid.pt 형식을 가정.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Utilities
# =========================================================
def sanitize_array(x: np.ndarray, clip_value: float = 1e12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)
    x = np.clip(x, -clip_value, clip_value)
    return x


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot <= 0:
        return 0.0

    return float(1.0 - ss_res / ss_tot)


def colebrook_like_F_np(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    F(x; a,b) = x + 2 log10(a + b x)
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    inside = a + b * x
    out = np.full_like(x, np.nan, dtype=np.float64)

    mask = inside > 0
    out[mask] = x[mask] + 2.0 * np.log10(inside[mask])

    return out


def save_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)

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


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
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


def detect_key(data: np.lib.npyio.NpzFile, candidates: List[str]) -> Optional[str]:
    keys = set(data.files)
    for k in candidates:
        if k in keys:
            return k
    return None


# =========================================================
# Scaler
# =========================================================
class SavedStandardizer:
    def __init__(self, saved: Dict):
        self.mean = np.asarray(saved["mean"], dtype=np.float64)
        self.std = np.asarray(saved["std"], dtype=np.float64)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = sanitize_array(X)
        Xs = (X - self.mean) / self.std
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        Xs = np.clip(Xs, -1e6, 1e6)
        return Xs.astype(np.float32)


# =========================================================
# Dataset
# =========================================================
class NPZRootDataset1D(Dataset):
    def __init__(
        self,
        npz_path: str,
        seq_scaler: SavedStandardizer,
        glob_scaler: SavedStandardizer,
        use_log_ab: bool = False,
        include_center: bool = True,
        include_ab: bool = True,
        target_key: Optional[str] = None,
    ):
        data = np.load(npz_path)

        if target_key is None:
            target_key = detect_key(data, ["root", "y", "target", "x_star", "x_true", "solution"])

        if target_key is None:
            raise KeyError("target key를 찾지 못했습니다. root/y/target/x_star/x_true/solution 중 하나가 필요합니다.")

        coeffs = sanitize_array(data["coeffs"].astype(np.float32))
        center = sanitize_array(data["center"].astype(np.float32)).reshape(-1, 1)
        a = sanitize_array(data["a"].astype(np.float32)).reshape(-1, 1)
        b = sanitize_array(data["b"].astype(np.float32)).reshape(-1, 1)
        y = sanitize_array(data[target_key].astype(np.float32)).reshape(-1, 1)

        seq_x = coeffs[..., None].astype(np.float32)

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

        if feats:
            glob_x = np.concatenate(feats, axis=1).astype(np.float32)
        else:
            glob_x = np.zeros((coeffs.shape[0], 0), dtype=np.float32)

        # Apply saved scalers
        seq_flat = seq_x.reshape(-1, seq_x.shape[-1])
        seq_x = seq_scaler.transform(seq_flat).reshape(seq_x.shape)

        if glob_x.shape[1] > 0:
            glob_x = glob_scaler.transform(glob_x).astype(np.float32)

        self.seq_x = torch.from_numpy(seq_x.astype(np.float32))
        self.glob_x = torch.from_numpy(glob_x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.a = torch.from_numpy(a.astype(np.float32))
        self.b = torch.from_numpy(b.astype(np.float32))

        self.y_np = y.reshape(-1).astype(np.float64)
        self.a_np = a.reshape(-1).astype(np.float64)
        self.b_np = b.reshape(-1).astype(np.float64)
        self.center_np = center.reshape(-1).astype(np.float64)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.seq_x[idx], self.glob_x[idx], self.y[idx], self.a[idx], self.b[idx]


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
    def __init__(
        self,
        seq_dim: int,
        glob_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        head_hidden: int = 128,
        head_layers: int = 2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            seq_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
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
    def __init__(
        self,
        seq_dim: int,
        glob_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        head_hidden: int = 128,
        head_layers: int = 2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            seq_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
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
    def __init__(
        self,
        seq_dim: int,
        seq_len: int,
        glob_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        ff_dim: int = 128,
        head_hidden: int = 128,
        head_layers: int = 2,
        use_cls_token: bool = True,
    ):
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


def build_model_from_ckpt(ckpt: Dict, device: torch.device):
    args = ckpt["args"]
    model_name = args["model"]
    seq_dim = ckpt["seq_dim"]
    seq_len = ckpt["seq_len"]
    glob_dim = ckpt["glob_dim"]
    dropout = args.get("dropout", 0.0)

    if model_name == "mlp":
        model = MLPRegressor(
            seq_dim=seq_dim,
            seq_len=seq_len,
            glob_dim=glob_dim,
            hidden_dims=args["hidden_dims"],
            dropout=dropout,
        )
    elif model_name == "lstm":
        model = LSTMRegressor(
            seq_dim=seq_dim,
            glob_dim=glob_dim,
            hidden_size=args["hidden_size"],
            num_layers=args["num_layers"],
            dropout=dropout,
            head_hidden=args["head_hidden"],
            head_layers=args["head_layers"],
        )
    elif model_name == "gru":
        model = GRURegressor(
            seq_dim=seq_dim,
            glob_dim=glob_dim,
            hidden_size=args["hidden_size"],
            num_layers=args["num_layers"],
            dropout=dropout,
            head_hidden=args["head_hidden"],
            head_layers=args["head_layers"],
        )
    elif model_name == "transformer":
        model = TransformerRegressor(
            seq_dim=seq_dim,
            seq_len=seq_len,
            glob_dim=glob_dim,
            d_model=args["d_model"],
            nhead=args["nhead"],
            num_layers=args["num_layers"],
            dropout=dropout,
            ff_dim=args["ff_dim"],
            head_hidden=args["head_hidden"],
            head_layers=args["head_layers"],
            use_cls_token=args["use_cls_token"],
        )
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model


# =========================================================
# Baseline
# =========================================================
def clip_initial_x(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=1.0, posinf=upper, neginf=lower)
    x = np.clip(x, lower, upper)
    return x.astype(np.float64)


def make_baseline(
    data: np.lib.npyio.NpzFile,
    method: str = "fixed",
    fixed_x0: float = 1.0,
    baseline_key: Optional[str] = None,
    init_clip_min: float = 0.0,
    init_clip_max: float = 50.0,
    eps: float = 1e-12,
):
    keys = set(data.files)

    a = sanitize_array(data["a"]).reshape(-1)
    b = sanitize_array(data["b"]).reshape(-1)

    if method == "fixed":
        return clip_initial_x(np.full_like(a, fixed_x0, dtype=np.float64), init_clip_min, init_clip_max)

    if method == "one_step":
        inside = a + b * fixed_x0
        inside = np.clip(inside, eps, None)
        x0 = -2.0 * np.log10(inside)
        return clip_initial_x(x0, init_clip_min, init_clip_max)

    if method == "a_over_b":
        denom = np.where(np.abs(b) < eps, eps, b)
        x0 = a / denom
        return clip_initial_x(x0, init_clip_min, init_clip_max)

    if method == "b_over_a":
        denom = np.where(np.abs(a) < eps, eps, a)
        x0 = b / denom
        return clip_initial_x(x0, init_clip_min, init_clip_max)

    if method == "key":
        if baseline_key is None:
            raise ValueError("--baseline_key is required when baseline_method=key")
        if baseline_key == "root":
            raise ValueError("baseline_key='root'는 ground-truth이므로 사용할 수 없습니다.")
        if baseline_key not in keys:
            raise KeyError(f"baseline_key={baseline_key} not found in NPZ.")
        return clip_initial_x(sanitize_array(data[baseline_key]).reshape(-1), init_clip_min, init_clip_max)

    if method == "center":
        if "center" not in keys:
            raise KeyError("center key not found.")
        return clip_initial_x(sanitize_array(data["center"]).reshape(-1), init_clip_min, init_clip_max)

    raise ValueError(f"Unknown baseline method: {method}")


# =========================================================
# Newton
# =========================================================
def newton_refine_batch(x0, a, b, tol=1e-12, max_iter=20, step_clip=5.0):
    x = sanitize_array(x0).astype(np.float64).copy()
    a = sanitize_array(a).astype(np.float64).reshape(-1)
    b = sanitize_array(b).astype(np.float64).reshape(-1)

    iters = np.zeros_like(x, dtype=np.int32)
    converged = np.zeros_like(x, dtype=bool)

    for i in range(len(x)):
        xi = float(x[i])
        ai = float(a[i])
        bi = float(b[i])

        for it in range(1, max_iter + 1):
            inside = ai + bi * xi

            if inside <= 0 or not np.isfinite(inside):
                break

            fx = xi + 2.0 * np.log10(inside)
            dfx = 1.0 + 2.0 * bi / (np.log(10.0) * inside)

            if not np.isfinite(fx) or not np.isfinite(dfx) or abs(dfx) < 1e-15:
                break

            step = np.clip(fx / dfx, -step_clip, step_clip)
            x_new = xi - step

            if ai + bi * x_new <= 0:
                x_half = xi - 0.5 * step
                if ai + bi * x_half > 0:
                    x_new = x_half
                else:
                    break

            xi = float(x_new)
            iters[i] = it

            inside_new = ai + bi * xi
            if inside_new > 0:
                f_new = xi + 2.0 * np.log10(inside_new)
                if np.isfinite(f_new) and abs(f_new) <= tol:
                    converged[i] = True
                    break

        x[i] = xi

    return x, iters, converged


# =========================================================
# Prediction
# =========================================================
def predict_model(model, loader, device):
    preds = []

    model.eval()
    with torch.no_grad():
        for seq_x, glob_x, y, a, b in loader:
            seq_x = seq_x.to(device)
            glob_x = glob_x.to(device)
            pred = model(seq_x, glob_x)
            preds.append(pred.detach().cpu().numpy())

    return np.concatenate(preds, axis=0).reshape(-1).astype(np.float64)


# =========================================================
# Metrics
# =========================================================
def summarize_basic(name: str, pred: np.ndarray, true: np.ndarray, a: np.ndarray, b: np.ndarray):
    pred = sanitize_array(pred).reshape(-1)
    true = sanitize_array(true).reshape(-1)
    a = sanitize_array(a).reshape(-1)
    b = sanitize_array(b).reshape(-1)

    err = np.abs(pred - true)
    f = np.abs(colebrook_like_F_np(pred, a, b))
    valid = np.isfinite(f)

    if np.any(valid):
        f_valid = f[valid]
        residual_mean = float(np.mean(f_valid))
        residual_median = float(np.median(f_valid))
        residual_p90 = float(np.percentile(f_valid, 90))
    else:
        residual_mean = float("inf")
        residual_median = float("inf")
        residual_p90 = float("inf")

    return {
        "name": name,
        "mae": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        "r2": float(r2_score_np(true, pred)),
        "residual_mean": residual_mean,
        "residual_median": residual_median,
        "residual_p90": residual_p90,
        "valid_ratio": float(np.mean(valid)),
    }


def analyze_correction(
    model_name: str,
    baseline_name: str,
    x_base: np.ndarray,
    x_model: np.ndarray,
    true: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    tol: float,
    max_newton_iter: int,
    step_clip: float,
    eps: float = 1e-12,
):
    x_base = sanitize_array(x_base).reshape(-1)
    x_model = sanitize_array(x_model).reshape(-1)
    true = sanitize_array(true).reshape(-1)
    a = sanitize_array(a).reshape(-1)
    b = sanitize_array(b).reshape(-1)

    # Direct error
    e_base = np.abs(x_base - true)
    e_model = np.abs(x_model - true)

    # Direct residual
    r_base = np.abs(colebrook_like_F_np(x_base, a, b))
    r_model = np.abs(colebrook_like_F_np(x_model, a, b))

    # finite masks
    e_ratio = e_model / (e_base + eps)
    r_ratio = r_model / (r_base + eps)

    correction_success = e_model < e_base
    residual_success = r_model < r_base

    # Newton refinement
    x_base_newton, iter_base, conv_base = newton_refine_batch(
        x_base, a, b, tol=tol, max_iter=max_newton_iter, step_clip=step_clip
    )
    x_model_newton, iter_model, conv_model = newton_refine_batch(
        x_model, a, b, tol=tol, max_iter=max_newton_iter, step_clip=step_clip
    )

    mean_iter_base = float(np.mean(iter_base))
    mean_iter_model = float(np.mean(iter_model))

    iter_saving = mean_iter_base - mean_iter_model
    iter_reduction_rate = iter_saving / (mean_iter_base + eps)

    # Fast convergence as empirical basin-proximity proxy
    fast_base_1 = np.mean((iter_base <= 1) & conv_base)
    fast_base_2 = np.mean((iter_base <= 2) & conv_base)
    fast_base_3 = np.mean((iter_base <= 3) & conv_base)

    fast_model_1 = np.mean((iter_model <= 1) & conv_model)
    fast_model_2 = np.mean((iter_model <= 2) & conv_model)
    fast_model_3 = np.mean((iter_model <= 3) & conv_model)

    row = {
        "model": model_name,
        "baseline": baseline_name,

        # Correction success
        "correction_success_rate": float(np.mean(correction_success)),
        "residual_success_rate": float(np.mean(residual_success)),

        # Error ratio
        "mean_error_ratio": float(np.mean(e_ratio)),
        "median_error_ratio": float(np.median(e_ratio)),
        "p90_error_ratio": float(np.percentile(e_ratio, 90)),

        # Residual ratio
        "mean_residual_ratio": float(np.mean(r_ratio[np.isfinite(r_ratio)])),
        "median_residual_ratio": float(np.median(r_ratio[np.isfinite(r_ratio)])),
        "p90_residual_ratio": float(np.percentile(r_ratio[np.isfinite(r_ratio)], 90)),

        # Absolute direct metrics
        "baseline_direct_rmse": float(np.sqrt(np.mean((x_base - true) ** 2))),
        "model_direct_rmse": float(np.sqrt(np.mean((x_model - true) ** 2))),
        "baseline_residual_mean": float(np.mean(r_base[np.isfinite(r_base)])),
        "model_residual_mean": float(np.mean(r_model[np.isfinite(r_model)])),

        # Newton metrics
        "baseline_newton_iter_mean": mean_iter_base,
        "model_newton_iter_mean": mean_iter_model,
        "iter_saving": float(iter_saving),
        "iter_reduction_rate": float(iter_reduction_rate),

        "baseline_iter_median": float(np.median(iter_base)),
        "model_iter_median": float(np.median(iter_model)),
        "baseline_iter_p90": float(np.percentile(iter_base, 90)),
        "model_iter_p90": float(np.percentile(iter_model, 90)),

        "baseline_converged_ratio": float(np.mean(conv_base)),
        "model_converged_ratio": float(np.mean(conv_model)),

        # Basin proxy
        "baseline_fastconv_at1": float(fast_base_1),
        "baseline_fastconv_at2": float(fast_base_2),
        "baseline_fastconv_at3": float(fast_base_3),

        "model_fastconv_at1": float(fast_model_1),
        "model_fastconv_at2": float(fast_model_2),
        "model_fastconv_at3": float(fast_model_3),

        "fastconv_at2_gain": float(fast_model_2 - fast_base_2),
        "fastconv_at3_gain": float(fast_model_3 - fast_base_3),
    }

    detail = {
        "e_base": e_base,
        "e_model": e_model,
        "r_base": r_base,
        "r_model": r_model,
        "e_ratio": e_ratio,
        "r_ratio": r_ratio,
        "correction_success": correction_success,
        "residual_success": residual_success,
        "iter_base": iter_base,
        "iter_model": iter_model,
        "conv_base": conv_base,
        "conv_model": conv_model,
        "x_base_newton": x_base_newton,
        "x_model_newton": x_model_newton,
    }

    return row, detail


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_npz", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--target_key", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument(
        "--baseline_method",
        type=str,
        default="fixed",
        choices=["fixed", "one_step", "a_over_b", "b_over_a", "key", "center"],
    )
    parser.add_argument("--baseline_key", type=str, default=None)
    parser.add_argument("--fixed_x0", type=float, default=1.0)
    parser.add_argument("--init_clip_min", type=float, default=0.0)
    parser.add_argument("--init_clip_max", type=float, default=50.0)

    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max_newton_iter", type=int, default=20)
    parser.add_argument("--step_clip", type=float, default=5.0)

    parser.add_argument("--save_details", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)

    ckpt_args = ckpt["args"]
    model_name = ckpt_args["model"]

    seq_scaler = SavedStandardizer(ckpt["seq_scaler"])
    glob_scaler = SavedStandardizer(ckpt["glob_scaler"])

    include_center = bool(ckpt_args.get("include_center", True))
    include_ab = bool(ckpt_args.get("include_ab", True))
    use_log_ab = bool(ckpt_args.get("use_log_ab", False))

    # Load dataset
    ds = NPZRootDataset1D(
        npz_path=args.test_npz,
        seq_scaler=seq_scaler,
        glob_scaler=glob_scaler,
        use_log_ab=use_log_ab,
        include_center=include_center,
        include_ab=include_ab,
        target_key=args.target_key,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Build and predict
    model = build_model_from_ckpt(ckpt, device)
    x_model = predict_model(model, loader, device)

    # Raw test data for baseline
    raw = np.load(args.test_npz)
    x_base = make_baseline(
        raw,
        method=args.baseline_method,
        fixed_x0=args.fixed_x0,
        baseline_key=args.baseline_key,
        init_clip_min=args.init_clip_min,
        init_clip_max=args.init_clip_max,
    )

    y_true = ds.y_np
    a = ds.a_np
    b = ds.b_np

    baseline_name = args.baseline_method
    if args.baseline_method == "fixed":
        baseline_name = f"fixed_x0_{args.fixed_x0}"
    elif args.baseline_method == "one_step":
        baseline_name = f"one_step_from_{args.fixed_x0}"

    # Basic summaries
    basic_model = summarize_basic(model_name, x_model, y_true, a, b)
    basic_base = summarize_basic(baseline_name, x_base, y_true, a, b)

    # Correction analysis
    row, detail = analyze_correction(
        model_name=model_name,
        baseline_name=baseline_name,
        x_base=x_base,
        x_model=x_model,
        true=y_true,
        a=a,
        b=b,
        tol=args.tol,
        max_newton_iter=args.max_newton_iter,
        step_clip=args.step_clip,
    )

    # Save
    save_json(out_dir / f"basic_model_{model_name}.json", basic_model)
    save_json(out_dir / f"basic_baseline_{baseline_name}.json", basic_base)
    save_json(out_dir / f"correction_effect_{model_name}_vs_{baseline_name}.json", row)
    save_csv(out_dir / f"correction_effect_{model_name}_vs_{baseline_name}.csv", [row])

    # Save predictions and details
    np.savez_compressed(
        out_dir / f"predictions_{model_name}_vs_{baseline_name}.npz",
        y_true=y_true,
        a=a,
        b=b,
        x_base=x_base,
        x_model=x_model,
    )

    if args.save_details:
        np.savez_compressed(
            out_dir / f"correction_details_{model_name}_vs_{baseline_name}.npz",
            **detail,
        )

    # Print
    print("\n================ BASIC BASELINE ================")
    print(json.dumps(json_safe(basic_base), ensure_ascii=False, indent=2))

    print("\n================ BASIC MODEL ================")
    print(json.dumps(json_safe(basic_model), ensure_ascii=False, indent=2))

    print("\n================ CORRECTION EFFECT ================")
    print(json.dumps(json_safe(row), ensure_ascii=False, indent=2))

    print("\n[DONE]")
    print(out_dir / f"correction_effect_{model_name}_vs_{baseline_name}.csv")
    print(out_dir / f"predictions_{model_name}_vs_{baseline_name}.npz")


if __name__ == "__main__":
    main()