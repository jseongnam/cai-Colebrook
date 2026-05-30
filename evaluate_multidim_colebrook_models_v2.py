#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

PI = math.pi
LN10 = math.log(10.0)


# =========================================================
# Utility
# =========================================================
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


def percentile(x, q):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def sanitize_array(x: np.ndarray, clip_value: float = 1e12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)
    x = np.clip(x, -clip_value, clip_value)
    return x


def apply_scaler(X: np.ndarray, scaler: Dict[str, np.ndarray], clip_out: float = 1e6) -> np.ndarray:
    X = sanitize_array(X, clip_value=1e12)
    Xs = (X - scaler["mean"]) / scaler["std"]
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = np.clip(Xs, -clip_out, clip_out)
    return Xs.astype(np.float32)


# =========================================================
# Physics / system
# =========================================================
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


def head_loss(Q, x, L, D, g):
    return 8.0 * L * (Q ** 2) / (g * (PI ** 2) * (D ** 5) * (x ** 2))


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

    Re1 = re_from_Q(Q1, rho, mu, D1)
    Re2 = re_from_Q(Q2, rho, mu, D2)

    rr1 = eps1 / D1
    rr2 = eps2 / D2

    F1 = colebrook_single_x_eq(x1, Re1, rr1)
    F2 = colebrook_single_x_eq(x2, Re2, rr2)
    F3 = head_loss(Q1, x1, L1, D1, g) - head_loss(Q2, x2, L2, D2, g)

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


# =========================================================
# Data
# =========================================================
def load_npz(npz_path: str):
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


def build_inputs(data: Dict[str, np.ndarray], use_log_features: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coeffs = np.asarray(data["coeffs"], dtype=np.float64)
    center = np.asarray(data["center"], dtype=np.float64)
    y = np.asarray(data["target"], dtype=np.float64)

    coeffs = sanitize_array(coeffs, clip_value=1e30)
    center = sanitize_array(center, clip_value=1e12)
    y = sanitize_array(y, clip_value=1e12)

    coeffs = signed_log1p(coeffs)

    center_expand = center[..., None]
    seq_x = np.concatenate([coeffs, center_expand], axis=2)
    seq_x = sanitize_array(seq_x, clip_value=1e12)

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
    globals_raw = [sanitize_array(x, clip_value=1e12) for x in globals_raw]

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

    glob_x = sanitize_array(glob_x, clip_value=1e12)
    return seq_x, glob_x, y


# =========================================================
# Models (v2-compatible)
# =========================================================
class MLPModel(nn.Module):
    def __init__(self, seq_dim: int, seq_len: int, glob_dim: int, hidden_dims=(256, 256, 128), dropout=0.1, out_dim=3):
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
    def __init__(self, seq_dim: int, glob_dim: int, hidden_size=128, num_layers=2, dropout=0.1, out_dim=3, head_hidden=128, head_layers=2):
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
        cur_hidden = hidden_dim
        for _ in range(head_layers - 1):
            layers += [nn.Linear(prev, cur_hidden), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = cur_hidden
            cur_hidden = max(cur_hidden // 2, 32)
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        _, (hn, _) = self.lstm(seq_x)
        h_last = hn[-1]
        return self.head(torch.cat([h_last, glob_x], dim=1))


class GRUModel(nn.Module):
    def __init__(self, seq_dim: int, glob_dim: int, hidden_size=128, num_layers=2, dropout=0.1, out_dim=3, head_hidden=128, head_layers=2):
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
        cur_hidden = hidden_dim
        for _ in range(head_layers - 1):
            layers += [nn.Linear(prev, cur_hidden), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = cur_hidden
            cur_hidden = max(cur_hidden // 2, 32)
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        _, hn = self.gru(seq_x)
        h_last = hn[-1]
        return self.head(torch.cat([h_last, glob_x], dim=1))


class TransformerModel(nn.Module):
    def __init__(self, seq_dim: int, seq_len: int, glob_dim: int,
                 d_model=96, nhead=4, num_layers=2, dropout=0.1,
                 out_dim=3, ff_dim=192, head_hidden=128,
                 head_layers=2, use_cls_token=True):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.input_proj = nn.Linear(seq_dim, d_model)

        total_len = seq_len + (1 if use_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, total_len, d_model))

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None

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
        cur_hidden = hidden_dim
        for _ in range(head_layers - 1):
            layers += [nn.Linear(prev, cur_hidden), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = cur_hidden
            cur_hidden = max(cur_hidden // 2, 32)
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        bsz = seq_x.size(0)
        x = self.input_proj(seq_x)

        if self.use_cls_token:
            cls = self.cls_token.expand(bsz, -1, -1)
            x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed[:, :x.size(1), :]
        h = self.encoder(x)
        h = self.norm(h)

        if self.use_cls_token:
            pooled = h[:, 0, :]
        else:
            pooled = h.mean(dim=1)

        return self.head(torch.cat([pooled, glob_x], dim=1))


def build_model_from_ckpt_args(ckpt_args: Dict, seq_dim: int, seq_len: int, glob_dim: int) -> nn.Module:
    model_name = ckpt_args["model"]

    if model_name == "mlp":
        return MLPModel(
            seq_dim=seq_dim,
            seq_len=seq_len,
            glob_dim=glob_dim,
            hidden_dims=tuple(ckpt_args.get("hidden_dims", [256, 256, 128])),
            dropout=float(ckpt_args.get("dropout", 0.1)),
            out_dim=3,
        )
    elif model_name == "lstm":
        return LSTMModel(
            seq_dim=seq_dim,
            glob_dim=glob_dim,
            hidden_size=int(ckpt_args.get("hidden_size", 128)),
            num_layers=int(ckpt_args.get("num_layers", 2)),
            dropout=float(ckpt_args.get("dropout", 0.1)),
            out_dim=3,
            head_hidden=int(ckpt_args.get("head_hidden", 128)),
            head_layers=int(ckpt_args.get("head_layers", 2)),
        )
    elif model_name == "gru":
        return GRUModel(
            seq_dim=seq_dim,
            glob_dim=glob_dim,
            hidden_size=int(ckpt_args.get("hidden_size", 128)),
            num_layers=int(ckpt_args.get("num_layers", 2)),
            dropout=float(ckpt_args.get("dropout", 0.1)),
            out_dim=3,
            head_hidden=int(ckpt_args.get("head_hidden", 128)),
            head_layers=int(ckpt_args.get("head_layers", 2)),
        )
    elif model_name == "transformer":
        return TransformerModel(
            seq_dim=seq_dim,
            seq_len=seq_len,
            glob_dim=glob_dim,
            d_model=int(ckpt_args.get("d_model", 96)),
            nhead=int(ckpt_args.get("nhead", 4)),
            num_layers=int(ckpt_args.get("num_layers", 2)),
            dropout=float(ckpt_args.get("dropout", 0.1)),
            out_dim=3,
            ff_dim=int(ckpt_args.get("ff_dim", 192)),
            head_hidden=int(ckpt_args.get("head_hidden", 128)),
            head_layers=int(ckpt_args.get("head_layers", 2)),
            use_cls_token=bool(ckpt_args.get("use_cls_token", False)),
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def load_model_checkpoint(ckpt_path: str, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt["args"]

    model = build_model_from_ckpt_args(
        ckpt_args=ckpt_args,
        seq_dim=int(ckpt["seq_dim"]),
        seq_len=int(ckpt["seq_len"]),
        glob_dim=int(ckpt["glob_dim"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    seq_scaler = {
        "mean": np.array(ckpt["seq_scaler"]["mean"], dtype=np.float64).reshape(1, -1),
        "std": np.array(ckpt["seq_scaler"]["std"], dtype=np.float64).reshape(1, -1),
    }
    glob_scaler = {
        "mean": np.array(ckpt["glob_scaler"]["mean"], dtype=np.float64).reshape(1, -1),
        "std": np.array(ckpt["glob_scaler"]["std"], dtype=np.float64).reshape(1, -1),
    }
    return ckpt, model, seq_scaler, glob_scaler


@torch.no_grad()
def predict_model(model, seq_x: np.ndarray, glob_x: np.ndarray, device="cpu", batch_size=4096):
    preds = []
    for i in range(0, len(seq_x), batch_size):
        s = torch.from_numpy(seq_x[i:i + batch_size]).to(device)
        g = torch.from_numpy(glob_x[i:i + batch_size]).to(device)
        preds.append(model(s, g).cpu().numpy())
    return np.concatenate(preds, axis=0)


# =========================================================
# Metrics
# =========================================================
def vector_metrics(pred, true):
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


def baseline_zero_like(data):
    QT = np.asarray(data["Q_total"], dtype=np.float64)
    z = np.zeros((len(QT), 3), dtype=np.float64)
    z[:, 0] = QT / 2.0
    z[:, 1] = 7.0
    z[:, 2] = 7.0
    return z


def baseline_heuristic(data):
    QT = np.asarray(data["Q_total"], dtype=np.float64)
    D1 = np.asarray(data["D1"], dtype=np.float64)
    D2 = np.asarray(data["D2"], dtype=np.float64)
    eps1 = np.asarray(data["eps1"], dtype=np.float64)
    eps2 = np.asarray(data["eps2"], dtype=np.float64)
    rho = np.asarray(data["rho"], dtype=np.float64)
    mu = np.asarray(data["mu"], dtype=np.float64)

    out = np.zeros((len(QT), 3), dtype=np.float64)
    out[:, 0] = QT / 2.0
    for i in range(len(QT)):
        out[i, 1] = solve_x_from_Q(out[i, 0], D1[i], eps1[i], rho[i], mu[i])
        out[i, 2] = solve_x_from_Q(out[i, 0], D2[i], eps2[i], rho[i], mu[i])
    return out


def refine_batch(z_init, data, tol=1e-12, max_iter=20):
    n = len(z_init)
    out = np.zeros_like(z_init, dtype=np.float64)
    iters = np.zeros(n, dtype=np.int32)
    conv = np.zeros(n, dtype=bool)

    for i in range(n):
        params_i = {k: float(np.asarray(data[k])[i]) for k in ["Q_total", "D1", "D2", "eps1", "eps2", "L1", "L2", "rho", "mu", "g"]}
        zf, it, ok = newton_system_single(z_init[i], params_i, tol=tol, max_iter=max_iter)
        out[i] = zf
        iters[i] = it
        conv[i] = ok

    return out, iters, conv


def summarize_method(name, pred, true, data, iters=None, conv=None):
    row = {"name": name}
    row.update(vector_metrics(pred, true))
    row.update(residual_metrics(pred, data))
    if iters is not None:
        row["newton_iter_mean"] = float(np.mean(iters))
        row["newton_iter_median"] = float(np.median(iters))
        row["newton_iter_p90"] = float(np.percentile(iters, 90))
    if conv is not None:
        row["newton_converged_ratio"] = float(np.mean(conv))
    return row


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_npz", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--max_newton_iter", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_npz(args.test_npz)
    y_true = sanitize_array(np.asarray(data["target"], dtype=np.float64), clip_value=1e12)

    ckpt, model, seq_scaler, glob_scaler = load_model_checkpoint(args.model, device=args.device)
    use_log_features = bool(ckpt["args"].get("use_log_features", False))
    model_name = ckpt["args"].get("model", ckpt.get("model_name", "unknown"))

    seq_x, glob_x, _ = build_inputs(data, use_log_features=use_log_features)

    seq_shape = seq_x.shape
    seq_x = apply_scaler(seq_x.reshape(-1, seq_x.shape[-1]), seq_scaler).reshape(seq_shape[0], seq_shape[1], seq_shape[2])
    glob_x = apply_scaler(glob_x, glob_scaler)

    pred_neural = predict_model(model, seq_x, glob_x, device=args.device).astype(np.float64)

    pred_zero = baseline_zero_like(data)
    pred_heur = baseline_heuristic(data)

    rows = []
    rows.append(summarize_method("zero_init_direct", pred_zero, y_true, data))
    z_ref, z_iter, z_conv = refine_batch(pred_zero, data, tol=args.tol, max_iter=args.max_newton_iter)
    rows.append(summarize_method("zero_init_plus_newton", z_ref, y_true, data, z_iter, z_conv))

    rows.append(summarize_method("heuristic_direct", pred_heur, y_true, data))
    h_ref, h_iter, h_conv = refine_batch(pred_heur, data, tol=args.tol, max_iter=args.max_newton_iter)
    rows.append(summarize_method("heuristic_plus_newton", h_ref, y_true, data, h_iter, h_conv))

    rows.append(summarize_method("neural_direct", pred_neural, y_true, data))
    n_ref, n_iter, n_conv = refine_batch(pred_neural, data, tol=args.tol, max_iter=args.max_newton_iter)
    rows.append(summarize_method("neural_plus_newton", n_ref, y_true, data, n_iter, n_conv))

    save_csv(out_dir / "summary_metrics.csv", rows)

    per_rows = []
    for i in range(len(y_true)):
        per_rows.append({
            "index": i,
            "true_Q1": float(y_true[i, 0]),
            "true_x1": float(y_true[i, 1]),
            "true_x2": float(y_true[i, 2]),
            "pred_Q1": float(pred_neural[i, 0]),
            "pred_x1": float(pred_neural[i, 1]),
            "pred_x2": float(pred_neural[i, 2]),
            "ref_Q1": float(n_ref[i, 0]),
            "ref_x1": float(n_ref[i, 1]),
            "ref_x2": float(n_ref[i, 2]),
            "iter": int(n_iter[i]),
            "converged": bool(n_conv[i]),
        })
    save_csv(out_dir / "per_sample_results.csv", per_rows)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "test_npz": args.test_npz,
            "model_ckpt": args.model,
            "model_name": model_name,
            "tol": args.tol,
            "max_newton_iter": args.max_newton_iter,
            "device": args.device,
            "use_log_features": use_log_features,
        }, f, ensure_ascii=False, indent=2)

    print("[DONE] Outputs saved to:", out_dir)
    print("  - summary_metrics.csv")
    print("  - per_sample_results.csv")
    print("  - config.json")
    print("\n=== Summary ===")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

PI = math.pi
LN10 = math.log(10.0)


# =========================================================
# Utility
# =========================================================
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


def percentile(x, q):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def sanitize_array(x: np.ndarray, clip_value: float = 1e12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)
    x = np.clip(x, -clip_value, clip_value)
    return x


def apply_scaler(X: np.ndarray, scaler: Dict[str, np.ndarray], clip_out: float = 1e6) -> np.ndarray:
    X = sanitize_array(X, clip_value=1e12)
    Xs = (X - scaler["mean"]) / scaler["std"]
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = np.clip(Xs, -clip_out, clip_out)
    return Xs.astype(np.float32)


# =========================================================
# Physics / system
# =========================================================
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


def head_loss(Q, x, L, D, g):
    return 8.0 * L * (Q ** 2) / (g * (PI ** 2) * (D ** 5) * (x ** 2))


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

    Re1 = re_from_Q(Q1, rho, mu, D1)
    Re2 = re_from_Q(Q2, rho, mu, D2)

    rr1 = eps1 / D1
    rr2 = eps2 / D2

    F1 = colebrook_single_x_eq(x1, Re1, rr1)
    F2 = colebrook_single_x_eq(x2, Re2, rr2)
    F3 = head_loss(Q1, x1, L1, D1, g) - head_loss(Q2, x2, L2, D2, g)

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


# =========================================================
# Data
# =========================================================
def load_npz(npz_path: str):
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


def build_inputs(data: Dict[str, np.ndarray], use_log_features: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coeffs = np.asarray(data["coeffs"], dtype=np.float64)
    center = np.asarray(data["center"], dtype=np.float64)
    y = np.asarray(data["target"], dtype=np.float64)

    coeffs = sanitize_array(coeffs, clip_value=1e30)
    center = sanitize_array(center, clip_value=1e12)
    y = sanitize_array(y, clip_value=1e12)

    coeffs = signed_log1p(coeffs)

    center_expand = center[..., None]
    seq_x = np.concatenate([coeffs, center_expand], axis=2)
    seq_x = sanitize_array(seq_x, clip_value=1e12)

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
    globals_raw = [sanitize_array(x, clip_value=1e12) for x in globals_raw]

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

    glob_x = sanitize_array(glob_x, clip_value=1e12)
    return seq_x, glob_x, y


# =========================================================
# Models (v2-compatible)
# =========================================================
class MLPModel(nn.Module):
    def __init__(self, seq_dim: int, seq_len: int, glob_dim: int, hidden_dims=(256, 256, 128), dropout=0.1, out_dim=3):
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
    def __init__(self, seq_dim: int, glob_dim: int, hidden_size=128, num_layers=2, dropout=0.1, out_dim=3, head_hidden=128, head_layers=2):
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
        cur_hidden = hidden_dim
        for _ in range(head_layers - 1):
            layers += [nn.Linear(prev, cur_hidden), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = cur_hidden
            cur_hidden = max(cur_hidden // 2, 32)
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        _, (hn, _) = self.lstm(seq_x)
        h_last = hn[-1]
        return self.head(torch.cat([h_last, glob_x], dim=1))


class GRUModel(nn.Module):
    def __init__(self, seq_dim: int, glob_dim: int, hidden_size=128, num_layers=2, dropout=0.1, out_dim=3, head_hidden=128, head_layers=2):
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
        cur_hidden = hidden_dim
        for _ in range(head_layers - 1):
            layers += [nn.Linear(prev, cur_hidden), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = cur_hidden
            cur_hidden = max(cur_hidden // 2, 32)
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        _, hn = self.gru(seq_x)
        h_last = hn[-1]
        return self.head(torch.cat([h_last, glob_x], dim=1))


class TransformerModel(nn.Module):
    def __init__(self, seq_dim: int, seq_len: int, glob_dim: int,
                 d_model=96, nhead=4, num_layers=2, dropout=0.1,
                 out_dim=3, ff_dim=192, head_hidden=128,
                 head_layers=2, use_cls_token=True):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.input_proj = nn.Linear(seq_dim, d_model)

        total_len = seq_len + (1 if use_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, total_len, d_model))

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None

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
        cur_hidden = hidden_dim
        for _ in range(head_layers - 1):
            layers += [nn.Linear(prev, cur_hidden), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = cur_hidden
            cur_hidden = max(cur_hidden // 2, 32)
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        bsz = seq_x.size(0)
        x = self.input_proj(seq_x)

        if self.use_cls_token:
            cls = self.cls_token.expand(bsz, -1, -1)
            x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed[:, :x.size(1), :]
        h = self.encoder(x)
        h = self.norm(h)

        if self.use_cls_token:
            pooled = h[:, 0, :]
        else:
            pooled = h.mean(dim=1)

        return self.head(torch.cat([pooled, glob_x], dim=1))


def build_model_from_ckpt_args(ckpt_args: Dict, seq_dim: int, seq_len: int, glob_dim: int) -> nn.Module:
    model_name = ckpt_args["model"]

    if model_name == "mlp":
        return MLPModel(
            seq_dim=seq_dim,
            seq_len=seq_len,
            glob_dim=glob_dim,
            hidden_dims=tuple(ckpt_args.get("hidden_dims", [256, 256, 128])),
            dropout=float(ckpt_args.get("dropout", 0.1)),
            out_dim=3,
        )
    elif model_name == "lstm":
        return LSTMModel(
            seq_dim=seq_dim,
            glob_dim=glob_dim,
            hidden_size=int(ckpt_args.get("hidden_size", 128)),
            num_layers=int(ckpt_args.get("num_layers", 2)),
            dropout=float(ckpt_args.get("dropout", 0.1)),
            out_dim=3,
            head_hidden=int(ckpt_args.get("head_hidden", 128)),
            head_layers=int(ckpt_args.get("head_layers", 2)),
        )
    elif model_name == "gru":
        return GRUModel(
            seq_dim=seq_dim,
            glob_dim=glob_dim,
            hidden_size=int(ckpt_args.get("hidden_size", 128)),
            num_layers=int(ckpt_args.get("num_layers", 2)),
            dropout=float(ckpt_args.get("dropout", 0.1)),
            out_dim=3,
            head_hidden=int(ckpt_args.get("head_hidden", 128)),
            head_layers=int(ckpt_args.get("head_layers", 2)),
        )
    elif model_name == "transformer":
        return TransformerModel(
            seq_dim=seq_dim,
            seq_len=seq_len,
            glob_dim=glob_dim,
            d_model=int(ckpt_args.get("d_model", 96)),
            nhead=int(ckpt_args.get("nhead", 4)),
            num_layers=int(ckpt_args.get("num_layers", 2)),
            dropout=float(ckpt_args.get("dropout", 0.1)),
            out_dim=3,
            ff_dim=int(ckpt_args.get("ff_dim", 192)),
            head_hidden=int(ckpt_args.get("head_hidden", 128)),
            head_layers=int(ckpt_args.get("head_layers", 2)),
            use_cls_token=bool(ckpt_args.get("use_cls_token", False)),
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def load_model_checkpoint(ckpt_path: str, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt["args"]

    model = build_model_from_ckpt_args(
        ckpt_args=ckpt_args,
        seq_dim=int(ckpt["seq_dim"]),
        seq_len=int(ckpt["seq_len"]),
        glob_dim=int(ckpt["glob_dim"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    seq_scaler = {
        "mean": np.array(ckpt["seq_scaler"]["mean"], dtype=np.float64).reshape(1, -1),
        "std": np.array(ckpt["seq_scaler"]["std"], dtype=np.float64).reshape(1, -1),
    }
    glob_scaler = {
        "mean": np.array(ckpt["glob_scaler"]["mean"], dtype=np.float64).reshape(1, -1),
        "std": np.array(ckpt["glob_scaler"]["std"], dtype=np.float64).reshape(1, -1),
    }
    return ckpt, model, seq_scaler, glob_scaler


@torch.no_grad()
def predict_model(model, seq_x: np.ndarray, glob_x: np.ndarray, device="cpu", batch_size=4096):
    preds = []
    for i in range(0, len(seq_x), batch_size):
        s = torch.from_numpy(seq_x[i:i + batch_size]).to(device)
        g = torch.from_numpy(glob_x[i:i + batch_size]).to(device)
        preds.append(model(s, g).cpu().numpy())
    return np.concatenate(preds, axis=0)


# =========================================================
# Metrics
# =========================================================
def vector_metrics(pred, true):
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


def baseline_zero_like(data):
    QT = np.asarray(data["Q_total"], dtype=np.float64)
    z = np.zeros((len(QT), 3), dtype=np.float64)
    z[:, 0] = QT / 2.0
    z[:, 1] = 7.0
    z[:, 2] = 7.0
    return z


def baseline_heuristic(data):
    QT = np.asarray(data["Q_total"], dtype=np.float64)
    D1 = np.asarray(data["D1"], dtype=np.float64)
    D2 = np.asarray(data["D2"], dtype=np.float64)
    eps1 = np.asarray(data["eps1"], dtype=np.float64)
    eps2 = np.asarray(data["eps2"], dtype=np.float64)
    rho = np.asarray(data["rho"], dtype=np.float64)
    mu = np.asarray(data["mu"], dtype=np.float64)

    out = np.zeros((len(QT), 3), dtype=np.float64)
    out[:, 0] = QT / 2.0
    for i in range(len(QT)):
        out[i, 1] = solve_x_from_Q(out[i, 0], D1[i], eps1[i], rho[i], mu[i])
        out[i, 2] = solve_x_from_Q(out[i, 0], D2[i], eps2[i], rho[i], mu[i])
    return out


def refine_batch(z_init, data, tol=1e-12, max_iter=20):
    n = len(z_init)
    out = np.zeros_like(z_init, dtype=np.float64)
    iters = np.zeros(n, dtype=np.int32)
    conv = np.zeros(n, dtype=bool)

    for i in range(n):
        params_i = {k: float(np.asarray(data[k])[i]) for k in ["Q_total", "D1", "D2", "eps1", "eps2", "L1", "L2", "rho", "mu", "g"]}
        zf, it, ok = newton_system_single(z_init[i], params_i, tol=tol, max_iter=max_iter)
        out[i] = zf
        iters[i] = it
        conv[i] = ok

    return out, iters, conv


def summarize_method(name, pred, true, data, iters=None, conv=None):
    row = {"name": name}
    row.update(vector_metrics(pred, true))
    row.update(residual_metrics(pred, data))
    if iters is not None:
        row["newton_iter_mean"] = float(np.mean(iters))
        row["newton_iter_median"] = float(np.median(iters))
        row["newton_iter_p90"] = float(np.percentile(iters, 90))
    if conv is not None:
        row["newton_converged_ratio"] = float(np.mean(conv))
    return row


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_npz", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--max_newton_iter", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_npz(args.test_npz)
    y_true = sanitize_array(np.asarray(data["target"], dtype=np.float64), clip_value=1e12)

    ckpt, model, seq_scaler, glob_scaler = load_model_checkpoint(args.model, device=args.device)
    use_log_features = bool(ckpt["args"].get("use_log_features", False))
    model_name = ckpt["args"].get("model", ckpt.get("model_name", "unknown"))

    seq_x, glob_x, _ = build_inputs(data, use_log_features=use_log_features)

    seq_shape = seq_x.shape
    seq_x = apply_scaler(seq_x.reshape(-1, seq_x.shape[-1]), seq_scaler).reshape(seq_shape[0], seq_shape[1], seq_shape[2])
    glob_x = apply_scaler(glob_x, glob_scaler)

    pred_neural = predict_model(model, seq_x, glob_x, device=args.device).astype(np.float64)

    pred_zero = baseline_zero_like(data)
    pred_heur = baseline_heuristic(data)

    rows = []
    rows.append(summarize_method("zero_init_direct", pred_zero, y_true, data))
    z_ref, z_iter, z_conv = refine_batch(pred_zero, data, tol=args.tol, max_iter=args.max_newton_iter)
    rows.append(summarize_method("zero_init_plus_newton", z_ref, y_true, data, z_iter, z_conv))

    rows.append(summarize_method("heuristic_direct", pred_heur, y_true, data))
    h_ref, h_iter, h_conv = refine_batch(pred_heur, data, tol=args.tol, max_iter=args.max_newton_iter)
    rows.append(summarize_method("heuristic_plus_newton", h_ref, y_true, data, h_iter, h_conv))

    rows.append(summarize_method("neural_direct", pred_neural, y_true, data))
    n_ref, n_iter, n_conv = refine_batch(pred_neural, data, tol=args.tol, max_iter=args.max_newton_iter)
    rows.append(summarize_method("neural_plus_newton", n_ref, y_true, data, n_iter, n_conv))

    save_csv(out_dir / "summary_metrics.csv", rows)

    per_rows = []
    for i in range(len(y_true)):
        per_rows.append({
            "index": i,
            "true_Q1": float(y_true[i, 0]),
            "true_x1": float(y_true[i, 1]),
            "true_x2": float(y_true[i, 2]),
            "pred_Q1": float(pred_neural[i, 0]),
            "pred_x1": float(pred_neural[i, 1]),
            "pred_x2": float(pred_neural[i, 2]),
            "ref_Q1": float(n_ref[i, 0]),
            "ref_x1": float(n_ref[i, 1]),
            "ref_x2": float(n_ref[i, 2]),
            "iter": int(n_iter[i]),
            "converged": bool(n_conv[i]),
        })
    save_csv(out_dir / "per_sample_results.csv", per_rows)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "test_npz": args.test_npz,
            "model_ckpt": args.model,
            "model_name": model_name,
            "tol": args.tol,
            "max_newton_iter": args.max_newton_iter,
            "device": args.device,
            "use_log_features": use_log_features,
        }, f, ensure_ascii=False, indent=2)

    print("[DONE] Outputs saved to:", out_dir)
    print("  - summary_metrics.csv")
    print("  - per_sample_results.csv")
    print("  - config.json")
    print("\n=== Summary ===")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()