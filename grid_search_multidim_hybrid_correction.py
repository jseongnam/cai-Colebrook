#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, itertools, json, math, random, time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PI = math.pi
LN10 = math.log(10.0)

# =========================
# Utility
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sanitize_array(x, clip_value=1e12):
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)
    return np.clip(x, -clip_value, clip_value)


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def percentile(x, q):
    x = np.asarray(x, dtype=np.float64)
    return float(np.percentile(x, q)) if x.size else float('nan')


def save_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        X = sanitize_array(X)
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0
        self.mean[~np.isfinite(self.mean)] = 0.0
        self.std[~np.isfinite(self.std)] = 1.0

    def transform(self, X):
        X = sanitize_array(X)
        Xs = (X - self.mean) / self.std
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(Xs, -1e6, 1e6).astype(np.float32)

    def save(self):
        return {'mean': self.mean.tolist(), 'std': self.std.tolist()}


def apply_scaler(X: np.ndarray, scaler: Dict[str, np.ndarray], clip_out: float = 1e6) -> np.ndarray:
    X = sanitize_array(X)
    Xs = (X - scaler['mean']) / scaler['std']
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(Xs, -clip_out, clip_out).astype(np.float32)


# =========================
# Physics
# =========================
def re_from_Q(Q, rho, mu, D):
    return 4.0 * rho * Q / (PI * mu * D)


def colebrook_single_x_eq(x, Re, rel_rough):
    x = np.asarray(x, dtype=np.float64)
    Re = np.asarray(Re, dtype=np.float64)
    rel_rough = np.asarray(rel_rough, dtype=np.float64)
    z = rel_rough / 3.7 + 2.51 * x / Re
    out = np.full_like(x, np.nan, dtype=np.float64)
    m = (Re > 0) & (z > 0)
    out[m] = x[m] + 2.0 * np.log10(z[m])
    return out


def colebrook_single_x_df(x, Re, rel_rough):
    x = np.asarray(x, dtype=np.float64)
    Re = np.asarray(Re, dtype=np.float64)
    rel_rough = np.asarray(rel_rough, dtype=np.float64)
    z = rel_rough / 3.7 + 2.51 * x / Re
    out = np.full_like(x, np.nan, dtype=np.float64)
    m = (Re > 0) & (z > 0)
    out[m] = 1.0 + 2.0 * ((2.51 / Re[m]) / (z[m] * LN10))
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
    QT = params['Q_total']; D1 = params['D1']; D2 = params['D2']
    eps1 = params['eps1']; eps2 = params['eps2']; L1 = params['L1']; L2 = params['L2']
    rho = params['rho']; mu = params['mu']; g = params['g']
    Q2 = QT - Q1
    Re1 = re_from_Q(Q1, rho, mu, D1)
    Re2 = re_from_Q(Q2, rho, mu, D2)
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
        fp = system_F(zp[None, :], p)[0]
        fm = system_F(zm[None, :], p)[0]
        J[:, j] = (fp - fm) / (2.0 * step)
    return J, f0


def project_feasible(z, p):
    z = np.asarray(z, dtype=np.float64).copy()
    QT = float(p['Q_total'])
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
                z_new, f_new = z_half, f_half
        z = z_new
        used_iter = k
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
        p = {k: float(np.asarray(data[k])[i]) for k in ['Q_total','D1','D2','eps1','eps2','L1','L2','rho','mu','g']}
        zf, it, ok = newton_system_single(z_init[i], p, tol=tol, max_iter=max_iter)
        out[i] = zf; iters[i] = it; conv[i] = ok
    return out, iters, conv


# =========================
# Data
# =========================
def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    required = ['coeffs','center','target','Q_total','D1','D2','eps1','eps2','L1','L2','rho','mu','g']
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {npz_path}. Available: {list(data.keys())}")
    return {k: np.asarray(data[k]) for k in required}


def build_inputs_and_baseline(data: Dict[str, np.ndarray], use_log_features: bool = True):
    coeffs = sanitize_array(np.asarray(data['coeffs'], dtype=np.float64), clip_value=1e30)
    center = sanitize_array(np.asarray(data['center'], dtype=np.float64), clip_value=1e12)
    y = sanitize_array(np.asarray(data['target'], dtype=np.float64), clip_value=1e12)
    coeffs = signed_log1p(coeffs)
    seq_x = np.concatenate([coeffs, center[..., None]], axis=2)
    seq_x = sanitize_array(seq_x, 1e12)

    globals_raw = [
        np.asarray(data['Q_total'], dtype=np.float64).reshape(-1,1),
        np.asarray(data['D1'], dtype=np.float64).reshape(-1,1),
        np.asarray(data['D2'], dtype=np.float64).reshape(-1,1),
        np.asarray(data['eps1'], dtype=np.float64).reshape(-1,1),
        np.asarray(data['eps2'], dtype=np.float64).reshape(-1,1),
        np.asarray(data['L1'], dtype=np.float64).reshape(-1,1),
        np.asarray(data['L2'], dtype=np.float64).reshape(-1,1),
        np.asarray(data['rho'], dtype=np.float64).reshape(-1,1),
        np.asarray(data['mu'], dtype=np.float64).reshape(-1,1),
        np.asarray(data['g'], dtype=np.float64).reshape(-1,1),
    ]
    globals_raw = [sanitize_array(g, 1e12) for g in globals_raw]
    if use_log_features:
        globals_proc = []
        for i, arr in enumerate(globals_raw):
            globals_proc.append(np.log(np.clip(arr, 1e-12, None)) if i < 9 else arr)
        glob_x = np.concatenate(globals_proc, axis=1)
    else:
        glob_x = np.concatenate(globals_raw, axis=1)
    glob_x = sanitize_array(glob_x, 1e12)

    QT = np.asarray(data['Q_total'], dtype=np.float64)
    D1 = np.asarray(data['D1'], dtype=np.float64); D2 = np.asarray(data['D2'], dtype=np.float64)
    eps1 = np.asarray(data['eps1'], dtype=np.float64); eps2 = np.asarray(data['eps2'], dtype=np.float64)
    rho = np.asarray(data['rho'], dtype=np.float64); mu = np.asarray(data['mu'], dtype=np.float64)
    n = len(QT)
    z0 = np.zeros((n, 3), dtype=np.float64)
    z0[:, 0] = QT / 2.0
    for i in range(n):
        qh = QT[i] / 2.0
        z0[i, 1] = solve_x_from_Q(qh, D1[i], eps1[i], rho[i], mu[i])
        z0[i, 2] = solve_x_from_Q(qh, D2[i], eps2[i], rho[i], mu[i])
    return seq_x, glob_x, y, z0


class HybridDataset(Dataset):
    def __init__(self, seq_x, glob_x, y, z0, raw_data: Dict[str, np.ndarray]):
        self.seq_x = torch.from_numpy(seq_x.astype(np.float32))
        self.glob_x = torch.from_numpy(glob_x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.z0 = torch.from_numpy(z0.astype(np.float32))
        self.raw = {}
        for k in ['Q_total','D1','D2','eps1','eps2','L1','L2','rho','mu','g']:
            self.raw[k] = torch.from_numpy(np.asarray(raw_data[k]).astype(np.float32).reshape(-1,1))

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        item = {'seq_x': self.seq_x[idx], 'glob_x': self.glob_x[idx], 'y': self.y[idx], 'z0': self.z0[idx]}
        for k, v in self.raw.items():
            item[k] = v[idx]
        return item


def standardize_datasets(train_ds, val_ds, test_ds):
    seq_scaler = Standardizer(); glob_scaler = Standardizer()
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
        glob_scaler.mean = np.zeros((1,0), dtype=np.float64)
        glob_scaler.std = np.ones((1,0), dtype=np.float64)
    return seq_scaler, glob_scaler


# =========================
# Models
# =========================
class MLPBackbone(nn.Module):
    def __init__(self, seq_dim, seq_len, glob_dim, hidden_dims=(256,256,128), dropout=0.1):
        super().__init__()
        in_dim = seq_dim * seq_len + glob_dim
        layers = []; prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0: layers.append(nn.Dropout(dropout))
            prev = h
        self.feat = nn.Sequential(*layers)
        self.out_dim = prev
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
            if dropout > 0: layers.append(nn.Dropout(dropout))
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
        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model)) if use_cls_token else None
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True, activation='gelu', norm_first=True)
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
        pooled = h[:,0,:] if self.use_cls_token else h.mean(dim=1)
        return self.fusion(torch.cat([pooled, glob_x], dim=1))


class HybridCorrectionModel(nn.Module):
    def __init__(self, model_name, seq_dim, seq_len, glob_dim, hp):
        super().__init__()
        if model_name == 'mlp':
            self.backbone = MLPBackbone(seq_dim, seq_len, glob_dim, hidden_dims=tuple(hp['hidden_dims']), dropout=hp['dropout'])
        elif model_name == 'lstm':
            self.backbone = LSTMBackbone(seq_dim, glob_dim, hidden_size=hp['hidden_size'], num_layers=hp['num_layers'], dropout=hp['dropout'], head_hidden=hp['head_hidden'], head_layers=hp['head_layers'])
        elif model_name == 'gru':
            self.backbone = GRUBackbone(seq_dim, glob_dim, hidden_size=hp['hidden_size'], num_layers=hp['num_layers'], dropout=hp['dropout'], head_hidden=hp['head_hidden'], head_layers=hp['head_layers'])
        elif model_name == 'transformer':
            self.backbone = TransformerBackbone(seq_dim, seq_len, glob_dim, d_model=hp['d_model'], nhead=hp['nhead'], num_layers=hp['num_layers'], dropout=hp['dropout'], ff_dim=hp['ff_dim'], head_hidden=hp['head_hidden'], head_layers=hp['head_layers'], use_cls_token=hp['use_cls_token'])
        else:
            raise ValueError(model_name)
        self.delta_head = nn.Linear(self.backbone.out_dim, 3)
        self.dr_scale = float(hp['dr_scale'])
        self.dx_scale = float(hp['dx_scale'])
    def forward(self, seq_x, glob_x, z0, Q_total):
        feat = self.backbone(seq_x, glob_x)
        delta_raw = self.delta_head(feat)
        q0 = z0[:,0]; x10 = z0[:,1]; x20 = z0[:,2]
        r0 = torch.clamp(q0 / Q_total.squeeze(1), 1e-5, 1.0 - 1e-5)
        logit_r0 = torch.log(r0 / (1.0 - r0))
        logit_r = logit_r0 + self.dr_scale * torch.tanh(delta_raw[:,0])
        r = torch.sigmoid(logit_r)
        q1 = r * Q_total.squeeze(1)
        x1 = torch.clamp(x10 + self.dx_scale * torch.tanh(delta_raw[:,1]), min=1e-3)
        x2 = torch.clamp(x20 + self.dx_scale * torch.tanh(delta_raw[:,2]), min=1e-3)
        pred = torch.stack([q1, x1, x2], dim=1)
        delta_vec = torch.stack([q1-q0, x1-x10, x2-x20], dim=1)
        return pred, delta_vec


def build_model_from_hp(hp, seq_dim, seq_len, glob_dim):
    return HybridCorrectionModel(hp['model'], seq_dim, seq_len, glob_dim, hp)


def load_model_checkpoint(ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device)
    hp = ckpt['hp'] if 'hp' in ckpt else ckpt['args']
    model = build_model_from_hp(hp, ckpt['seq_dim'], ckpt['seq_len'], ckpt['glob_dim'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device); model.eval()
    seq_scaler = {'mean': np.array(ckpt['seq_scaler']['mean'], dtype=np.float64), 'std': np.array(ckpt['seq_scaler']['std'], dtype=np.float64)}
    glob_scaler = {'mean': np.array(ckpt['glob_scaler']['mean'], dtype=np.float64), 'std': np.array(ckpt['glob_scaler']['std'], dtype=np.float64)}
    return ckpt, model, seq_scaler, glob_scaler


# =========================
# Loss / Eval
# =========================
def system_F_torch(z, batch):
    Q1 = z[:,0]; x1 = z[:,1]; x2 = z[:,2]
    QT = batch['Q_total'].squeeze(1); D1 = batch['D1'].squeeze(1); D2 = batch['D2'].squeeze(1)
    eps1 = batch['eps1'].squeeze(1); eps2 = batch['eps2'].squeeze(1); L1 = batch['L1'].squeeze(1); L2 = batch['L2'].squeeze(1)
    rho = batch['rho'].squeeze(1); mu = batch['mu'].squeeze(1); g = batch['g'].squeeze(1)
    Q2 = QT - Q1
    Re1 = 4.0 * rho * Q1 / (PI * mu * D1)
    Re2 = 4.0 * rho * Q2 / (PI * mu * D2)
    rr1 = eps1 / D1; rr2 = eps2 / D2
    eps_safe = 1e-12
    z1 = torch.clamp(rr1/3.7 + 2.51 * x1 / torch.clamp(Re1, min=eps_safe), min=eps_safe)
    z2 = torch.clamp(rr2/3.7 + 2.51 * x2 / torch.clamp(Re2, min=eps_safe), min=eps_safe)
    x1 = torch.clamp(x1, min=1e-6); x2 = torch.clamp(x2, min=1e-6)
    F1 = x1 + 2.0 * torch.log10(z1)
    F2 = x2 + 2.0 * torch.log10(z2)
    H1 = 8.0 * L1 * (Q1**2) / (g * (PI**2) * (D1**5) * (x1**2))
    H2 = 8.0 * L2 * (Q2**2) / (g * (PI**2) * (D2**5) * (x2**2))
    F3 = H1 - H2
    return torch.stack([F1, F2, F3], dim=1)


def hybrid_loss(pred, y, delta_vec, batch, hp):
    l_sup = torch.mean(torch.abs(pred - y))
    F = system_F_torch(pred, batch)
    Q_total = batch['Q_total'].squeeze(1); D1 = batch['D1'].squeeze(1); L1 = batch['L1'].squeeze(1); g = batch['g'].squeeze(1)
    q_ref = torch.clamp(Q_total, min=1e-6)
    x_ref = torch.ones_like(q_ref)
    h_ref = 8.0 * L1 * (0.5 * Q_total)**2 / (g * (PI**2) * (D1**5) * (7.0**2))
    h_ref = torch.clamp(torch.abs(h_ref), min=1e-6)
    l_res = torch.mean(torch.abs(F[:,0]) / x_ref + torch.abs(F[:,1]) / x_ref + torch.abs(F[:,2]) / h_ref)
    l_delta = torch.mean(torch.abs(delta_vec[:,0]) / q_ref + 0.5 * torch.abs(delta_vec[:,1]) + 0.5 * torch.abs(delta_vec[:,2]))
    total = hp['lambda_sup'] * l_sup + hp['lambda_res'] * l_res + hp['lambda_delta'] * l_delta
    return total, {'loss_sup': float(l_sup.detach().cpu().item()), 'loss_res': float(l_res.detach().cpu().item()), 'loss_delta': float(l_delta.detach().cpu().item())}


def vector_metrics(pred, true):
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((true - true.mean(axis=0, keepdims=True))**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mae_Q1': float(np.mean(np.abs(err[:,0]))), 'mae_x1': float(np.mean(np.abs(err[:,1]))), 'mae_x2': float(np.mean(np.abs(err[:,2]))), 'max_abs_error': float(np.max(np.abs(err)))}


def residual_metrics(pred, data):
    params = {k: np.asarray(data[k], dtype=np.float64) for k in ['Q_total','D1','D2','eps1','eps2','L1','L2','rho','mu','g']}
    F = system_F(pred.astype(np.float64), params)
    norms_inf = np.max(np.abs(F), axis=1)
    valid = np.all(np.isfinite(F), axis=1)
    return {'valid_ratio': float(np.mean(valid)), 'residual_mean': float(np.nanmean(norms_inf)), 'residual_median': float(np.nanmedian(norms_inf)), 'residual_p90': percentile(norms_inf[np.isfinite(norms_inf)], 90)}


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    preds = []; trues = []; total_loss = 0.0; total_n = 0
    hp = loader.hp
    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device)
        pred, delta_vec = model(batch['seq_x'], batch['glob_x'], batch['z0'], batch['Q_total'])
        loss, _ = hybrid_loss(pred, batch['y'], delta_vec, batch, hp)
        bs = pred.shape[0]
        total_loss += float(loss.detach().cpu().item()) * bs
        total_n += bs
        preds.append(pred.detach().cpu().numpy()); trues.append(batch['y'].detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    m = vector_metrics(pred, true)
    m['loss'] = total_loss / max(total_n, 1)
    return m, pred, true


def heuristic_pred_from_data(data):
    _, _, _, z0 = build_inputs_and_baseline(data, use_log_features=True)
    return z0


# =========================
# Search space
# =========================
def grid_product(grid: Dict[str, list]):
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in itertools.product(*vals):
        yield {k: v for k, v in zip(keys, combo)}


def build_search_space(selected_models):
    configs = []
    common = {
        'use_log_features': [True],
        'optimizer': ['adamw'],
        'dropout': [0.05, 0.1],
        'lr': [5e-4, 1e-3],
        'weight_decay': [1e-4],
        'lambda_sup': [1.0],
        'lambda_res': [0, 0],
        'lambda_delta': [0, 0],
        'dr_scale': [1.0, 2.0],
        'dx_scale': [0.25, 0.5],
        'hidden_dims': [[256,256,128]],
        'hidden_size': [128],
        'num_layers': [2],
        'head_hidden': [128],
        'head_layers': [2],
        'd_model': [96],
        'nhead': [4],
        'ff_dim': [128],
        'use_cls_token': [True],
    }
    if 'mlp' in selected_models:
        g = dict(common); g['model'] = ['mlp']; g['hidden_dims'] = [[256,256,128],[256,128,64]]
        configs.extend(list(grid_product(g)))
    if 'lstm' in selected_models:
        g = dict(common); g['model'] = ['lstm']; g['hidden_size'] = [96,128]; g['num_layers'] = [1,2]; g['head_hidden'] = [64,128]
        configs.extend(list(grid_product(g)))
    if 'gru' in selected_models:
        g = dict(common); g['model'] = ['gru']; g['hidden_size'] = [96,128]; g['num_layers'] = [1,2]; g['head_hidden'] = [64,128]
        configs.extend(list(grid_product(g)))
    if 'transformer' in selected_models:
        g = dict(common); g['model'] = ['transformer']; g['d_model'] = [64,96]; g['num_layers'] = [1,2]; g['ff_dim'] = [128,192]; g['use_cls_token'] = [False,True]
        configs.extend(list(grid_product(g)))
    return configs


# =========================
# Main grid search
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_npz', required=True)
    parser.add_argument('--val_npz', required=True)
    parser.add_argument('--test_npz', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--models', nargs='+', default=['mlp','lstm','gru','transformer'])
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--tol', type=float, default=1e-12)
    parser.add_argument('--max_newton_iter', type=int, default=20)
    parser.add_argument('--rank_metric', default='plus_newton_r2', choices=['direct_r2','direct_rmse','direct_mae','plus_newton_r2','plus_newton_rmse','plus_newton_mae','plus_newton_converged_ratio'])
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_raw = load_npz(args.train_npz)
    val_raw = load_npz(args.val_npz)
    test_raw = load_npz(args.test_npz)

    search_space = build_search_space(args.models)
    all_rows = []
    best_metric = None
    best_row = None
    best_ckpt = None

    for trial_id, hp in enumerate(search_space, start=1):
        trial_name = f"trial_{trial_id:03d}_{hp['model']}"
        print(f"\n========== {trial_name} ==========")
        print(json.dumps(hp, ensure_ascii=False))
        start_t = time.time()

        tr_seq, tr_glob, tr_y, tr_z0 = build_inputs_and_baseline(train_raw, use_log_features=hp['use_log_features'])
        va_seq, va_glob, va_y, va_z0 = build_inputs_and_baseline(val_raw, use_log_features=hp['use_log_features'])
        te_seq, te_glob, te_y, te_z0 = build_inputs_and_baseline(test_raw, use_log_features=hp['use_log_features'])

        train_ds = HybridDataset(tr_seq, tr_glob, tr_y, tr_z0, train_raw)
        val_ds = HybridDataset(va_seq, va_glob, va_y, va_z0, val_raw)
        test_ds = HybridDataset(te_seq, te_glob, te_y, te_z0, test_raw)
        seq_scaler, glob_scaler = standardize_datasets(train_ds, val_ds, test_ds)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        train_loader.hp = hp; val_loader.hp = hp; test_loader.hp = hp

        model = HybridCorrectionModel(hp['model'], train_ds.seq_x.shape[2], train_ds.seq_x.shape[1], train_ds.glob_x.shape[1], hp).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay']) if hp['optimizer'] == 'adamw' else torch.optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])

        best_val_rmse = float('inf'); best_epoch = -1; best_state = None; wait = 0
        for epoch in range(1, args.epochs + 1):
            model.train(); train_loss_sum = 0.0; train_n = 0
            for batch in train_loader:
                for k in batch:
                    batch[k] = batch[k].to(device)
                pred, delta_vec = model(batch['seq_x'], batch['glob_x'], batch['z0'], batch['Q_total'])
                loss, _ = hybrid_loss(pred, batch['y'], delta_vec, batch, hp)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                bs = pred.shape[0]
                train_loss_sum += float(loss.detach().cpu().item()) * bs
                train_n += bs
            val_metrics, _, _ = run_eval(model, val_loader, device)
            print(f"[{trial_name}] epoch={epoch:03d} train_loss={train_loss_sum/max(train_n,1):.6f} val_rmse={val_metrics['rmse']:.6f} val_r2={val_metrics['r2']:.6f}")
            if val_metrics['rmse'] < best_val_rmse:
                best_val_rmse = val_metrics['rmse']
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= args.patience:
                    break

        if best_state is None:
            continue

        model.load_state_dict(best_state)
        direct_metrics, pred_direct, true = run_eval(model, test_loader, device)
        direct_metrics.update(residual_metrics(pred_direct, test_raw))

        pred_ref, pred_iter, pred_conv = refine_batch(pred_direct.astype(np.float64), test_raw, tol=args.tol, max_iter=args.max_newton_iter)
        plus = vector_metrics(pred_ref, true.astype(np.float64))
        plus.update(residual_metrics(pred_ref, test_raw))
        plus['newton_iter_mean'] = float(np.mean(pred_iter))
        plus['newton_iter_median'] = float(np.median(pred_iter))
        plus['newton_iter_p90'] = float(np.percentile(pred_iter, 90))
        plus['newton_converged_ratio'] = float(np.mean(pred_conv))

        elapsed = time.time() - start_t
        row = {
            'trial_id': trial_id, 'trial_name': trial_name, 'model': hp['model'], 'best_epoch': best_epoch, 'elapsed_sec': elapsed,
            'direct_mae': direct_metrics['mae'], 'direct_rmse': direct_metrics['rmse'], 'direct_r2': direct_metrics['r2'],
            'direct_valid_ratio': direct_metrics['valid_ratio'], 'direct_residual_mean': direct_metrics['residual_mean'], 'direct_residual_median': direct_metrics['residual_median'], 'direct_residual_p90': direct_metrics['residual_p90'],
            'plus_newton_mae': plus['mae'], 'plus_newton_rmse': plus['rmse'], 'plus_newton_r2': plus['r2'],
            'plus_newton_valid_ratio': plus['valid_ratio'], 'plus_newton_residual_mean': plus['residual_mean'], 'plus_newton_residual_median': plus['residual_median'], 'plus_newton_residual_p90': plus['residual_p90'],
            'plus_newton_newton_iter_mean': plus['newton_iter_mean'], 'plus_newton_newton_iter_median': plus['newton_iter_median'], 'plus_newton_newton_iter_p90': plus['newton_iter_p90'], 'plus_newton_converged_ratio': plus['newton_converged_ratio'],
            'hp_use_log_features': hp['use_log_features'], 'hp_optimizer': hp['optimizer'], 'hp_dropout': hp['dropout'], 'hp_lr': hp['lr'], 'hp_weight_decay': hp['weight_decay'],
            'hp_lambda_sup': hp['lambda_sup'], 'hp_lambda_res': hp['lambda_res'], 'hp_lambda_delta': hp['lambda_delta'], 'hp_dr_scale': hp['dr_scale'], 'hp_dx_scale': hp['dx_scale'],
            'hp_hidden_dims': json.dumps(hp['hidden_dims']), 'hp_hidden_size': hp['hidden_size'], 'hp_num_layers': hp['num_layers'], 'hp_head_hidden': hp['head_hidden'], 'hp_head_layers': hp['head_layers'],
            'hp_d_model': hp['d_model'], 'hp_nhead': hp['nhead'], 'hp_ff_dim': hp['ff_dim'], 'hp_use_cls_token': hp['use_cls_token'],
        }
        all_rows.append(row)

        cur_metric = row[args.rank_metric]
        if best_metric is None:
            better = True
        else:
            better = cur_metric < best_metric if args.rank_metric in ['direct_rmse','direct_mae','plus_newton_rmse','plus_newton_mae'] else cur_metric > best_metric
        if better:
            best_metric = cur_metric
            best_row = dict(row)
            best_ckpt = {
                'model_state_dict': deepcopy(model.state_dict()), 'seq_scaler': seq_scaler.save(), 'glob_scaler': glob_scaler.save(), 'hp': hp,
                'seq_dim': train_ds.seq_x.shape[2], 'seq_len': train_ds.seq_x.shape[1], 'glob_dim': train_ds.glob_x.shape[1], 'best_val_rmse': best_val_rmse, 'best_epoch': best_epoch,
            }

        with open(out_dir / f"{trial_name}.json", 'w', encoding='utf-8') as f:
            json.dump(row, f, ensure_ascii=False, indent=2)

    if not all_rows:
        raise RuntimeError('No successful trials completed.')

    reverse = args.rank_metric not in ['direct_rmse','direct_mae','plus_newton_rmse','plus_newton_mae']
    all_rows_sorted = sorted(all_rows, key=lambda r: r[args.rank_metric], reverse=reverse)
    save_csv(out_dir / 'all_trials.csv', all_rows_sorted)
    with open(out_dir / 'best_result.json', 'w', encoding='utf-8') as f:
        json.dump(best_row, f, ensure_ascii=False, indent=2)
    if best_ckpt is not None:
        torch.save(best_ckpt, out_dir / 'best_model_by_grid.pt')

    print('\n================ FINAL RANKING ================')
    for row in all_rows_sorted[:10]:
        print({'trial_id': row['trial_id'], 'model': row['model'], args.rank_metric: row[args.rank_metric], 'plus_newton_rmse': row['plus_newton_rmse'], 'plus_newton_r2': row['plus_newton_r2'], 'plus_newton_converged_ratio': row['plus_newton_converged_ratio']})
    print('\n[DONE]')
    print(out_dir / 'all_trials.csv')
    print(out_dir / 'best_result.json')
    print(out_dir / 'best_model_by_grid.pt')


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, itertools, json, math, random, time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PI = math.pi
LN10 = math.log(10.0)

# =========================
# Utility
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sanitize_array(x, clip_value=1e12):
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)
    return np.clip(x, -clip_value, clip_value)


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def percentile(x, q):
    x = np.asarray(x, dtype=np.float64)
    return float(np.percentile(x, q)) if x.size else float('nan')


def save_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        X = sanitize_array(X)
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0
        self.mean[~np.isfinite(self.mean)] = 0.0
        self.std[~np.isfinite(self.std)] = 1.0

    def transform(self, X):
        X = sanitize_array(X)
        Xs = (X - self.mean) / self.std
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(Xs, -1e6, 1e6).astype(np.float32)

    def save(self):
        return {'mean': self.mean.tolist(), 'std': self.std.tolist()}


def apply_scaler(X: np.ndarray, scaler: Dict[str, np.ndarray], clip_out: float = 1e6) -> np.ndarray:
    X = sanitize_array(X)
    Xs = (X - scaler['mean']) / scaler['std']
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(Xs, -clip_out, clip_out).astype(np.float32)


# =========================
# Physics
# =========================
def re_from_Q(Q, rho, mu, D):
    return 4.0 * rho * Q / (PI * mu * D)


def colebrook_single_x_eq(x, Re, rel_rough):
    x = np.asarray(x, dtype=np.float64)
    Re = np.asarray(Re, dtype=np.float64)
    rel_rough = np.asarray(rel_rough, dtype=np.float64)
    z = rel_rough / 3.7 + 2.51 * x / Re
    out = np.full_like(x, np.nan, dtype=np.float64)
    m = (Re > 0) & (z > 0)
    out[m] = x[m] + 2.0 * np.log10(z[m])
    return out


def colebrook_single_x_df(x, Re, rel_rough):
    x = np.asarray(x, dtype=np.float64)
    Re = np.asarray(Re, dtype=np.float64)
    rel_rough = np.asarray(rel_rough, dtype=np.float64)
    z = rel_rough / 3.7 + 2.51 * x / Re
    out = np.full_like(x, np.nan, dtype=np.float64)
    m = (Re > 0) & (z > 0)
    out[m] = 1.0 + 2.0 * ((2.51 / Re[m]) / (z[m] * LN10))
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
    QT = params['Q_total']; D1 = params['D1']; D2 = params['D2']
    eps1 = params['eps1']; eps2 = params['eps2']; L1 = params['L1']; L2 = params['L2']
    rho = params['rho']; mu = params['mu']; g = params['g']
    Q2 = QT - Q1
    Re1 = re_from_Q(Q1, rho, mu, D1)
    Re2 = re_from_Q(Q2, rho, mu, D2)
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
        fp = system_F(zp[None, :], p)[0]
        fm = system_F(zm[None, :], p)[0]
        J[:, j] = (fp - fm) / (2.0 * step)
    return J, f0


def project_feasible(z, p):
    z = np.asarray(z, dtype=np.float64).copy()
    QT = float(p['Q_total'])
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
                z_new, f_new = z_half, f_half
        z = z_new
        used_iter = k
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
        p = {k: float(np.asarray(data[k])[i]) for k in ['Q_total','D1','D2','eps1','eps2','L1','L2','rho','mu','g']}
        zf, it, ok = newton_system_single(z_init[i], p, tol=tol, max_iter=max_iter)
        out[i] = zf; iters[i] = it; conv[i] = ok
    return out, iters, conv


# =========================
# Data
# =========================
def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    required = ['coeffs','center','target','Q_total','D1','D2','eps1','eps2','L1','L2','rho','mu','g']
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {npz_path}. Available: {list(data.keys())}")
    return {k: np.asarray(data[k]) for k in required}


def build_inputs_and_baseline(data: Dict[str, np.ndarray], use_log_features: bool = True):
    coeffs = sanitize_array(np.asarray(data['coeffs'], dtype=np.float64), clip_value=1e30)
    center = sanitize_array(np.asarray(data['center'], dtype=np.float64), clip_value=1e12)
    y = sanitize_array(np.asarray(data['target'], dtype=np.float64), clip_value=1e12)
    coeffs = signed_log1p(coeffs)
    seq_x = np.concatenate([coeffs, center[..., None]], axis=2)
    seq_x = sanitize_array(seq_x, 1e12)

    globals_raw = [
        np.asarray(data['Q_total'], dtype=np.float64).reshape(-1,1),
        np.asarray(data['D1'], dtype=np.float64).reshape(-1,1),
        np.asarray(data['D2'], dtype=np.float64).reshape(-1,1),
        np.asarray(data['eps1'], dtype=np.float64).reshape(-1,1),
        np.asarray(data['eps2'], dtype=np.float64).reshape(-1,1),
        np.asarray(data['L1'], dtype=np.float64).reshape(-1,1),
        np.asarray(data['L2'], dtype=np.float64).reshape(-1,1),
        np.asarray(data['rho'], dtype=np.float64).reshape(-1,1),
        np.asarray(data['mu'], dtype=np.float64).reshape(-1,1),
        np.asarray(data['g'], dtype=np.float64).reshape(-1,1),
    ]
    globals_raw = [sanitize_array(g, 1e12) for g in globals_raw]
    if use_log_features:
        globals_proc = []
        for i, arr in enumerate(globals_raw):
            globals_proc.append(np.log(np.clip(arr, 1e-12, None)) if i < 9 else arr)
        glob_x = np.concatenate(globals_proc, axis=1)
    else:
        glob_x = np.concatenate(globals_raw, axis=1)
    glob_x = sanitize_array(glob_x, 1e12)

    QT = np.asarray(data['Q_total'], dtype=np.float64)
    D1 = np.asarray(data['D1'], dtype=np.float64); D2 = np.asarray(data['D2'], dtype=np.float64)
    eps1 = np.asarray(data['eps1'], dtype=np.float64); eps2 = np.asarray(data['eps2'], dtype=np.float64)
    rho = np.asarray(data['rho'], dtype=np.float64); mu = np.asarray(data['mu'], dtype=np.float64)
    n = len(QT)
    z0 = np.zeros((n, 3), dtype=np.float64)
    z0[:, 0] = QT / 2.0
    for i in range(n):
        qh = QT[i] / 2.0
        z0[i, 1] = solve_x_from_Q(qh, D1[i], eps1[i], rho[i], mu[i])
        z0[i, 2] = solve_x_from_Q(qh, D2[i], eps2[i], rho[i], mu[i])
    return seq_x, glob_x, y, z0


class HybridDataset(Dataset):
    def __init__(self, seq_x, glob_x, y, z0, raw_data: Dict[str, np.ndarray]):
        self.seq_x = torch.from_numpy(seq_x.astype(np.float32))
        self.glob_x = torch.from_numpy(glob_x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.z0 = torch.from_numpy(z0.astype(np.float32))
        self.raw = {}
        for k in ['Q_total','D1','D2','eps1','eps2','L1','L2','rho','mu','g']:
            self.raw[k] = torch.from_numpy(np.asarray(raw_data[k]).astype(np.float32).reshape(-1,1))

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        item = {'seq_x': self.seq_x[idx], 'glob_x': self.glob_x[idx], 'y': self.y[idx], 'z0': self.z0[idx]}
        for k, v in self.raw.items():
            item[k] = v[idx]
        return item


def standardize_datasets(train_ds, val_ds, test_ds):
    seq_scaler = Standardizer(); glob_scaler = Standardizer()
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
        glob_scaler.mean = np.zeros((1,0), dtype=np.float64)
        glob_scaler.std = np.ones((1,0), dtype=np.float64)
    return seq_scaler, glob_scaler


# =========================
# Models
# =========================
class MLPBackbone(nn.Module):
    def __init__(self, seq_dim, seq_len, glob_dim, hidden_dims=(256,256,128), dropout=0.1):
        super().__init__()
        in_dim = seq_dim * seq_len + glob_dim
        layers = []; prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0: layers.append(nn.Dropout(dropout))
            prev = h
        self.feat = nn.Sequential(*layers)
        self.out_dim = prev
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
            if dropout > 0: layers.append(nn.Dropout(dropout))
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
        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model)) if use_cls_token else None
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True, activation='gelu', norm_first=True)
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
        pooled = h[:,0,:] if self.use_cls_token else h.mean(dim=1)
        return self.fusion(torch.cat([pooled, glob_x], dim=1))


class HybridCorrectionModel(nn.Module):
    def __init__(self, model_name, seq_dim, seq_len, glob_dim, hp):
        super().__init__()
        if model_name == 'mlp':
            self.backbone = MLPBackbone(seq_dim, seq_len, glob_dim, hidden_dims=tuple(hp['hidden_dims']), dropout=hp['dropout'])
        elif model_name == 'lstm':
            self.backbone = LSTMBackbone(seq_dim, glob_dim, hidden_size=hp['hidden_size'], num_layers=hp['num_layers'], dropout=hp['dropout'], head_hidden=hp['head_hidden'], head_layers=hp['head_layers'])
        elif model_name == 'gru':
            self.backbone = GRUBackbone(seq_dim, glob_dim, hidden_size=hp['hidden_size'], num_layers=hp['num_layers'], dropout=hp['dropout'], head_hidden=hp['head_hidden'], head_layers=hp['head_layers'])
        elif model_name == 'transformer':
            self.backbone = TransformerBackbone(seq_dim, seq_len, glob_dim, d_model=hp['d_model'], nhead=hp['nhead'], num_layers=hp['num_layers'], dropout=hp['dropout'], ff_dim=hp['ff_dim'], head_hidden=hp['head_hidden'], head_layers=hp['head_layers'], use_cls_token=hp['use_cls_token'])
        else:
            raise ValueError(model_name)
        self.delta_head = nn.Linear(self.backbone.out_dim, 3)
        self.dr_scale = float(hp['dr_scale'])
        self.dx_scale = float(hp['dx_scale'])
    def forward(self, seq_x, glob_x, z0, Q_total):
        feat = self.backbone(seq_x, glob_x)
        delta_raw = self.delta_head(feat)
        q0 = z0[:,0]; x10 = z0[:,1]; x20 = z0[:,2]
        r0 = torch.clamp(q0 / Q_total.squeeze(1), 1e-5, 1.0 - 1e-5)
        logit_r0 = torch.log(r0 / (1.0 - r0))
        logit_r = logit_r0 + self.dr_scale * torch.tanh(delta_raw[:,0])
        r = torch.sigmoid(logit_r)
        q1 = r * Q_total.squeeze(1)
        x1 = torch.clamp(x10 + self.dx_scale * torch.tanh(delta_raw[:,1]), min=1e-3)
        x2 = torch.clamp(x20 + self.dx_scale * torch.tanh(delta_raw[:,2]), min=1e-3)
        pred = torch.stack([q1, x1, x2], dim=1)
        delta_vec = torch.stack([q1-q0, x1-x10, x2-x20], dim=1)
        return pred, delta_vec


def build_model_from_hp(hp, seq_dim, seq_len, glob_dim):
    return HybridCorrectionModel(hp['model'], seq_dim, seq_len, glob_dim, hp)


def load_model_checkpoint(ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device)
    hp = ckpt['hp'] if 'hp' in ckpt else ckpt['args']
    model = build_model_from_hp(hp, ckpt['seq_dim'], ckpt['seq_len'], ckpt['glob_dim'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device); model.eval()
    seq_scaler = {'mean': np.array(ckpt['seq_scaler']['mean'], dtype=np.float64), 'std': np.array(ckpt['seq_scaler']['std'], dtype=np.float64)}
    glob_scaler = {'mean': np.array(ckpt['glob_scaler']['mean'], dtype=np.float64), 'std': np.array(ckpt['glob_scaler']['std'], dtype=np.float64)}
    return ckpt, model, seq_scaler, glob_scaler


# =========================
# Loss / Eval
# =========================
def system_F_torch(z, batch):
    Q1 = z[:,0]; x1 = z[:,1]; x2 = z[:,2]
    QT = batch['Q_total'].squeeze(1); D1 = batch['D1'].squeeze(1); D2 = batch['D2'].squeeze(1)
    eps1 = batch['eps1'].squeeze(1); eps2 = batch['eps2'].squeeze(1); L1 = batch['L1'].squeeze(1); L2 = batch['L2'].squeeze(1)
    rho = batch['rho'].squeeze(1); mu = batch['mu'].squeeze(1); g = batch['g'].squeeze(1)
    Q2 = QT - Q1
    Re1 = 4.0 * rho * Q1 / (PI * mu * D1)
    Re2 = 4.0 * rho * Q2 / (PI * mu * D2)
    rr1 = eps1 / D1; rr2 = eps2 / D2
    eps_safe = 1e-12
    z1 = torch.clamp(rr1/3.7 + 2.51 * x1 / torch.clamp(Re1, min=eps_safe), min=eps_safe)
    z2 = torch.clamp(rr2/3.7 + 2.51 * x2 / torch.clamp(Re2, min=eps_safe), min=eps_safe)
    x1 = torch.clamp(x1, min=1e-6); x2 = torch.clamp(x2, min=1e-6)
    F1 = x1 + 2.0 * torch.log10(z1)
    F2 = x2 + 2.0 * torch.log10(z2)
    H1 = 8.0 * L1 * (Q1**2) / (g * (PI**2) * (D1**5) * (x1**2))
    H2 = 8.0 * L2 * (Q2**2) / (g * (PI**2) * (D2**5) * (x2**2))
    F3 = H1 - H2
    return torch.stack([F1, F2, F3], dim=1)


def hybrid_loss(pred, y, delta_vec, batch, hp):
    l_sup = torch.mean(torch.abs(pred - y))
    F = system_F_torch(pred, batch)
    Q_total = batch['Q_total'].squeeze(1); D1 = batch['D1'].squeeze(1); L1 = batch['L1'].squeeze(1); g = batch['g'].squeeze(1)
    q_ref = torch.clamp(Q_total, min=1e-6)
    x_ref = torch.ones_like(q_ref)
    h_ref = 8.0 * L1 * (0.5 * Q_total)**2 / (g * (PI**2) * (D1**5) * (7.0**2))
    h_ref = torch.clamp(torch.abs(h_ref), min=1e-6)
    l_res = torch.mean(torch.abs(F[:,0]) / x_ref + torch.abs(F[:,1]) / x_ref + torch.abs(F[:,2]) / h_ref)
    l_delta = torch.mean(torch.abs(delta_vec[:,0]) / q_ref + 0.5 * torch.abs(delta_vec[:,1]) + 0.5 * torch.abs(delta_vec[:,2]))
    total = hp['lambda_sup'] * l_sup + hp['lambda_res'] * l_res + hp['lambda_delta'] * l_delta
    return total, {'loss_sup': float(l_sup.detach().cpu().item()), 'loss_res': float(l_res.detach().cpu().item()), 'loss_delta': float(l_delta.detach().cpu().item())}


def vector_metrics(pred, true):
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((true - true.mean(axis=0, keepdims=True))**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mae_Q1': float(np.mean(np.abs(err[:,0]))), 'mae_x1': float(np.mean(np.abs(err[:,1]))), 'mae_x2': float(np.mean(np.abs(err[:,2]))), 'max_abs_error': float(np.max(np.abs(err)))}


def residual_metrics(pred, data):
    params = {k: np.asarray(data[k], dtype=np.float64) for k in ['Q_total','D1','D2','eps1','eps2','L1','L2','rho','mu','g']}
    F = system_F(pred.astype(np.float64), params)
    norms_inf = np.max(np.abs(F), axis=1)
    valid = np.all(np.isfinite(F), axis=1)
    return {'valid_ratio': float(np.mean(valid)), 'residual_mean': float(np.nanmean(norms_inf)), 'residual_median': float(np.nanmedian(norms_inf)), 'residual_p90': percentile(norms_inf[np.isfinite(norms_inf)], 90)}


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    preds = []; trues = []; total_loss = 0.0; total_n = 0
    hp = loader.hp
    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device)
        pred, delta_vec = model(batch['seq_x'], batch['glob_x'], batch['z0'], batch['Q_total'])
        loss, _ = hybrid_loss(pred, batch['y'], delta_vec, batch, hp)
        bs = pred.shape[0]
        total_loss += float(loss.detach().cpu().item()) * bs
        total_n += bs
        preds.append(pred.detach().cpu().numpy()); trues.append(batch['y'].detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    m = vector_metrics(pred, true)
    m['loss'] = total_loss / max(total_n, 1)
    return m, pred, true


def heuristic_pred_from_data(data):
    _, _, _, z0 = build_inputs_and_baseline(data, use_log_features=True)
    return z0


# =========================
# Search space
# =========================
def grid_product(grid: Dict[str, list]):
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in itertools.product(*vals):
        yield {k: v for k, v in zip(keys, combo)}


def build_search_space(selected_models):
    configs = []
    common = {
        'use_log_features': [True],
        'optimizer': ['adamw'],
        'dropout': [0.05, 0.1],
        'lr': [5e-4, 1e-3],
        'weight_decay': [1e-4],
        'lambda_sup': [1.0],
        'lambda_res': [0, 0],
        'lambda_delta': [0, 0],
        'dr_scale': [1.0, 2.0],
        'dx_scale': [0.25, 0.5],
        'hidden_dims': [[256,256,128]],
        'hidden_size': [128],
        'num_layers': [2],
        'head_hidden': [128],
        'head_layers': [2],
        'd_model': [96],
        'nhead': [4],
        'ff_dim': [128],
        'use_cls_token': [True],
    }
    if 'mlp' in selected_models:
        g = dict(common); g['model'] = ['mlp']; g['hidden_dims'] = [[256,256,128],[256,128,64]]
        configs.extend(list(grid_product(g)))
    if 'lstm' in selected_models:
        g = dict(common); g['model'] = ['lstm']; g['hidden_size'] = [96,128]; g['num_layers'] = [1,2]; g['head_hidden'] = [64,128]
        configs.extend(list(grid_product(g)))
    if 'gru' in selected_models:
        g = dict(common); g['model'] = ['gru']; g['hidden_size'] = [96,128]; g['num_layers'] = [1,2]; g['head_hidden'] = [64,128]
        configs.extend(list(grid_product(g)))
    if 'transformer' in selected_models:
        g = dict(common); g['model'] = ['transformer']; g['d_model'] = [64,96]; g['num_layers'] = [1,2]; g['ff_dim'] = [128,192]; g['use_cls_token'] = [False,True]
        configs.extend(list(grid_product(g)))
    return configs


# =========================
# Main grid search
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_npz', required=True)
    parser.add_argument('--val_npz', required=True)
    parser.add_argument('--test_npz', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--models', nargs='+', default=['mlp','lstm','gru','transformer'])
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--tol', type=float, default=1e-12)
    parser.add_argument('--max_newton_iter', type=int, default=20)
    parser.add_argument('--rank_metric', default='plus_newton_r2', choices=['direct_r2','direct_rmse','direct_mae','plus_newton_r2','plus_newton_rmse','plus_newton_mae','plus_newton_converged_ratio'])
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_raw = load_npz(args.train_npz)
    val_raw = load_npz(args.val_npz)
    test_raw = load_npz(args.test_npz)

    search_space = build_search_space(args.models)
    all_rows = []
    best_metric = None
    best_row = None
    best_ckpt = None

    for trial_id, hp in enumerate(search_space, start=1):
        trial_name = f"trial_{trial_id:03d}_{hp['model']}"
        print(f"\n========== {trial_name} ==========")
        print(json.dumps(hp, ensure_ascii=False))
        start_t = time.time()

        tr_seq, tr_glob, tr_y, tr_z0 = build_inputs_and_baseline(train_raw, use_log_features=hp['use_log_features'])
        va_seq, va_glob, va_y, va_z0 = build_inputs_and_baseline(val_raw, use_log_features=hp['use_log_features'])
        te_seq, te_glob, te_y, te_z0 = build_inputs_and_baseline(test_raw, use_log_features=hp['use_log_features'])

        train_ds = HybridDataset(tr_seq, tr_glob, tr_y, tr_z0, train_raw)
        val_ds = HybridDataset(va_seq, va_glob, va_y, va_z0, val_raw)
        test_ds = HybridDataset(te_seq, te_glob, te_y, te_z0, test_raw)
        seq_scaler, glob_scaler = standardize_datasets(train_ds, val_ds, test_ds)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        train_loader.hp = hp; val_loader.hp = hp; test_loader.hp = hp

        model = HybridCorrectionModel(hp['model'], train_ds.seq_x.shape[2], train_ds.seq_x.shape[1], train_ds.glob_x.shape[1], hp).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay']) if hp['optimizer'] == 'adamw' else torch.optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])

        best_val_rmse = float('inf'); best_epoch = -1; best_state = None; wait = 0
        for epoch in range(1, args.epochs + 1):
            model.train(); train_loss_sum = 0.0; train_n = 0
            for batch in train_loader:
                for k in batch:
                    batch[k] = batch[k].to(device)
                pred, delta_vec = model(batch['seq_x'], batch['glob_x'], batch['z0'], batch['Q_total'])
                loss, _ = hybrid_loss(pred, batch['y'], delta_vec, batch, hp)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                bs = pred.shape[0]
                train_loss_sum += float(loss.detach().cpu().item()) * bs
                train_n += bs
            val_metrics, _, _ = run_eval(model, val_loader, device)
            print(f"[{trial_name}] epoch={epoch:03d} train_loss={train_loss_sum/max(train_n,1):.6f} val_rmse={val_metrics['rmse']:.6f} val_r2={val_metrics['r2']:.6f}")
            if val_metrics['rmse'] < best_val_rmse:
                best_val_rmse = val_metrics['rmse']
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= args.patience:
                    break

        if best_state is None:
            continue

        model.load_state_dict(best_state)
        direct_metrics, pred_direct, true = run_eval(model, test_loader, device)
        direct_metrics.update(residual_metrics(pred_direct, test_raw))

        pred_ref, pred_iter, pred_conv = refine_batch(pred_direct.astype(np.float64), test_raw, tol=args.tol, max_iter=args.max_newton_iter)
        plus = vector_metrics(pred_ref, true.astype(np.float64))
        plus.update(residual_metrics(pred_ref, test_raw))
        plus['newton_iter_mean'] = float(np.mean(pred_iter))
        plus['newton_iter_median'] = float(np.median(pred_iter))
        plus['newton_iter_p90'] = float(np.percentile(pred_iter, 90))
        plus['newton_converged_ratio'] = float(np.mean(pred_conv))

        elapsed = time.time() - start_t
        row = {
            'trial_id': trial_id, 'trial_name': trial_name, 'model': hp['model'], 'best_epoch': best_epoch, 'elapsed_sec': elapsed,
            'direct_mae': direct_metrics['mae'], 'direct_rmse': direct_metrics['rmse'], 'direct_r2': direct_metrics['r2'],
            'direct_valid_ratio': direct_metrics['valid_ratio'], 'direct_residual_mean': direct_metrics['residual_mean'], 'direct_residual_median': direct_metrics['residual_median'], 'direct_residual_p90': direct_metrics['residual_p90'],
            'plus_newton_mae': plus['mae'], 'plus_newton_rmse': plus['rmse'], 'plus_newton_r2': plus['r2'],
            'plus_newton_valid_ratio': plus['valid_ratio'], 'plus_newton_residual_mean': plus['residual_mean'], 'plus_newton_residual_median': plus['residual_median'], 'plus_newton_residual_p90': plus['residual_p90'],
            'plus_newton_newton_iter_mean': plus['newton_iter_mean'], 'plus_newton_newton_iter_median': plus['newton_iter_median'], 'plus_newton_newton_iter_p90': plus['newton_iter_p90'], 'plus_newton_converged_ratio': plus['newton_converged_ratio'],
            'hp_use_log_features': hp['use_log_features'], 'hp_optimizer': hp['optimizer'], 'hp_dropout': hp['dropout'], 'hp_lr': hp['lr'], 'hp_weight_decay': hp['weight_decay'],
            'hp_lambda_sup': hp['lambda_sup'], 'hp_lambda_res': hp['lambda_res'], 'hp_lambda_delta': hp['lambda_delta'], 'hp_dr_scale': hp['dr_scale'], 'hp_dx_scale': hp['dx_scale'],
            'hp_hidden_dims': json.dumps(hp['hidden_dims']), 'hp_hidden_size': hp['hidden_size'], 'hp_num_layers': hp['num_layers'], 'hp_head_hidden': hp['head_hidden'], 'hp_head_layers': hp['head_layers'],
            'hp_d_model': hp['d_model'], 'hp_nhead': hp['nhead'], 'hp_ff_dim': hp['ff_dim'], 'hp_use_cls_token': hp['use_cls_token'],
        }
        all_rows.append(row)

        cur_metric = row[args.rank_metric]
        if best_metric is None:
            better = True
        else:
            better = cur_metric < best_metric if args.rank_metric in ['direct_rmse','direct_mae','plus_newton_rmse','plus_newton_mae'] else cur_metric > best_metric
        if better:
            best_metric = cur_metric
            best_row = dict(row)
            best_ckpt = {
                'model_state_dict': deepcopy(model.state_dict()), 'seq_scaler': seq_scaler.save(), 'glob_scaler': glob_scaler.save(), 'hp': hp,
                'seq_dim': train_ds.seq_x.shape[2], 'seq_len': train_ds.seq_x.shape[1], 'glob_dim': train_ds.glob_x.shape[1], 'best_val_rmse': best_val_rmse, 'best_epoch': best_epoch,
            }

        with open(out_dir / f"{trial_name}.json", 'w', encoding='utf-8') as f:
            json.dump(row, f, ensure_ascii=False, indent=2)

    if not all_rows:
        raise RuntimeError('No successful trials completed.')

    reverse = args.rank_metric not in ['direct_rmse','direct_mae','plus_newton_rmse','plus_newton_mae']
    all_rows_sorted = sorted(all_rows, key=lambda r: r[args.rank_metric], reverse=reverse)
    save_csv(out_dir / 'all_trials.csv', all_rows_sorted)
    with open(out_dir / 'best_result.json', 'w', encoding='utf-8') as f:
        json.dump(best_row, f, ensure_ascii=False, indent=2)
    if best_ckpt is not None:
        torch.save(best_ckpt, out_dir / 'best_model_by_grid.pt')

    print('\n================ FINAL RANKING ================')
    for row in all_rows_sorted[:10]:
        print({'trial_id': row['trial_id'], 'model': row['model'], args.rank_metric: row[args.rank_metric], 'plus_newton_rmse': row['plus_newton_rmse'], 'plus_newton_r2': row['plus_newton_r2'], 'plus_newton_converged_ratio': row['plus_newton_converged_ratio']})
    print('\n[DONE]')
    print(out_dir / 'all_trials.csv')
    print(out_dir / 'best_result.json')
    print(out_dir / 'best_model_by_grid.pt')


if __name__ == '__main__':
    main()
