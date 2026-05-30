#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Utility
# =========================================================
def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def sanitize_array(x: np.ndarray, clip_value: float = 1e12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)
    x = np.clip(x, -clip_value, clip_value)
    return x


def fit_standard_scaler(X: np.ndarray) -> Dict[str, np.ndarray]:
    X = sanitize_array(X, clip_value=1e12)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    mean[~np.isfinite(mean)] = 0.0
    std[~np.isfinite(std)] = 1.0
    std[std < 1e-8] = 1.0
    return {"mean": mean.astype(np.float64), "std": std.astype(np.float64)}


def apply_scaler(X: np.ndarray, scaler: Dict[str, np.ndarray], clip_out: float = 1e6) -> np.ndarray:
    X = sanitize_array(X, clip_value=1e12)
    Xs = (X - scaler["mean"]) / scaler["std"]
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = np.clip(Xs, -clip_out, clip_out)
    return Xs.astype(np.float32)


def filter_valid_rows(seq_x: np.ndarray, glob_x: np.ndarray, y: np.ndarray):
    seq_ok = np.all(np.isfinite(seq_x.reshape(seq_x.shape[0], -1)), axis=1)
    glob_ok = np.all(np.isfinite(glob_x), axis=1)
    y_ok = np.all(np.isfinite(y), axis=1)
    mask = seq_ok & glob_ok & y_ok
    return seq_x[mask], glob_x[mask], y[mask]


# =========================================================
# Data
# =========================================================
def load_npz(npz_path: str) -> Dict[str, np.ndarray]:
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
    """
    Returns
    -------
    seq_x  : (N, 3, token_dim)
    glob_x : (N, G)
    y      : (N, 3)
    """
    coeffs = np.asarray(data["coeffs"], dtype=np.float64)
    center = np.asarray(data["center"], dtype=np.float64)
    y = np.asarray(data["target"], dtype=np.float64)

    coeffs = sanitize_array(coeffs, clip_value=1e30)
    center = sanitize_array(center, clip_value=1e12)
    y = sanitize_array(y, clip_value=1e12)

    coeffs = signed_log1p(coeffs)

    # token_i = [coeff_i, center_i]
    center_expand = center[..., None]   # (N, 3, 1)
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
    globals_raw = [sanitize_array(g, clip_value=1e12) for g in globals_raw]

    if use_log_features:
        globals_proc = []
        for i, arr in enumerate(globals_raw):
            if i < 9:  # g 제외
                globals_proc.append(np.log(np.clip(arr, 1e-12, None)))
            else:
                globals_proc.append(arr)
        glob_x = np.concatenate(globals_proc, axis=1)
    else:
        glob_x = np.concatenate(globals_raw, axis=1)

    glob_x = sanitize_array(glob_x, clip_value=1e12)
    return seq_x, glob_x, y


class MultiInputDataset(Dataset):
    def __init__(self, seq_x: np.ndarray, glob_x: np.ndarray, y: np.ndarray):
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
    def __init__(self, seq_dim: int, seq_len: int, glob_dim: int,
                 hidden_dims=(256, 256, 128), dropout=0.1, out_dim=3):
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
    def __init__(self, seq_dim: int, glob_dim: int,
                 hidden_size=128, num_layers=2, dropout=0.1,
                 out_dim=3, head_hidden=128, head_layers=2):
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
        for _ in range(head_layers - 1):
            layers += [nn.Linear(prev, hidden_dim), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden_dim
            hidden_dim = max(hidden_dim // 2, 32)
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        _, (hn, _) = self.lstm(seq_x)
        h_last = hn[-1]
        x = torch.cat([h_last, glob_x], dim=1)
        return self.head(x)


class GRUModel(nn.Module):
    def __init__(self, seq_dim: int, glob_dim: int,
                 hidden_size=128, num_layers=2, dropout=0.1,
                 out_dim=3, head_hidden=128, head_layers=2):
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
        for _ in range(head_layers - 1):
            layers += [nn.Linear(prev, hidden_dim), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden_dim
            hidden_dim = max(hidden_dim // 2, 32)
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        _, hn = self.gru(seq_x)
        h_last = hn[-1]
        x = torch.cat([h_last, glob_x], dim=1)
        return self.head(x)


class TransformerModel(nn.Module):
    def __init__(self, seq_dim: int, seq_len: int, glob_dim: int,
                 d_model=96, nhead=4, num_layers=2, dropout=0.1,
                 out_dim=3, ff_dim=192, head_hidden=128,
                 head_layers=2, use_cls_token=True):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.seq_len = seq_len

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

        z = torch.cat([pooled, glob_x], dim=1)
        return self.head(z)


def build_model(model_name: str,
                seq_dim: int,
                seq_len: int,
                glob_dim: int,
                args) -> nn.Module:
    if model_name == "mlp":
        return MLPModel(
            seq_dim=seq_dim,
            seq_len=seq_len,
            glob_dim=glob_dim,
            hidden_dims=tuple(args.hidden_dims),
            dropout=args.dropout,
            out_dim=3,
        )
    elif model_name == "lstm":
        return LSTMModel(
            seq_dim=seq_dim,
            glob_dim=glob_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            out_dim=3,
            head_hidden=args.head_hidden,
            head_layers=args.head_layers,
        )
    elif model_name == "gru":
        return GRUModel(
            seq_dim=seq_dim,
            glob_dim=glob_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            out_dim=3,
            head_hidden=args.head_hidden,
            head_layers=args.head_layers,
        )
    elif model_name == "transformer":
        return TransformerModel(
            seq_dim=seq_dim,
            seq_len=seq_len,
            glob_dim=glob_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
            out_dim=3,
            ff_dim=args.ff_dim,
            head_hidden=args.head_hidden,
            head_layers=args.head_layers,
            use_cls_token=args.use_cls_token,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")


# =========================================================
# Metrics
# =========================================================
def regression_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
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
    }


# =========================================================
# Train / Eval
# =========================================================
def build_optimizer(model, args):
    opt_name = args.optimizer.lower()
    if opt_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif opt_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def run_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.SmoothL1Loss(beta=0.1)
    total_loss = 0.0
    n = 0

    for seq_x, glob_x, yb in loader:
        seq_x = seq_x.to(device)
        glob_x = glob_x.to(device)
        yb = yb.to(device)

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
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.SmoothL1Loss(beta=0.1)

    total_loss = 0.0
    n = 0
    preds = []
    trues = []

    for seq_x, glob_x, yb in loader:
        seq_x = seq_x.to(device)
        glob_x = glob_x.to(device)
        yb = yb.to(device)

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
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", choices=["mlp", "lstm", "gru", "transformer"], required=True)
    ap.add_argument("--train_npz", required=True)
    ap.add_argument("--val_npz", required=True)
    ap.add_argument("--test_npz", required=True)
    ap.add_argument("--save_dir", required=True)

    ap.add_argument("--use_log_features", action="store_true")

    # common
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=35)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)

    # optimizer
    ap.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    # mlp
    ap.add_argument("--hidden_dims", nargs="+", type=int, default=[256, 256, 128])

    # lstm / gru / transformer common-ish
    ap.add_argument("--hidden_size", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--head_hidden", type=int, default=128)
    ap.add_argument("--head_layers", type=int, default=2)

    # transformer
    ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--ff_dim", type=int, default=192)
    ap.add_argument("--use_cls_token", action="store_true")

    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_data = load_npz(args.train_npz)
    val_data = load_npz(args.val_npz)
    test_data = load_npz(args.test_npz)

    tr_seq, tr_glob, tr_y = build_inputs(train_data, use_log_features=args.use_log_features)
    va_seq, va_glob, va_y = build_inputs(val_data, use_log_features=args.use_log_features)
    te_seq, te_glob, te_y = build_inputs(test_data, use_log_features=args.use_log_features)

    tr_seq, tr_glob, tr_y = filter_valid_rows(tr_seq, tr_glob, tr_y)
    va_seq, va_glob, va_y = filter_valid_rows(va_seq, va_glob, va_y)
    te_seq, te_glob, te_y = filter_valid_rows(te_seq, te_glob, te_y)

    print(f"[INFO] train valid samples: {len(tr_seq)}")
    print(f"[INFO] val valid samples:   {len(va_seq)}")
    print(f"[INFO] test valid samples:  {len(te_seq)}")

    # separate scalers
    tr_seq_flat = tr_seq.reshape(-1, tr_seq.shape[-1])
    seq_scaler = fit_standard_scaler(tr_seq_flat)
    glob_scaler = fit_standard_scaler(tr_glob)

    tr_seq = apply_scaler(tr_seq.reshape(-1, tr_seq.shape[-1]), seq_scaler).reshape(tr_seq.shape[0], tr_seq.shape[1], tr_seq.shape[2])
    va_seq = apply_scaler(va_seq.reshape(-1, va_seq.shape[-1]), seq_scaler).reshape(va_seq.shape[0], va_seq.shape[1], va_seq.shape[2])
    te_seq = apply_scaler(te_seq.reshape(-1, te_seq.shape[-1]), seq_scaler).reshape(te_seq.shape[0], te_seq.shape[1], te_seq.shape[2])

    tr_glob = apply_scaler(tr_glob, glob_scaler)
    va_glob = apply_scaler(va_glob, glob_scaler)
    te_glob = apply_scaler(te_glob, glob_scaler)

    train_loader = DataLoader(MultiInputDataset(tr_seq, tr_glob, tr_y), batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(MultiInputDataset(va_seq, va_glob, va_y), batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(MultiInputDataset(te_seq, te_glob, te_y), batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = build_model(
        model_name=args.model,
        seq_dim=tr_seq.shape[2],
        seq_len=tr_seq.shape[1],
        glob_dim=tr_glob.shape[1],
        args=args,
    ).to(args.device)

    optimizer = build_optimizer(model, args)

    best_val_rmse = float("inf")
    best_epoch = -1
    best_state = None
    wait = 0
    logs = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, args.device)
        val_metrics, _, _ = evaluate(model, val_loader, args.device)

        line = (
            f"[Epoch {epoch:04d}] "
            f"train_loss={train_loss:.8f} "
            f"val_loss={val_metrics['loss']:.8f} "
            f"val_mae={val_metrics['mae']:.8f} "
            f"val_rmse={val_metrics['rmse']:.8f} "
            f"val_r2={val_metrics['r2']:.8f} "
            f"val_mae_Q1={val_metrics['mae_Q1']:.8f} "
            f"val_mae_x1={val_metrics['mae_x1']:.8f} "
            f"val_mae_x2={val_metrics['mae_x2']:.8f}"
        )
        print(line)
        logs.append(line)

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            wait = 0
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_name": args.model,
                "dropout": args.dropout,
                "seq_dim": int(tr_seq.shape[2]),
                "seq_len": int(tr_seq.shape[1]),
                "glob_dim": int(tr_glob.shape[1]),
                "output_dim": 3,
                "seq_scaler": {
                    "mean": seq_scaler["mean"].reshape(-1).tolist(),
                    "std": seq_scaler["std"].reshape(-1).tolist(),
                },
                "glob_scaler": {
                    "mean": glob_scaler["mean"].reshape(-1).tolist(),
                    "std": glob_scaler["std"].reshape(-1).tolist(),
                },
                "args": vars(args),
            }
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[Early stopping] patience={args.patience}")
                break

    if best_state is None:
        raise RuntimeError("Training failed: no valid best_state was produced.")

    model.load_state_dict(best_state["model_state_dict"])
    val_metrics, _, _ = evaluate(model, val_loader, args.device)
    test_metrics, _, _ = evaluate(model, test_loader, args.device)

    torch.save(best_state, save_dir / "best_model.pt")

    with open(save_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write("=== Best Validation / Test Summary ===\n")
        f.write(f"model: {args.model}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"best_val_rmse: {best_val_rmse:.8f}\n\n")
        f.write("[Validation]\n")
        for k, v in val_metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n[Test]\n")
        for k, v in test_metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n[Epoch Logs]\n")
        for line in logs:
            f.write(line + "\n")

    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "train_npz": args.train_npz,
            "val_npz": args.val_npz,
            "test_npz": args.test_npz,
            "save_dir": str(save_dir),
            "use_log_features": args.use_log_features,
            "seed": args.seed,
            "device": args.device,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "patience": args.patience,
            "optimizer": args.optimizer,
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
            "seq_dim": int(tr_seq.shape[2]),
            "seq_len": int(tr_seq.shape[1]),
            "glob_dim": int(tr_glob.shape[1]),
        }, f, ensure_ascii=False, indent=2)

    print("\n=== Best Validation / Test Summary ===")
    print(f"model: {args.model}")
    print(f"best_epoch: {best_epoch}")
    print(f"best_val_rmse: {best_val_rmse:.8f}")
    print("\n[Validation]")
    for k, v in val_metrics.items():
        print(f"{k}: {v}")
    print("\n[Test]")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    print(f"\n[DONE] Outputs saved to: {save_dir}")
    print("  - best_model.pt")
    print("  - metrics.txt")
    print("  - config.json")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Utility
# =========================================================
def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def sanitize_array(x: np.ndarray, clip_value: float = 1e12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)
    x = np.clip(x, -clip_value, clip_value)
    return x


def fit_standard_scaler(X: np.ndarray) -> Dict[str, np.ndarray]:
    X = sanitize_array(X, clip_value=1e12)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    mean[~np.isfinite(mean)] = 0.0
    std[~np.isfinite(std)] = 1.0
    std[std < 1e-8] = 1.0
    return {"mean": mean.astype(np.float64), "std": std.astype(np.float64)}


def apply_scaler(X: np.ndarray, scaler: Dict[str, np.ndarray], clip_out: float = 1e6) -> np.ndarray:
    X = sanitize_array(X, clip_value=1e12)
    Xs = (X - scaler["mean"]) / scaler["std"]
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = np.clip(Xs, -clip_out, clip_out)
    return Xs.astype(np.float32)


def filter_valid_rows(seq_x: np.ndarray, glob_x: np.ndarray, y: np.ndarray):
    seq_ok = np.all(np.isfinite(seq_x.reshape(seq_x.shape[0], -1)), axis=1)
    glob_ok = np.all(np.isfinite(glob_x), axis=1)
    y_ok = np.all(np.isfinite(y), axis=1)
    mask = seq_ok & glob_ok & y_ok
    return seq_x[mask], glob_x[mask], y[mask]


# =========================================================
# Data
# =========================================================
def load_npz(npz_path: str) -> Dict[str, np.ndarray]:
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
    """
    Returns
    -------
    seq_x  : (N, 3, token_dim)
    glob_x : (N, G)
    y      : (N, 3)
    """
    coeffs = np.asarray(data["coeffs"], dtype=np.float64)
    center = np.asarray(data["center"], dtype=np.float64)
    y = np.asarray(data["target"], dtype=np.float64)

    coeffs = sanitize_array(coeffs, clip_value=1e30)
    center = sanitize_array(center, clip_value=1e12)
    y = sanitize_array(y, clip_value=1e12)

    coeffs = signed_log1p(coeffs)

    # token_i = [coeff_i, center_i]
    center_expand = center[..., None]   # (N, 3, 1)
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
    globals_raw = [sanitize_array(g, clip_value=1e12) for g in globals_raw]

    if use_log_features:
        globals_proc = []
        for i, arr in enumerate(globals_raw):
            if i < 9:  # g 제외
                globals_proc.append(np.log(np.clip(arr, 1e-12, None)))
            else:
                globals_proc.append(arr)
        glob_x = np.concatenate(globals_proc, axis=1)
    else:
        glob_x = np.concatenate(globals_raw, axis=1)

    glob_x = sanitize_array(glob_x, clip_value=1e12)
    return seq_x, glob_x, y


class MultiInputDataset(Dataset):
    def __init__(self, seq_x: np.ndarray, glob_x: np.ndarray, y: np.ndarray):
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
    def __init__(self, seq_dim: int, seq_len: int, glob_dim: int,
                 hidden_dims=(256, 256, 128), dropout=0.1, out_dim=3):
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
    def __init__(self, seq_dim: int, glob_dim: int,
                 hidden_size=128, num_layers=2, dropout=0.1,
                 out_dim=3, head_hidden=128, head_layers=2):
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
        for _ in range(head_layers - 1):
            layers += [nn.Linear(prev, hidden_dim), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden_dim
            hidden_dim = max(hidden_dim // 2, 32)
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        _, (hn, _) = self.lstm(seq_x)
        h_last = hn[-1]
        x = torch.cat([h_last, glob_x], dim=1)
        return self.head(x)


class GRUModel(nn.Module):
    def __init__(self, seq_dim: int, glob_dim: int,
                 hidden_size=128, num_layers=2, dropout=0.1,
                 out_dim=3, head_hidden=128, head_layers=2):
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
        for _ in range(head_layers - 1):
            layers += [nn.Linear(prev, hidden_dim), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden_dim
            hidden_dim = max(hidden_dim // 2, 32)
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, seq_x, glob_x):
        _, hn = self.gru(seq_x)
        h_last = hn[-1]
        x = torch.cat([h_last, glob_x], dim=1)
        return self.head(x)


class TransformerModel(nn.Module):
    def __init__(self, seq_dim: int, seq_len: int, glob_dim: int,
                 d_model=96, nhead=4, num_layers=2, dropout=0.1,
                 out_dim=3, ff_dim=192, head_hidden=128,
                 head_layers=2, use_cls_token=True):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.seq_len = seq_len

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

        z = torch.cat([pooled, glob_x], dim=1)
        return self.head(z)


def build_model(model_name: str,
                seq_dim: int,
                seq_len: int,
                glob_dim: int,
                args) -> nn.Module:
    if model_name == "mlp":
        return MLPModel(
            seq_dim=seq_dim,
            seq_len=seq_len,
            glob_dim=glob_dim,
            hidden_dims=tuple(args.hidden_dims),
            dropout=args.dropout,
            out_dim=3,
        )
    elif model_name == "lstm":
        return LSTMModel(
            seq_dim=seq_dim,
            glob_dim=glob_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            out_dim=3,
            head_hidden=args.head_hidden,
            head_layers=args.head_layers,
        )
    elif model_name == "gru":
        return GRUModel(
            seq_dim=seq_dim,
            glob_dim=glob_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            out_dim=3,
            head_hidden=args.head_hidden,
            head_layers=args.head_layers,
        )
    elif model_name == "transformer":
        return TransformerModel(
            seq_dim=seq_dim,
            seq_len=seq_len,
            glob_dim=glob_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
            out_dim=3,
            ff_dim=args.ff_dim,
            head_hidden=args.head_hidden,
            head_layers=args.head_layers,
            use_cls_token=args.use_cls_token,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")


# =========================================================
# Metrics
# =========================================================
def regression_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
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
    }


# =========================================================
# Train / Eval
# =========================================================
def build_optimizer(model, args):
    opt_name = args.optimizer.lower()
    if opt_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif opt_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def run_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.SmoothL1Loss(beta=0.1)
    total_loss = 0.0
    n = 0

    for seq_x, glob_x, yb in loader:
        seq_x = seq_x.to(device)
        glob_x = glob_x.to(device)
        yb = yb.to(device)

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
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.SmoothL1Loss(beta=0.1)

    total_loss = 0.0
    n = 0
    preds = []
    trues = []

    for seq_x, glob_x, yb in loader:
        seq_x = seq_x.to(device)
        glob_x = glob_x.to(device)
        yb = yb.to(device)

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
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", choices=["mlp", "lstm", "gru", "transformer"], required=True)
    ap.add_argument("--train_npz", required=True)
    ap.add_argument("--val_npz", required=True)
    ap.add_argument("--test_npz", required=True)
    ap.add_argument("--save_dir", required=True)

    ap.add_argument("--use_log_features", action="store_true")

    # common
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=35)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)

    # optimizer
    ap.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    # mlp
    ap.add_argument("--hidden_dims", nargs="+", type=int, default=[256, 256, 128])

    # lstm / gru / transformer common-ish
    ap.add_argument("--hidden_size", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--head_hidden", type=int, default=128)
    ap.add_argument("--head_layers", type=int, default=2)

    # transformer
    ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--ff_dim", type=int, default=192)
    ap.add_argument("--use_cls_token", action="store_true")

    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_data = load_npz(args.train_npz)
    val_data = load_npz(args.val_npz)
    test_data = load_npz(args.test_npz)

    tr_seq, tr_glob, tr_y = build_inputs(train_data, use_log_features=args.use_log_features)
    va_seq, va_glob, va_y = build_inputs(val_data, use_log_features=args.use_log_features)
    te_seq, te_glob, te_y = build_inputs(test_data, use_log_features=args.use_log_features)

    tr_seq, tr_glob, tr_y = filter_valid_rows(tr_seq, tr_glob, tr_y)
    va_seq, va_glob, va_y = filter_valid_rows(va_seq, va_glob, va_y)
    te_seq, te_glob, te_y = filter_valid_rows(te_seq, te_glob, te_y)

    print(f"[INFO] train valid samples: {len(tr_seq)}")
    print(f"[INFO] val valid samples:   {len(va_seq)}")
    print(f"[INFO] test valid samples:  {len(te_seq)}")

    # separate scalers
    tr_seq_flat = tr_seq.reshape(-1, tr_seq.shape[-1])
    seq_scaler = fit_standard_scaler(tr_seq_flat)
    glob_scaler = fit_standard_scaler(tr_glob)

    tr_seq = apply_scaler(tr_seq.reshape(-1, tr_seq.shape[-1]), seq_scaler).reshape(tr_seq.shape[0], tr_seq.shape[1], tr_seq.shape[2])
    va_seq = apply_scaler(va_seq.reshape(-1, va_seq.shape[-1]), seq_scaler).reshape(va_seq.shape[0], va_seq.shape[1], va_seq.shape[2])
    te_seq = apply_scaler(te_seq.reshape(-1, te_seq.shape[-1]), seq_scaler).reshape(te_seq.shape[0], te_seq.shape[1], te_seq.shape[2])

    tr_glob = apply_scaler(tr_glob, glob_scaler)
    va_glob = apply_scaler(va_glob, glob_scaler)
    te_glob = apply_scaler(te_glob, glob_scaler)

    train_loader = DataLoader(MultiInputDataset(tr_seq, tr_glob, tr_y), batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(MultiInputDataset(va_seq, va_glob, va_y), batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(MultiInputDataset(te_seq, te_glob, te_y), batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = build_model(
        model_name=args.model,
        seq_dim=tr_seq.shape[2],
        seq_len=tr_seq.shape[1],
        glob_dim=tr_glob.shape[1],
        args=args,
    ).to(args.device)

    optimizer = build_optimizer(model, args)

    best_val_rmse = float("inf")
    best_epoch = -1
    best_state = None
    wait = 0
    logs = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, args.device)
        val_metrics, _, _ = evaluate(model, val_loader, args.device)

        line = (
            f"[Epoch {epoch:04d}] "
            f"train_loss={train_loss:.8f} "
            f"val_loss={val_metrics['loss']:.8f} "
            f"val_mae={val_metrics['mae']:.8f} "
            f"val_rmse={val_metrics['rmse']:.8f} "
            f"val_r2={val_metrics['r2']:.8f} "
            f"val_mae_Q1={val_metrics['mae_Q1']:.8f} "
            f"val_mae_x1={val_metrics['mae_x1']:.8f} "
            f"val_mae_x2={val_metrics['mae_x2']:.8f}"
        )
        print(line)
        logs.append(line)

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            wait = 0
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_name": args.model,
                "dropout": args.dropout,
                "seq_dim": int(tr_seq.shape[2]),
                "seq_len": int(tr_seq.shape[1]),
                "glob_dim": int(tr_glob.shape[1]),
                "output_dim": 3,
                "seq_scaler": {
                    "mean": seq_scaler["mean"].reshape(-1).tolist(),
                    "std": seq_scaler["std"].reshape(-1).tolist(),
                },
                "glob_scaler": {
                    "mean": glob_scaler["mean"].reshape(-1).tolist(),
                    "std": glob_scaler["std"].reshape(-1).tolist(),
                },
                "args": vars(args),
            }
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[Early stopping] patience={args.patience}")
                break

    if best_state is None:
        raise RuntimeError("Training failed: no valid best_state was produced.")

    model.load_state_dict(best_state["model_state_dict"])
    val_metrics, _, _ = evaluate(model, val_loader, args.device)
    test_metrics, _, _ = evaluate(model, test_loader, args.device)

    torch.save(best_state, save_dir / "best_model.pt")

    with open(save_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write("=== Best Validation / Test Summary ===\n")
        f.write(f"model: {args.model}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"best_val_rmse: {best_val_rmse:.8f}\n\n")
        f.write("[Validation]\n")
        for k, v in val_metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n[Test]\n")
        for k, v in test_metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n[Epoch Logs]\n")
        for line in logs:
            f.write(line + "\n")

    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "train_npz": args.train_npz,
            "val_npz": args.val_npz,
            "test_npz": args.test_npz,
            "save_dir": str(save_dir),
            "use_log_features": args.use_log_features,
            "seed": args.seed,
            "device": args.device,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "patience": args.patience,
            "optimizer": args.optimizer,
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
            "seq_dim": int(tr_seq.shape[2]),
            "seq_len": int(tr_seq.shape[1]),
            "glob_dim": int(tr_glob.shape[1]),
        }, f, ensure_ascii=False, indent=2)

    print("\n=== Best Validation / Test Summary ===")
    print(f"model: {args.model}")
    print(f"best_epoch: {best_epoch}")
    print(f"best_val_rmse: {best_val_rmse:.8f}")
    print("\n[Validation]")
    for k, v in val_metrics.items():
        print(f"{k}: {v}")
    print("\n[Test]")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    print(f"\n[DONE] Outputs saved to: {save_dir}")
    print("  - best_model.pt")
    print("  - metrics.txt")
    print("  - config.json")


if __name__ == "__main__":
    main()