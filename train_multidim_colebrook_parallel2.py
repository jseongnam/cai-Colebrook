#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    required = ["coeffs", "center", "target", "Q_total", "D1", "D2", "eps1", "eps2", "L1", "L2", "rho", "mu", "g"]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {npz_path}. Available: {list(data.keys())}")
    return {k: np.asarray(data[k]) for k in required}

def build_features(data, use_log_features=False):
    coeffs = np.asarray(data["coeffs"], dtype=np.float32)
    center = np.asarray(data["center"], dtype=np.float32)
    coeffs_flat = coeffs.reshape(coeffs.shape[0], -1)

    globals_raw = [
        np.asarray(data["Q_total"], dtype=np.float32).reshape(-1, 1),
        np.asarray(data["D1"], dtype=np.float32).reshape(-1, 1),
        np.asarray(data["D2"], dtype=np.float32).reshape(-1, 1),
        np.asarray(data["eps1"], dtype=np.float32).reshape(-1, 1),
        np.asarray(data["eps2"], dtype=np.float32).reshape(-1, 1),
        np.asarray(data["L1"], dtype=np.float32).reshape(-1, 1),
        np.asarray(data["L2"], dtype=np.float32).reshape(-1, 1),
        np.asarray(data["rho"], dtype=np.float32).reshape(-1, 1),
        np.asarray(data["mu"], dtype=np.float32).reshape(-1, 1),
        np.asarray(data["g"], dtype=np.float32).reshape(-1, 1),
    ]

    if use_log_features:
        globals_proc = []
        for i, arr in enumerate(globals_raw):
            if i < 9:
                globals_proc.append(np.log(np.clip(arr, 1e-12, None)))
            else:
                globals_proc.append(arr)
        globals_cat = np.concatenate(globals_proc, axis=1)
    else:
        globals_cat = np.concatenate(globals_raw, axis=1)

    return np.concatenate([coeffs_flat, center, globals_cat], axis=1).astype(np.float32)

def build_target(data):
    return np.asarray(data["target"], dtype=np.float32)

def fit_standard_scaler(X):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}

def apply_scaler(X, scaler):
    return ((X - scaler["mean"]) / scaler["std"]).astype(np.float32)

class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dims=(256, 256, 128), dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

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
    }

def run_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        bs = xb.shape[0]
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    n = 0
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        bs = xb.shape[0]
        total_loss += loss.item() * bs
        n += bs
        preds.append(pred.cpu().numpy())
        trues.append(yb.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    m = regression_metrics(preds, trues)
    m["loss"] = total_loss / max(n, 1)
    return m, preds, trues

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz", required=True)
    ap.add_argument("--val_npz", required=True)
    ap.add_argument("--test_npz", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--use_log_features", action="store_true")
    ap.add_argument("--hidden_dims", nargs="+", type=int, default=[256, 256, 128])
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_data = load_npz(args.train_npz)
    val_data = load_npz(args.val_npz)
    test_data = load_npz(args.test_npz)

    X_train = build_features(train_data, use_log_features=args.use_log_features)
    X_val = build_features(val_data, use_log_features=args.use_log_features)
    X_test = build_features(test_data, use_log_features=args.use_log_features)

    y_train = build_target(train_data)
    y_val = build_target(val_data)
    y_test = build_target(test_data)

    scaler = fit_standard_scaler(X_train)
    X_train = apply_scaler(X_train, scaler)
    X_val = apply_scaler(X_val, scaler)
    X_test = apply_scaler(X_test, scaler)

    train_loader = DataLoader(NumpyDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(NumpyDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(NumpyDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    model = MLPRegressor(
        input_dim=X_train.shape[1],
        output_dim=3,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
                "hidden_dims": list(args.hidden_dims),
                "dropout": args.dropout,
                "input_dim": int(X_train.shape[1]),
                "output_dim": 3,
                "scaler": {
                    "mean": scaler["mean"].reshape(-1).tolist(),
                    "std": scaler["std"].reshape(-1).tolist(),
                },
                "args": {
                    "use_log_features": bool(args.use_log_features),
                    "feature_layout": "flatten(coeffs)+center+globals",
                },
            }
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[Early stopping] patience={args.patience}")
                break

    model.load_state_dict(best_state["model_state_dict"])
    val_metrics, _, _ = evaluate(model, val_loader, args.device)
    test_metrics, _, _ = evaluate(model, test_loader, args.device)

    torch.save(best_state, save_dir / "best_model.pt")

    with open(save_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write("=== Best Validation / Test Summary ===\n")
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
            "train_npz": args.train_npz,
            "val_npz": args.val_npz,
            "test_npz": args.test_npz,
            "use_log_features": args.use_log_features,
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "patience": args.patience,
            "seed": args.seed,
            "device": args.device,
            "input_dim": int(X_train.shape[1]),
        }, f, ensure_ascii=False, indent=2)

    print("\n=== Best Validation / Test Summary ===")
    print(f"best_epoch: {best_epoch}")
    print(f"best_val_rmse: {best_val_rmse:.8f}")
    print("\n[Validation]")
    for k, v in val_metrics.items():
        print(f"{k}: {v}")
    print("\n[Test]")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")
    print(f"\n[DONE] Outputs saved to: {save_dir}")

if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    required = ["coeffs", "center", "target", "Q_total", "D1", "D2", "eps1", "eps2", "L1", "L2", "rho", "mu", "g"]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {npz_path}. Available: {list(data.keys())}")
    return {k: np.asarray(data[k]) for k in required}

def build_features(data, use_log_features=False):
    coeffs = np.asarray(data["coeffs"], dtype=np.float32)
    center = np.asarray(data["center"], dtype=np.float32)
    coeffs_flat = coeffs.reshape(coeffs.shape[0], -1)

    globals_raw = [
        np.asarray(data["Q_total"], dtype=np.float32).reshape(-1, 1),
        np.asarray(data["D1"], dtype=np.float32).reshape(-1, 1),
        np.asarray(data["D2"], dtype=np.float32).reshape(-1, 1),
        np.asarray(data["eps1"], dtype=np.float32).reshape(-1, 1),
        np.asarray(data["eps2"], dtype=np.float32).reshape(-1, 1),
        np.asarray(data["L1"], dtype=np.float32).reshape(-1, 1),
        np.asarray(data["L2"], dtype=np.float32).reshape(-1, 1),
        np.asarray(data["rho"], dtype=np.float32).reshape(-1, 1),
        np.asarray(data["mu"], dtype=np.float32).reshape(-1, 1),
        np.asarray(data["g"], dtype=np.float32).reshape(-1, 1),
    ]

    if use_log_features:
        globals_proc = []
        for i, arr in enumerate(globals_raw):
            if i < 9:
                globals_proc.append(np.log(np.clip(arr, 1e-12, None)))
            else:
                globals_proc.append(arr)
        globals_cat = np.concatenate(globals_proc, axis=1)
    else:
        globals_cat = np.concatenate(globals_raw, axis=1)

    return np.concatenate([coeffs_flat, center, globals_cat], axis=1).astype(np.float32)

def build_target(data):
    return np.asarray(data["target"], dtype=np.float32)

def fit_standard_scaler(X):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}

def apply_scaler(X, scaler):
    return ((X - scaler["mean"]) / scaler["std"]).astype(np.float32)

class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dims=(256, 256, 128), dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

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
    }

def run_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        bs = xb.shape[0]
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    n = 0
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        bs = xb.shape[0]
        total_loss += loss.item() * bs
        n += bs
        preds.append(pred.cpu().numpy())
        trues.append(yb.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    m = regression_metrics(preds, trues)
    m["loss"] = total_loss / max(n, 1)
    return m, preds, trues

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz", required=True)
    ap.add_argument("--val_npz", required=True)
    ap.add_argument("--test_npz", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--use_log_features", action="store_true")
    ap.add_argument("--hidden_dims", nargs="+", type=int, default=[256, 256, 128])
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_data = load_npz(args.train_npz)
    val_data = load_npz(args.val_npz)
    test_data = load_npz(args.test_npz)

    X_train = build_features(train_data, use_log_features=args.use_log_features)
    X_val = build_features(val_data, use_log_features=args.use_log_features)
    X_test = build_features(test_data, use_log_features=args.use_log_features)

    y_train = build_target(train_data)
    y_val = build_target(val_data)
    y_test = build_target(test_data)

    scaler = fit_standard_scaler(X_train)
    X_train = apply_scaler(X_train, scaler)
    X_val = apply_scaler(X_val, scaler)
    X_test = apply_scaler(X_test, scaler)

    train_loader = DataLoader(NumpyDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(NumpyDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(NumpyDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    model = MLPRegressor(
        input_dim=X_train.shape[1],
        output_dim=3,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
                "hidden_dims": list(args.hidden_dims),
                "dropout": args.dropout,
                "input_dim": int(X_train.shape[1]),
                "output_dim": 3,
                "scaler": {
                    "mean": scaler["mean"].reshape(-1).tolist(),
                    "std": scaler["std"].reshape(-1).tolist(),
                },
                "args": {
                    "use_log_features": bool(args.use_log_features),
                    "feature_layout": "flatten(coeffs)+center+globals",
                },
            }
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[Early stopping] patience={args.patience}")
                break

    model.load_state_dict(best_state["model_state_dict"])
    val_metrics, _, _ = evaluate(model, val_loader, args.device)
    test_metrics, _, _ = evaluate(model, test_loader, args.device)

    torch.save(best_state, save_dir / "best_model.pt")

    with open(save_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write("=== Best Validation / Test Summary ===\n")
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
            "train_npz": args.train_npz,
            "val_npz": args.val_npz,
            "test_npz": args.test_npz,
            "use_log_features": args.use_log_features,
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "patience": args.patience,
            "seed": args.seed,
            "device": args.device,
            "input_dim": int(X_train.shape[1]),
        }, f, ensure_ascii=False, indent=2)

    print("\n=== Best Validation / Test Summary ===")
    print(f"best_epoch: {best_epoch}")
    print(f"best_val_rmse: {best_val_rmse:.8f}")
    print("\n[Validation]")
    for k, v in val_metrics.items():
        print(f"{k}: {v}")
    print("\n[Test]")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")
    print(f"\n[DONE] Outputs saved to: {save_dir}")

if __name__ == "__main__":
    main()