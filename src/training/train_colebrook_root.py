#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_colebrook_root.py

Dataset format (from colebrook_like_dataset.py):
    npz keys:
        - coeffs   : (N, degree+1)
        - center   : (N,)
        - a        : (N,)
        - b        : (N,)
        - root     : (N,)
        - residual : (N,)

Input features:
    default:
        [c0, c1, ..., c25, x0, a, b]

    optional:
        [c0, ..., c25, x0, log(a), log(b)]

Task:
    regress root x* for
        F(x; a, b) = x + 2*log10(a + b*x) = 0
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def colebrook_like_F_np(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x + 2.0 * np.log10(a + b * x)


class NPZRootDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        use_log_ab: bool = False,
        include_center: bool = True,
        include_ab: bool = True,
        target_key: str = "root",
    ):
        data = np.load(npz_path)

        coeffs = data["coeffs"].astype(np.float32)
        center = data["center"].astype(np.float32).reshape(-1, 1)
        a = data["a"].astype(np.float32).reshape(-1, 1)
        b = data["b"].astype(np.float32).reshape(-1, 1)
        y = data[target_key].astype(np.float32).reshape(-1, 1)

        feats = [coeffs]

        if include_center:
            feats.append(center)

        if include_ab:
            if use_log_ab:
                feats.append(np.log(a))
                feats.append(np.log(b))
            else:
                feats.append(a)
                feats.append(b)

        X = np.concatenate(feats, axis=1).astype(np.float32)

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.a = torch.from_numpy(a)
        self.b = torch.from_numpy(b)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.a[idx], self.b[idx]


class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def save(self) -> Dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.0):
        super().__init__()

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    count = 0

    ys, preds, as_, bs_ = [], [], [], []

    for X, y, a, b in loader:
        X = X.to(device)
        y = y.to(device)
        a = a.to(device)
        b = b.to(device)

        with torch.set_grad_enabled(train):
            pred = model(X)
            loss = criterion(pred, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * X.size(0)
        count += X.size(0)

        ys.append(y.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())
        as_.append(a.detach().cpu().numpy())
        bs_.append(b.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0).reshape(-1)
    y_pred = np.concatenate(preds, axis=0).reshape(-1)
    a_np = np.concatenate(as_, axis=0).reshape(-1)
    b_np = np.concatenate(bs_, axis=0).reshape(-1)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = r2_score_np(y_true, y_pred)

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
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "residual_mean": residual_mean,
        "residual_median": residual_median,
        "residual_p90": residual_p90,
        "valid_ratio": valid_ratio,
    }


def standardize_dataset(train_ds, val_ds, test_ds):
    scaler = Standardizer()
    scaler.fit(train_ds.X.numpy())

    train_ds.X = torch.from_numpy(scaler.transform(train_ds.X.numpy()).astype(np.float32))
    val_ds.X = torch.from_numpy(scaler.transform(val_ds.X.numpy()).astype(np.float32))
    test_ds.X = torch.from_numpy(scaler.transform(test_ds.X.numpy()).astype(np.float32))
    return scaler


def save_text(path: Path, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_npz", type=str, required=True)
    parser.add_argument("--val_npz", type=str, required=True)
    parser.add_argument("--test_npz", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256, 128])

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=100)

    parser.add_argument("--use_log_ab", action="store_true")
    parser.add_argument("--include_center", action="store_true", default=True)
    parser.add_argument("--include_ab", action="store_true", default=True)
    parser.add_argument("--no_include_center", action="store_true")
    parser.add_argument("--no_include_ab", action="store_true")

    parser.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()

    if args.no_include_center:
        args.include_center = False
    if args.no_include_ab:
        args.include_ab = False

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = NPZRootDataset(
        args.train_npz,
        use_log_ab=args.use_log_ab,
        include_center=args.include_center,
        include_ab=args.include_ab,
    )
    val_ds = NPZRootDataset(
        args.val_npz,
        use_log_ab=args.use_log_ab,
        include_center=args.include_center,
        include_ab=args.include_ab,
    )
    test_ds = NPZRootDataset(
        args.test_npz,
        use_log_ab=args.use_log_ab,
        include_center=args.include_center,
        include_ab=args.include_ab,
    )

    scaler = standardize_dataset(train_ds, val_ds, test_ds)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    input_dim = train_ds.X.shape[1]

    model = MLPRegressor(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_epoch = -1
    wait = 0
    log_lines = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, criterion, device, train=False)

        line = (
            f"[Epoch {epoch:04d}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_rmse={train_metrics['rmse']:.6f} "
            f"val_rmse={val_metrics['rmse']:.6f} "
            f"val_mae={val_metrics['mae']:.6f} "
            f"val_r2={val_metrics['r2']:.6f} "
            f"val_residual_mean={val_metrics['residual_mean']:.3e} "
            f"valid_ratio={val_metrics['valid_ratio']:.4f}"
        )
        print(line)
        log_lines.append(line)

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "scaler": scaler.save(),
                    "args": vars(args),
                    "input_dim": input_dim,
                    "hidden_dims": args.hidden_dims,
                    "best_val_rmse": best_val_rmse,
                    "best_epoch": best_epoch,
                },
                out_dir / "best_model.pt",
            )
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[Early Stop] patience reached at epoch {epoch}")
                log_lines.append(f"[Early Stop] patience reached at epoch {epoch}")
                break

    checkpoint = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_best = run_epoch(model, val_loader, optimizer=None, criterion=criterion, device=device, train=False)
    test_best = run_epoch(model, test_loader, optimizer=None, criterion=criterion, device=device, train=False)

    summary = []
    summary.append("=== Best Validation / Test Summary ===")
    summary.append(f"best_epoch: {checkpoint['best_epoch']}")
    summary.append(f"best_val_rmse: {checkpoint['best_val_rmse']:.8f}")
    summary.append("")
    summary.append("[Validation]")
    for k, v in val_best.items():
        summary.append(f"{k}: {v}")
    summary.append("")
    summary.append("[Test]")
    for k, v in test_best.items():
        summary.append(f"{k}: {v}")

    print("\n".join(summary))
    log_lines.extend(["", *summary])

    save_text(out_dir / "metrics.txt", "\n".join(log_lines))
    save_text(out_dir / "config.txt", json.dumps(vars(args), indent=2, ensure_ascii=False))

    print(f"\n[DONE] Outputs saved to: {out_dir.resolve()}")
    print("  - best_model.pt")
    print("  - metrics.txt")
    print("  - config.txt")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_colebrook_root.py

Dataset format (from colebrook_like_dataset.py):
    npz keys:
        - coeffs   : (N, degree+1)
        - center   : (N,)
        - a        : (N,)
        - b        : (N,)
        - root     : (N,)
        - residual : (N,)

Input features:
    default:
        [c0, c1, ..., c25, x0, a, b]

    optional:
        [c0, ..., c25, x0, log(a), log(b)]

Task:
    regress root x* for
        F(x; a, b) = x + 2*log10(a + b*x) = 0
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def colebrook_like_F_np(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x + 2.0 * np.log10(a + b * x)


class NPZRootDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        use_log_ab: bool = False,
        include_center: bool = True,
        include_ab: bool = True,
        target_key: str = "root",
    ):
        data = np.load(npz_path)

        coeffs = data["coeffs"].astype(np.float32)
        center = data["center"].astype(np.float32).reshape(-1, 1)
        a = data["a"].astype(np.float32).reshape(-1, 1)
        b = data["b"].astype(np.float32).reshape(-1, 1)
        y = data[target_key].astype(np.float32).reshape(-1, 1)

        feats = [coeffs]

        if include_center:
            feats.append(center)

        if include_ab:
            if use_log_ab:
                feats.append(np.log(a))
                feats.append(np.log(b))
            else:
                feats.append(a)
                feats.append(b)

        X = np.concatenate(feats, axis=1).astype(np.float32)

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.a = torch.from_numpy(a)
        self.b = torch.from_numpy(b)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.a[idx], self.b[idx]


class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def save(self) -> Dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.0):
        super().__init__()

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    count = 0

    ys, preds, as_, bs_ = [], [], [], []

    for X, y, a, b in loader:
        X = X.to(device)
        y = y.to(device)
        a = a.to(device)
        b = b.to(device)

        with torch.set_grad_enabled(train):
            pred = model(X)
            loss = criterion(pred, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * X.size(0)
        count += X.size(0)

        ys.append(y.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())
        as_.append(a.detach().cpu().numpy())
        bs_.append(b.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0).reshape(-1)
    y_pred = np.concatenate(preds, axis=0).reshape(-1)
    a_np = np.concatenate(as_, axis=0).reshape(-1)
    b_np = np.concatenate(bs_, axis=0).reshape(-1)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = r2_score_np(y_true, y_pred)

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
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "residual_mean": residual_mean,
        "residual_median": residual_median,
        "residual_p90": residual_p90,
        "valid_ratio": valid_ratio,
    }


def standardize_dataset(train_ds, val_ds, test_ds):
    scaler = Standardizer()
    scaler.fit(train_ds.X.numpy())

    train_ds.X = torch.from_numpy(scaler.transform(train_ds.X.numpy()).astype(np.float32))
    val_ds.X = torch.from_numpy(scaler.transform(val_ds.X.numpy()).astype(np.float32))
    test_ds.X = torch.from_numpy(scaler.transform(test_ds.X.numpy()).astype(np.float32))
    return scaler


def save_text(path: Path, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_npz", type=str, required=True)
    parser.add_argument("--val_npz", type=str, required=True)
    parser.add_argument("--test_npz", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256, 128])

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=100)

    parser.add_argument("--use_log_ab", action="store_true")
    parser.add_argument("--include_center", action="store_true", default=True)
    parser.add_argument("--include_ab", action="store_true", default=True)
    parser.add_argument("--no_include_center", action="store_true")
    parser.add_argument("--no_include_ab", action="store_true")

    parser.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()

    if args.no_include_center:
        args.include_center = False
    if args.no_include_ab:
        args.include_ab = False

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = NPZRootDataset(
        args.train_npz,
        use_log_ab=args.use_log_ab,
        include_center=args.include_center,
        include_ab=args.include_ab,
    )
    val_ds = NPZRootDataset(
        args.val_npz,
        use_log_ab=args.use_log_ab,
        include_center=args.include_center,
        include_ab=args.include_ab,
    )
    test_ds = NPZRootDataset(
        args.test_npz,
        use_log_ab=args.use_log_ab,
        include_center=args.include_center,
        include_ab=args.include_ab,
    )

    scaler = standardize_dataset(train_ds, val_ds, test_ds)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    input_dim = train_ds.X.shape[1]

    model = MLPRegressor(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_epoch = -1
    wait = 0
    log_lines = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, criterion, device, train=False)

        line = (
            f"[Epoch {epoch:04d}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_rmse={train_metrics['rmse']:.6f} "
            f"val_rmse={val_metrics['rmse']:.6f} "
            f"val_mae={val_metrics['mae']:.6f} "
            f"val_r2={val_metrics['r2']:.6f} "
            f"val_residual_mean={val_metrics['residual_mean']:.3e} "
            f"valid_ratio={val_metrics['valid_ratio']:.4f}"
        )
        print(line)
        log_lines.append(line)

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "scaler": scaler.save(),
                    "args": vars(args),
                    "input_dim": input_dim,
                    "hidden_dims": args.hidden_dims,
                    "best_val_rmse": best_val_rmse,
                    "best_epoch": best_epoch,
                },
                out_dir / "best_model.pt",
            )
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[Early Stop] patience reached at epoch {epoch}")
                log_lines.append(f"[Early Stop] patience reached at epoch {epoch}")
                break

    checkpoint = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_best = run_epoch(model, val_loader, optimizer=None, criterion=criterion, device=device, train=False)
    test_best = run_epoch(model, test_loader, optimizer=None, criterion=criterion, device=device, train=False)

    summary = []
    summary.append("=== Best Validation / Test Summary ===")
    summary.append(f"best_epoch: {checkpoint['best_epoch']}")
    summary.append(f"best_val_rmse: {checkpoint['best_val_rmse']:.8f}")
    summary.append("")
    summary.append("[Validation]")
    for k, v in val_best.items():
        summary.append(f"{k}: {v}")
    summary.append("")
    summary.append("[Test]")
    for k, v in test_best.items():
        summary.append(f"{k}: {v}")

    print("\n".join(summary))
    log_lines.extend(["", *summary])

    save_text(out_dir / "metrics.txt", "\n".join(log_lines))
    save_text(out_dir / "config.txt", json.dumps(vars(args), indent=2, ensure_ascii=False))

    print(f"\n[DONE] Outputs saved to: {out_dir.resolve()}")
    print("  - best_model.pt")
    print("  - metrics.txt")
    print("  - config.txt")


if __name__ == "__main__":
    main()
