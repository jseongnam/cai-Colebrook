#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

LN10 = math.log(10.0)


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


def colebrook_f(x, a, b):
    x = np.asarray(x, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = a + b * x
    out = np.full_like(x, np.nan, dtype=np.float64)
    mask = z > 0
    out[mask] = x[mask] + 2.0 * np.log10(z[mask])
    return out


def colebrook_df(x, a, b):
    x = np.asarray(x, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = a + b * x
    out = np.full_like(x, np.nan, dtype=np.float64)
    mask = z > 0
    out[mask] = 1.0 + 2.0 * b[mask] / (z[mask] * LN10)
    return out


def project_to_domain_np(x, a, b, eps=1e-10):
    left = -a / b + eps
    return np.maximum(x, left)


def newton_refine_original_vectorized(
    x0,
    a,
    b,
    tol=1e-12,
    max_iter=20,
    damping=1.0,
    deriv_eps=1e-14,
    max_step=5.0,
):
    x = project_to_domain_np(np.asarray(x0, dtype=np.float64).copy(), a, b, eps=1e-10)
    n = len(x)
    iters_used = np.zeros(n, dtype=np.int32)
    converged = np.zeros(n, dtype=bool)
    active = np.ones(n, dtype=bool)

    for k in range(1, max_iter + 1):
        if not np.any(active):
            break

        idx = np.where(active)[0]
        xa = x[idx]
        aa = a[idx]
        bb = b[idx]

        xa = project_to_domain_np(xa, aa, bb, eps=1e-10)

        fx = colebrook_f(xa, aa, bb)
        dfx = colebrook_df(xa, aa, bb)

        small = np.abs(dfx) < deriv_eps
        dfx[small] = np.where(dfx[small] >= 0, deriv_eps, -deriv_eps)

        step = damping * fx / dfx
        step = np.clip(step, -max_step, max_step)
        x_new = xa - step
        x_new = project_to_domain_np(x_new, aa, bb, eps=1e-10)

        fx_new = np.abs(colebrook_f(x_new, aa, bb))
        worse = fx_new > np.abs(fx)
        if np.any(worse):
            x_half = xa[worse] - 0.5 * step[worse]
            x_half = project_to_domain_np(x_half, aa[worse], bb[worse], eps=1e-10)
            fx_half = np.abs(colebrook_f(x_half, aa[worse], bb[worse]))
            better_half = fx_half < fx_new[worse]
            x_new[worse] = np.where(better_half, x_half, x_new[worse])

        x[idx] = x_new
        iters_used[idx] = k

        fx_now = np.abs(colebrook_f(x_new, aa, bb))
        done = fx_now <= tol
        converged[idx[done]] = True
        active[idx[done]] = False

    return x, iters_used, converged


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
        return self.net(x).squeeze(-1)


def infer_hidden_dims_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, List[int]]:
    linear_weights = []
    for k, v in state_dict.items():
        if k.endswith(".weight") and v.ndim == 2:
            linear_weights.append((k, v.shape))
    linear_weights = sorted(linear_weights, key=lambda kv: kv[0])

    if not linear_weights:
        raise RuntimeError("No linear layers found in state_dict")

    dims = []
    for _, shape in linear_weights:
        out_dim, in_dim = shape
        dims.append((in_dim, out_dim))

    input_dim = dims[0][0]
    hidden_dims = [out_dim for (_, out_dim) in dims[:-1]]
    return input_dim, hidden_dims


def load_model_checkpoint(path: str, device="cpu"):
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise RuntimeError("Unsupported checkpoint format")

    if isinstance(ckpt, dict) and "hidden_dims" in ckpt:
        hidden_dims = list(ckpt["hidden_dims"])
        input_dim = None
        for k, v in state_dict.items():
            if k.endswith(".weight") and v.ndim == 2:
                input_dim = int(v.shape[1])
                break
        if input_dim is None:
            input_dim, _ = infer_hidden_dims_from_state_dict(state_dict)
    else:
        input_dim, hidden_dims = infer_hidden_dims_from_state_dict(state_dict)

    dropout = 0.0
    if isinstance(ckpt, dict) and "args" in ckpt and isinstance(ckpt["args"], dict):
        dropout = float(ckpt["args"].get("dropout", 0.0))

    model = MLPRegressor(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return ckpt, model


def load_npz(path: str):
    data = np.load(path, allow_pickle=True)
    available = list(data.keys())

    for k in ["coeffs", "a", "b", "root"]:
        if k not in data:
            raise KeyError(f"Missing key '{k}'; available: {available}")

    if "x0" in data:
        x0_arr = np.asarray(data["x0"], dtype=np.float64).reshape(-1)
    elif "center" in data:
        x0_arr = np.asarray(data["center"], dtype=np.float64).reshape(-1)
    else:
        raise KeyError(f"Missing key 'x0' (or alias 'center'); available: {available}")

    return {
        "coeffs": np.asarray(data["coeffs"], dtype=np.float64),
        "x0": x0_arr,
        "a": np.asarray(data["a"], dtype=np.float64).reshape(-1),
        "b": np.asarray(data["b"], dtype=np.float64).reshape(-1),
        "root": np.asarray(data["root"], dtype=np.float64).reshape(-1),
    }


def build_features_from_checkpoint_args(coeffs, x0, a, b, ckpt_args):
    use_log_ab = bool(ckpt_args.get("use_log_ab", False))
    include_center = bool(ckpt_args.get("include_center", True))
    include_ab = bool(ckpt_args.get("include_ab", True))

    feats = [coeffs.astype(np.float32)]

    if include_center:
        feats.append(x0.reshape(-1, 1).astype(np.float32))

    if include_ab:
        if use_log_ab:
            feats.append(np.log(a).reshape(-1, 1).astype(np.float32))
            feats.append(np.log(b).reshape(-1, 1).astype(np.float32))
        else:
            feats.append(a.reshape(-1, 1).astype(np.float32))
            feats.append(b.reshape(-1, 1).astype(np.float32))

    return np.concatenate(feats, axis=1).astype(np.float32)


def apply_scaler(X, scaler_dict):
    mean = np.array(scaler_dict["mean"], dtype=np.float32)
    std = np.array(scaler_dict["std"], dtype=np.float32)
    return (X - mean) / std


def predict_with_checkpoint(coeffs, x0, a, b, ckpt, model, device="cpu", batch_size=4096):
    X = build_features_from_checkpoint_args(coeffs, x0, a, b, ckpt["args"])
    X = apply_scaler(X, ckpt["scaler"])

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).to(device)
            yb = model(xb).detach().cpu().numpy().reshape(-1)
            preds.append(yb)

    return np.concatenate(preds, axis=0).astype(np.float64)


def compute_basin_ratio_for_sample(
    a: float,
    b: float,
    true_root: float,
    radius: float = 1.0,
    grid_points: int = 81,
    tol: float = 1e-12,
    max_iter: int = 20,
):
    grid = np.linspace(true_root - radius, true_root + radius, grid_points)
    aa = np.full(grid_points, a, dtype=np.float64)
    bb = np.full(grid_points, b, dtype=np.float64)

    ref, _, _ = newton_refine_original_vectorized(grid, aa, bb, max_iter=max_iter, tol=tol)
    conv_to_true = np.abs(ref - true_root) <= 1e-6
    return float(np.mean(conv_to_true))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_npz", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--grid_points", type=int, default=81)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max_newton_iter", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_npz(args.test_npz)
    ckpt, model = load_model_checkpoint(args.model, device=args.device)

    coeffs = data["coeffs"]
    x0 = data["x0"]
    a = data["a"]
    b = data["b"]
    root = data["root"]

    pred_all = predict_with_checkpoint(coeffs, x0, a, b, ckpt, model, device=args.device)

    n = len(root)
    k = min(args.num_samples, n)
    selected = np.sort(rng.choice(np.arange(n), size=k, replace=False))

    rows = []
    for rank, idx in enumerate(selected, start=1):
        basin_ratio = compute_basin_ratio_for_sample(
            a=float(a[idx]),
            b=float(b[idx]),
            true_root=float(root[idx]),
            radius=args.radius,
            grid_points=args.grid_points,
            tol=args.tol,
            max_iter=args.max_newton_iter,
        )

        pred = float(pred_all[idx])
        true_r = float(root[idx])

        ref, _, _ = newton_refine_original_vectorized(
            np.array([pred], dtype=np.float64),
            np.array([a[idx]], dtype=np.float64),
            np.array([b[idx]], dtype=np.float64),
            max_iter=args.max_newton_iter,
            tol=args.tol,
        )
        inside = bool(np.abs(ref[0] - true_r) <= 1e-6)

        rows.append({
            "Sample ID": rank,
            "Original Index": int(idx),
            "True Root": true_r,
            "Predicted Initial Value": pred,
            "Basin Ratio": basin_ratio,
            "Is Prediction Inside Basin?": "Yes" if inside else "No",
            "Absolute Init Error": abs(pred - true_r),
            "a": float(a[idx]),
            "b": float(b[idx]),
        })

    csv_path = out_dir / "table3_basin_samples.csv"
    save_csv(csv_path, rows)

    txt_path = out_dir / "table3_basin_samples.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Table III candidate samples (randomly selected)\n")
        f.write("=" * 90 + "\n")
        f.write(f"test_npz: {args.test_npz}\n")
        f.write(f"model: {args.model}\n")
        f.write(f"num_samples: {k}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"radius: {args.radius}\n")
        f.write(f"grid_points: {args.grid_points}\n")
        f.write("=" * 90 + "\n\n")
        for row in rows:
            f.write(
                f"Sample ID={row['Sample ID']} | "
                f"Original Index={row['Original Index']} | "
                f"True Root={row['True Root']:.12f} | "
                f"Predicted Initial Value={row['Predicted Initial Value']:.12f} | "
                f"Basin Ratio={row['Basin Ratio']:.6f} | "
                f"Inside Basin={row['Is Prediction Inside Basin?']} | "
                f"Abs Init Error={row['Absolute Init Error']:.12f} | "
                f"a={row['a']:.12f} | b={row['b']:.12f}\n"
            )

    print(f"[DONE] Saved to: {out_dir.resolve()}")
    print(f"  - {csv_path.name}")
    print(f"  - {txt_path.name}")
    print("\nPreview:")
    for row in rows:
        print(
            f"{row['Sample ID']}, "
            f"{row['True Root']:.6f}, "
            f"{row['Predicted Initial Value']:.6f}, "
            f"{row['Basin Ratio']:.3f}, "
            f"{row['Is Prediction Inside Basin?']}"
        )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

LN10 = math.log(10.0)


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


def colebrook_f(x, a, b):
    x = np.asarray(x, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = a + b * x
    out = np.full_like(x, np.nan, dtype=np.float64)
    mask = z > 0
    out[mask] = x[mask] + 2.0 * np.log10(z[mask])
    return out


def colebrook_df(x, a, b):
    x = np.asarray(x, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = a + b * x
    out = np.full_like(x, np.nan, dtype=np.float64)
    mask = z > 0
    out[mask] = 1.0 + 2.0 * b[mask] / (z[mask] * LN10)
    return out


def project_to_domain_np(x, a, b, eps=1e-10):
    left = -a / b + eps
    return np.maximum(x, left)


def newton_refine_original_vectorized(
    x0,
    a,
    b,
    tol=1e-12,
    max_iter=20,
    damping=1.0,
    deriv_eps=1e-14,
    max_step=5.0,
):
    x = project_to_domain_np(np.asarray(x0, dtype=np.float64).copy(), a, b, eps=1e-10)
    n = len(x)
    iters_used = np.zeros(n, dtype=np.int32)
    converged = np.zeros(n, dtype=bool)
    active = np.ones(n, dtype=bool)

    for k in range(1, max_iter + 1):
        if not np.any(active):
            break

        idx = np.where(active)[0]
        xa = x[idx]
        aa = a[idx]
        bb = b[idx]

        xa = project_to_domain_np(xa, aa, bb, eps=1e-10)

        fx = colebrook_f(xa, aa, bb)
        dfx = colebrook_df(xa, aa, bb)

        small = np.abs(dfx) < deriv_eps
        dfx[small] = np.where(dfx[small] >= 0, deriv_eps, -deriv_eps)

        step = damping * fx / dfx
        step = np.clip(step, -max_step, max_step)
        x_new = xa - step
        x_new = project_to_domain_np(x_new, aa, bb, eps=1e-10)

        fx_new = np.abs(colebrook_f(x_new, aa, bb))
        worse = fx_new > np.abs(fx)
        if np.any(worse):
            x_half = xa[worse] - 0.5 * step[worse]
            x_half = project_to_domain_np(x_half, aa[worse], bb[worse], eps=1e-10)
            fx_half = np.abs(colebrook_f(x_half, aa[worse], bb[worse]))
            better_half = fx_half < fx_new[worse]
            x_new[worse] = np.where(better_half, x_half, x_new[worse])

        x[idx] = x_new
        iters_used[idx] = k

        fx_now = np.abs(colebrook_f(x_new, aa, bb))
        done = fx_now <= tol
        converged[idx[done]] = True
        active[idx[done]] = False

    return x, iters_used, converged


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
        return self.net(x).squeeze(-1)


def infer_hidden_dims_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, List[int]]:
    linear_weights = []
    for k, v in state_dict.items():
        if k.endswith(".weight") and v.ndim == 2:
            linear_weights.append((k, v.shape))
    linear_weights = sorted(linear_weights, key=lambda kv: kv[0])

    if not linear_weights:
        raise RuntimeError("No linear layers found in state_dict")

    dims = []
    for _, shape in linear_weights:
        out_dim, in_dim = shape
        dims.append((in_dim, out_dim))

    input_dim = dims[0][0]
    hidden_dims = [out_dim for (_, out_dim) in dims[:-1]]
    return input_dim, hidden_dims


def load_model_checkpoint(path: str, device="cpu"):
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise RuntimeError("Unsupported checkpoint format")

    if isinstance(ckpt, dict) and "hidden_dims" in ckpt:
        hidden_dims = list(ckpt["hidden_dims"])
        input_dim = None
        for k, v in state_dict.items():
            if k.endswith(".weight") and v.ndim == 2:
                input_dim = int(v.shape[1])
                break
        if input_dim is None:
            input_dim, _ = infer_hidden_dims_from_state_dict(state_dict)
    else:
        input_dim, hidden_dims = infer_hidden_dims_from_state_dict(state_dict)

    dropout = 0.0
    if isinstance(ckpt, dict) and "args" in ckpt and isinstance(ckpt["args"], dict):
        dropout = float(ckpt["args"].get("dropout", 0.0))

    model = MLPRegressor(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return ckpt, model


def load_npz(path: str):
    data = np.load(path, allow_pickle=True)
    available = list(data.keys())

    for k in ["coeffs", "a", "b", "root"]:
        if k not in data:
            raise KeyError(f"Missing key '{k}'; available: {available}")

    if "x0" in data:
        x0_arr = np.asarray(data["x0"], dtype=np.float64).reshape(-1)
    elif "center" in data:
        x0_arr = np.asarray(data["center"], dtype=np.float64).reshape(-1)
    else:
        raise KeyError(f"Missing key 'x0' (or alias 'center'); available: {available}")

    return {
        "coeffs": np.asarray(data["coeffs"], dtype=np.float64),
        "x0": x0_arr,
        "a": np.asarray(data["a"], dtype=np.float64).reshape(-1),
        "b": np.asarray(data["b"], dtype=np.float64).reshape(-1),
        "root": np.asarray(data["root"], dtype=np.float64).reshape(-1),
    }


def build_features_from_checkpoint_args(coeffs, x0, a, b, ckpt_args):
    use_log_ab = bool(ckpt_args.get("use_log_ab", False))
    include_center = bool(ckpt_args.get("include_center", True))
    include_ab = bool(ckpt_args.get("include_ab", True))

    feats = [coeffs.astype(np.float32)]

    if include_center:
        feats.append(x0.reshape(-1, 1).astype(np.float32))

    if include_ab:
        if use_log_ab:
            feats.append(np.log(a).reshape(-1, 1).astype(np.float32))
            feats.append(np.log(b).reshape(-1, 1).astype(np.float32))
        else:
            feats.append(a.reshape(-1, 1).astype(np.float32))
            feats.append(b.reshape(-1, 1).astype(np.float32))

    return np.concatenate(feats, axis=1).astype(np.float32)


def apply_scaler(X, scaler_dict):
    mean = np.array(scaler_dict["mean"], dtype=np.float32)
    std = np.array(scaler_dict["std"], dtype=np.float32)
    return (X - mean) / std


def predict_with_checkpoint(coeffs, x0, a, b, ckpt, model, device="cpu", batch_size=4096):
    X = build_features_from_checkpoint_args(coeffs, x0, a, b, ckpt["args"])
    X = apply_scaler(X, ckpt["scaler"])

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).to(device)
            yb = model(xb).detach().cpu().numpy().reshape(-1)
            preds.append(yb)

    return np.concatenate(preds, axis=0).astype(np.float64)


def compute_basin_ratio_for_sample(
    a: float,
    b: float,
    true_root: float,
    radius: float = 1.0,
    grid_points: int = 81,
    tol: float = 1e-12,
    max_iter: int = 20,
):
    grid = np.linspace(true_root - radius, true_root + radius, grid_points)
    aa = np.full(grid_points, a, dtype=np.float64)
    bb = np.full(grid_points, b, dtype=np.float64)

    ref, _, _ = newton_refine_original_vectorized(grid, aa, bb, max_iter=max_iter, tol=tol)
    conv_to_true = np.abs(ref - true_root) <= 1e-6
    return float(np.mean(conv_to_true))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_npz", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--grid_points", type=int, default=81)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max_newton_iter", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_npz(args.test_npz)
    ckpt, model = load_model_checkpoint(args.model, device=args.device)

    coeffs = data["coeffs"]
    x0 = data["x0"]
    a = data["a"]
    b = data["b"]
    root = data["root"]

    pred_all = predict_with_checkpoint(coeffs, x0, a, b, ckpt, model, device=args.device)

    n = len(root)
    k = min(args.num_samples, n)
    selected = np.sort(rng.choice(np.arange(n), size=k, replace=False))

    rows = []
    for rank, idx in enumerate(selected, start=1):
        basin_ratio = compute_basin_ratio_for_sample(
            a=float(a[idx]),
            b=float(b[idx]),
            true_root=float(root[idx]),
            radius=args.radius,
            grid_points=args.grid_points,
            tol=args.tol,
            max_iter=args.max_newton_iter,
        )

        pred = float(pred_all[idx])
        true_r = float(root[idx])

        ref, _, _ = newton_refine_original_vectorized(
            np.array([pred], dtype=np.float64),
            np.array([a[idx]], dtype=np.float64),
            np.array([b[idx]], dtype=np.float64),
            max_iter=args.max_newton_iter,
            tol=args.tol,
        )
        inside = bool(np.abs(ref[0] - true_r) <= 1e-6)

        rows.append({
            "Sample ID": rank,
            "Original Index": int(idx),
            "True Root": true_r,
            "Predicted Initial Value": pred,
            "Basin Ratio": basin_ratio,
            "Is Prediction Inside Basin?": "Yes" if inside else "No",
            "Absolute Init Error": abs(pred - true_r),
            "a": float(a[idx]),
            "b": float(b[idx]),
        })

    csv_path = out_dir / "table3_basin_samples.csv"
    save_csv(csv_path, rows)

    txt_path = out_dir / "table3_basin_samples.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Table III candidate samples (randomly selected)\n")
        f.write("=" * 90 + "\n")
        f.write(f"test_npz: {args.test_npz}\n")
        f.write(f"model: {args.model}\n")
        f.write(f"num_samples: {k}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"radius: {args.radius}\n")
        f.write(f"grid_points: {args.grid_points}\n")
        f.write("=" * 90 + "\n\n")
        for row in rows:
            f.write(
                f"Sample ID={row['Sample ID']} | "
                f"Original Index={row['Original Index']} | "
                f"True Root={row['True Root']:.12f} | "
                f"Predicted Initial Value={row['Predicted Initial Value']:.12f} | "
                f"Basin Ratio={row['Basin Ratio']:.6f} | "
                f"Inside Basin={row['Is Prediction Inside Basin?']} | "
                f"Abs Init Error={row['Absolute Init Error']:.12f} | "
                f"a={row['a']:.12f} | b={row['b']:.12f}\n"
            )

    print(f"[DONE] Saved to: {out_dir.resolve()}")
    print(f"  - {csv_path.name}")
    print(f"  - {txt_path.name}")
    print("\nPreview:")
    for row in rows:
        print(
            f"{row['Sample ID']}, "
            f"{row['True Root']:.6f}, "
            f"{row['Predicted Initial Value']:.6f}, "
            f"{row['Basin Ratio']:.3f}, "
            f"{row['Is Prediction Inside Basin?']}"
        )


if __name__ == "__main__":
    main()
