#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

def percentile(x, q):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))

def save_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def colebrook_f(x, a, b):
    x = np.asarray(x, dtype=np.float64)
    z = a + b * x
    out = np.full_like(x, np.nan, dtype=np.float64)
    m = z > 0
    out[m] = x[m] + 2.0 * np.log10(z[m])
    return out

def colebrook_df(x, a, b):
    x = np.asarray(x, dtype=np.float64)
    z = a + b * x
    out = np.full_like(x, np.nan, dtype=np.float64)
    m = z > 0
    out[m] = 1.0 + 2.0 * (b[m] / (z[m] * np.log(10.0)))
    return out

def newton_refine_batch(x_init, a, b, max_iter=20, tol=1e-12):
    x = np.asarray(x_init, dtype=np.float64).copy()
    n = len(x)
    iters = np.zeros(n, dtype=np.int32)
    converged = np.zeros(n, dtype=bool)
    for t in range(max_iter):
        fx = colebrook_f(x, a, b)
        dfx = colebrook_df(x, a, b)
        done = np.isfinite(fx) & (np.abs(fx) <= tol)
        converged |= done
        active = (~converged) & np.isfinite(fx) & np.isfinite(dfx) & (np.abs(dfx) > 1e-15)
        if not np.any(active):
            break
        x[active] = x[active] - fx[active] / dfx[active]
        iters[active] = t + 1
    fx = colebrook_f(x, a, b)
    converged = np.isfinite(fx) & (np.abs(fx) <= tol)
    return x, iters, converged

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
def infer_hidden_dims(state_dict):
    ws = []
    for k, v in state_dict.items():
        if k.endswith(".weight") and v.ndim == 2:
            ws.append((k, tuple(v.shape)))
    ws = sorted(ws, key=lambda x: x[0])
    if not ws:
        raise RuntimeError("No Linear weights found in checkpoint.")
    input_dim = ws[0][1][1]
    hidden_dims = [shape[0] for _, shape in ws[:-1]]
    return input_dim, hidden_dims
def load_model_checkpoint(path, device="cpu"):
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
            input_dim, _ = infer_hidden_dims(state_dict)
    else:
        input_dim, hidden_dims = infer_hidden_dims(state_dict)

    dropout = 0.0
    if isinstance(ckpt, dict) and "args" in ckpt and isinstance(ckpt["args"], dict):
        dropout = float(ckpt["args"].get("dropout", 0.0))

    model = MLPRegressor(input_dim, hidden_dims, dropout=dropout)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return ckpt, model, {
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
    }
def load_npz(path):
    data = np.load(path, allow_pickle=True)

    available = list(data.keys())
    if "coeffs" not in data:
        raise KeyError(f"Missing key 'coeffs'; available: {available}")
    if "a" not in data:
        raise KeyError(f"Missing key 'a'; available: {available}")
    if "b" not in data:
        raise KeyError(f"Missing key 'b'; available: {available}")
    if "root" not in data:
        raise KeyError(f"Missing key 'root'; available: {available}")

    # x0 alias handling:
    # preferred: x0
    # fallback: center
    if "x0" in data:
        x0_arr = np.asarray(data["x0"], dtype=np.float64).reshape(-1)
        x0_source = "x0"
    elif "center" in data:
        x0_arr = np.asarray(data["center"], dtype=np.float64).reshape(-1)
        x0_source = "center"
    else:
        raise KeyError(
            "Missing key 'x0' (or fallback alias 'center'); "
            f"available: {available}"
        )

    return {
        "coeffs": np.asarray(data["coeffs"], dtype=np.float64),
        "x0": x0_arr,
        "a": np.asarray(data["a"], dtype=np.float64).reshape(-1),
        "b": np.asarray(data["b"], dtype=np.float64).reshape(-1),
        "root": np.asarray(data["root"], dtype=np.float64).reshape(-1),
        "_x0_source": x0_source,
    }

def build_features(mode, coeffs, x0, a, b):
    if mode == "coeffs_only":
        X = coeffs
    elif mode == "coeffs_x0":
        X = np.concatenate([coeffs, x0[:, None]], axis=1)
    elif mode == "coeffs_x0_ab":
        X = np.concatenate([coeffs, x0[:, None], a[:, None], b[:, None]], axis=1)
    elif mode == "coeffs_x0_logab":
        la = np.log(np.clip(a, 1e-15, None))
        lb = np.log(np.clip(b, 1e-15, None))
        X = np.concatenate([coeffs, x0[:, None], la[:, None], lb[:, None]], axis=1)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return X.astype(np.float32)

def predict_model(model, X, device="cpu", batch_size=4096):
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).to(device)
            yb = model(xb).detach().cpu().numpy().reshape(-1)
            outs.append(yb)
    return np.concatenate(outs, axis=0)

def compute_metrics(pred, true_root, a, b):
    pred = np.asarray(pred, dtype=np.float64)
    true_root = np.asarray(true_root, dtype=np.float64)
    mae = float(np.mean(np.abs(pred - true_root)))
    rmse = float(np.sqrt(np.mean((pred - true_root) ** 2)))
    ss_res = float(np.sum((pred - true_root) ** 2))
    ss_tot = float(np.sum((true_root - np.mean(true_root)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    residual = np.abs(colebrook_f(pred, a, b))
    valid = np.isfinite(residual)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2),
        "valid_ratio": float(np.mean(valid)),
        "residual_mean": float(np.nanmean(residual)),
        "residual_median": float(np.nanmedian(residual)),
        "residual_p90": percentile(residual[np.isfinite(residual)], 90),
        "max_abs_error": float(np.max(np.abs(pred - true_root))),
    }

def baseline_zero_init(a):
    return np.zeros_like(a, dtype=np.float64)

def baseline_heuristic_init(a):
    return -2.0 * np.log10(np.clip(a, 1e-15, None))

def eval_basic(data, models, out_dir, device="cpu", newton_iter=20, tol=1e-12):
    coeffs, x0, a, b, root = data["coeffs"], data["x0"], data["a"], data["b"], data["root"]
    rows = []
    zero = baseline_zero_init(a)
    rows.append({"name": "zero_init_direct", **compute_metrics(zero, root, a, b)})
    zref, ziter, zconv = newton_refine_batch(zero, a, b, max_iter=newton_iter, tol=tol)
    rows.append({"name": "zero_init_plus_newton", **compute_metrics(zref, root, a, b),
                 "newton_iter_mean": float(np.mean(ziter)),
                 "newton_iter_median": float(np.median(ziter)),
                 "newton_iter_p90": percentile(ziter, 90),
                 "newton_converged_ratio": float(np.mean(zconv))})
    heur = baseline_heuristic_init(a)
    rows.append({"name": "heuristic_direct", **compute_metrics(heur, root, a, b)})
    href, hiter, hconv = newton_refine_batch(heur, a, b, max_iter=newton_iter, tol=tol)
    rows.append({"name": "heuristic_plus_newton", **compute_metrics(href, root, a, b),
                 "newton_iter_mean": float(np.mean(hiter)),
                 "newton_iter_median": float(np.median(hiter)),
                 "newton_iter_p90": percentile(hiter, 90),
                 "newton_converged_ratio": float(np.mean(hconv))})
    for mode, pack in models.items():
        X = build_features(mode, coeffs, x0, a, b)
        pred = predict_model(pack["model"], X, device=device)
        rows.append({"name": f"{mode}_direct", **compute_metrics(pred, root, a, b)})
        pref, piter, pconv = newton_refine_batch(pred, a, b, max_iter=newton_iter, tol=tol)
        rows.append({"name": f"{mode}_plus_newton", **compute_metrics(pref, root, a, b),
                     "newton_iter_mean": float(np.mean(piter)),
                     "newton_iter_median": float(np.median(piter)),
                     "newton_iter_p90": percentile(piter, 90),
                     "newton_converged_ratio": float(np.mean(pconv))})
    save_csv(out_dir / "basic_summary.csv", rows)

def eval_newton_budget(data, models, out_dir, device="cpu", max_budget=10, tol=1e-12):
    coeffs, x0, a, b, root = data["coeffs"], data["x0"], data["a"], data["b"], data["root"]
    rows = []
    inits = {"zero_init": baseline_zero_init(a), "heuristic": baseline_heuristic_init(a)}
    for mode, pack in models.items():
        X = build_features(mode, coeffs, x0, a, b)
        inits[mode] = predict_model(pack["model"], X, device=device)
    for name, init in inits.items():
        for budget in range(max_budget + 1):
            if budget == 0:
                pred = init.copy()
                iters = np.zeros_like(root, dtype=np.int32)
                conv = np.isfinite(colebrook_f(pred, a, b)) & (np.abs(colebrook_f(pred, a, b)) <= tol)
            else:
                pred, iters, conv = newton_refine_batch(init, a, b, max_iter=budget, tol=tol)
            rows.append({"method": name, "newton_budget": budget,
                         **compute_metrics(pred, root, a, b),
                         "iter_mean": float(np.mean(iters)),
                         "iter_median": float(np.median(iters)),
                         "converged_ratio": float(np.mean(conv))})
    save_csv(out_dir / "newton_budget.csv", rows)

def eval_noise(data, models, out_dir, device="cpu"):
    coeffs, x0, a, b, root = data["coeffs"], data["x0"], data["a"], data["b"], data["root"]
    rows = []
    coeff_noise_list = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
    ab_noise_list = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
    coeff_std = np.std(coeffs, axis=0, keepdims=True) + 1e-12
    for mode, pack in models.items():
        for sigma in coeff_noise_list:
            noisy_coeffs = coeffs + np.random.randn(*coeffs.shape) * coeff_std * sigma
            X = build_features(mode, noisy_coeffs, x0, a, b)
            pred = predict_model(pack["model"], X, device=device)
            rows.append({"mode": mode, "noise_type": "coeff_gaussian", "noise_level": sigma,
                         **compute_metrics(pred, root, a, b)})
        for sigma in ab_noise_list:
            na = np.clip(a * (1.0 + sigma * np.random.randn(*a.shape)), 1e-12, None)
            nb = np.clip(b * (1.0 + sigma * np.random.randn(*b.shape)), 1e-12, None)
            X = build_features(mode, coeffs, x0, na, nb)
            pred = predict_model(pack["model"], X, device=device)
            nres = np.abs(colebrook_f(pred, na, nb))
            rows.append({"mode": mode, "noise_type": "ab_multiplicative", "noise_level": sigma,
                         **compute_metrics(pred, root, a, b),
                         "noisy_equation_residual_mean": float(np.nanmean(nres)),
                         "noisy_equation_residual_p90": percentile(nres[np.isfinite(nres)], 90)})
    save_csv(out_dir / "noise_robustness.csv", rows)

def make_ood_masks(data):
    a, b, root = data["a"], data["b"], data["root"]
    rq = np.abs(root)
    aq80, bq80, rq80 = np.quantile(a, 0.8), np.quantile(b, 0.8), np.quantile(rq, 0.8)
    return {
        "ID_all": np.ones_like(a, dtype=bool),
        "OOD_large_a": a >= aq80,
        "OOD_large_b": b >= bq80,
        "OOD_large_root_abs": rq >= rq80,
        "HARD_union": (a >= aq80) | (b >= bq80) | (rq >= rq80),
        "EASY_intersection": (a < np.quantile(a, 0.5)) & (b < np.quantile(b, 0.5)) & (rq < np.quantile(rq, 0.5))
    }

def eval_ood(data, models, out_dir, device="cpu", newton_iter=20, tol=1e-12):
    coeffs, x0, a, b, root = data["coeffs"], data["x0"], data["a"], data["b"], data["root"]
    masks = make_ood_masks(data)
    rows = []
    for split, mask in masks.items():
        if np.sum(mask) == 0:
            continue
        for init_name, init in {"zero_init": baseline_zero_init(a), "heuristic": baseline_heuristic_init(a)}.items():
            rows.append({"split": split, "method": f"{init_name}_direct", "n": int(np.sum(mask)),
                         **compute_metrics(init[mask], root[mask], a[mask], b[mask])})
            ref, iters, conv = newton_refine_batch(init[mask], a[mask], b[mask], max_iter=newton_iter, tol=tol)
            rows.append({"split": split, "method": f"{init_name}_plus_newton", "n": int(np.sum(mask)),
                         **compute_metrics(ref, root[mask], a[mask], b[mask]),
                         "iter_mean": float(np.mean(iters)), "iter_median": float(np.median(iters)),
                         "converged_ratio": float(np.mean(conv))})
        for mode, pack in models.items():
            X = build_features(mode, coeffs, x0, a, b)
            pred_all = predict_model(pack["model"], X, device=device)
            rows.append({"split": split, "method": f"{mode}_direct", "n": int(np.sum(mask)),
                         **compute_metrics(pred_all[mask], root[mask], a[mask], b[mask])})
            ref, iters, conv = newton_refine_batch(pred_all[mask], a[mask], b[mask], max_iter=newton_iter, tol=tol)
            rows.append({"split": split, "method": f"{mode}_plus_newton", "n": int(np.sum(mask)),
                         **compute_metrics(ref, root[mask], a[mask], b[mask]),
                         "iter_mean": float(np.mean(iters)), "iter_median": float(np.median(iters)),
                         "converged_ratio": float(np.mean(conv))})
    save_csv(out_dir / "ood_analysis.csv", rows)

def eval_basin(data, models, out_dir, device="cpu", n_cases=25, radius=1.0, grid_points=81, tol=1e-12, max_iter=20):
    a, b, root = data["a"], data["b"], data["root"]
    coeffs, x0 = data["coeffs"], data["x0"]
    idxs = np.linspace(0, len(root)-1, min(n_cases, len(root))).astype(int)
    pred_cache = {}
    for mode, pack in models.items():
        X = build_features(mode, coeffs, x0, a, b)
        pred_cache[mode] = predict_model(pack["model"], X, device=device)
    rows = []
    for idx in idxs:
        tr = root[idx]
        grid = np.linspace(tr - radius, tr + radius, grid_points)
        ai = np.full(grid_points, a[idx], dtype=np.float64)
        bi = np.full(grid_points, b[idx], dtype=np.float64)
        ref, iters, conv = newton_refine_batch(grid, ai, bi, max_iter=max_iter, tol=tol)
        conv_to_true = np.abs(ref - tr) <= 1e-6
        row = {
            "sample_index": int(idx),
            "a": float(a[idx]),
            "b": float(b[idx]),
            "true_root": float(tr),
            "basin_ratio_around_true_root": float(np.mean(conv_to_true)),
            "grid_left": float(grid[0]),
            "grid_right": float(grid[-1]),
        }
        for mode, pred_all in pred_cache.items():
            row[f"{mode}_pred"] = float(pred_all[idx])
            row[f"{mode}_abs_init_error"] = float(abs(pred_all[idx] - tr))
        rows.append(row)
    save_csv(out_dir / "basin_analysis.csv", rows)

def time_callable(fn, repeats=10000):
    t0 = time.perf_counter_ns()
    for _ in range(repeats):
        fn()
    t1 = time.perf_counter_ns()
    return (t1 - t0) / repeats / 1000.0

def eval_timing(data, models, out_dir, device="cpu", repeats=20000):
    coeffs, x0, a, b = data["coeffs"], data["x0"], data["a"], data["b"]
    idx = 0
    ai = np.array([a[idx]], dtype=np.float64)
    bi = np.array([b[idx]], dtype=np.float64)
    rows = []
    zero = baseline_zero_init(ai)
    heur = baseline_heuristic_init(ai)
    rows.append({"method": "zero_init_direct", "time_mean_us": time_callable(lambda: baseline_zero_init(ai), repeats)})
    rows.append({"method": "heuristic_direct", "time_mean_us": time_callable(lambda: baseline_heuristic_init(ai), repeats)})
    rows.append({"method": "zero_init_plus_newton", "time_mean_us": time_callable(lambda: newton_refine_batch(zero, ai, bi, 20, 1e-12), repeats)})
    rows.append({"method": "heuristic_plus_newton", "time_mean_us": time_callable(lambda: newton_refine_batch(heur, ai, bi, 20, 1e-12), repeats)})
    for mode, pack in models.items():
        X = build_features(mode, coeffs[idx:idx+1], x0[idx:idx+1], a[idx:idx+1], b[idx:idx+1])
        pred = predict_model(pack["model"], X, device=device, batch_size=1)
        rows.append({"method": f"{mode}_direct", "time_mean_us": time_callable(lambda: predict_model(pack["model"], X, device=device, batch_size=1), repeats)})
        rows.append({"method": f"{mode}_plus_newton_refine_only", "time_mean_us": time_callable(lambda: newton_refine_batch(pred, ai, bi, 20, 1e-12), repeats)})
    save_csv(out_dir / "timing_microbench.csv", rows)

def parse_model_args(items):
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--model must be mode=path, got {item}")
        k, v = item.split("=", 1)
        out[k] = v
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_npz", required=True)
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--max_newton_iter", type=int, default=20)
    ap.add_argument("--run_all", action="store_true")
    ap.add_argument("--eval_basic", action="store_true")
    ap.add_argument("--eval_newton_budget", action="store_true")
    ap.add_argument("--eval_noise", action="store_true")
    ap.add_argument("--eval_ood", action="store_true")
    ap.add_argument("--eval_basin", action="store_true")
    ap.add_argument("--eval_timing", action="store_true")
    ap.add_argument("--newton_budget_max", type=int, default=10)
    ap.add_argument("--basin_cases", type=int, default=25)
    ap.add_argument("--basin_radius", type=float, default=1.0)
    ap.add_argument("--basin_grid_points", type=int, default=81)
    ap.add_argument("--timing_repeats", type=int, default=20000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_npz(args.test_npz)

    model_paths = parse_model_args(args.model)
    models = {}
    for mode, path in model_paths.items():
        ckpt, model, meta = load_model_checkpoint(path, device=args.device)
        models[mode] = {"ckpt": ckpt, "model": model, "meta": meta, "path": path}

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"test_npz": args.test_npz, "models": model_paths}, f, ensure_ascii=False, indent=2)

    if args.run_all or args.eval_basic:
        print("[1] basic evaluation")
        eval_basic(data, models, out_dir, device=args.device, newton_iter=args.max_newton_iter, tol=args.tol)
    if args.run_all or args.eval_newton_budget:
        print("[2] newton budget")
        eval_newton_budget(data, models, out_dir, device=args.device, max_budget=args.newton_budget_max, tol=args.tol)
    if args.run_all or args.eval_noise:
        print("[3] noise robustness")
        eval_noise(data, models, out_dir, device=args.device)
    if args.run_all or args.eval_ood:
        print("[4] ood analysis")
        eval_ood(data, models, out_dir, device=args.device, newton_iter=args.max_newton_iter, tol=args.tol)
    if args.run_all or args.eval_basin:
        print("[5] basin analysis")
        eval_basin(data, models, out_dir, device=args.device, n_cases=args.basin_cases, radius=args.basin_radius,
                   grid_points=args.basin_grid_points, tol=args.tol, max_iter=args.max_newton_iter)
    if args.run_all or args.eval_timing:
        print("[6] timing")
        eval_timing(data, models, out_dir, device=args.device, repeats=args.timing_repeats)

    print(f"[DONE] saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

def percentile(x, q):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))

def save_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def colebrook_f(x, a, b):
    x = np.asarray(x, dtype=np.float64)
    z = a + b * x
    out = np.full_like(x, np.nan, dtype=np.float64)
    m = z > 0
    out[m] = x[m] + 2.0 * np.log10(z[m])
    return out

def colebrook_df(x, a, b):
    x = np.asarray(x, dtype=np.float64)
    z = a + b * x
    out = np.full_like(x, np.nan, dtype=np.float64)
    m = z > 0
    out[m] = 1.0 + 2.0 * (b[m] / (z[m] * np.log(10.0)))
    return out

def newton_refine_batch(x_init, a, b, max_iter=20, tol=1e-12):
    x = np.asarray(x_init, dtype=np.float64).copy()
    n = len(x)
    iters = np.zeros(n, dtype=np.int32)
    converged = np.zeros(n, dtype=bool)
    for t in range(max_iter):
        fx = colebrook_f(x, a, b)
        dfx = colebrook_df(x, a, b)
        done = np.isfinite(fx) & (np.abs(fx) <= tol)
        converged |= done
        active = (~converged) & np.isfinite(fx) & np.isfinite(dfx) & (np.abs(dfx) > 1e-15)
        if not np.any(active):
            break
        x[active] = x[active] - fx[active] / dfx[active]
        iters[active] = t + 1
    fx = colebrook_f(x, a, b)
    converged = np.isfinite(fx) & (np.abs(fx) <= tol)
    return x, iters, converged

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
def infer_hidden_dims(state_dict):
    ws = []
    for k, v in state_dict.items():
        if k.endswith(".weight") and v.ndim == 2:
            ws.append((k, tuple(v.shape)))
    ws = sorted(ws, key=lambda x: x[0])
    if not ws:
        raise RuntimeError("No Linear weights found in checkpoint.")
    input_dim = ws[0][1][1]
    hidden_dims = [shape[0] for _, shape in ws[:-1]]
    return input_dim, hidden_dims
def load_model_checkpoint(path, device="cpu"):
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
            input_dim, _ = infer_hidden_dims(state_dict)
    else:
        input_dim, hidden_dims = infer_hidden_dims(state_dict)

    dropout = 0.0
    if isinstance(ckpt, dict) and "args" in ckpt and isinstance(ckpt["args"], dict):
        dropout = float(ckpt["args"].get("dropout", 0.0))

    model = MLPRegressor(input_dim, hidden_dims, dropout=dropout)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return ckpt, model, {
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
    }
def load_npz(path):
    data = np.load(path, allow_pickle=True)

    available = list(data.keys())
    if "coeffs" not in data:
        raise KeyError(f"Missing key 'coeffs'; available: {available}")
    if "a" not in data:
        raise KeyError(f"Missing key 'a'; available: {available}")
    if "b" not in data:
        raise KeyError(f"Missing key 'b'; available: {available}")
    if "root" not in data:
        raise KeyError(f"Missing key 'root'; available: {available}")

    # x0 alias handling:
    # preferred: x0
    # fallback: center
    if "x0" in data:
        x0_arr = np.asarray(data["x0"], dtype=np.float64).reshape(-1)
        x0_source = "x0"
    elif "center" in data:
        x0_arr = np.asarray(data["center"], dtype=np.float64).reshape(-1)
        x0_source = "center"
    else:
        raise KeyError(
            "Missing key 'x0' (or fallback alias 'center'); "
            f"available: {available}"
        )

    return {
        "coeffs": np.asarray(data["coeffs"], dtype=np.float64),
        "x0": x0_arr,
        "a": np.asarray(data["a"], dtype=np.float64).reshape(-1),
        "b": np.asarray(data["b"], dtype=np.float64).reshape(-1),
        "root": np.asarray(data["root"], dtype=np.float64).reshape(-1),
        "_x0_source": x0_source,
    }

def build_features(mode, coeffs, x0, a, b):
    if mode == "coeffs_only":
        X = coeffs
    elif mode == "coeffs_x0":
        X = np.concatenate([coeffs, x0[:, None]], axis=1)
    elif mode == "coeffs_x0_ab":
        X = np.concatenate([coeffs, x0[:, None], a[:, None], b[:, None]], axis=1)
    elif mode == "coeffs_x0_logab":
        la = np.log(np.clip(a, 1e-15, None))
        lb = np.log(np.clip(b, 1e-15, None))
        X = np.concatenate([coeffs, x0[:, None], la[:, None], lb[:, None]], axis=1)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return X.astype(np.float32)

def predict_model(model, X, device="cpu", batch_size=4096):
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).to(device)
            yb = model(xb).detach().cpu().numpy().reshape(-1)
            outs.append(yb)
    return np.concatenate(outs, axis=0)

def compute_metrics(pred, true_root, a, b):
    pred = np.asarray(pred, dtype=np.float64)
    true_root = np.asarray(true_root, dtype=np.float64)
    mae = float(np.mean(np.abs(pred - true_root)))
    rmse = float(np.sqrt(np.mean((pred - true_root) ** 2)))
    ss_res = float(np.sum((pred - true_root) ** 2))
    ss_tot = float(np.sum((true_root - np.mean(true_root)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    residual = np.abs(colebrook_f(pred, a, b))
    valid = np.isfinite(residual)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2),
        "valid_ratio": float(np.mean(valid)),
        "residual_mean": float(np.nanmean(residual)),
        "residual_median": float(np.nanmedian(residual)),
        "residual_p90": percentile(residual[np.isfinite(residual)], 90),
        "max_abs_error": float(np.max(np.abs(pred - true_root))),
    }

def baseline_zero_init(a):
    return np.zeros_like(a, dtype=np.float64)

def baseline_heuristic_init(a):
    return -2.0 * np.log10(np.clip(a, 1e-15, None))

def eval_basic(data, models, out_dir, device="cpu", newton_iter=20, tol=1e-12):
    coeffs, x0, a, b, root = data["coeffs"], data["x0"], data["a"], data["b"], data["root"]
    rows = []
    zero = baseline_zero_init(a)
    rows.append({"name": "zero_init_direct", **compute_metrics(zero, root, a, b)})
    zref, ziter, zconv = newton_refine_batch(zero, a, b, max_iter=newton_iter, tol=tol)
    rows.append({"name": "zero_init_plus_newton", **compute_metrics(zref, root, a, b),
                 "newton_iter_mean": float(np.mean(ziter)),
                 "newton_iter_median": float(np.median(ziter)),
                 "newton_iter_p90": percentile(ziter, 90),
                 "newton_converged_ratio": float(np.mean(zconv))})
    heur = baseline_heuristic_init(a)
    rows.append({"name": "heuristic_direct", **compute_metrics(heur, root, a, b)})
    href, hiter, hconv = newton_refine_batch(heur, a, b, max_iter=newton_iter, tol=tol)
    rows.append({"name": "heuristic_plus_newton", **compute_metrics(href, root, a, b),
                 "newton_iter_mean": float(np.mean(hiter)),
                 "newton_iter_median": float(np.median(hiter)),
                 "newton_iter_p90": percentile(hiter, 90),
                 "newton_converged_ratio": float(np.mean(hconv))})
    for mode, pack in models.items():
        X = build_features(mode, coeffs, x0, a, b)
        pred = predict_model(pack["model"], X, device=device)
        rows.append({"name": f"{mode}_direct", **compute_metrics(pred, root, a, b)})
        pref, piter, pconv = newton_refine_batch(pred, a, b, max_iter=newton_iter, tol=tol)
        rows.append({"name": f"{mode}_plus_newton", **compute_metrics(pref, root, a, b),
                     "newton_iter_mean": float(np.mean(piter)),
                     "newton_iter_median": float(np.median(piter)),
                     "newton_iter_p90": percentile(piter, 90),
                     "newton_converged_ratio": float(np.mean(pconv))})
    save_csv(out_dir / "basic_summary.csv", rows)

def eval_newton_budget(data, models, out_dir, device="cpu", max_budget=10, tol=1e-12):
    coeffs, x0, a, b, root = data["coeffs"], data["x0"], data["a"], data["b"], data["root"]
    rows = []
    inits = {"zero_init": baseline_zero_init(a), "heuristic": baseline_heuristic_init(a)}
    for mode, pack in models.items():
        X = build_features(mode, coeffs, x0, a, b)
        inits[mode] = predict_model(pack["model"], X, device=device)
    for name, init in inits.items():
        for budget in range(max_budget + 1):
            if budget == 0:
                pred = init.copy()
                iters = np.zeros_like(root, dtype=np.int32)
                conv = np.isfinite(colebrook_f(pred, a, b)) & (np.abs(colebrook_f(pred, a, b)) <= tol)
            else:
                pred, iters, conv = newton_refine_batch(init, a, b, max_iter=budget, tol=tol)
            rows.append({"method": name, "newton_budget": budget,
                         **compute_metrics(pred, root, a, b),
                         "iter_mean": float(np.mean(iters)),
                         "iter_median": float(np.median(iters)),
                         "converged_ratio": float(np.mean(conv))})
    save_csv(out_dir / "newton_budget.csv", rows)

def eval_noise(data, models, out_dir, device="cpu"):
    coeffs, x0, a, b, root = data["coeffs"], data["x0"], data["a"], data["b"], data["root"]
    rows = []
    coeff_noise_list = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
    ab_noise_list = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
    coeff_std = np.std(coeffs, axis=0, keepdims=True) + 1e-12
    for mode, pack in models.items():
        for sigma in coeff_noise_list:
            noisy_coeffs = coeffs + np.random.randn(*coeffs.shape) * coeff_std * sigma
            X = build_features(mode, noisy_coeffs, x0, a, b)
            pred = predict_model(pack["model"], X, device=device)
            rows.append({"mode": mode, "noise_type": "coeff_gaussian", "noise_level": sigma,
                         **compute_metrics(pred, root, a, b)})
        for sigma in ab_noise_list:
            na = np.clip(a * (1.0 + sigma * np.random.randn(*a.shape)), 1e-12, None)
            nb = np.clip(b * (1.0 + sigma * np.random.randn(*b.shape)), 1e-12, None)
            X = build_features(mode, coeffs, x0, na, nb)
            pred = predict_model(pack["model"], X, device=device)
            nres = np.abs(colebrook_f(pred, na, nb))
            rows.append({"mode": mode, "noise_type": "ab_multiplicative", "noise_level": sigma,
                         **compute_metrics(pred, root, a, b),
                         "noisy_equation_residual_mean": float(np.nanmean(nres)),
                         "noisy_equation_residual_p90": percentile(nres[np.isfinite(nres)], 90)})
    save_csv(out_dir / "noise_robustness.csv", rows)

def make_ood_masks(data):
    a, b, root = data["a"], data["b"], data["root"]
    rq = np.abs(root)
    aq80, bq80, rq80 = np.quantile(a, 0.8), np.quantile(b, 0.8), np.quantile(rq, 0.8)
    return {
        "ID_all": np.ones_like(a, dtype=bool),
        "OOD_large_a": a >= aq80,
        "OOD_large_b": b >= bq80,
        "OOD_large_root_abs": rq >= rq80,
        "HARD_union": (a >= aq80) | (b >= bq80) | (rq >= rq80),
        "EASY_intersection": (a < np.quantile(a, 0.5)) & (b < np.quantile(b, 0.5)) & (rq < np.quantile(rq, 0.5))
    }

def eval_ood(data, models, out_dir, device="cpu", newton_iter=20, tol=1e-12):
    coeffs, x0, a, b, root = data["coeffs"], data["x0"], data["a"], data["b"], data["root"]
    masks = make_ood_masks(data)
    rows = []
    for split, mask in masks.items():
        if np.sum(mask) == 0:
            continue
        for init_name, init in {"zero_init": baseline_zero_init(a), "heuristic": baseline_heuristic_init(a)}.items():
            rows.append({"split": split, "method": f"{init_name}_direct", "n": int(np.sum(mask)),
                         **compute_metrics(init[mask], root[mask], a[mask], b[mask])})
            ref, iters, conv = newton_refine_batch(init[mask], a[mask], b[mask], max_iter=newton_iter, tol=tol)
            rows.append({"split": split, "method": f"{init_name}_plus_newton", "n": int(np.sum(mask)),
                         **compute_metrics(ref, root[mask], a[mask], b[mask]),
                         "iter_mean": float(np.mean(iters)), "iter_median": float(np.median(iters)),
                         "converged_ratio": float(np.mean(conv))})
        for mode, pack in models.items():
            X = build_features(mode, coeffs, x0, a, b)
            pred_all = predict_model(pack["model"], X, device=device)
            rows.append({"split": split, "method": f"{mode}_direct", "n": int(np.sum(mask)),
                         **compute_metrics(pred_all[mask], root[mask], a[mask], b[mask])})
            ref, iters, conv = newton_refine_batch(pred_all[mask], a[mask], b[mask], max_iter=newton_iter, tol=tol)
            rows.append({"split": split, "method": f"{mode}_plus_newton", "n": int(np.sum(mask)),
                         **compute_metrics(ref, root[mask], a[mask], b[mask]),
                         "iter_mean": float(np.mean(iters)), "iter_median": float(np.median(iters)),
                         "converged_ratio": float(np.mean(conv))})
    save_csv(out_dir / "ood_analysis.csv", rows)

def eval_basin(data, models, out_dir, device="cpu", n_cases=25, radius=1.0, grid_points=81, tol=1e-12, max_iter=20):
    a, b, root = data["a"], data["b"], data["root"]
    coeffs, x0 = data["coeffs"], data["x0"]
    idxs = np.linspace(0, len(root)-1, min(n_cases, len(root))).astype(int)
    pred_cache = {}
    for mode, pack in models.items():
        X = build_features(mode, coeffs, x0, a, b)
        pred_cache[mode] = predict_model(pack["model"], X, device=device)
    rows = []
    for idx in idxs:
        tr = root[idx]
        grid = np.linspace(tr - radius, tr + radius, grid_points)
        ai = np.full(grid_points, a[idx], dtype=np.float64)
        bi = np.full(grid_points, b[idx], dtype=np.float64)
        ref, iters, conv = newton_refine_batch(grid, ai, bi, max_iter=max_iter, tol=tol)
        conv_to_true = np.abs(ref - tr) <= 1e-6
        row = {
            "sample_index": int(idx),
            "a": float(a[idx]),
            "b": float(b[idx]),
            "true_root": float(tr),
            "basin_ratio_around_true_root": float(np.mean(conv_to_true)),
            "grid_left": float(grid[0]),
            "grid_right": float(grid[-1]),
        }
        for mode, pred_all in pred_cache.items():
            row[f"{mode}_pred"] = float(pred_all[idx])
            row[f"{mode}_abs_init_error"] = float(abs(pred_all[idx] - tr))
        rows.append(row)
    save_csv(out_dir / "basin_analysis.csv", rows)

def time_callable(fn, repeats=10000):
    t0 = time.perf_counter_ns()
    for _ in range(repeats):
        fn()
    t1 = time.perf_counter_ns()
    return (t1 - t0) / repeats / 1000.0

def eval_timing(data, models, out_dir, device="cpu", repeats=20000):
    coeffs, x0, a, b = data["coeffs"], data["x0"], data["a"], data["b"]
    idx = 0
    ai = np.array([a[idx]], dtype=np.float64)
    bi = np.array([b[idx]], dtype=np.float64)
    rows = []
    zero = baseline_zero_init(ai)
    heur = baseline_heuristic_init(ai)
    rows.append({"method": "zero_init_direct", "time_mean_us": time_callable(lambda: baseline_zero_init(ai), repeats)})
    rows.append({"method": "heuristic_direct", "time_mean_us": time_callable(lambda: baseline_heuristic_init(ai), repeats)})
    rows.append({"method": "zero_init_plus_newton", "time_mean_us": time_callable(lambda: newton_refine_batch(zero, ai, bi, 20, 1e-12), repeats)})
    rows.append({"method": "heuristic_plus_newton", "time_mean_us": time_callable(lambda: newton_refine_batch(heur, ai, bi, 20, 1e-12), repeats)})
    for mode, pack in models.items():
        X = build_features(mode, coeffs[idx:idx+1], x0[idx:idx+1], a[idx:idx+1], b[idx:idx+1])
        pred = predict_model(pack["model"], X, device=device, batch_size=1)
        rows.append({"method": f"{mode}_direct", "time_mean_us": time_callable(lambda: predict_model(pack["model"], X, device=device, batch_size=1), repeats)})
        rows.append({"method": f"{mode}_plus_newton_refine_only", "time_mean_us": time_callable(lambda: newton_refine_batch(pred, ai, bi, 20, 1e-12), repeats)})
    save_csv(out_dir / "timing_microbench.csv", rows)

def parse_model_args(items):
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--model must be mode=path, got {item}")
        k, v = item.split("=", 1)
        out[k] = v
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_npz", required=True)
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--max_newton_iter", type=int, default=20)
    ap.add_argument("--run_all", action="store_true")
    ap.add_argument("--eval_basic", action="store_true")
    ap.add_argument("--eval_newton_budget", action="store_true")
    ap.add_argument("--eval_noise", action="store_true")
    ap.add_argument("--eval_ood", action="store_true")
    ap.add_argument("--eval_basin", action="store_true")
    ap.add_argument("--eval_timing", action="store_true")
    ap.add_argument("--newton_budget_max", type=int, default=10)
    ap.add_argument("--basin_cases", type=int, default=25)
    ap.add_argument("--basin_radius", type=float, default=1.0)
    ap.add_argument("--basin_grid_points", type=int, default=81)
    ap.add_argument("--timing_repeats", type=int, default=20000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_npz(args.test_npz)

    model_paths = parse_model_args(args.model)
    models = {}
    for mode, path in model_paths.items():
        ckpt, model, meta = load_model_checkpoint(path, device=args.device)
        models[mode] = {"ckpt": ckpt, "model": model, "meta": meta, "path": path}

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"test_npz": args.test_npz, "models": model_paths}, f, ensure_ascii=False, indent=2)

    if args.run_all or args.eval_basic:
        print("[1] basic evaluation")
        eval_basic(data, models, out_dir, device=args.device, newton_iter=args.max_newton_iter, tol=args.tol)
    if args.run_all or args.eval_newton_budget:
        print("[2] newton budget")
        eval_newton_budget(data, models, out_dir, device=args.device, max_budget=args.newton_budget_max, tol=args.tol)
    if args.run_all or args.eval_noise:
        print("[3] noise robustness")
        eval_noise(data, models, out_dir, device=args.device)
    if args.run_all or args.eval_ood:
        print("[4] ood analysis")
        eval_ood(data, models, out_dir, device=args.device, newton_iter=args.max_newton_iter, tol=args.tol)
    if args.run_all or args.eval_basin:
        print("[5] basin analysis")
        eval_basin(data, models, out_dir, device=args.device, n_cases=args.basin_cases, radius=args.basin_radius,
                   grid_points=args.basin_grid_points, tol=args.tol, max_iter=args.max_newton_iter)
    if args.run_all or args.eval_timing:
        print("[6] timing")
        eval_timing(data, models, out_dir, device=args.device, repeats=args.timing_repeats)

    print(f"[DONE] saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
