#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_best_model_tables_with_baseline.py

목적:
- trial_001_gru.json, trial_002_lstm.json ... 전체 trial JSON을 읽는다.
- 모델 재훈련 없음.
- 모델 checkpoint 재평가 없음.
- test_npz에서 heuristic baseline direct / baseline + Newton만 계산한다.
- 최종 output은 다음 표 형태로 저장한다.

1) best_direct_table_with_baseline.csv
   Heuristic baseline
   MLP
   LSTM
   GRU
   Transformer

2) best_newton_table_with_baseline.csv
   Heuristic baseline + Newton
   MLP + Newton
   LSTM + Newton
   GRU + Newton
   Transformer + Newton

지원 baseline:
- fixed     : 모든 sample에 동일한 x0 사용, 기본 fixed_x0=1.0
- one_step  : Colebrook fixed-point one-step heuristic, x0 = -2 log10(a + b)
- a_over_b  : scale heuristic, x0 = a / b
- b_over_a  : scale heuristic, x0 = b / a
- key       : NPZ 안의 특정 key 사용
- center    : NPZ 안의 center 사용
- auto      : xbase, baseline, x0 등 자동 탐색

주의:
- root는 ground-truth이므로 baseline_key로 사용하면 안 됨.
- center도 Taylor expansion center라 root 정보가 섞였을 수 있으므로 논문용 baseline으로는 주의.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# =========================================================
# Basic utils
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
    f(x; a,b) = x + 2 log10(a + b x)
    """
    x = np.asarray(x, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    inside = a + b * x
    out = np.full_like(x, np.nan, dtype=np.float64)

    mask = inside > 0
    out[mask] = x[mask] + 2.0 * np.log10(inside[mask])

    return out


def to_float_safe(x: Any, default=np.nan) -> float:
    try:
        if x is None:
            return default
        if x == "":
            return default
        if x == "inf":
            return float("inf")
        if x == "-inf":
            return float("-inf")
        return float(x)
    except Exception:
        return default


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
            clean = {}
            for k in keys:
                v = row.get(k, "")
                if isinstance(v, (dict, list)):
                    clean[k] = json.dumps(v, ensure_ascii=False)
                elif v is None:
                    clean[k] = ""
                else:
                    clean[k] = v
            writer.writerow(clean)


def detect_key(data: np.lib.npyio.NpzFile, candidates: List[str]) -> Optional[str]:
    keys = set(data.files)
    for k in candidates:
        if k in keys:
            return k
    return None


def clip_initial_x(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=1.0, posinf=upper, neginf=lower)
    x = np.clip(x, lower, upper)
    return x.astype(np.float64)


# =========================================================
# Load test NPZ
# =========================================================
def load_test_npz(test_npz: str, target_key: Optional[str] = None):
    data = np.load(test_npz)

    print("\n[NPZ KEYS]")
    print(data.files)

    if target_key is not None:
        y_key = target_key
    else:
        y_key = detect_key(
            data,
            ["root", "y", "target", "x_star", "x_true", "solution"],
        )

    if y_key is None:
        raise KeyError(
            "target/root key를 찾지 못했습니다. "
            "가능한 key 이름: root, y, target, x_star, x_true, solution"
        )

    a_key = detect_key(data, ["a", "A"])
    b_key = detect_key(data, ["b", "B"])

    if a_key is None or b_key is None:
        raise KeyError("test_npz 안에 a, b key가 필요합니다.")

    y_true = sanitize_array(data[y_key]).reshape(-1)
    a = sanitize_array(data[a_key]).reshape(-1)
    b = sanitize_array(data[b_key]).reshape(-1)

    print("\n[LOAD TEST]")
    print(f"target key = {y_key}")
    print(f"a key      = {a_key}")
    print(f"b key      = {b_key}")
    print(f"N test     = {len(y_true)}")

    return data, y_true, a, b


# =========================================================
# Baseline
# =========================================================
def make_baseline_prediction(
    data: np.lib.npyio.NpzFile,
    method: str = "one_step",
    baseline_key: Optional[str] = None,
    fixed_x0: float = 1.0,
    init_clip_min: float = 0.0,
    init_clip_max: float = 50.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    baseline initializer 생성.

    method:
    - fixed:
        x0 = fixed_x0

    - one_step:
        Colebrook fixed-point relation
            x = -2 log10(a + b x)
        에서 x=1을 한 번 대입:
            x0 = -2 log10(a + b)
        root/center를 사용하지 않는 nonlearned heuristic.

    - a_over_b:
        a와 bx의 scale balance a ≈ bx에서 나온 x0 = a/b.
        단, root approximation으로 항상 좋은 것은 아니므로 test용.

    - b_over_a:
        x0 = b/a. 대부분 x scale로는 덜 자연스럽지만 비교용.

    - key:
        NPZ 안의 baseline_key 사용.
        주의: baseline_key=root는 금지해야 함.

    - center:
        NPZ 안의 center 사용.
        주의: center가 root 기반이면 data leakage 가능성 있음.

    - auto:
        xbase, x_base, baseline, x0 등 자동 탐색.
        center는 자동 후보에서 제외했다.
    """
    keys = set(data.files)

    if "a" not in keys or "b" not in keys:
        raise KeyError("baseline 계산을 위해 NPZ에 a, b key가 필요합니다.")

    a = sanitize_array(data["a"]).reshape(-1)
    b = sanitize_array(data["b"]).reshape(-1)
    n = len(a)

    # -----------------------------------------------------
    # Fixed x0
    # -----------------------------------------------------
    if method == "fixed":
        print(f"[BASELINE] using fixed x0={fixed_x0}")
        x0 = np.full(shape=(n,), fill_value=fixed_x0, dtype=np.float64)
        return clip_initial_x(x0, init_clip_min, init_clip_max)

    # -----------------------------------------------------
    # Colebrook fixed-point one-step heuristic
    # x0 = -2 log10(a + b)
    # -----------------------------------------------------
    if method == "one_step":
        inside = a + b * fixed_x0
        inside = np.clip(inside, eps, None)

        x0 = -2.0 * np.log10(inside)
        x0 = clip_initial_x(x0, init_clip_min, init_clip_max)

        print(
            "[BASELINE] using one_step heuristic: "
            f"x0 = -2 log10(a + b * {fixed_x0}), "
            f"clip=[{init_clip_min}, {init_clip_max}]"
        )
        return x0

    # -----------------------------------------------------
    # a / b scale heuristic
    # -----------------------------------------------------
    if method == "a_over_b":
        denom = np.where(np.abs(b) < eps, np.sign(b) * eps + (b == 0) * eps, b)
        x0 = a / denom
        x0 = clip_initial_x(x0, init_clip_min, init_clip_max)

        print(
            "[BASELINE] using scale heuristic: x0 = a / b, "
            f"clip=[{init_clip_min}, {init_clip_max}]"
        )
        return x0

    # -----------------------------------------------------
    # b / a scale heuristic
    # -----------------------------------------------------
    if method == "b_over_a":
        denom = np.where(np.abs(a) < eps, np.sign(a) * eps + (a == 0) * eps, a)
        x0 = b / denom
        x0 = clip_initial_x(x0, init_clip_min, init_clip_max)

        print(
            "[BASELINE] using scale heuristic: x0 = b / a, "
            f"clip=[{init_clip_min}, {init_clip_max}]"
        )
        return x0

    # -----------------------------------------------------
    # User-given key
    # -----------------------------------------------------
    if method == "key":
        if baseline_key is None:
            raise ValueError("--baseline_key가 필요합니다.")
        if baseline_key == "root":
            raise ValueError(
                "baseline_key='root'는 ground-truth label이므로 baseline으로 사용할 수 없습니다."
            )
        if baseline_key not in keys:
            raise KeyError(f"baseline_key={baseline_key}가 NPZ에 없습니다.")
        print(f"[BASELINE] using key='{baseline_key}'")
        x0 = sanitize_array(data[baseline_key]).reshape(-1)
        return clip_initial_x(x0, init_clip_min, init_clip_max)

    # -----------------------------------------------------
    # Center key
    # -----------------------------------------------------
    if method == "center":
        if "center" not in keys:
            raise KeyError("method=center를 사용하려면 NPZ에 center key가 있어야 합니다.")
        print(
            "[BASELINE] using key='center' "
            "(주의: center가 root 기반이면 논문용 baseline으로 부적절할 수 있음)"
        )
        x0 = sanitize_array(data["center"]).reshape(-1)
        return clip_initial_x(x0, init_clip_min, init_clip_max)

    # -----------------------------------------------------
    # Auto key detection
    # -----------------------------------------------------
    if method == "auto":
        if baseline_key is not None:
            if baseline_key == "root":
                raise ValueError(
                    "baseline_key='root'는 ground-truth label이므로 baseline으로 사용할 수 없습니다."
                )
            if baseline_key not in keys:
                raise KeyError(f"baseline_key={baseline_key}가 NPZ에 없습니다.")
            print(f"[BASELINE] using user baseline_key='{baseline_key}'")
            x0 = sanitize_array(data[baseline_key]).reshape(-1)
            return clip_initial_x(x0, init_clip_min, init_clip_max)

        candidate_keys = [
            "xbase",
            "x_base",
            "baseline",
            "baseline_x",
            "x0_base",
            "x_init",
            "init",
            "initial",
            "x0",
            # "center"는 자동 후보에서 제외.
            # center는 Taylor expansion center라 data leakage 가능성이 있음.
        ]

        found = detect_key(data, candidate_keys)

        if found is None:
            raise KeyError(
                "baseline key를 자동으로 찾지 못했습니다. "
                "NPZ에 xbase/x_base/baseline/x0 중 하나가 필요하거나 "
                "--baseline_method fixed/one_step/a_over_b 등을 사용하세요."
            )

        print(f"[BASELINE] auto-detected key='{found}'")
        x0 = sanitize_array(data[found]).reshape(-1)
        return clip_initial_x(x0, init_clip_min, init_clip_max)

    raise ValueError(f"Unknown baseline method: {method}")


def newton_refine_batch(
    x0: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 20,
    step_clip: float = 5.0,
):
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

            step = fx / dfx
            step = float(np.clip(step, -step_clip, step_clip))

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


def summarize_prediction(
    pred: np.ndarray,
    true: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    iters: Optional[np.ndarray] = None,
    conv: Optional[np.ndarray] = None,
):
    pred = sanitize_array(pred).reshape(-1)
    true = sanitize_array(true).reshape(-1)
    a = sanitize_array(a).reshape(-1)
    b = sanitize_array(b).reshape(-1)

    abs_err = np.abs(true - pred)

    inside = a + b * pred
    valid_mask = inside > 0

    residual_all = np.full_like(pred, np.inf, dtype=np.float64)

    if np.any(valid_mask):
        fvals = colebrook_like_F_np(pred[valid_mask], a[valid_mask], b[valid_mask])
        residual_all[valid_mask] = np.abs(fvals)

    finite_mask = np.isfinite(residual_all)

    if np.any(finite_mask):
        residual = residual_all[finite_mask]
        residual_mean = float(np.mean(residual))
        residual_median = float(np.median(residual))
        residual_p90 = float(np.percentile(residual, 90))
    else:
        residual_mean = float("inf")
        residual_median = float("inf")
        residual_p90 = float("inf")

    row = {
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean((true - pred) ** 2))),
        "r2": float(r2_score_np(true, pred)),
        "valid_ratio": float(np.mean(valid_mask)),
        "residual_mean": residual_mean,
        "residual_median": residual_median,
        "residual_p90": residual_p90,
        "max_abs_error": float(np.max(abs_err)),
    }

    if iters is not None:
        row["newton_iter_mean"] = float(np.mean(iters))
        row["newton_iter_median"] = float(np.median(iters))
        row["newton_iter_p90"] = float(np.percentile(iters, 90))

    if conv is not None:
        row["newton_converged_ratio"] = float(np.mean(conv))

    return row


def make_baseline_row(
    y_true: np.ndarray,
    x_base: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    tol: float,
    max_newton_iter: int,
    step_clip: float,
    baseline_method: str,
):
    direct = summarize_prediction(
        pred=x_base,
        true=y_true,
        a=a,
        b=b,
    )

    x_newton, iters, conv = newton_refine_batch(
        x0=x_base,
        a=a,
        b=b,
        tol=tol,
        max_iter=max_newton_iter,
        step_clip=step_clip,
    )

    plus = summarize_prediction(
        pred=x_newton,
        true=y_true,
        a=a,
        b=b,
        iters=iters,
        conv=conv,
    )

    row = {
        "trial_id": 0,
        "trial_name": f"trial_000_{baseline_method}_baseline",
        "model": "heuristic_baseline",
        "baseline_method": baseline_method,
        "best_epoch": None,
        "elapsed_sec": None,

        "direct_mae": direct["mae"],
        "direct_rmse": direct["rmse"],
        "direct_r2": direct["r2"],
        "direct_valid_ratio": direct["valid_ratio"],
        "direct_residual_mean": direct["residual_mean"],
        "direct_residual_median": direct["residual_median"],
        "direct_residual_p90": direct["residual_p90"],

        "plus_newton_mae": plus["mae"],
        "plus_newton_rmse": plus["rmse"],
        "plus_newton_r2": plus["r2"],
        "plus_newton_valid_ratio": plus["valid_ratio"],
        "plus_newton_residual_mean": plus["residual_mean"],
        "plus_newton_residual_median": plus["residual_median"],
        "plus_newton_residual_p90": plus["residual_p90"],
        "plus_newton_newton_iter_mean": plus["newton_iter_mean"],
        "plus_newton_newton_iter_median": plus["newton_iter_median"],
        "plus_newton_newton_iter_p90": plus["newton_iter_p90"],
        "plus_newton_converged_ratio": plus["newton_converged_ratio"],

        "hp_use_log_ab": None,
        "hp_optimizer": None,
        "hp_criterion": None,
        "hp_dropout": None,
        "hp_lr": None,
        "hp_weight_decay": None,
        "hp_hidden_dims": None,
        "hp_hidden_size": None,
        "hp_num_layers": None,
        "hp_head_hidden": None,
        "hp_head_layers": None,
        "hp_d_model": None,
        "hp_nhead": None,
        "hp_ff_dim": None,
        "hp_use_cls_token": None,

        "is_baseline": True,
    }

    return row


# =========================================================
# Trial JSON
# =========================================================
def parse_trial_sort_key(path: Path):
    m = re.search(r"trial_(\d+)_", path.name)
    if m:
        return int(m.group(1))
    return 10**18


def load_trial_jsons(trial_json_dir: Path, pattern: str) -> List[Dict]:
    paths = sorted(trial_json_dir.glob(pattern), key=parse_trial_sort_key)

    rows = []

    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                row = json.load(f)
        except Exception as e:
            print(f"[WARN] failed to read {p}: {e}")
            continue

        if not isinstance(row, dict):
            continue

        if "model" not in row:
            continue

        row["is_baseline"] = False
        row["json_path"] = str(p)

        rows.append(row)

    if not rows:
        raise RuntimeError(f"No trial json files found: {trial_json_dir}/{pattern}")

    print("\n[TRIAL JSON]")
    print(f"dir     = {trial_json_dir}")
    print(f"pattern = {pattern}")
    print(f"loaded  = {len(rows)} files")

    return rows


# =========================================================
# Best selection
# =========================================================
def is_lower_better(metric: str) -> bool:
    higher_better = {
        "direct_r2",
        "plus_newton_r2",
        "plus_newton_converged_ratio",
        "direct_valid_ratio",
        "plus_newton_valid_ratio",
    }
    return metric not in higher_better


def select_best_by_model(rows: List[Dict], metric: str) -> List[Dict]:
    lower = is_lower_better(metric)
    best = {}

    for row in rows:
        model = str(row.get("model", "unknown"))
        value = to_float_safe(row.get(metric), default=np.nan)

        if not np.isfinite(value):
            continue

        if model not in best:
            best[model] = row
            continue

        old = to_float_safe(best[model].get(metric), default=np.nan)

        if lower:
            if value < old:
                best[model] = row
        else:
            if value > old:
                best[model] = row

    order = [
        "heuristic_baseline",
        "mlp",
        "lstm",
        "gru",
        "transformer",
    ]

    selected = []
    for m in order:
        if m in best:
            selected.append(best[m])

    for m, row in best.items():
        if m not in order:
            selected.append(row)

    return selected


def baseline_display_name(row: Dict, with_newton: bool = False) -> str:
    method = str(row.get("baseline_method", "heuristic"))

    name_map = {
        "fixed": "Fixed baseline",
        "one_step": "One-step Colebrook heuristic",
        "a_over_b": "Scale heuristic (a/b)",
        "b_over_a": "Scale heuristic (b/a)",
        "center": "Center baseline",
        "key": "Key baseline",
        "auto": "Auto-detected baseline",
    }

    base = name_map.get(method, "Heuristic baseline")

    if with_newton:
        return base + " + Newton"
    return base


def make_direct_table(best_rows: List[Dict]) -> List[Dict]:
    name_map = {
        "mlp": "MLP",
        "lstm": "LSTM",
        "gru": "GRU",
        "transformer": "Transformer",
    }

    table = []

    for row in best_rows:
        model = str(row.get("model", "unknown"))

        if model == "heuristic_baseline":
            model_name = baseline_display_name(row, with_newton=False)
        else:
            model_name = name_map.get(model, model)

        table.append({
            "Model": model_name,
            "Best Trial": row.get("trial_name"),
            "MAE": row.get("direct_mae"),
            "RMSE": row.get("direct_rmse"),
            "R2": row.get("direct_r2"),
            "Residual Mean": row.get("direct_residual_mean"),
            "Residual Median": row.get("direct_residual_median"),
            "Residual p90": row.get("direct_residual_p90"),
            "Valid Ratio": row.get("direct_valid_ratio"),
            "Selected By": "direct table best",
        })

    return table


def make_newton_table(best_rows: List[Dict]) -> List[Dict]:
    name_map = {
        "mlp": "MLP + Newton",
        "lstm": "LSTM + Newton",
        "gru": "GRU + Newton",
        "transformer": "Transformer + Newton",
    }

    table = []

    for row in best_rows:
        model = str(row.get("model", "unknown"))

        if model == "heuristic_baseline":
            model_name = baseline_display_name(row, with_newton=True)
        else:
            model_name = name_map.get(model, model + " + Newton")

        table.append({
            "Model": model_name,
            "Best Trial": row.get("trial_name"),
            "Final MAE": row.get("plus_newton_mae"),
            "Final RMSE": row.get("plus_newton_rmse"),
            "R2": row.get("plus_newton_r2"),
            "Residual Mean": row.get("plus_newton_residual_mean"),
            "Residual Median": row.get("plus_newton_residual_median"),
            "Residual p90": row.get("plus_newton_residual_p90"),
            "Iter. Mean": row.get("plus_newton_newton_iter_mean"),
            "Iter. Median": row.get("plus_newton_newton_iter_median"),
            "Iter. p90": row.get("plus_newton_newton_iter_p90"),
            "Converged Ratio": row.get("plus_newton_converged_ratio"),
            "Selected By": "newton table best",
        })

    return table


def print_table(title: str, rows: List[Dict]):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    if not rows:
        print("(empty)")
        return

    keys = list(rows[0].keys())

    print(" | ".join(keys))
    print("-" * 100)

    for row in rows:
        vals = []
        for k in keys:
            v = row.get(k, "")
            if isinstance(v, float):
                if abs(v) >= 1e4 or (abs(v) > 0 and abs(v) < 1e-4):
                    vals.append(f"{v:.6e}")
                else:
                    vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        print(" | ".join(vals))


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_npz", type=str, required=True)
    parser.add_argument("--trial_json_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--json_pattern", type=str, default="trial_*.json")
    parser.add_argument("--target_key", type=str, default=None)

    parser.add_argument(
        "--baseline_method",
        type=str,
        default="one_step",
        choices=[
            "auto",
            "key",
            "center",
            "fixed",
            "one_step",
            "a_over_b",
            "b_over_a",
        ],
    )
    parser.add_argument("--baseline_key", type=str, default=None)
    parser.add_argument(
        "--fixed_x0",
        type=float,
        default=1.0,
        help=(
            "fixed baseline에서는 x0 값. "
            "one_step에서는 x0 = -2 log10(a + b * fixed_x0)의 내부 기준값."
        ),
    )
    parser.add_argument("--init_clip_min", type=float, default=0.0)
    parser.add_argument("--init_clip_max", type=float, default=50.0)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max_newton_iter", type=int, default=20)
    parser.add_argument("--step_clip", type=float, default=5.0)

    parser.add_argument(
        "--direct_best_metric",
        type=str,
        default="direct_rmse",
        help="Direct table에서 모델별 최고 trial을 고르는 기준",
    )

    parser.add_argument(
        "--newton_best_metric",
        type=str,
        default="plus_newton_newton_iter_mean",
        help="Newton table에서 모델별 최고 trial을 고르는 기준",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load trial jsons
    trial_rows = load_trial_jsons(
        trial_json_dir=Path(args.trial_json_dir),
        pattern=args.json_pattern,
    )

    # 2. Compute baseline
    data, y_true, a, b = load_test_npz(
        test_npz=args.test_npz,
        target_key=args.target_key,
    )

    x_base = make_baseline_prediction(
        data=data,
        method=args.baseline_method,
        baseline_key=args.baseline_key,
        fixed_x0=args.fixed_x0,
        init_clip_min=args.init_clip_min,
        init_clip_max=args.init_clip_max,
    )

    if len(x_base) != len(y_true):
        raise ValueError(
            f"baseline length mismatch: len(x_base)={len(x_base)}, len(y_true)={len(y_true)}"
        )

    baseline_row = make_baseline_row(
        y_true=y_true,
        x_base=x_base,
        a=a,
        b=b,
        tol=args.tol,
        max_newton_iter=args.max_newton_iter,
        step_clip=args.step_clip,
        baseline_method=args.baseline_method,
    )

    # 3. Combine baseline + trials
    all_rows = [baseline_row] + trial_rows

    # 4. Best per model
    best_direct_rows = select_best_by_model(
        rows=all_rows,
        metric=args.direct_best_metric,
    )

    best_newton_rows = select_best_by_model(
        rows=all_rows,
        metric=args.newton_best_metric,
    )

    # 5. Paper tables
    direct_table = make_direct_table(best_direct_rows)
    newton_table = make_newton_table(best_newton_rows)

    # 6. Save outputs
    save_json(out_dir / "baseline_trial_000.json", baseline_row)
    save_json(out_dir / "all_trials_plus_baseline.json", all_rows)
    save_csv(out_dir / "all_trials_plus_baseline.csv", all_rows)

    save_json(out_dir / "best_direct_rows_raw.json", best_direct_rows)
    save_json(out_dir / "best_newton_rows_raw.json", best_newton_rows)

    save_csv(out_dir / "best_direct_table_with_baseline.csv", direct_table)
    save_csv(out_dir / "best_newton_table_with_baseline.csv", newton_table)

    save_json(out_dir / "best_direct_table_with_baseline.json", direct_table)
    save_json(out_dir / "best_newton_table_with_baseline.json", newton_table)

    save_json(out_dir / "run_config.json", {
        "test_npz": args.test_npz,
        "trial_json_dir": args.trial_json_dir,
        "json_pattern": args.json_pattern,
        "baseline_method": args.baseline_method,
        "baseline_key": args.baseline_key,
        "fixed_x0": args.fixed_x0,
        "init_clip_min": args.init_clip_min,
        "init_clip_max": args.init_clip_max,
        "tol": args.tol,
        "max_newton_iter": args.max_newton_iter,
        "step_clip": args.step_clip,
        "direct_best_metric": args.direct_best_metric,
        "newton_best_metric": args.newton_best_metric,
        "n_trial_jsons": len(trial_rows),
        "n_total_rows_with_baseline": len(all_rows),
    })

    # 7. Print final tables
    print_table(
        f"DIRECT TABLE: best per model by {args.direct_best_metric}",
        direct_table,
    )

    print_table(
        f"NEWTON TABLE: best per model by {args.newton_best_metric}",
        newton_table,
    )

    print("\n[DONE]")
    print(f"Saved: {out_dir / 'best_direct_table_with_baseline.csv'}")
    print(f"Saved: {out_dir / 'best_newton_table_with_baseline.csv'}")


if __name__ == "__main__":
    main()