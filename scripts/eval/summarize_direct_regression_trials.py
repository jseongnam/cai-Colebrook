#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize direct-regression JSON trials by Taylor degree and model.

Expected path structure:
  /root/project/dataset/math_03_14/results/{degree}degree/coupled_grid_1000ep/trial_{number}_{model}.json

Example:
  /root/project/dataset/math_03_14/results/25degree/coupled_grid_1000ep/trial_003_lstm.json

Outputs:
  - direct_regression_all_trials.csv
  - direct_regression_best_by_degree_model.csv
  - direct_regression_model_average.csv
  - direct_regression_degree_average.csv
  - direct_regression_best_by_degree_model.tex
  - direct_regression_model_average.tex
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


# ------------------------------------------------------------
# 1. Utility: flatten nested JSON
# ------------------------------------------------------------

def flatten_dict(
    obj: Dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    """
    Flatten nested dictionaries.

    Example:
      {"direct": {"rmse": 0.1}} -> {"direct.rmse": 0.1}
    """
    items = {}
    for key, value in obj.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)

        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = value

    return items


def to_float_or_none(value: Any) -> Optional[float]:
    """Convert value to float if possible."""
    if value is None:
        return None

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None

    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return None
        try:
            x = float(s)
            if math.isfinite(x):
                return x
        except ValueError:
            return None

    return None


def find_first_numeric(flat: Dict[str, Any], candidates: Iterable[str]) -> Optional[float]:
    """
    Search candidate key names in flattened JSON.
    Supports exact match first, then suffix match.
    """
    # exact match
    for key in candidates:
        if key in flat:
            x = to_float_or_none(flat[key])
            if x is not None:
                return x

    # case-insensitive exact
    lower_map = {k.lower(): k for k in flat.keys()}
    for key in candidates:
        lk = key.lower()
        if lk in lower_map:
            x = to_float_or_none(flat[lower_map[lk]])
            if x is not None:
                return x

    # suffix match: e.g., metrics.direct_rmse -> direct_rmse
    for key in candidates:
        lk = key.lower()
        for actual_key, value in flat.items():
            ak = actual_key.lower()
            if ak.endswith(lk) or ak.endswith("." + lk):
                x = to_float_or_none(value)
                if x is not None:
                    return x

    return None


# ------------------------------------------------------------
# 2. Key aliases
# ------------------------------------------------------------

ALIASES = {
    # Direct-stage metrics
    "direct_mae": [
        "direct_mae", "mae", "direct.mae", "metrics.direct_mae",
        "direct_metrics.mae",
    ],
    "direct_rmse": [
        "direct_rmse", "rmse", "direct.rmse", "metrics.direct_rmse",
        "direct_metrics.rmse",
    ],
    "direct_r2": [
        "direct_r2", "r2", "direct.r2", "direct_R2",
        "metrics.direct_r2", "direct_metrics.r2",
    ],
    "direct_valid_ratio": [
        "direct_valid_ratio", "valid_ratio", "direct.valid_ratio",
        "metrics.direct_valid_ratio",
    ],
    "direct_residual_mean": [
        "direct_residual_mean", "residual_mean", "direct.residual_mean",
        "direct_residual.mean", "metrics.direct_residual_mean",
    ],
    "direct_residual_median": [
        "direct_residual_median", "residual_median", "direct.residual_median",
        "direct_residual.median",
    ],
    "direct_residual_p90": [
        "direct_residual_p90", "residual_p90", "direct.residual_p90",
        "direct_residual.p90",
    ],
    "max_abs_error": [
        "max_abs_error", "direct_max_abs_error", "direct.max_abs_error",
    ],

    # Newton-refined metrics
    "newton_rmse": [
        "plus_newton_rmse", "newton_rmse", "neural_newton_rmse",
        "refined_rmse", "post_newton_rmse", "newton.rmse",
    ],
    "newton_mae": [
        "plus_newton_mae", "newton_mae", "neural_newton_mae",
        "refined_mae", "post_newton_mae", "newton.mae",
    ],
    "newton_residual_mean": [
        "plus_newton_residual_mean", "newton_residual_mean",
        "neural_newton_residual_mean", "refined_residual_mean",
        "post_newton_residual_mean", "newton.residual_mean",
    ],
    "newton_residual_median": [
        "plus_newton_residual_median", "newton_residual_median",
        "neural_newton_residual_median", "refined_residual_median",
        "post_newton_residual_median",
    ],
    "newton_residual_p90": [
        "plus_newton_residual_p90", "newton_residual_p90",
        "neural_newton_residual_p90", "refined_residual_p90",
        "post_newton_residual_p90",
    ],
    "newton_iter_mean": [
        "plus_newton_iter_mean", "newton_iter_mean", "neural_iter",
        "neural_iter_mean", "iter_mean", "newton.iter_mean",
        "newton_iterations_mean",
    ],
    "newton_iter_median": [
        "plus_newton_iter_median", "newton_iter_median",
        "iter_median", "newton.iter_median",
    ],
    "newton_iter_p90": [
        "plus_newton_iter_p90", "newton_iter_p90",
        "iter_p90", "newton.iter_p90",
    ],
    "newton_converged_ratio": [
        "plus_newton_converged_ratio", "newton_converged_ratio",
        "neural_conv", "neural_converged_ratio", "conv_ratio",
        "converged_ratio", "newton.converged_ratio",
    ],

    # Timing metrics, if present
    "forward_gpu_ms_per_sample": [
        "forward_gpu_ms_per_sample", "forward_gpu_ms", "forward_ms",
        "forward_time_ms_per_sample",
    ],
    "forward_copy_ms_per_sample": [
        "forward_copy_ms_per_sample", "forward_plus_copy_ms_per_sample",
        "forward_copy_ms", "forward+copy",
    ],
    "direct_total_ms_per_sample": [
        "direct_total_ms_per_sample", "direct_total_ms",
        "direct_time_ms_per_sample",
    ],
    "neural_newton_ms_per_sample": [
        "neural_newton_ms_per_sample", "neural+newton_ms_per_sample",
        "plus_newton_ms_per_sample", "total_ms_per_sample",
        "newton_total_ms_per_sample",
    ],

    # Training information, if present
    "best_epoch": [
        "best_epoch", "epoch", "selected_epoch", "checkpoint_epoch",
    ],
    "train_time_sec": [
        "train_time_sec", "elapsed_sec", "elapsed_time_sec",
        "training_time_sec",
    ],
    "lr": [
        "lr", "learning_rate", "optimizer.lr",
    ],
    "weight_decay": [
        "weight_decay", "optimizer.weight_decay",
    ],
}


# ------------------------------------------------------------
# 3. File-name parsing
# ------------------------------------------------------------

def parse_degree_from_path(path: Path) -> Optional[int]:
    """
    Extract degree from directory name like '25degree'.
    """
    for part in path.parts:
        m = re.match(r"^(\d+)\s*degree$", part, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def parse_trial_model_from_filename(path: Path) -> tuple[Optional[int], Optional[str]]:
    """
    Parse file names like:
      trial_003_lstm.json
      trial_3_TRANSFORMER.json
      trial_12_mlp.json
    """
    stem = path.stem

    m = re.match(r"^trial[_\-]?(\d+)[_\-](.+)$", stem, flags=re.IGNORECASE)
    if m:
        trial_id = int(m.group(1))
        model = m.group(2).strip().lower()
        return trial_id, normalize_model_name(model)

    # fallback: find trial number
    nums = re.findall(r"\d+", stem)
    trial_id = int(nums[0]) if nums else None

    model = None
    for candidate in ["transformer", "trans", "lstm", "gru", "mlp", "ann"]:
        if candidate in stem.lower():
            model = normalize_model_name(candidate)
            break

    return trial_id, model


def normalize_model_name(model: Optional[str]) -> Optional[str]:
    if model is None:
        return None

    m = model.lower()
    m = m.replace("model_", "")
    m = m.replace("trial_", "")
    m = m.replace("-", "_")

    if "transformer" in m or m == "trans":
        return "Transformer"
    if "lstm" in m:
        return "LSTM"
    if "gru" in m:
        return "GRU"
    if "mlp" in m:
        return "MLP"
    if "ann" in m:
        return "ANN"

    return model.upper()


# ------------------------------------------------------------
# 4. JSON row extraction
# ------------------------------------------------------------

def extract_row(json_path: Path) -> Dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    flat = flatten_dict(raw)

    degree = parse_degree_from_path(json_path)
    trial_id, model = parse_trial_model_from_filename(json_path)

    # JSON 내부에 degree/model/trial이 있으면 파일명보다 우선할 수도 있음
    json_degree = find_first_numeric(flat, ["degree", "taylor_degree", "K", "taylor_deg"])
    if json_degree is not None:
        degree = int(json_degree)

    # model은 문자열이라 별도 처리
    for key in ["model", "model_name", "backbone", "arch", "architecture"]:
        if key in flat and isinstance(flat[key], str):
            model = normalize_model_name(flat[key])
            break

    row = {
        "degree": degree,
        "model": model,
        "trial": trial_id,
        "json_path": str(json_path),
    }

    for canonical_key, candidates in ALIASES.items():
        row[canonical_key] = find_first_numeric(flat, candidates)

    # 원본 JSON의 모든 numeric key도 보존하고 싶으면 extra_*로 저장
    for key, value in flat.items():
        x = to_float_or_none(value)
        if x is not None:
            safe_key = "raw_" + re.sub(r"[^0-9a-zA-Z_]+", "_", key).strip("_")
            if safe_key not in row:
                row[safe_key] = x

    return row


# ------------------------------------------------------------
# 5. Ranking / aggregation
# ------------------------------------------------------------

def add_selection_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lower score is better.
    Priority:
      1) higher convergence ratio
      2) lower Newton iteration mean
      3) lower direct RMSE
      4) lower Newton RMSE
      5) lower residual mean
    """
    out = df.copy()

    conv = out["newton_converged_ratio"].fillna(0.0)
    iter_mean = out["newton_iter_mean"].fillna(1e9)
    direct_rmse = out["direct_rmse"].fillna(1e9)
    newton_rmse = out["newton_rmse"].fillna(1e9)
    residual = out["newton_residual_mean"].fillna(1e9)

    # convergence는 높을수록 좋으므로 음수화
    out["_selection_score"] = (
        (-conv * 1e6)
        + (iter_mean * 1e3)
        + (direct_rmse * 1e1)
        + (newton_rmse * 1e2)
        + residual
    )

    return out


def aggregate_best_by_degree_model(df: pd.DataFrame) -> pd.DataFrame:
    df2 = add_selection_score(df)

    sort_cols = ["degree", "model", "_selection_score"]
    df2 = df2.sort_values(sort_cols, ascending=[True, True, True])

    best = (
        df2.groupby(["degree", "model"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    return best.drop(columns=["_selection_score"], errors="ignore")


def aggregate_mean_by_model(best_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "direct_rmse",
        "direct_r2",
        "direct_residual_mean",
        "newton_rmse",
        "newton_residual_mean",
        "newton_iter_mean",
        "newton_converged_ratio",
        "forward_gpu_ms_per_sample",
        "forward_copy_ms_per_sample",
        "direct_total_ms_per_sample",
        "neural_newton_ms_per_sample",
    ]

    cols = ["model"] + [c for c in metric_cols if c in best_df.columns]

    agg = (
        best_df[cols]
        .groupby("model", as_index=False)
        .mean(numeric_only=True)
    )

    return agg


def aggregate_mean_by_degree(best_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "direct_rmse",
        "direct_r2",
        "direct_residual_mean",
        "newton_rmse",
        "newton_residual_mean",
        "newton_iter_mean",
        "newton_converged_ratio",
        "forward_gpu_ms_per_sample",
        "forward_copy_ms_per_sample",
        "direct_total_ms_per_sample",
        "neural_newton_ms_per_sample",
    ]

    cols = ["degree"] + [c for c in metric_cols if c in best_df.columns]

    agg = (
        best_df[cols]
        .groupby("degree", as_index=False)
        .mean(numeric_only=True)
    )

    return agg


# ------------------------------------------------------------
# 6. Formatting
# ------------------------------------------------------------

def fmt_sci(x: Any, digits: int = 3) -> str:
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):.{digits}e}"
    except Exception:
        return str(x)


def fmt_fixed(x: Any, digits: int = 4) -> str:
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def make_compact_table(best_df: pd.DataFrame) -> pd.DataFrame:
    """
    논문에 바로 넣기 좋은 compact table.
    """
    cols = [
        "degree",
        "model",
        "trial",
        "direct_rmse",
        "direct_r2",
        "newton_rmse",
        "newton_residual_mean",
        "newton_iter_mean",
        "newton_converged_ratio",
    ]

    available = [c for c in cols if c in best_df.columns]
    tab = best_df[available].copy()

    rename = {
        "degree": "Degree",
        "model": "Model",
        "trial": "Trial",
        "direct_rmse": "Direct RMSE",
        "direct_r2": "Direct R2",
        "newton_rmse": "Newton RMSE",
        "newton_residual_mean": "Newton Residual Mean",
        "newton_iter_mean": "Newton Iter.",
        "newton_converged_ratio": "Conv. Ratio",
    }
    tab = tab.rename(columns=rename)

    for c in tab.columns:
        if "RMSE" in c or "Residual" in c:
            tab[c] = tab[c].map(lambda x: fmt_sci(x, 3))
        elif "R2" in c or "Conv" in c:
            tab[c] = tab[c].map(lambda x: fmt_fixed(x, 5))
        elif "Iter" in c:
            tab[c] = tab[c].map(lambda x: fmt_fixed(x, 3))

    return tab


def save_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    latex = df.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        longtable=False,
    )

    # CAI/논문용으로 필요하면 table*로 감싸기
    latex = latex.replace("\\begin{table}", "\\begin{table*}")
    latex = latex.replace("\\end{table}", "\\end{table*}")

    path.write_text(latex, encoding="utf-8")


# ------------------------------------------------------------
# 7. Main
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default="/root/project/dataset/math_03_14/results",
        help="Root directory containing {degree}degree/coupled_grid_1000ep folders.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="/root/project/dataset/math_03_14/results/direct_regression_summary",
        help="Output directory for CSV and LaTeX summary files.",
    )

    parser.add_argument(
        "--degrees",
        type=str,
        default="10,15,20,25,30,35",
        help="Comma-separated Taylor degrees to scan. Use 'all' to scan all *degree folders.",
    )

    parser.add_argument(
        "--subdir",
        type=str,
        default="coupled_grid_1000ep",
        help="Subdirectory under each {degree}degree folder.",
    )

    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.degrees.lower() == "all":
        degree_dirs = sorted(root.glob("*degree"))
    else:
        degree_list = [int(x.strip()) for x in args.degrees.split(",") if x.strip()]
        degree_dirs = [root / f"{d}degree" for d in degree_list]

    json_files = []
    for degree_dir in degree_dirs:
        run_dir = degree_dir / args.subdir
        json_files.extend(sorted(run_dir.glob("trial_*.json")))

    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found under {root}/{{degree}}degree/{args.subdir}/trial_*.json"
        )

    print(f"[INFO] Found {len(json_files)} JSON files.")

    rows = []
    failed = []

    for jp in json_files:
        try:
            rows.append(extract_row(jp))
        except Exception as e:
            failed.append((str(jp), repr(e)))

    if failed:
        fail_path = out_dir / "failed_json_files.txt"
        fail_path.write_text(
            "\n".join([f"{p}\t{err}" for p, err in failed]),
            encoding="utf-8",
        )
        print(f"[WARN] Failed to parse {len(failed)} files. See {fail_path}")

    df = pd.DataFrame(rows)

    # 기본 정렬
    sort_cols = [c for c in ["degree", "model", "trial"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    all_csv = out_dir / "direct_regression_all_trials.csv"
    df.to_csv(all_csv, index=False)
    print(f"[SAVE] {all_csv}")

    best = aggregate_best_by_degree_model(df)
    best = best.sort_values(["degree", "model"]).reset_index(drop=True)

    best_csv = out_dir / "direct_regression_best_by_degree_model.csv"
    best.to_csv(best_csv, index=False)
    print(f"[SAVE] {best_csv}")

    model_avg = aggregate_mean_by_model(best)
    model_avg_csv = out_dir / "direct_regression_model_average.csv"
    model_avg.to_csv(model_avg_csv, index=False)
    print(f"[SAVE] {model_avg_csv}")

    degree_avg = aggregate_mean_by_degree(best)
    degree_avg_csv = out_dir / "direct_regression_degree_average.csv"
    degree_avg.to_csv(degree_avg_csv, index=False)
    print(f"[SAVE] {degree_avg_csv}")

    # 논문용 compact table
    compact = make_compact_table(best)
    compact_csv = out_dir / "direct_regression_best_by_degree_model_compact.csv"
    compact.to_csv(compact_csv, index=False)
    print(f"[SAVE] {compact_csv}")

    save_latex_table(
        compact,
        out_dir / "direct_regression_best_by_degree_model.tex",
        caption=(
            "Direct-regression ablation results grouped by Taylor degree and model backbone. "
            "The best trial for each degree-model pair is selected by convergence ratio, "
            "Newton iteration count, and direct RMSE."
        ),
        label="tab:direct_regression_by_degree_model",
    )
    print(f"[SAVE] {out_dir / 'direct_regression_best_by_degree_model.tex'}")

    # model average compact
    model_avg_compact = model_avg.copy()
    rename = {
        "model": "Model",
        "direct_rmse": "Mean Direct RMSE",
        "direct_r2": "Mean Direct R2",
        "newton_rmse": "Mean Newton RMSE",
        "newton_residual_mean": "Mean Newton Residual",
        "newton_iter_mean": "Mean Iter.",
        "newton_converged_ratio": "Mean Conv.",
        "neural_newton_ms_per_sample": "Mean Neural+Newton ms",
    }
    model_avg_compact = model_avg_compact.rename(columns=rename)

    for c in model_avg_compact.columns:
        if "RMSE" in c or "Residual" in c:
            model_avg_compact[c] = model_avg_compact[c].map(lambda x: fmt_sci(x, 3))
        elif "R2" in c or "Conv" in c:
            model_avg_compact[c] = model_avg_compact[c].map(lambda x: fmt_fixed(x, 5))
        elif "Iter" in c or "ms" in c:
            model_avg_compact[c] = model_avg_compact[c].map(lambda x: fmt_fixed(x, 4))

    model_avg_compact_csv = out_dir / "direct_regression_model_average_compact.csv"
    model_avg_compact.to_csv(model_avg_compact_csv, index=False)
    print(f"[SAVE] {model_avg_compact_csv}")

    save_latex_table(
        model_avg_compact,
        out_dir / "direct_regression_model_average.tex",
        caption=(
            "Model-wise average performance of the direct-regression ablation "
            "over the tested Taylor degrees."
        ),
        label="tab:direct_regression_model_average",
    )
    print(f"[SAVE] {out_dir / 'direct_regression_model_average.tex'}")

    # summary
    print("\n[DONE]")
    print(f"Parsed JSON files: {len(rows)}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()