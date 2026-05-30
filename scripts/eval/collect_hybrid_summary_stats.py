#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_hybrid_summary_stats_v2.py

목적
----
hybrid_params2_runs 아래에 흩어진 summary_stats.json을 전부 읽어서
논문에 넣을 핵심 지표만 한 번에 비교 가능한 CSV / Markdown / LaTeX 표로 만든다.

이 버전의 핵심 수정
-------------------
기존 collector에서는 논문용 표를 만들 때 RMSE를 fmt_float(..., 4)로 포맷해서
Newton 이후 작은 RMSE 값이 0.0000으로 보이는 문제가 있었다.

이 버전에서는:
- RMSE 계열: scientific notation, 예: 3.421e-09
- residual 계열: scientific notation
- R2: 소수점 8자리
- iteration: 소수점 3자리
- convergence ratio: 소수점 5자리
- reduction: percentage

입력 summary_stats.json 구조 예시
---------------------------------
{
  "heuristic_direct": {
    "rmse": {"mean": ..., "std": ..., "min": ..., "max": ...},
    ...
  },
  "heuristic_plus_newton": {...},
  "neural_direct": {...},
  "neural_plus_newton": {...}
}

출력
----
out_dir/
  collected_all_metrics.csv        # raw float 수치 전체
  paper_main_table.csv             # 논문용 포맷 전체 degree/model
  paper_main_table.md
  paper_main_table.tex
  best_by_degree_raw.csv
  best_by_degree.csv
  best_by_degree.md
  best_by_degree.tex
  best_by_model_raw.csv
  best_by_model.csv
  best_by_model.md
  best_by_model.tex
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Any, List


# =========================================================
# Basic utilities
# =========================================================
def safe_float(x, default=float("nan")):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def get_stat(stats: Dict[str, Any], group: str, metric: str, stat: str = "mean"):
    """
    summary_stats.json에서 stats[group][metric][stat]를 안전하게 꺼낸다.
    없으면 NaN 반환.
    """
    try:
        return safe_float(stats[group][metric][stat])
    except Exception:
        return float("nan")


def save_csv(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"[WARN] no rows to save: {path}")
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


def parse_degree_and_model(path: Path):
    """
    path 예:
    /home/seokjun/math_03_14/hybrid_params2_runs/deg25/lstm_trial_030_lstm/summary_stats.json

    반환:
    degree = 25
    model = lstm
    trial_dir = lstm_trial_030_lstm
    """
    degree = None

    for part in path.parts:
        m = re.match(r"deg(\d+)$", part)
        if m:
            degree = int(m.group(1))

    trial_dir = path.parent.name
    model = trial_dir.split("_")[0].lower()

    return degree, model, trial_dir


# =========================================================
# Formatting for paper tables
# =========================================================
def fmt_sci(x, digits=3):
    """
    작은 값이 0.0000으로 죽지 않도록 scientific notation 사용.
    """
    x = safe_float(x)
    if not math.isfinite(x):
        return "-"
    return f"{x:.{digits}e}"


def fmt_fixed(x, digits=4):
    x = safe_float(x)
    if not math.isfinite(x):
        return "-"
    return f"{x:.{digits}f}"


def fmt_r2(x):
    x = safe_float(x)
    if not math.isfinite(x):
        return "-"
    return f"{x:.8f}"


def fmt_ratio(x):
    x = safe_float(x)
    if not math.isfinite(x):
        return "-"
    return f"{x:.5f}"


def fmt_iter(x):
    x = safe_float(x)
    if not math.isfinite(x):
        return "-"
    return f"{x:.3f}"


def fmt_percent_fraction(x):
    """
    0.9972 -> 99.72\\%
    """
    x = safe_float(x)
    if not math.isfinite(x):
        return "-"
    return f"{100.0 * x:.2f}\\%"


def markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for row in rows:
        vals = []
        for c in columns:
            vals.append(str(row.get(c, "")))
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def latex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\%", "%TEMP_PERCENT%")
        .replace("\\", r"\textbackslash{}")
        .replace("%TEMP_PERCENT%", r"\%")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def latex_table(
    rows: List[Dict[str, Any]],
    columns: List[str],
    caption: str,
    label: str,
    table_star: bool = True,
):
    env = "table*" if table_star else "table"
    colspec = "l" * len(columns)

    lines = []
    lines.append(r"\begin{" + env + r"}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + colspec + r"}")
    lines.append(r"\hline")
    lines.append(" & ".join([latex_escape(c) for c in columns]) + r" \\")
    lines.append(r"\hline")

    for row in rows:
        vals = [latex_escape(row.get(c, "")) for c in columns]
        lines.append(" & ".join(vals) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{" + env + r"}")

    return "\n".join(lines)


# =========================================================
# Extraction
# =========================================================
def extract_row(json_path: Path):
    degree, model, trial_dir = parse_degree_and_model(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    groups = [
        "heuristic_direct",
        "heuristic_plus_newton",
        "neural_direct",
        "neural_plus_newton",
    ]

    row = {
        "degree": degree,
        "model": model,
        "trial_dir": trial_dir,
        "json_path": str(json_path),
    }

    # summary_stats.json의 주요 지표를 모두 raw float로 수집
    for group in groups:
        for metric in [
            "mae",
            "rmse",
            "r2",
            "valid_ratio",
            "residual_mean",
            "residual_median",
            "residual_p90",
            "max_abs_error",
            "newton_iter_mean",
            "newton_iter_median",
            "newton_iter_p90",
            "newton_converged_ratio",
        ]:
            row[f"{group}_{metric}_mean"] = get_stat(stats, group, metric, "mean")
            row[f"{group}_{metric}_std"] = get_stat(stats, group, metric, "std")
            row[f"{group}_{metric}_min"] = get_stat(stats, group, metric, "min")
            row[f"{group}_{metric}_max"] = get_stat(stats, group, metric, "max")

    # -----------------------------------------------------
    # Derived metrics
    # -----------------------------------------------------
    # Direct RMSE reduction
    h_rmse = row["heuristic_direct_rmse_mean"]
    n_rmse = row["neural_direct_rmse_mean"]
    row["direct_rmse_reduction_vs_heuristic"] = (
        1.0 - n_rmse / h_rmse
        if math.isfinite(h_rmse) and h_rmse != 0.0 and math.isfinite(n_rmse)
        else float("nan")
    )

    # Direct MAE reduction
    h_mae = row["heuristic_direct_mae_mean"]
    n_mae = row["neural_direct_mae_mean"]
    row["direct_mae_reduction_vs_heuristic"] = (
        1.0 - n_mae / h_mae
        if math.isfinite(h_mae) and h_mae != 0.0 and math.isfinite(n_mae)
        else float("nan")
    )

    # Direct residual reduction
    h_res = row["heuristic_direct_residual_mean_mean"]
    n_res = row["neural_direct_residual_mean_mean"]
    row["direct_residual_reduction_vs_heuristic"] = (
        1.0 - n_res / h_res
        if math.isfinite(h_res) and h_res != 0.0 and math.isfinite(n_res)
        else float("nan")
    )

    # Newton iteration reduction
    h_iter = row["heuristic_plus_newton_newton_iter_mean_mean"]
    n_iter = row["neural_plus_newton_newton_iter_mean_mean"]
    row["newton_iter_reduction_vs_heuristic"] = (
        1.0 - n_iter / h_iter
        if math.isfinite(h_iter) and h_iter != 0.0 and math.isfinite(n_iter)
        else float("nan")
    )

    # Newton RMSE ratio
    h_plus_rmse = row["heuristic_plus_newton_rmse_mean"]
    n_plus_rmse = row["neural_plus_newton_rmse_mean"]
    row["plus_newton_rmse_ratio_neural_over_heuristic"] = (
        n_plus_rmse / h_plus_rmse
        if math.isfinite(h_plus_rmse) and h_plus_rmse != 0.0 and math.isfinite(n_plus_rmse)
        else float("nan")
    )

    return row


def make_paper_row(row: Dict[str, Any]):
    """
    논문에 바로 넣기 좋은 축약 행.
    RMSE/residual 계열은 scientific notation으로 표시한다.
    """
    return {
        "Degree": row["degree"],
        "Model": str(row["model"]).upper(),

        # Direct initializer quality
        "Neural Direct RMSE": fmt_sci(row["neural_direct_rmse_mean"], 3),
        "Heuristic Direct RMSE": fmt_sci(row["heuristic_direct_rmse_mean"], 3),
        "RMSE Reduction": fmt_percent_fraction(row["direct_rmse_reduction_vs_heuristic"]),
        "Neural Direct R2": fmt_r2(row["neural_direct_r2_mean"]),

        # Newton-refined accuracy
        "Neural+Newton RMSE": fmt_sci(row["neural_plus_newton_rmse_mean"], 3),
        "Heuristic+Newton RMSE": fmt_sci(row["heuristic_plus_newton_rmse_mean"], 3),

        # Residuals
        "Neural Direct Residual": fmt_sci(row["neural_direct_residual_mean_mean"], 3),
        "Heuristic Direct Residual": fmt_sci(row["heuristic_direct_residual_mean_mean"], 3),
        "Neural+Newton Residual": fmt_sci(row["neural_plus_newton_residual_mean_mean"], 3),
        "Heuristic+Newton Residual": fmt_sci(row["heuristic_plus_newton_residual_mean_mean"], 3),

        # Newton efficiency
        "Neural Iter.": fmt_iter(row["neural_plus_newton_newton_iter_mean_mean"]),
        "Heuristic Iter.": fmt_iter(row["heuristic_plus_newton_newton_iter_mean_mean"]),
        "Iter. Reduction": fmt_percent_fraction(row["newton_iter_reduction_vs_heuristic"]),

        # Robustness
        "Neural Conv.": fmt_ratio(row["neural_plus_newton_newton_converged_ratio_mean"]),
        "Heuristic Conv.": fmt_ratio(row["heuristic_plus_newton_newton_converged_ratio_mean"]),
    }


def choose_best_by_degree(rows: List[Dict[str, Any]], criterion: str):
    """
    degree별 best model 선택.
    """
    best = {}

    higher_is_better = {
        "neural_plus_newton_newton_converged_ratio_mean",
        "neural_direct_valid_ratio_mean",
        "neural_direct_r2_mean",
        "direct_rmse_reduction_vs_heuristic",
        "newton_iter_reduction_vs_heuristic",
    }

    lower_is_better = criterion not in higher_is_better

    for row in rows:
        deg = row["degree"]
        val = safe_float(row.get(criterion))

        if deg is None or not math.isfinite(val):
            continue

        if deg not in best:
            best[deg] = row
            continue

        old = safe_float(best[deg].get(criterion))

        if lower_is_better:
            if val < old:
                best[deg] = row
        else:
            if val > old:
                best[deg] = row

    return [best[k] for k in sorted(best.keys())]


def aggregate_by_model(rows: List[Dict[str, Any]]):
    """
    모델별 degree 평균 요약.
    """
    models = sorted(set(r["model"] for r in rows if r["model"]))

    metrics = [
        "neural_direct_rmse_mean",
        "heuristic_direct_rmse_mean",
        "direct_rmse_reduction_vs_heuristic",
        "neural_direct_r2_mean",

        "neural_plus_newton_rmse_mean",
        "heuristic_plus_newton_rmse_mean",
        "neural_plus_newton_residual_mean_mean",
        "heuristic_plus_newton_residual_mean_mean",

        "neural_plus_newton_newton_iter_mean_mean",
        "heuristic_plus_newton_newton_iter_mean_mean",
        "newton_iter_reduction_vs_heuristic",

        "neural_plus_newton_newton_converged_ratio_mean",
        "heuristic_plus_newton_newton_converged_ratio_mean",
    ]

    out = []

    for model in models:
        sub = [r for r in rows if r["model"] == model]
        row = {
            "model": model,
            "n_degrees": len(sub),
        }

        for metric in metrics:
            vals = [safe_float(r.get(metric)) for r in sub]
            vals = [v for v in vals if math.isfinite(v)]

            if vals:
                row[f"{metric}_avg"] = sum(vals) / len(vals)
                row[f"{metric}_min"] = min(vals)
                row[f"{metric}_max"] = max(vals)
            else:
                row[f"{metric}_avg"] = float("nan")
                row[f"{metric}_min"] = float("nan")
                row[f"{metric}_max"] = float("nan")

        out.append(row)

    return out


def make_model_average_row(row: Dict[str, Any]):
    return {
        "Model": str(row["model"]).upper(),
        "N Degrees": row["n_degrees"],

        "Avg Neural Direct RMSE": fmt_sci(row["neural_direct_rmse_mean_avg"], 3),
        "Avg Heuristic Direct RMSE": fmt_sci(row["heuristic_direct_rmse_mean_avg"], 3),
        "Avg RMSE Reduction": fmt_percent_fraction(row["direct_rmse_reduction_vs_heuristic_avg"]),
        "Avg Neural Direct R2": fmt_r2(row["neural_direct_r2_mean_avg"]),

        "Avg Neural+Newton RMSE": fmt_sci(row["neural_plus_newton_rmse_mean_avg"], 3),
        "Avg Heuristic+Newton RMSE": fmt_sci(row["heuristic_plus_newton_rmse_mean_avg"], 3),

        "Avg Neural+Newton Residual": fmt_sci(row["neural_plus_newton_residual_mean_mean_avg"], 3),
        "Avg Heuristic+Newton Residual": fmt_sci(row["heuristic_plus_newton_residual_mean_mean_avg"], 3),

        "Avg Neural Iter.": fmt_iter(row["neural_plus_newton_newton_iter_mean_mean_avg"]),
        "Avg Heuristic Iter.": fmt_iter(row["heuristic_plus_newton_newton_iter_mean_mean_avg"]),
        "Avg Iter. Reduction": fmt_percent_fraction(row["newton_iter_reduction_vs_heuristic_avg"]),

        "Avg Neural Conv.": fmt_ratio(row["neural_plus_newton_newton_converged_ratio_mean_avg"]),
        "Avg Heuristic Conv.": fmt_ratio(row["heuristic_plus_newton_newton_converged_ratio_mean_avg"]),
    }


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help=(
            "summary_stats.json들이 들어있는 루트 폴더. "
            "예: /home/seokjun/math_03_14/hybrid_params2_runs"
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="합친 결과 저장 폴더",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="summary_stats.json",
    )
    parser.add_argument(
        "--best_criterion",
        type=str,
        default="neural_direct_rmse_mean",
        help=(
            "degree별 best model 선택 기준. "
            "예: neural_direct_rmse_mean, "
            "neural_plus_newton_rmse_mean, "
            "newton_iter_reduction_vs_heuristic"
        ),
    )

    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(root.rglob(args.pattern))

    if not paths:
        raise FileNotFoundError(f"No summary_stats.json found under: {root}")

    print(f"[INFO] root = {root}")
    print(f"[INFO] found {len(paths)} summary_stats.json files")

    rows = []
    for path in paths:
        try:
            row = extract_row(path)
            rows.append(row)
            print(f"[OK] degree={row['degree']} model={row['model']} path={path}")
        except Exception as e:
            print(f"[WARN] failed: {path} -> {e}")

    if not rows:
        raise RuntimeError("No valid summary_stats.json files were parsed.")

    rows = sorted(
        rows,
        key=lambda r: (
            r["degree"] if r["degree"] is not None else 999999,
            str(r["model"]),
            str(r["trial_dir"]),
        ),
    )

    # -----------------------------------------------------
    # 1. Raw full metrics
    # -----------------------------------------------------
    save_csv(out_dir / "collected_all_metrics.csv", rows)

    # -----------------------------------------------------
    # 2. Paper table: all degree/model combinations
    # -----------------------------------------------------
    paper_rows = [make_paper_row(r) for r in rows]

    paper_columns = [
        "Degree",
        "Model",
        "Neural Direct RMSE",
        "Heuristic Direct RMSE",
        "RMSE Reduction",
        "Neural Direct R2",
        "Neural+Newton RMSE",
        "Heuristic+Newton RMSE",
        "Neural+Newton Residual",
        "Heuristic+Newton Residual",
        "Neural Iter.",
        "Heuristic Iter.",
        "Iter. Reduction",
        "Neural Conv.",
        "Heuristic Conv.",
    ]

    save_csv(out_dir / "paper_main_table.csv", paper_rows)
    (out_dir / "paper_main_table.md").write_text(
        markdown_table(paper_rows, paper_columns),
        encoding="utf-8",
    )
    (out_dir / "paper_main_table.tex").write_text(
        latex_table(
            paper_rows,
            paper_columns,
            caption=(
                "Degree-wise comparison of the hybrid neural correction initializer "
                "against the nonlearned heuristic initializer in the multi-dimensional setting."
            ),
            label="tab:multidim_degreewise_hybrid_summary",
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # 3. Best model by degree
    # -----------------------------------------------------
    best_degree_rows_raw = choose_best_by_degree(
        rows,
        criterion=args.best_criterion,
    )
    best_degree_rows = [make_paper_row(r) for r in best_degree_rows_raw]

    save_csv(out_dir / "best_by_degree_raw.csv", best_degree_rows_raw)
    save_csv(out_dir / "best_by_degree.csv", best_degree_rows)
    (out_dir / "best_by_degree.md").write_text(
        markdown_table(best_degree_rows, paper_columns),
        encoding="utf-8",
    )
    (out_dir / "best_by_degree.tex").write_text(
        latex_table(
            best_degree_rows,
            paper_columns,
            caption=(
                "Best model for each Taylor degree selected by "
                + args.best_criterion.replace("_", r"\_")
                + "."
            ),
            label="tab:multidim_best_by_degree",
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # 4. Model-wise average across degrees
    # -----------------------------------------------------
    model_rows_raw = aggregate_by_model(rows)
    model_rows = [make_model_average_row(r) for r in model_rows_raw]

    model_columns = [
        "Model",
        "N Degrees",
        "Avg Neural Direct RMSE",
        "Avg Heuristic Direct RMSE",
        "Avg RMSE Reduction",
        "Avg Neural Direct R2",
        "Avg Neural+Newton RMSE",
        "Avg Heuristic+Newton RMSE",
        "Avg Neural+Newton Residual",
        "Avg Heuristic+Newton Residual",
        "Avg Neural Iter.",
        "Avg Heuristic Iter.",
        "Avg Iter. Reduction",
        "Avg Neural Conv.",
        "Avg Heuristic Conv.",
    ]

    save_csv(out_dir / "best_by_model_raw.csv", model_rows_raw)
    save_csv(out_dir / "best_by_model.csv", model_rows)
    (out_dir / "best_by_model.md").write_text(
        markdown_table(model_rows, model_columns),
        encoding="utf-8",
    )
    (out_dir / "best_by_model.tex").write_text(
        latex_table(
            model_rows,
            model_columns,
            caption=(
                "Model-wise average performance of the hybrid neural correction "
                "initializer across Taylor degrees."
            ),
            label="tab:multidim_model_average_hybrid_summary",
        ),
        encoding="utf-8",
    )

    print("\n[DONE]")
    print("Saved files:")
    for name in [
        "collected_all_metrics.csv",
        "paper_main_table.csv",
        "paper_main_table.md",
        "paper_main_table.tex",
        "best_by_degree_raw.csv",
        "best_by_degree.csv",
        "best_by_degree.md",
        "best_by_degree.tex",
        "best_by_model_raw.csv",
        "best_by_model.csv",
        "best_by_model.md",
        "best_by_model.tex",
    ]:
        print(" -", out_dir / name)


if __name__ == "__main__":
    main()