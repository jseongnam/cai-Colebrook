#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METRICS = [
    "direct_residual_mean",
    "direct_residual_p90",
    "newton_iterations_mean",
    "newton_iterations_p90",
    "convergence_ratio",
    "q_relative_rmse",
    "x_rmse",
]


def confidence_interval(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2 or np.allclose(np.std(values, ddof=1), 0.0):
        mean = float(np.mean(values))
        return mean, mean
    sem = stats.sem(values)
    radius = float(stats.t.ppf(0.975, len(values) - 1) * sem)
    mean = float(np.mean(values))
    return mean - radius, mean + radius


def paired_rows(frame: pd.DataFrame, label: str, cell_a: tuple[str, str],
                cell_b: tuple[str, str]) -> list[dict]:
    keys = ["branches", "regime", "input_mode", "seed"]
    a = frame[(frame.target_mode == cell_a[0]) &
              (frame.flow_parameterization == cell_a[1])]
    b = frame[(frame.target_mode == cell_b[0]) &
              (frame.flow_parameterization == cell_b[1])]
    rows = []
    for group_values, group_a in a.groupby(keys[:-1], dropna=False):
        group_b = b
        for key, value in zip(keys[:-1], group_values):
            group_b = group_b[group_b[key] == value]
        merged = group_a.merge(group_b, on=keys, suffixes=("_a", "_b"))
        for metric in METRICS:
            if f"{metric}_a" not in merged or merged.empty:
                continue
            differences = (merged[f"{metric}_a"] - merged[f"{metric}_b"]).to_numpy(float)
            ci_low, ci_high = confidence_interval(differences)
            p_value = (float(stats.ttest_rel(merged[f"{metric}_a"], merged[f"{metric}_b"]).pvalue)
                       if len(merged) >= 2 and not np.allclose(differences, differences[0])
                       else np.nan)
            row = dict(zip(keys[:-1], group_values))
            row.update({
                "comparison": label,
                "cell_a": f"{cell_a[0]}_{cell_a[1]}",
                "cell_b": f"{cell_b[0]}_{cell_b[1]}",
                "metric": metric,
                "n_seeds": len(merged),
                "mean_a": float(merged[f"{metric}_a"].mean()),
                "mean_b": float(merged[f"{metric}_b"].mean()),
                "paired_difference_a_minus_b": float(np.mean(differences)),
                "paired_difference_std": float(np.std(differences, ddof=1)) if len(differences) > 1 else np.nan,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "paired_t_pvalue": p_value,
                "seed_differences": ";".join(f"{value:.12g}" for value in differences),
            })
            rows.append(row)
    return rows


def interaction_rows(frame: pd.DataFrame) -> list[dict]:
    keys = ["branches", "regime", "input_mode", "seed"]
    rows = []
    for metric in METRICS:
        pivot = frame.pivot_table(
            index=keys, columns=["target_mode", "flow_parameterization"],
            values=metric, aggfunc="first")
        required = [
            ("correction", "logit"), ("direct", "logit"),
            ("correction", "raw_ratio"), ("direct", "raw_ratio"),
        ]
        if not all(cell in pivot.columns for cell in required):
            continue
        effect = ((pivot[("correction", "logit")] - pivot[("direct", "logit")]) -
                  (pivot[("correction", "raw_ratio")] - pivot[("direct", "raw_ratio")]))
        table = effect.rename("effect").reset_index()
        for group_values, group in table.groupby(keys[:-1], dropna=False):
            values = group.effect.to_numpy(float)
            ci_low, ci_high = confidence_interval(values)
            row = dict(zip(keys[:-1], group_values))
            row.update({
                "comparison": "target_x_parameterization_interaction",
                "cell_a": "difference_in_differences",
                "cell_b": "",
                "metric": metric,
                "n_seeds": len(values),
                "mean_a": np.nan,
                "mean_b": np.nan,
                "paired_difference_a_minus_b": float(np.mean(values)),
                "paired_difference_std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "paired_t_pvalue": (float(stats.ttest_1samp(values, 0.0).pvalue)
                                      if len(values) >= 2 and not np.allclose(values, values[0])
                                      else np.nan),
                "seed_differences": ";".join(f"{value:.12g}" for value in values),
            })
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired 2x2 factorial comparisons across seeds")
    parser.add_argument("--results", default="results_extended")
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.results)
    files = sorted(root.rglob("eval_*.csv"))
    if not files:
        raise SystemExit(f"No eval_*.csv files found under {root}")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    frame = frame[frame.method == "neural_plus_newton"].copy()
    for column, default in [("target_mode", "correction"),
                            ("flow_parameterization", "logit")]:
        if column not in frame:
            frame[column] = default
        else:
            frame[column] = frame[column].fillna(default)
    if "experiment_variant" not in frame:
        frame["experiment_variant"] = frame["checkpoint"].map(
            lambda value: Path(str(value)).parent.parent.name)
    else:
        missing = frame["experiment_variant"].isna()
        frame.loc[missing, "experiment_variant"] = frame.loc[missing, "checkpoint"].map(
            lambda value: Path(str(value)).parent.parent.name)
    factorial_variants = {
        "correction_logit", "direct_logit", "correction_raw_ratio", "direct_raw_ratio"
    }
    frame = frame[frame.experiment_variant.isin(factorial_variants)]
    rows = []
    rows.extend(paired_rows(
        frame, "correction_effect_with_logit",
        ("correction", "logit"), ("direct", "logit")))
    rows.extend(paired_rows(
        frame, "logit_effect_with_correction",
        ("correction", "logit"), ("correction", "raw_ratio")))
    rows.extend(interaction_rows(frame))
    output = Path(args.output) if args.output else root / "factorial_paired_comparisons.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
