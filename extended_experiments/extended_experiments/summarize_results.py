#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize(files: list[Path], keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [pd.read_csv(p) for p in files]
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if raw.empty:
        return raw, raw
    defaults = {
        "target_mode": "correction",
        "flow_parameterization": "logit",
        "inference_batch_size": 0,
    }
    for column, default in defaults.items():
        if column not in raw:
            raw[column] = default
        else:
            raw[column] = raw[column].fillna(default)
    if "experiment_variant" not in raw:
        raw["experiment_variant"] = raw["checkpoint"].map(
            lambda value: Path(str(value)).parent.parent.name)
    elif "checkpoint" in raw:
        missing = raw["experiment_variant"].isna()
        raw.loc[missing, "experiment_variant"] = raw.loc[missing, "checkpoint"].map(
            lambda value: Path(str(value)).parent.parent.name)
    numeric = [c for c in raw.select_dtypes(include="number").columns if c not in keys + ["seed"]]
    grouped = raw.groupby([k for k in keys if k in raw], dropna=False)[numeric].agg(["mean", "std"])
    grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
    return raw, grouped.reset_index()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results_extended")
    args = p.parse_args()
    root = Path(args.results)
    root.mkdir(parents=True, exist_ok=True)
    eval_files = sorted(root.rglob("eval_*.csv"))
    runtime_files = sorted(root.rglob("runtime_*.csv"))
    raw, summary = summarize(
        eval_files,
        ["branches", "regime", "input_mode", "experiment_variant", "target_mode",
         "flow_parameterization", "method"],
    )
    raw.to_csv(root / "all_evaluation_rows.csv", index=False)
    summary.to_csv(root / "paper_evaluation_summary.csv", index=False)
    if not summary.empty:
        (root / "paper_evaluation_summary.tex").write_text(
            summary.to_latex(index=False, float_format=lambda x: f"{x:.5g}"), encoding="utf-8"
        )
    raw_rt, summary_rt = summarize(
        runtime_files,
        ["branches", "regime", "input_mode", "experiment_variant", "target_mode",
         "flow_parameterization", "device", "inference_batch_size"],
    )
    raw_rt.to_csv(root / "all_runtime_rows.csv", index=False)
    summary_rt.to_csv(root / "paper_runtime_summary.csv", index=False)
    if not summary_rt.empty:
        (root / "paper_runtime_summary.tex").write_text(
            summary_rt.to_latex(index=False, float_format=lambda x: f"{x:.5g}"), encoding="utf-8"
        )
    print(f"evaluation files={len(eval_files)}, runtime files={len(runtime_files)}")


if __name__ == "__main__":
    main()
