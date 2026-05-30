#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_multidim_model_results.py

목적
----
여러 모델의 평가 결과(summary_metrics.csv)를 자동으로 읽어서
하나의 비교 CSV로 합친다.

지원 입력
--------
1) 여러 eval 폴더 지정:
   --eval_dir ./eval_runs/multidim_mlp_deg8
   --eval_dir ./eval_runs/multidim_lstm_deg8
   --eval_dir ./eval_runs/multidim_gru_deg8
   --eval_dir ./eval_runs/multidim_transformer_deg8

2) 또는 summary csv 직접 지정:
   --summary_csv ./eval_runs/multidim_mlp_deg8/summary_metrics.csv
   --summary_csv ./eval_runs/multidim_lstm_deg8/summary_metrics.csv

출력
----
- combined_all_rows.csv
    각 summary_metrics.csv의 모든 method row를 합친 파일
- neural_methods_only.csv
    neural_direct / neural_plus_newton 만 모은 파일
- neural_pivot_wide.csv
    모델별 neural_direct / neural_plus_newton 주요 지표를 wide 형태로 정리한 파일
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def read_csv(path: Path) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def write_csv(path: Path, rows: List[Dict]):
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


def try_read_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def to_float_if_possible(v):
    if v is None:
        return v
    if isinstance(v, (int, float)):
        return v
    s = str(v).strip()
    if s == "":
        return s
    try:
        return float(s)
    except Exception:
        return v


def normalize_row(row: Dict[str, str]) -> Dict:
    return {k: to_float_if_possible(v) for k, v in row.items()}


def infer_model_name(eval_dir: Path, config: Dict = None) -> str:
    if config is not None:
        if "model_name" in config:
            return str(config["model_name"])
        if "model" in config:
            return str(config["model"])
    return eval_dir.name


def collect_from_eval_dir(eval_dir: Path) -> Tuple[str, List[Dict]]:
    summary_path = eval_dir / "summary_metrics.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary_metrics.csv in {eval_dir}")

    config = try_read_json(eval_dir / "config.json")
    model_name = infer_model_name(eval_dir, config=config)

    rows = read_csv(summary_path)
    rows = [normalize_row(r) for r in rows]

    enriched = []
    for r in rows:
        rr = dict(r)
        rr["source_eval_dir"] = str(eval_dir)
        rr["model_name"] = model_name
        enriched.append(rr)

    return model_name, enriched


def collect_from_summary_csv(summary_csv: Path) -> Tuple[str, List[Dict]]:
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing file: {summary_csv}")

    eval_dir = summary_csv.parent
    config = try_read_json(eval_dir / "config.json")
    model_name = infer_model_name(eval_dir, config=config)

    rows = read_csv(summary_csv)
    rows = [normalize_row(r) for r in rows]

    enriched = []
    for r in rows:
        rr = dict(r)
        rr["source_eval_dir"] = str(eval_dir)
        rr["model_name"] = model_name
        enriched.append(rr)

    return model_name, enriched


def build_neural_pivot(neural_rows: List[Dict]) -> List[Dict]:
    grouped = {}
    for row in neural_rows:
        model_name = row["model_name"]
        method = row["name"]

        if model_name not in grouped:
            grouped[model_name] = {}
        grouped[model_name][method] = row

    out = []
    for model_name, methods in grouped.items():
        row = {"model_name": model_name}

        direct = methods.get("neural_direct", {})
        refine = methods.get("neural_plus_newton", {})

        for src, prefix in [(direct, "direct"), (refine, "plus_newton")]:
            for key, value in src.items():
                if key in ["name", "model_name", "source_eval_dir"]:
                    continue
                row[f"{prefix}_{key}"] = value

        if "direct_rmse" in row and "plus_newton_rmse" in row:
            try:
                row["delta_rmse"] = row["plus_newton_rmse"] - row["direct_rmse"]
            except Exception:
                pass

        if "direct_mae" in row and "plus_newton_mae" in row:
            try:
                row["delta_mae"] = row["plus_newton_mae"] - row["direct_mae"]
            except Exception:
                pass

        if "direct_r2" in row and "plus_newton_r2" in row:
            try:
                row["delta_r2"] = row["plus_newton_r2"] - row["direct_r2"]
            except Exception:
                pass

        if "plus_newton_newton_iter_mean" in row:
            row["iter_mean"] = row["plus_newton_newton_iter_mean"]

        if "plus_newton_newton_converged_ratio" in row:
            row["converged_ratio"] = row["plus_newton_newton_converged_ratio"]

        out.append(row)

    def sort_key(r):
        rmse = r.get("plus_newton_rmse", float("inf"))
        conv = r.get("converged_ratio", -1.0)
        try:
            return (float(rmse), -float(conv))
        except Exception:
            return (float("inf"), 1.0)

    out.sort(key=sort_key)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", action="append", default=[], help="evaluation directory containing summary_metrics.csv")
    ap.add_argument("--summary_csv", action="append", default=[], help="direct path to summary_metrics.csv")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    if not args.eval_dir and not args.summary_csv:
        raise ValueError("Provide at least one --eval_dir or --summary_csv")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    seen = set()

    for d in args.eval_dir:
        eval_dir = Path(d)
        key = ("dir", str(eval_dir.resolve()))
        if key in seen:
            continue
        seen.add(key)

        _, rows = collect_from_eval_dir(eval_dir)
        all_rows.extend(rows)

    for s in args.summary_csv:
        summary_csv = Path(s)
        key = ("csv", str(summary_csv.resolve()))
        if key in seen:
            continue
        seen.add(key)

        _, rows = collect_from_summary_csv(summary_csv)
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No rows collected.")

    combined_all_path = out_dir / "combined_all_rows.csv"
    write_csv(combined_all_path, all_rows)

    neural_rows = [r for r in all_rows if r.get("name") in ["neural_direct", "neural_plus_newton"]]
    neural_rows_path = out_dir / "neural_methods_only.csv"
    write_csv(neural_rows_path, neural_rows)

    pivot_rows = build_neural_pivot(neural_rows)
    pivot_path = out_dir / "neural_pivot_wide.csv"
    write_csv(pivot_path, pivot_rows)

    print("[DONE]")
    print(f"  - {combined_all_path}")
    print(f"  - {neural_rows_path}")
    print(f"  - {pivot_path}")

    print("\nTop models by plus_newton_rmse:")
    for row in pivot_rows[:10]:
        print({
            "model_name": row.get("model_name"),
            "plus_newton_rmse": row.get("plus_newton_rmse"),
            "plus_newton_mae": row.get("plus_newton_mae"),
            "converged_ratio": row.get("converged_ratio"),
            "iter_mean": row.get("iter_mean"),
        })


if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_multidim_model_results.py

목적
----
여러 모델의 평가 결과(summary_metrics.csv)를 자동으로 읽어서
하나의 비교 CSV로 합친다.

지원 입력
--------
1) 여러 eval 폴더 지정:
   --eval_dir ./eval_runs/multidim_mlp_deg8
   --eval_dir ./eval_runs/multidim_lstm_deg8
   --eval_dir ./eval_runs/multidim_gru_deg8
   --eval_dir ./eval_runs/multidim_transformer_deg8

2) 또는 summary csv 직접 지정:
   --summary_csv ./eval_runs/multidim_mlp_deg8/summary_metrics.csv
   --summary_csv ./eval_runs/multidim_lstm_deg8/summary_metrics.csv

출력
----
- combined_all_rows.csv
    각 summary_metrics.csv의 모든 method row를 합친 파일
- neural_methods_only.csv
    neural_direct / neural_plus_newton 만 모은 파일
- neural_pivot_wide.csv
    모델별 neural_direct / neural_plus_newton 주요 지표를 wide 형태로 정리한 파일
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def read_csv(path: Path) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def write_csv(path: Path, rows: List[Dict]):
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


def try_read_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def to_float_if_possible(v):
    if v is None:
        return v
    if isinstance(v, (int, float)):
        return v
    s = str(v).strip()
    if s == "":
        return s
    try:
        return float(s)
    except Exception:
        return v


def normalize_row(row: Dict[str, str]) -> Dict:
    return {k: to_float_if_possible(v) for k, v in row.items()}


def infer_model_name(eval_dir: Path, config: Dict = None) -> str:
    if config is not None:
        if "model_name" in config:
            return str(config["model_name"])
        if "model" in config:
            return str(config["model"])
    return eval_dir.name


def collect_from_eval_dir(eval_dir: Path) -> Tuple[str, List[Dict]]:
    summary_path = eval_dir / "summary_metrics.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary_metrics.csv in {eval_dir}")

    config = try_read_json(eval_dir / "config.json")
    model_name = infer_model_name(eval_dir, config=config)

    rows = read_csv(summary_path)
    rows = [normalize_row(r) for r in rows]

    enriched = []
    for r in rows:
        rr = dict(r)
        rr["source_eval_dir"] = str(eval_dir)
        rr["model_name"] = model_name
        enriched.append(rr)

    return model_name, enriched


def collect_from_summary_csv(summary_csv: Path) -> Tuple[str, List[Dict]]:
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing file: {summary_csv}")

    eval_dir = summary_csv.parent
    config = try_read_json(eval_dir / "config.json")
    model_name = infer_model_name(eval_dir, config=config)

    rows = read_csv(summary_csv)
    rows = [normalize_row(r) for r in rows]

    enriched = []
    for r in rows:
        rr = dict(r)
        rr["source_eval_dir"] = str(eval_dir)
        rr["model_name"] = model_name
        enriched.append(rr)

    return model_name, enriched


def build_neural_pivot(neural_rows: List[Dict]) -> List[Dict]:
    grouped = {}
    for row in neural_rows:
        model_name = row["model_name"]
        method = row["name"]

        if model_name not in grouped:
            grouped[model_name] = {}
        grouped[model_name][method] = row

    out = []
    for model_name, methods in grouped.items():
        row = {"model_name": model_name}

        direct = methods.get("neural_direct", {})
        refine = methods.get("neural_plus_newton", {})

        for src, prefix in [(direct, "direct"), (refine, "plus_newton")]:
            for key, value in src.items():
                if key in ["name", "model_name", "source_eval_dir"]:
                    continue
                row[f"{prefix}_{key}"] = value

        if "direct_rmse" in row and "plus_newton_rmse" in row:
            try:
                row["delta_rmse"] = row["plus_newton_rmse"] - row["direct_rmse"]
            except Exception:
                pass

        if "direct_mae" in row and "plus_newton_mae" in row:
            try:
                row["delta_mae"] = row["plus_newton_mae"] - row["direct_mae"]
            except Exception:
                pass

        if "direct_r2" in row and "plus_newton_r2" in row:
            try:
                row["delta_r2"] = row["plus_newton_r2"] - row["direct_r2"]
            except Exception:
                pass

        if "plus_newton_newton_iter_mean" in row:
            row["iter_mean"] = row["plus_newton_newton_iter_mean"]

        if "plus_newton_newton_converged_ratio" in row:
            row["converged_ratio"] = row["plus_newton_newton_converged_ratio"]

        out.append(row)

    def sort_key(r):
        rmse = r.get("plus_newton_rmse", float("inf"))
        conv = r.get("converged_ratio", -1.0)
        try:
            return (float(rmse), -float(conv))
        except Exception:
            return (float("inf"), 1.0)

    out.sort(key=sort_key)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", action="append", default=[], help="evaluation directory containing summary_metrics.csv")
    ap.add_argument("--summary_csv", action="append", default=[], help="direct path to summary_metrics.csv")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    if not args.eval_dir and not args.summary_csv:
        raise ValueError("Provide at least one --eval_dir or --summary_csv")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    seen = set()

    for d in args.eval_dir:
        eval_dir = Path(d)
        key = ("dir", str(eval_dir.resolve()))
        if key in seen:
            continue
        seen.add(key)

        _, rows = collect_from_eval_dir(eval_dir)
        all_rows.extend(rows)

    for s in args.summary_csv:
        summary_csv = Path(s)
        key = ("csv", str(summary_csv.resolve()))
        if key in seen:
            continue
        seen.add(key)

        _, rows = collect_from_summary_csv(summary_csv)
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No rows collected.")

    combined_all_path = out_dir / "combined_all_rows.csv"
    write_csv(combined_all_path, all_rows)

    neural_rows = [r for r in all_rows if r.get("name") in ["neural_direct", "neural_plus_newton"]]
    neural_rows_path = out_dir / "neural_methods_only.csv"
    write_csv(neural_rows_path, neural_rows)

    pivot_rows = build_neural_pivot(neural_rows)
    pivot_path = out_dir / "neural_pivot_wide.csv"
    write_csv(pivot_path, pivot_rows)

    print("[DONE]")
    print(f"  - {combined_all_path}")
    print(f"  - {neural_rows_path}")
    print(f"  - {pivot_path}")

    print("\nTop models by plus_newton_rmse:")
    for row in pivot_rows[:10]:
        print({
            "model_name": row.get("model_name"),
            "plus_newton_rmse": row.get("plus_newton_rmse"),
            "plus_newton_mae": row.get("plus_newton_mae"),
            "converged_ratio": row.get("converged_ratio"),
            "iter_mean": row.get("iter_mean"),
        })


if __name__ == "__main__":
    main()