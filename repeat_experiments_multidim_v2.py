#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


def read_summary_metrics(csv_path: Path):
    rows = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["name"]] = row
    return rows


def to_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def mean_std(values):
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0, "min": vals[0], "max": vals[0]}
    return {
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals),
        "min": min(vals),
        "max": max(vals),
    }


def aggregate_records(records, metric_names):
    out = {}
    for metric in metric_names:
        vals = [to_float(r.get(metric, math.nan)) for r in records]
        out[metric] = mean_std(vals)
    return out


def main():
    parser = argparse.ArgumentParser()

    # 공통 경로
    parser.add_argument("--train_script", required=True,
                        help="예: train_multidim_hybrid_correction_v2.py")
    parser.add_argument("--eval_script", required=True,
                        help="예: evaluate_multidim_hybrid_correction_v2.py")

    parser.add_argument("--train_npz", required=True)
    parser.add_argument("--val_npz", required=True)
    parser.add_argument("--test_npz", required=True)

    parser.add_argument("--output_root", required=True,
                        help="반복 실험 결과 저장 루트 폴더")

    # 반복 횟수
    parser.add_argument("--num_runs", type=int, default=50)
    parser.add_argument("--seed_start", type=int, default=1)

    # 모델/학습 파라미터
    parser.add_argument("--model", choices=["mlp", "lstm", "gru", "transformer"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)

    parser.add_argument("--use_log_features", action="store_true")
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--loss_name", choices=["smoothl1", "mse"], default="smoothl1")

    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[256, 256, 128])
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--head_hidden", type=int, default=128)
    parser.add_argument("--head_layers", type=int, default=2)

    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=128)
    parser.add_argument("--use_cls_token", action="store_true")

    # eval 파라미터
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max_newton_iter", type=int, default=20)

    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_run_rows = []

    for i in range(args.num_runs):
        seed = args.seed_start + i

        run_dir = output_root / f"run_{seed:03d}"
        train_dir = run_dir / "train"
        eval_dir = run_dir / "eval"
        train_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            sys.executable, args.train_script,
            "--model", args.model,
            "--train_npz", args.train_npz,
            "--val_npz", args.val_npz,
            "--test_npz", args.test_npz,
            "--save_dir", str(train_dir),
            "--optimizer", args.optimizer,
            "--lr", str(args.lr),
            "--weight_decay", str(args.weight_decay),
            "--dropout", str(args.dropout),
            "--loss_name", args.loss_name,
            "--batch_size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--device", args.device,
            "--seed", str(seed),
            "--num_workers", "0",
            "--hidden_size", str(args.hidden_size),
            "--num_layers", str(args.num_layers),
            "--head_hidden", str(args.head_hidden),
            "--head_layers", str(args.head_layers),
            "--d_model", str(args.d_model),
            "--nhead", str(args.nhead),
            "--ff_dim", str(args.ff_dim),
        ]

        if args.use_log_features:
            train_cmd.append("--use_log_features")
        if args.use_cls_token:
            train_cmd.append("--use_cls_token")

        # hidden_dims는 mlp에서만 의미 있지만 그냥 넣어도 argparse에서 받음
        if args.hidden_dims:
            train_cmd += ["--hidden_dims"] + [str(x) for x in args.hidden_dims]

        print(f"\n[RUN {i+1}/{args.num_runs}] seed={seed}")
        print("TRAIN CMD:", " ".join(train_cmd))
        subprocess.run(train_cmd, check=True)

        model_ckpt = train_dir / "best_model.pt"

        eval_cmd = [
            sys.executable, args.eval_script,
            "--test_npz", args.test_npz,
            "--model", str(model_ckpt),
            "--out_dir", str(eval_dir),
            "--tol", str(args.tol),
            "--max_newton_iter", str(args.max_newton_iter),
            "--device", args.device,
        ]

        print("EVAL CMD:", " ".join(eval_cmd))
        subprocess.run(eval_cmd, check=True)

        summary_csv = eval_dir / "summary_metrics.csv"
        summary_rows = read_summary_metrics(summary_csv)

        row = {
            "seed": seed,
            "train_dir": str(train_dir),
            "eval_dir": str(eval_dir),
        }

        # 논문에서 주로 쓸 항목들
        target_names = [
            "heuristic_direct",
            "heuristic_plus_newton",
            "neural_direct",
            "neural_plus_newton",
        ]

        keep_metrics = [
            "mae", "rmse", "r2",
            "valid_ratio",
            "residual_mean", "residual_median", "residual_p90",
            "max_abs_error",
            "newton_iter_mean", "newton_iter_median", "newton_iter_p90",
            "newton_converged_ratio",
        ]

        for name in target_names:
            if name not in summary_rows:
                continue
            for m in keep_metrics:
                if m in summary_rows[name]:
                    row[f"{name}__{m}"] = summary_rows[name][m]

        all_run_rows.append(row)

    # run별 raw 저장
    raw_csv = output_root / "all_runs_raw.csv"
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(all_run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_run_rows)

    # 집계
    grouped_summary = {}

    for prefix in [
        "heuristic_direct",
        "heuristic_plus_newton",
        "neural_direct",
        "neural_plus_newton",
    ]:
        records = []
        for row in all_run_rows:
            rec = {}
            for k, v in row.items():
                if k.startswith(prefix + "__"):
                    rec[k[len(prefix)+2:]] = v
            if rec:
                records.append(rec)

        if not records:
            continue

        grouped_summary[prefix] = aggregate_records(
            records,
            metric_names=[
                "mae", "rmse", "r2",
                "valid_ratio",
                "residual_mean", "residual_median", "residual_p90",
                "max_abs_error",
                "newton_iter_mean", "newton_iter_median", "newton_iter_p90",
                "newton_converged_ratio",
            ]
        )

    # json 저장
    summary_json = output_root / "summary_stats.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(grouped_summary, f, ensure_ascii=False, indent=2)

    # 논문 표 넣기 좋은 csv 저장
    paper_rows = []
    for group_name, metrics in grouped_summary.items():
        row = {"group": group_name}
        for metric_name, stat in metrics.items():
            row[f"{metric_name}_mean"] = stat["mean"]
            row[f"{metric_name}_std"] = stat["std"]
            row[f"{metric_name}_min"] = stat["min"]
            row[f"{metric_name}_max"] = stat["max"]
        paper_rows.append(row)

    paper_csv = output_root / "summary_stats_for_paper.csv"
    with open(paper_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=paper_rows[0].keys())
        writer.writeheader()
        writer.writerows(paper_rows)

    print("\n[DONE]")
    print("raw:", raw_csv)
    print("summary json:", summary_json)
    print("paper csv:", paper_csv)


if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


def read_summary_metrics(csv_path: Path):
    rows = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["name"]] = row
    return rows


def to_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def mean_std(values):
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0, "min": vals[0], "max": vals[0]}
    return {
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals),
        "min": min(vals),
        "max": max(vals),
    }


def aggregate_records(records, metric_names):
    out = {}
    for metric in metric_names:
        vals = [to_float(r.get(metric, math.nan)) for r in records]
        out[metric] = mean_std(vals)
    return out


def main():
    parser = argparse.ArgumentParser()

    # 공통 경로
    parser.add_argument("--train_script", required=True,
                        help="예: train_multidim_hybrid_correction_v2.py")
    parser.add_argument("--eval_script", required=True,
                        help="예: evaluate_multidim_hybrid_correction_v2.py")

    parser.add_argument("--train_npz", required=True)
    parser.add_argument("--val_npz", required=True)
    parser.add_argument("--test_npz", required=True)

    parser.add_argument("--output_root", required=True,
                        help="반복 실험 결과 저장 루트 폴더")

    # 반복 횟수
    parser.add_argument("--num_runs", type=int, default=50)
    parser.add_argument("--seed_start", type=int, default=1)

    # 모델/학습 파라미터
    parser.add_argument("--model", choices=["mlp", "lstm", "gru", "transformer"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)

    parser.add_argument("--use_log_features", action="store_true")
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--loss_name", choices=["smoothl1", "mse"], default="smoothl1")

    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[256, 256, 128])
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--head_hidden", type=int, default=128)
    parser.add_argument("--head_layers", type=int, default=2)

    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=128)
    parser.add_argument("--use_cls_token", action="store_true")

    # eval 파라미터
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max_newton_iter", type=int, default=20)

    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_run_rows = []

    for i in range(args.num_runs):
        seed = args.seed_start + i

        run_dir = output_root / f"run_{seed:03d}"
        train_dir = run_dir / "train"
        eval_dir = run_dir / "eval"
        train_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            sys.executable, args.train_script,
            "--model", args.model,
            "--train_npz", args.train_npz,
            "--val_npz", args.val_npz,
            "--test_npz", args.test_npz,
            "--save_dir", str(train_dir),
            "--optimizer", args.optimizer,
            "--lr", str(args.lr),
            "--weight_decay", str(args.weight_decay),
            "--dropout", str(args.dropout),
            "--loss_name", args.loss_name,
            "--batch_size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--device", args.device,
            "--seed", str(seed),
            "--num_workers", "0",
            "--hidden_size", str(args.hidden_size),
            "--num_layers", str(args.num_layers),
            "--head_hidden", str(args.head_hidden),
            "--head_layers", str(args.head_layers),
            "--d_model", str(args.d_model),
            "--nhead", str(args.nhead),
            "--ff_dim", str(args.ff_dim),
        ]

        if args.use_log_features:
            train_cmd.append("--use_log_features")
        if args.use_cls_token:
            train_cmd.append("--use_cls_token")

        # hidden_dims는 mlp에서만 의미 있지만 그냥 넣어도 argparse에서 받음
        if args.hidden_dims:
            train_cmd += ["--hidden_dims"] + [str(x) for x in args.hidden_dims]

        print(f"\n[RUN {i+1}/{args.num_runs}] seed={seed}")
        print("TRAIN CMD:", " ".join(train_cmd))
        subprocess.run(train_cmd, check=True)

        model_ckpt = train_dir / "best_model.pt"

        eval_cmd = [
            sys.executable, args.eval_script,
            "--test_npz", args.test_npz,
            "--model", str(model_ckpt),
            "--out_dir", str(eval_dir),
            "--tol", str(args.tol),
            "--max_newton_iter", str(args.max_newton_iter),
            "--device", args.device,
        ]

        print("EVAL CMD:", " ".join(eval_cmd))
        subprocess.run(eval_cmd, check=True)

        summary_csv = eval_dir / "summary_metrics.csv"
        summary_rows = read_summary_metrics(summary_csv)

        row = {
            "seed": seed,
            "train_dir": str(train_dir),
            "eval_dir": str(eval_dir),
        }

        # 논문에서 주로 쓸 항목들
        target_names = [
            "heuristic_direct",
            "heuristic_plus_newton",
            "neural_direct",
            "neural_plus_newton",
        ]

        keep_metrics = [
            "mae", "rmse", "r2",
            "valid_ratio",
            "residual_mean", "residual_median", "residual_p90",
            "max_abs_error",
            "newton_iter_mean", "newton_iter_median", "newton_iter_p90",
            "newton_converged_ratio",
        ]

        for name in target_names:
            if name not in summary_rows:
                continue
            for m in keep_metrics:
                if m in summary_rows[name]:
                    row[f"{name}__{m}"] = summary_rows[name][m]

        all_run_rows.append(row)

    # run별 raw 저장
    raw_csv = output_root / "all_runs_raw.csv"
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(all_run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_run_rows)

    # 집계
    grouped_summary = {}

    for prefix in [
        "heuristic_direct",
        "heuristic_plus_newton",
        "neural_direct",
        "neural_plus_newton",
    ]:
        records = []
        for row in all_run_rows:
            rec = {}
            for k, v in row.items():
                if k.startswith(prefix + "__"):
                    rec[k[len(prefix)+2:]] = v
            if rec:
                records.append(rec)

        if not records:
            continue

        grouped_summary[prefix] = aggregate_records(
            records,
            metric_names=[
                "mae", "rmse", "r2",
                "valid_ratio",
                "residual_mean", "residual_median", "residual_p90",
                "max_abs_error",
                "newton_iter_mean", "newton_iter_median", "newton_iter_p90",
                "newton_converged_ratio",
            ]
        )

    # json 저장
    summary_json = output_root / "summary_stats.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(grouped_summary, f, ensure_ascii=False, indent=2)

    # 논문 표 넣기 좋은 csv 저장
    paper_rows = []
    for group_name, metrics in grouped_summary.items():
        row = {"group": group_name}
        for metric_name, stat in metrics.items():
            row[f"{metric_name}_mean"] = stat["mean"]
            row[f"{metric_name}_std"] = stat["std"]
            row[f"{metric_name}_min"] = stat["min"]
            row[f"{metric_name}_max"] = stat["max"]
        paper_rows.append(row)

    paper_csv = output_root / "summary_stats_for_paper.csv"
    with open(paper_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=paper_rows[0].keys())
        writer.writeheader()
        writer.writerows(paper_rows)

    print("\n[DONE]")
    print("raw:", raw_csv)
    print("summary json:", summary_json)
    print("paper csv:", paper_csv)


if __name__ == "__main__":
    main()