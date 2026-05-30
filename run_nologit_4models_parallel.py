#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel launcher for no-logit correction ablation.

This script runs the no-logit correction ablation for:
    - MLP
    - LSTM
    - GRU
    - Transformer

Each model is launched as an independent subprocess. This avoids CUDA fork/reinitialization
issues and isolates failures per model.

Required trainer:
    /root/project/dataset/math_03_14/train_coupled_nologit_ablation.py

Main output:
    /root/project/dataset/math_03_14/results/25degree/coupled_nologit_1000ep/
        mlp/
        lstm/
        gru/
        transformer/
        _logs/
        nologit_4models_summary.json
        nologit_4models_summary.csv
        nologit_4models_table.tex
        parallel_launch_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# 1. General utilities
# ============================================================

def shell_join(cmd: List[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def parse_gpu_list(gpus: Optional[str]) -> List[str]:
    if gpus is None:
        return []
    gpus = gpus.strip()
    if gpus == "":
        return []
    return [x.strip() for x in gpus.split(",") if x.strip()]


def json_safe_record(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a process tracking record into a JSON-serializable dict.

    Important:
      subprocess.Popen object must not be stored in JSON.
    """
    safe: Dict[str, Any] = {}

    for k, v in item.items():
        if k == "process":
            continue

        if isinstance(v, Path):
            safe[k] = str(v)
        elif isinstance(v, (str, int, float, bool)) or v is None:
            safe[k] = v
        else:
            safe[k] = str(v)

    if "pid" in safe and safe["pid"] is not None:
        try:
            safe["pid"] = int(safe["pid"])
        except Exception:
            safe["pid"] = str(safe["pid"])

    if "returncode" in safe and safe["returncode"] is not None:
        try:
            safe["returncode"] = int(safe["returncode"])
        except Exception:
            safe["returncode"] = str(safe["returncode"])

    for key in ["start_time", "end_time", "duration_sec"]:
        if key in safe and safe[key] is not None:
            try:
                safe[key] = float(safe[key])
            except Exception:
                safe[key] = str(safe[key])

    return safe


def fmt_sci(x: Any, nd: int = 3) -> str:
    if x is None:
        return "--"
    try:
        return f"{float(x):.{nd}e}"
    except Exception:
        return "--"


def fmt_float(x: Any, nd: int = 4) -> str:
    if x is None:
        return "--"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "--"


def display_model(m: str) -> str:
    ml = m.lower()
    if ml == "mlp":
        return "MLP"
    if ml == "lstm":
        return "LSTM"
    if ml == "gru":
        return "GRU"
    if ml == "transformer":
        return "Transformer"
    return m


# ============================================================
# 2. Command builder
# ============================================================

def build_command(
    python_bin: str,
    trainer: Path,
    degree: int,
    model: str,
    train_npz: Path,
    val_npz: Path,
    test_npz: Path,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    device: str,
    seed: int,
    trial_id: int,
    num_workers: int,
    patience: int,
    newton_tol: float,
    newton_max_iter: int,
    baseline_mode: str,
    explicit_formula: str,
    coeff_key: Optional[str],
    target_key: Optional[str],
    baseline_key: Optional[str],
    amp: bool,
    extra_args: List[str],
) -> List[str]:

    cmd = [
        python_bin,
        str(trainer),
        "--degree", str(degree),
        "--model", model,
        "--train-npz", str(train_npz),
        "--val-npz", str(val_npz),
        "--test-npz", str(test_npz),
        "--out-dir", str(out_dir),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--eval-batch-size", str(eval_batch_size),
        "--device", device,
        "--seed", str(seed),
        "--trial-id", str(trial_id),
        "--num-workers", str(num_workers),
        "--patience", str(patience),
        "--newton-tol", str(newton_tol),
        "--newton-max-iter", str(newton_max_iter),
        "--baseline-mode", baseline_mode,
        "--explicit-formula", explicit_formula,
    ]

    if coeff_key:
        cmd += ["--coeff-key", coeff_key]
    if target_key:
        cmd += ["--target-key", target_key]
    if baseline_key:
        cmd += ["--baseline-key", baseline_key]
    if amp:
        cmd += ["--amp"]

    cmd += extra_args
    return cmd


# ============================================================
# 3. Launch / monitor
# ============================================================

def launch_process(
    cmd: List[str],
    log_path: Path,
    env: Dict[str, str],
) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    log_file.write("=" * 120 + "\n")
    log_file.write("[COMMAND]\n")
    log_file.write(shell_join(cmd) + "\n")
    log_file.write("=" * 120 + "\n\n")
    log_file.flush()

    p = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    )

    return p


def update_running_jobs(
    running: List[Dict[str, Any]],
    finished_records: List[Dict[str, Any]],
) -> None:
    still_running: List[Dict[str, Any]] = []

    for item in running:
        p: subprocess.Popen = item["process"]
        ret = p.poll()

        if ret is None:
            still_running.append(item)
        else:
            item["returncode"] = int(ret)
            item["end_time"] = time.time()
            item["duration_sec"] = item["end_time"] - item["start_time"]

            safe = json_safe_record(item)
            finished_records.append(safe)

            print(
                f"[DONE] model={safe.get('model')} "
                f"returncode={safe.get('returncode')} "
                f"duration={safe.get('duration_sec'):.1f}s "
                f"log={safe.get('log_path')}"
            )

    running[:] = still_running


def wait_for_slot(
    running: List[Dict[str, Any]],
    finished_records: List[Dict[str, Any]],
    max_parallel: int,
    poll_sec: float = 5.0,
) -> None:
    while True:
        update_running_jobs(running, finished_records)

        if len(running) < max_parallel:
            return

        print(f"[WAIT] {len(running)} jobs running. Waiting {poll_sec:.1f}s...")
        time.sleep(poll_sec)


# ============================================================
# 4. Result collection
# ============================================================

def load_model_result(out_root: Path, model: str) -> Dict[str, Any]:
    model_dir = out_root / model
    json_files = sorted(model_dir.glob("*nologit*.json"))

    if not json_files:
        return {
            "status": "missing_json",
            "model_dir": str(model_dir),
        }

    newest = max(json_files, key=lambda p: p.stat().st_mtime)

    try:
        data = json.loads(newest.read_text(encoding="utf-8"))
    except Exception as e:
        return {
            "status": "json_read_error",
            "json_path": str(newest),
            "error": repr(e),
        }

    keep_keys = [
        "experiment",
        "degree",
        "model",
        "trial",
        "seed",
        "best_epoch",
        "best_val_loss",
        "direct_mae",
        "direct_rmse",
        "direct_r2",
        "direct_valid_ratio",
        "direct_residual_mean",
        "direct_residual_median",
        "direct_residual_p90",
        "plus_newton_mae",
        "plus_newton_rmse",
        "plus_newton_r2",
        "plus_newton_residual_mean",
        "plus_newton_residual_median",
        "plus_newton_residual_p90",
        "plus_newton_iter_mean",
        "plus_newton_iter_median",
        "plus_newton_iter_p90",
        "plus_newton_converged_ratio",
        "forward_ms_per_sample",
        "forward_ms_per_sample_std",
        "plus_newton_ms_per_sample",
        "train_time_sec",
    ]

    row = {k: data.get(k, None) for k in keep_keys}
    row["json_path"] = str(newest)
    row["status"] = "ok"
    return row


def collect_results(out_root: Path, models: List[str], degree: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "degree": degree,
        "models": {},
    }

    for model in models:
        summary["models"][model] = load_model_result(out_root, model)

    return summary


def write_csv(summary: Dict[str, Any], csv_path: Path) -> None:
    rows: List[Dict[str, Any]] = []

    for model, row in summary["models"].items():
        out = {"model_key": model}
        out.update(row)
        rows.append(out)

    if not rows:
        return

    keys: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_latex(summary: Dict[str, Any], tex_path: Path) -> None:
    lines: List[str] = []

    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{No-logit correction ablation results on the coupled pipe-flow test set at Taylor degree $K=25$. "
        r"The baseline-aware correction target is retained, but the flow-ratio correction is performed in the raw ratio space rather than logit space.}"
    )
    lines.append(r"\label{tab:nologit_ablation_k25}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lccccccc}")
    lines.append(r"\hline")
    lines.append(
        r"Model & Direct RMSE & Direct Residual Mean & Newton RMSE & Newton Residual Mean & Iter. Mean & Conv. Ratio & ms/sample \\"
    )
    lines.append(r"\hline")

    for model in ["mlp", "lstm", "gru", "transformer"]:
        row = summary["models"].get(model, {})

        if row.get("status") != "ok":
            lines.append(
                f"{display_model(model)} & -- & -- & -- & -- & -- & -- & -- \\\\"
            )
            continue

        lines.append(
            f"{display_model(model)} & "
            f"{fmt_sci(row.get('direct_rmse'))} & "
            f"{fmt_sci(row.get('direct_residual_mean'))} & "
            f"{fmt_sci(row.get('plus_newton_rmse'))} & "
            f"{fmt_sci(row.get('plus_newton_residual_mean'))} & "
            f"{fmt_float(row.get('plus_newton_iter_mean'), 3)} & "
            f"{fmt_float(row.get('plus_newton_converged_ratio'), 5)} & "
            f"{fmt_float(row.get('plus_newton_ms_per_sample'), 5)} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    lines.append("")

    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 5. Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trainer",
        type=str,
        default="/root/project/dataset/math_03_14/train_coupled_nologit_ablation.py",
    )

    parser.add_argument("--degree", type=int, default=25)

    parser.add_argument("--train-npz", type=str, required=True)
    parser.add_argument("--val-npz", type=str, required=True)
    parser.add_argument("--test-npz", type=str, required=True)

    parser.add_argument(
        "--out-root",
        type=str,
        default="/root/project/dataset/math_03_14/results/25degree/coupled_nologit_1000ep",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["mlp", "lstm", "gru", "transformer"],
        choices=["mlp", "lstm", "gru", "transformer"],
    )

    parser.add_argument("--max-parallel", type=int, default=4)

    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        help="Comma-separated GPU ids, e.g., '0,1,2,3'. Empty means inherit current CUDA_VISIBLE_DEVICES.",
    )

    parser.add_argument("--python-bin", type=str, default=sys.executable)

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--newton-tol", type=float, default=1e-12)
    parser.add_argument("--newton-max-iter", type=int, default=20)

    parser.add_argument(
        "--baseline-mode",
        type=str,
        default="heuristic",
        choices=["heuristic", "equal", "conductance"],
    )

    parser.add_argument(
        "--explicit-formula",
        type=str,
        default="haaland",
        choices=["haaland", "swamee_jain", "serghides"],
    )

    parser.add_argument("--coeff-key", type=str, default=None)
    parser.add_argument("--target-key", type=str, default=None)
    parser.add_argument("--baseline-key", type=str, default=None)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing model output folders and logs before launching.",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a model if a *nologit*.json already exists in its output folder.",
    )

    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments passed to trainer after '--'.",
    )

    args = parser.parse_args()

    trainer = Path(args.trainer)
    train_npz = Path(args.train_npz)
    val_npz = Path(args.val_npz)
    test_npz = Path(args.test_npz)
    out_root = Path(args.out_root)
    logs_dir = out_root / "_logs"

    if not trainer.exists():
        raise FileNotFoundError(f"Trainer not found: {trainer}")
    if not train_npz.exists():
        raise FileNotFoundError(f"Train NPZ not found: {train_npz}")
    if not val_npz.exists():
        raise FileNotFoundError(f"Val NPZ not found: {val_npz}")
    if not test_npz.exists():
        raise FileNotFoundError(f"Test NPZ not found: {test_npz}")

    if args.clean and out_root.exists():
        print(f"[CLEAN] Removing existing output root: {out_root}")
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = parse_gpu_list(args.gpus)

    print("=" * 120)
    print("[PARALLEL NO-LOGIT CORRECTION ABLATION]")
    print(f"degree       : {args.degree}")
    print(f"models       : {args.models}")
    print(f"max_parallel : {args.max_parallel}")
    print(f"gpus         : {gpu_ids if gpu_ids else 'inherit current CUDA_VISIBLE_DEVICES'}")
    print(f"trainer      : {trainer}")
    print(f"train_npz    : {train_npz}")
    print(f"val_npz      : {val_npz}")
    print(f"test_npz     : {test_npz}")
    print(f"out_root     : {out_root}")
    print(f"clean        : {args.clean}")
    print(f"skip_existing: {args.skip_existing}")
    print("=" * 120)

    running: List[Dict[str, Any]] = []
    finished_records: List[Dict[str, Any]] = []
    skipped_records: List[Dict[str, Any]] = []

    launch_start = time.time()

    for idx, model in enumerate(args.models):
        model_out = out_root / model
        model_out.mkdir(parents=True, exist_ok=True)

        existing = sorted(model_out.glob("*nologit*.json"))
        if args.skip_existing and existing:
            print(f"[SKIP] model={model}, existing_json={existing[-1]}")
            skipped_records.append(
                {
                    "model": model,
                    "reason": "existing_json",
                    "json_path": str(existing[-1]),
                    "out_dir": str(model_out),
                }
            )
            continue

        wait_for_slot(
            running=running,
            finished_records=finished_records,
            max_parallel=args.max_parallel,
            poll_sec=5.0,
        )

        trial_id = idx + 1

        cmd = build_command(
            python_bin=args.python_bin,
            trainer=trainer,
            degree=args.degree,
            model=model,
            train_npz=train_npz,
            val_npz=val_npz,
            test_npz=test_npz,
            out_dir=model_out,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            device=args.device,
            seed=args.seed,
            trial_id=trial_id,
            num_workers=args.num_workers,
            patience=args.patience,
            newton_tol=args.newton_tol,
            newton_max_iter=args.newton_max_iter,
            baseline_mode=args.baseline_mode,
            explicit_formula=args.explicit_formula,
            coeff_key=args.coeff_key,
            target_key=args.target_key,
            baseline_key=args.baseline_key,
            amp=args.amp,
            extra_args=args.extra_args,
        )

        env = os.environ.copy()

        if gpu_ids:
            assigned_gpu = gpu_ids[idx % len(gpu_ids)]
            env["CUDA_VISIBLE_DEVICES"] = assigned_gpu
        else:
            assigned_gpu = env.get("CUDA_VISIBLE_DEVICES", "inherited")

        log_path = logs_dir / f"nologit_deg{args.degree}_{model}.log"

        print(f"[LAUNCH] model={model} gpu={assigned_gpu}")
        print(f"         out={model_out}")
        print(f"         log={log_path}")
        print(f"         cmd={shell_join(cmd)}")

        p = launch_process(cmd=cmd, log_path=log_path, env=env)

        running.append(
            {
                "model": model,
                "pid": int(p.pid),
                "gpu": assigned_gpu,
                "log_path": str(log_path),
                "out_dir": str(model_out),
                "start_time": time.time(),
                "process": p,
            }
        )

    while running:
        update_running_jobs(running, finished_records)
        if running:
            print(f"[WAIT] {len(running)} jobs still running...")
            time.sleep(10.0)

    launch_end = time.time()

    # Collect model JSON outputs even if some jobs failed
    result_summary = collect_results(
        out_root=out_root,
        models=args.models,
        degree=args.degree,
    )

    result_json_path = out_root / "nologit_4models_summary.json"
    result_csv_path = out_root / "nologit_4models_summary.csv"
    result_tex_path = out_root / "nologit_4models_table.tex"

    result_json_path.write_text(
        json.dumps(result_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(result_summary, result_csv_path)
    write_latex(result_summary, result_tex_path)

    launch_summary = {
        "degree": args.degree,
        "models": args.models,
        "max_parallel": args.max_parallel,
        "gpus": gpu_ids,
        "trainer": str(trainer),
        "train_npz": str(train_npz),
        "val_npz": str(val_npz),
        "test_npz": str(test_npz),
        "out_root": str(out_root),
        "start_time": float(launch_start),
        "end_time": float(launch_end),
        "total_duration_sec": float(launch_end - launch_start),
        "finished_records": finished_records,
        "skipped_records": skipped_records,
    }

    launch_summary_path = out_root / "parallel_launch_summary.json"
    launch_summary_path.write_text(
        json.dumps(launch_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 120)
    print("[ALL DONE]")
    print(f"Launch summary : {launch_summary_path}")
    print(f"Result summary : {result_json_path}")
    print(f"CSV summary    : {result_csv_path}")
    print(f"LaTeX table    : {result_tex_path}")
    print("=" * 120)

    for model in args.models:
        row = result_summary["models"].get(model, {})
        print("-" * 120)
        print(f"MODEL: {model}")
        print(f"status                         : {row.get('status')}")
        print(f"json_path                      : {row.get('json_path')}")
        print(f"direct_rmse                    : {row.get('direct_rmse')}")
        print(f"direct_residual_mean           : {row.get('direct_residual_mean')}")
        print(f"plus_newton_rmse               : {row.get('plus_newton_rmse')}")
        print(f"plus_newton_residual_mean      : {row.get('plus_newton_residual_mean')}")
        print(f"plus_newton_iter_mean          : {row.get('plus_newton_iter_mean')}")
        print(f"plus_newton_converged_ratio    : {row.get('plus_newton_converged_ratio')}")
        print(f"plus_newton_ms_per_sample      : {row.get('plus_newton_ms_per_sample')}")

    failed = [r for r in finished_records if r.get("returncode") != 0]
    if failed:
        print("\n[WARNING] Some jobs failed. Check logs:")
        for r in failed:
            print(
                f"  model={r.get('model')} "
                f"returncode={r.get('returncode')} "
                f"log={r.get('log_path')}"
            )
        sys.exit(1)

    missing = [
        m for m in args.models
        if result_summary["models"].get(m, {}).get("status") != "ok"
    ]
    if missing:
        print("\n[WARNING] Some model results are missing:")
        for m in missing:
            print(f"  {m}: {result_summary['models'].get(m)}")
        sys.exit(1)


if __name__ == "__main__":
    main()