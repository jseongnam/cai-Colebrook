#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
measure_hybrid_inference_time_parallel.py

목적
----
학습이 끝난 hybrid correction 모델(best_model.pt)을 재훈련 없이 불러와서
degree/model/run별 inference time을 병렬로 측정한다.

지원 구조
--------
models_root/
  deg10/
    mlp_trial_013_mlp/
      run_042/
        train/
          best_model.pt
    lstm_trial_030_lstm/
      run_042/
        train/
          best_model.pt
  deg15/
  ...

필요한 hybrid script
-------------------
예:
  /home/seokjun/math_03_14/repeat_experiments_multidim_allinone_v2.py

해당 파일 안에 아래 함수/클래스가 있어야 한다.
- load_npz
- load_model_checkpoint
- build_inputs_and_baseline
- refine_batch

그리고 model forward는 다음 형태를 가정한다.
  pred, delta_norm, delta_real = model(seq, glob, z0, q_total, delta_scaler_t)

실행 모드
--------
1) launcher 모드:
   전체 best_model.pt를 찾아서 병렬 worker 실행

2) worker 모드:
   checkpoint 하나만 측정하고 json/csv 저장

출력
----
out_dir/
  worker_results/
    deg10_mlp_run042.json
    ...
  logs/
    deg10_mlp_run042.log
    ...
  inference_time_all_raw.csv
  inference_time_degree_model_summary_raw.csv
  inference_time_paper_table.csv
  inference_time_paper_table.md
  inference_time_paper_table.tex
  inference_time_model_average.csv
  inference_time_model_average.md
  inference_time_model_average.tex
  inference_time_best_by_degree.csv
  inference_time_best_by_degree.md
  inference_time_best_by_degree.tex

주의
----
- 논문에 넣을 "정확한 isolated GPU forward time"은 max_parallel=1로 측정하는 것이 가장 엄밀하다.
- 빠르게 전체 경향을 보려면 max_parallel=4 또는 6으로 병렬 측정하면 된다.
- Newton refinement는 CPU loop가 크므로 병렬화 효과가 크다.
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


# =========================================================
# 공통 유틸
# =========================================================
def safe_float(x, default=float("nan")):
    try:
        if x is None:
            return default
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
    if isinstance(obj, np.ndarray):
        return obj.tolist()
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


def load_module_from_path(path: str):
    path = str(path)
    spec = importlib.util.spec_from_file_location("hybrid_module_for_timing", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cuda_sync(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def fmt_sci(x, digits=3):
    x = safe_float(x)
    if not math.isfinite(x):
        return "-"
    return f"{x:.{digits}e}"


def fmt_ms(x, digits=5):
    x = safe_float(x)
    if not math.isfinite(x):
        return "-"
    return f"{x:.{digits}f}"


def fmt_ratio(x, digits=5):
    x = safe_float(x)
    if not math.isfinite(x):
        return "-"
    return f"{x:.{digits}f}"


def markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for row in rows:
        vals = [str(row.get(c, "")) for c in columns]
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


def latex_table(rows: List[Dict[str, Any]], columns: List[str], caption: str, label: str):
    colspec = "l" * len(columns)
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + colspec + r"}")
    lines.append(r"\hline")
    lines.append(" & ".join(latex_escape(c) for c in columns) + r" \\")
    lines.append(r"\hline")

    for row in rows:
        vals = [latex_escape(row.get(c, "")) for c in columns]
        lines.append(" & ".join(vals) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# =========================================================
# 경로 파싱
# =========================================================
def parse_model_path(model_path: Path):
    """
    예:
    /home/seokjun/math_03_14/hybrid_params2_runs/deg25/lstm_trial_030_lstm/run_042/train/best_model.pt
    """
    degree = None
    model = None
    trial_dir = None
    run_dir = None
    seed = None

    for part in model_path.parts:
        m = re.match(r"deg(\d+)$", part)
        if m:
            degree = int(m.group(1))

        r = re.match(r"run_(\d+)$", part)
        if r:
            run_dir = part
            seed = int(r.group(1))

    try:
        trial_dir = model_path.parent.parent.parent.name
        model = trial_dir.split("_")[0].lower()
    except Exception:
        trial_dir = None
        model = None

    return degree, model, trial_dir, run_dir, seed


def find_model_paths(root: Path, degrees: List[int], models: List[str]):
    degrees_set = set(int(d) for d in degrees)
    models_set = set(m.lower() for m in models)

    all_paths = sorted(root.rglob("best_model.pt"))
    selected = []

    for p in all_paths:
        deg, model, trial_dir, run_dir, seed = parse_model_path(p)
        if deg is None or model is None:
            continue
        if deg not in degrees_set:
            continue
        if model not in models_set:
            continue
        selected.append(p)

    return selected


def test_npz_for_degree(data_root: Path, degree: int):
    return (
        data_root
        / f"multi_colebrook_data_deg{degree}"
        / f"parallel2_colebrook_deg{degree}_test.npz"
    )


def worker_result_name(model_path: Path):
    deg, model, trial_dir, run_dir, seed = parse_model_path(model_path)
    seed_str = f"run{seed:03d}" if seed is not None else "runxxx"
    return f"deg{deg}_{model}_{seed_str}"


# =========================================================
# 스케일러 / forward
# =========================================================
def scaler_transform(X: np.ndarray, scaler: Dict[str, Any]):
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["std"], dtype=np.float64)
    std = np.where(np.abs(std) < 1e-12, 1.0, std)

    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e12, neginf=-1e12)
    Xs = (X - mean) / std
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = np.clip(Xs, -1e6, 1e6)
    return Xs.astype(np.float32)


def make_batches(seq_x, glob_x, z0, q_total, batch_size: int, device: str):
    batches = []
    n = len(seq_x)

    for i in range(0, n, batch_size):
        s = torch.from_numpy(seq_x[i:i + batch_size].astype(np.float32)).to(device, non_blocking=True)
        g = torch.from_numpy(glob_x[i:i + batch_size].astype(np.float32)).to(device, non_blocking=True)
        z = torch.from_numpy(z0[i:i + batch_size].astype(np.float32)).to(device, non_blocking=True)
        qt = torch.from_numpy(
            np.asarray(q_total[i:i + batch_size]).astype(np.float32).reshape(-1, 1)
        ).to(device, non_blocking=True)

        batches.append((s, g, z, qt))

    return batches


@torch.no_grad()
def forward_only(model, batches, delta_scaler_t):
    last = None
    for s, g, z, qt in batches:
        pred, delta_norm, delta_real = model(s, g, z, qt, delta_scaler_t)
        last = pred
    return last


@torch.no_grad()
def forward_with_copy(model, batches, delta_scaler_t):
    preds = []
    for s, g, z, qt in batches:
        pred, delta_norm, delta_real = model(s, g, z, qt, delta_scaler_t)
        preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def measure_forward_gpu(model, batches, delta_scaler_t, n_samples, device, repeats, warmup):
    for _ in range(warmup):
        forward_only(model, batches, delta_scaler_t)
    cuda_sync(device)

    times = []
    for _ in range(repeats):
        cuda_sync(device)
        t0 = time.perf_counter()
        forward_only(model, batches, delta_scaler_t)
        cuda_sync(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0 / n_samples)

    return {
        "forward_gpu_ms_per_sample_mean": float(np.mean(times)),
        "forward_gpu_ms_per_sample_std": float(np.std(times)),
        "forward_gpu_ms_per_sample_min": float(np.min(times)),
        "forward_gpu_ms_per_sample_max": float(np.max(times)),
    }


def measure_forward_copy(model, batches, delta_scaler_t, n_samples, device, repeats, warmup):
    for _ in range(warmup):
        _ = forward_with_copy(model, batches, delta_scaler_t)
    cuda_sync(device)

    times = []
    pred = None
    for _ in range(repeats):
        cuda_sync(device)
        t0 = time.perf_counter()
        pred = forward_with_copy(model, batches, delta_scaler_t)
        cuda_sync(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0 / n_samples)

    return {
        "forward_with_copy_ms_per_sample_mean": float(np.mean(times)),
        "forward_with_copy_ms_per_sample_std": float(np.std(times)),
        "forward_with_copy_ms_per_sample_min": float(np.min(times)),
        "forward_with_copy_ms_per_sample_max": float(np.max(times)),
        "pred_direct": pred,
    }


def vector_metrics(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)

    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return mae, rmse, r2


def time_cpu_call(fn, n_samples: int, repeats: int):
    times = []
    last = None

    for _ in range(repeats):
        t0 = time.perf_counter()
        last = fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0 / n_samples)

    return {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
        "last": last,
    }


# =========================================================
# worker: checkpoint 하나 측정
# =========================================================
def worker_main(args):
    model_path = Path(args.model_path)
    test_npz = Path(args.test_npz)
    out_json = Path(args.out_json)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    deg, model_name, trial_dir, run_dir, seed = parse_model_path(model_path)

    print("=" * 100)
    print(f"[WORKER] degree={deg} model={model_name} seed={seed}")
    print(f"model_path = {model_path}")
    print(f"test_npz   = {test_npz}")
    print(f"device     = {args.device}")
    print(f"batch_size = {args.batch_size}")

    mod = load_module_from_path(args.hybrid_script)

    data = mod.load_npz(str(test_npz))
    ckpt, model, seq_scaler, glob_scaler, delta_scaler = mod.load_model_checkpoint(
        str(model_path),
        device=args.device,
    )

    hp = ckpt.get("hp", {})
    use_log_features = bool(hp.get("use_log_features", False))

    # -----------------------------------------------------
    # Preprocessing: baseline z0 생성 + 입력 생성 + scaling
    # -----------------------------------------------------
    t0 = time.perf_counter()
    seq_x, glob_x, y_true, z0, extra = mod.build_inputs_and_baseline(
        data,
        use_log_features=use_log_features,
    )

    seq_shape = seq_x.shape
    seq_x = scaler_transform(seq_x.reshape(-1, seq_x.shape[-1]), seq_scaler).reshape(seq_shape)
    glob_x = scaler_transform(glob_x, glob_scaler)

    t1 = time.perf_counter()

    n_samples = len(seq_x)
    preprocessing_ms_per_sample = (t1 - t0) * 1000.0 / n_samples

    # -----------------------------------------------------
    # Torch batch 구성
    # -----------------------------------------------------
    delta_scaler_t = {
        "mean": torch.tensor(np.asarray(delta_scaler["mean"], dtype=np.float32), device=args.device),
        "std": torch.tensor(np.asarray(delta_scaler["std"], dtype=np.float32), device=args.device),
    }

    batches = make_batches(
        seq_x=seq_x,
        glob_x=glob_x,
        z0=z0,
        q_total=np.asarray(data["Q_total"]),
        batch_size=args.batch_size,
        device=args.device,
    )

    model.eval()

    # -----------------------------------------------------
    # Neural forward 시간
    # -----------------------------------------------------
    gpu_stats = measure_forward_gpu(
        model=model,
        batches=batches,
        delta_scaler_t=delta_scaler_t,
        n_samples=n_samples,
        device=args.device,
        repeats=args.repeats,
        warmup=args.warmup,
    )

    copy_stats = measure_forward_copy(
        model=model,
        batches=batches,
        delta_scaler_t=delta_scaler_t,
        n_samples=n_samples,
        device=args.device,
        repeats=args.repeats,
        warmup=args.warmup,
    )

    pred_direct = copy_stats.pop("pred_direct")

    neural_direct_total_ms_per_sample = (
        preprocessing_ms_per_sample
        + copy_stats["forward_with_copy_ms_per_sample_mean"]
    )

    direct_mae, direct_rmse, direct_r2 = vector_metrics(pred_direct, y_true)

    # -----------------------------------------------------
    # Newton 시간
    # -----------------------------------------------------
    heuristic_newton_ms = float("nan")
    neural_newton_ms = float("nan")
    neural_end_to_end_ms = neural_direct_total_ms_per_sample

    heuristic_iter_mean = float("nan")
    neural_iter_mean = float("nan")
    heuristic_conv = float("nan")
    neural_conv = float("nan")

    if args.include_newton:
        def run_heuristic_newton():
            return mod.refine_batch(
                z0.astype(np.float64),
                data,
                tol=args.tol,
                max_iter=args.max_newton_iter,
            )

        h_time = time_cpu_call(
            run_heuristic_newton,
            n_samples=n_samples,
            repeats=args.newton_repeats,
        )
        href, hit, hconv = h_time["last"]
        heuristic_newton_ms = h_time["mean"]
        heuristic_iter_mean = float(np.mean(hit))
        heuristic_conv = float(np.mean(hconv))

        def run_neural_newton():
            return mod.refine_batch(
                pred_direct.astype(np.float64),
                data,
                tol=args.tol,
                max_iter=args.max_newton_iter,
            )

        n_time = time_cpu_call(
            run_neural_newton,
            n_samples=n_samples,
            repeats=args.newton_repeats,
        )
        nref, nit, nconv = n_time["last"]
        neural_newton_ms = n_time["mean"]
        neural_iter_mean = float(np.mean(nit))
        neural_conv = float(np.mean(nconv))

        neural_end_to_end_ms = neural_direct_total_ms_per_sample + neural_newton_ms

    row = {
        "degree": deg,
        "model": model_name,
        "trial_dir": trial_dir,
        "run_dir": run_dir,
        "seed": seed,
        "n_samples": n_samples,
        "device": args.device,
        "batch_size": args.batch_size,
        "model_path": str(model_path),
        "test_npz": str(test_npz),

        "best_epoch": ckpt.get("best_epoch", ""),
        "best_val_rmse": ckpt.get("best_val_rmse", ""),

        "direct_mae": direct_mae,
        "direct_rmse": direct_rmse,
        "direct_r2": direct_r2,

        "preprocessing_ms_per_sample": preprocessing_ms_per_sample,

        **gpu_stats,
        **copy_stats,

        "neural_direct_total_ms_per_sample": neural_direct_total_ms_per_sample,

        "heuristic_newton_ms_per_sample": heuristic_newton_ms,
        "neural_newton_ms_per_sample": neural_newton_ms,
        "neural_end_to_end_ms_per_sample": neural_end_to_end_ms,

        "heuristic_newton_iter_mean": heuristic_iter_mean,
        "neural_newton_iter_mean": neural_iter_mean,
        "heuristic_newton_converged_ratio": heuristic_conv,
        "neural_newton_converged_ratio": neural_conv,

        "forward_repeats": args.repeats,
        "forward_warmup": args.warmup,
        "newton_repeats": args.newton_repeats if args.include_newton else 0,
    }

    save_json(out_json, row)
    save_csv(out_json.with_suffix(".csv"), [row])

    print("[DONE WORKER]")
    print(json.dumps({
        "degree": row["degree"],
        "model": row["model"],
        "seed": row["seed"],
        "direct_rmse": row["direct_rmse"],
        "forward_gpu_ms_per_sample": row["forward_gpu_ms_per_sample_mean"],
        "forward_with_copy_ms_per_sample": row["forward_with_copy_ms_per_sample_mean"],
        "direct_total_ms_per_sample": row["neural_direct_total_ms_per_sample"],
        "neural_newton_ms_per_sample": row["neural_newton_ms_per_sample"],
        "end_to_end_ms_per_sample": row["neural_end_to_end_ms_per_sample"],
        "neural_iter": row["neural_newton_iter_mean"],
        "neural_conv": row["neural_newton_converged_ratio"],
    }, ensure_ascii=False, indent=2))


# =========================================================
# launcher: 병렬 실행
# =========================================================
def launcher_main(args):
    models_root = Path(args.models_root)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)

    worker_dir = out_dir / "worker_results"
    log_dir = out_dir / "logs"
    worker_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    model_paths = find_model_paths(
        root=models_root,
        degrees=args.degrees,
        models=args.models,
    )

    if not model_paths:
        raise FileNotFoundError(f"No best_model.pt found under {models_root}")

    print(f"[INFO] found {len(model_paths)} checkpoints")

    jobs = []
    gpu_ids = [x.strip() for x in args.gpu_ids.split(",") if x.strip()]
    if not gpu_ids:
        gpu_ids = ["0"]

    this_script = Path(__file__).resolve()

    for idx, model_path in enumerate(model_paths):
        deg, model_name, trial_dir, run_dir, seed = parse_model_path(model_path)
        test_npz = test_npz_for_degree(data_root, deg)

        if not test_npz.exists():
            print(f"[WARN] missing test npz, skip: {test_npz}")
            continue

        name = worker_result_name(model_path)
        out_json = worker_dir / f"{name}.json"
        log_path = log_dir / f"{name}.log"

        if args.skip_existing and out_json.exists():
            print(f"[SKIP existing] {out_json}")
            continue

        cmd = [
            args.python_bin,
            "-u",
            str(this_script),
            "--mode", "worker",
            "--hybrid_script", args.hybrid_script,
            "--model_path", str(model_path),
            "--test_npz", str(test_npz),
            "--out_json", str(out_json),
            "--device", args.device,
            "--batch_size", str(args.batch_size),
            "--repeats", str(args.repeats),
            "--warmup", str(args.warmup),
            "--newton_repeats", str(args.newton_repeats),
            "--tol", str(args.tol),
            "--max_newton_iter", str(args.max_newton_iter),
        ]

        if args.include_newton:
            cmd.append("--include_newton")

        gpu_id = gpu_ids[idx % len(gpu_ids)]

        jobs.append({
            "degree": deg,
            "model": model_name,
            "seed": seed,
            "cmd": cmd,
            "log_path": log_path,
            "out_json": out_json,
            "gpu_id": gpu_id,
        })

    print(f"[INFO] jobs to run   = {len(jobs)}")
    print(f"[INFO] max_parallel  = {args.max_parallel}")
    print(f"[INFO] gpu_ids       = {gpu_ids}")
    print(f"[INFO] include_newton= {args.include_newton}")

    running = []
    finished = []
    failed = []
    job_idx = 0

    while job_idx < len(jobs) or running:
        while job_idx < len(jobs) and len(running) < args.max_parallel:
            job = jobs[job_idx]
            job_idx += 1

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = job["gpu_id"]
            env["PYTHONUNBUFFERED"] = "1"

            env.setdefault("OMP_NUM_THREADS", str(args.cpu_threads_per_job))
            env.setdefault("MKL_NUM_THREADS", str(args.cpu_threads_per_job))
            env.setdefault("OPENBLAS_NUM_THREADS", str(args.cpu_threads_per_job))
            env.setdefault("NUMEXPR_NUM_THREADS", str(args.cpu_threads_per_job))

            log_f = open(job["log_path"], "w", encoding="utf-8")

            print("\n[START]")
            print(f"degree={job['degree']} model={job['model']} seed={job['seed']} gpu={job['gpu_id']}")
            print("cmd =", " ".join(shlex.quote(x) for x in job["cmd"]))
            print("log =", job["log_path"])

            proc = subprocess.Popen(
                job["cmd"],
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
            )

            job["proc"] = proc
            job["log_f"] = log_f
            job["start_time"] = time.time()
            running.append(job)

        still_running = []

        for job in running:
            ret = job["proc"].poll()

            if ret is None:
                still_running.append(job)
                continue

            job["log_f"].close()
            elapsed = time.time() - job["start_time"]

            if ret == 0 and job["out_json"].exists():
                print(f"[DONE] deg={job['degree']} model={job['model']} seed={job['seed']} time={elapsed:.1f}s")
                finished.append(job)
            else:
                print(
                    f"[FAIL] deg={job['degree']} model={job['model']} seed={job['seed']} "
                    f"ret={ret} time={elapsed:.1f}s log={job['log_path']}"
                )
                failed.append(job)

        running = still_running

        print(
            f"[STATUS] launched={job_idx}/{len(jobs)} "
            f"running={len(running)} finished={len(finished)} failed={len(failed)}"
        )

        time.sleep(args.sleep_sec)

    print("\n================ LAUNCH SUMMARY ================")
    print(f"finished = {len(finished)}")
    print(f"failed   = {len(failed)}")

    if failed:
        print("\nFailed jobs:")
        for job in failed:
            print(f"degree={job['degree']} model={job['model']} seed={job['seed']} log={job['log_path']}")

        if args.fail_on_error:
            raise SystemExit(1)

    collect_results(out_dir=out_dir, best_criterion=args.best_criterion)


# =========================================================
# collect: worker 결과 취합
# =========================================================
def aggregate_mean_std(rows: List[Dict[str, Any]], group_keys: List[str], metrics: List[str]):
    grouped = {}

    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        grouped.setdefault(key, []).append(row)

    out = []

    for key, sub in sorted(grouped.items()):
        r = {k: v for k, v in zip(group_keys, key)}
        r["n_runs"] = len(sub)

        for metric in metrics:
            vals = [safe_float(x.get(metric)) for x in sub]
            vals = [v for v in vals if math.isfinite(v)]

            if vals:
                r[f"{metric}_mean"] = float(np.mean(vals))
                r[f"{metric}_std"] = float(np.std(vals))
                r[f"{metric}_min"] = float(np.min(vals))
                r[f"{metric}_max"] = float(np.max(vals))
            else:
                r[f"{metric}_mean"] = float("nan")
                r[f"{metric}_std"] = float("nan")
                r[f"{metric}_min"] = float("nan")
                r[f"{metric}_max"] = float("nan")

        out.append(r)

    return out


def make_paper_rows(summary_rows):
    paper = []

    for r in summary_rows:
        paper.append({
            "Degree": r["degree"],
            "Model": str(r["model"]).upper(),
            "Direct RMSE": fmt_sci(r["direct_rmse_mean"], 3),
            "Forward GPU ms/sample": fmt_ms(r["forward_gpu_ms_per_sample_mean_mean"], 5),
            "Forward+Copy ms/sample": fmt_ms(r["forward_with_copy_ms_per_sample_mean_mean"], 5),
            "Direct Total ms/sample": fmt_ms(r["neural_direct_total_ms_per_sample_mean"], 5),
            "Neural+Newton ms/sample": fmt_ms(r["neural_end_to_end_ms_per_sample_mean"], 5),
            "Neural Iter.": fmt_ms(r["neural_newton_iter_mean_mean"], 3),
            "Neural Conv.": fmt_ratio(r["neural_newton_converged_ratio_mean"], 5),
        })

    return paper


def make_model_average_rows(summary_rows):
    """
    degree-model summary_rows를 받아서 model별 평균을 만든다.

    주의:
    summary_rows는 이미 aggregate_mean_std(raw_rows, ["degree", "model"], metrics)를 거친 결과이므로
    각 metric 이름이 다음처럼 되어 있다.

    direct_rmse_mean
    forward_gpu_ms_per_sample_mean_mean
    forward_with_copy_ms_per_sample_mean_mean
    neural_direct_total_ms_per_sample_mean
    neural_end_to_end_ms_per_sample_mean
    neural_newton_iter_mean_mean
    neural_newton_converged_ratio_mean

    따라서 여기서 aggregate_mean_std를 다시 쓰면 key가 _mean_mean_mean처럼 한 번 더 꼬인다.
    이 함수에서는 직접 모델별 평균을 계산한다.
    """

    def avg_of(rows, key):
        vals = [safe_float(r.get(key)) for r in rows]
        vals = [v for v in vals if math.isfinite(v)]
        if not vals:
            return float("nan")
        return float(np.mean(vals))

    models = sorted(set(str(r.get("model", "")).lower() for r in summary_rows if r.get("model")))

    paper = []

    for model in models:
        sub = [r for r in summary_rows if str(r.get("model", "")).lower() == model]

        paper.append({
            "Model": model.upper(),
            "N Degree-Model Rows": len(sub),

            "Avg Direct RMSE": fmt_sci(avg_of(sub, "direct_rmse_mean"), 3),

            "Avg Forward GPU ms/sample": fmt_ms(
                avg_of(sub, "forward_gpu_ms_per_sample_mean_mean"), 5
            ),

            "Avg Forward+Copy ms/sample": fmt_ms(
                avg_of(sub, "forward_with_copy_ms_per_sample_mean_mean"), 5
            ),

            "Avg Direct Total ms/sample": fmt_ms(
                avg_of(sub, "neural_direct_total_ms_per_sample_mean"), 5
            ),

            "Avg Neural+Newton ms/sample": fmt_ms(
                avg_of(sub, "neural_end_to_end_ms_per_sample_mean"), 5
            ),

            "Avg Iter.": fmt_ms(
                avg_of(sub, "neural_newton_iter_mean_mean"), 3
            ),

            "Avg Conv.": fmt_ratio(
                avg_of(sub, "neural_newton_converged_ratio_mean"), 5
            ),
        })

    return paper

def choose_best_by_degree(summary_rows, criterion: str):
    lower_is_better = criterion not in ["neural_newton_converged_ratio_mean"]
    best = {}

    for r in summary_rows:
        deg = r["degree"]
        val = safe_float(r.get(criterion))
        if deg is None or not math.isfinite(val):
            continue

        if deg not in best:
            best[deg] = r
            continue

        old = safe_float(best[deg].get(criterion))
        if lower_is_better:
            if val < old:
                best[deg] = r
        else:
            if val > old:
                best[deg] = r

    return [best[d] for d in sorted(best.keys())]


def collect_results(out_dir: Path, best_criterion: str):
    worker_dir = out_dir / "worker_results"
    json_paths = sorted(worker_dir.glob("*.json"))

    if not json_paths:
        print(f"[WARN] no worker json found: {worker_dir}")
        return

    raw_rows = []
    for p in json_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                raw_rows.append(json.load(f))
        except Exception as e:
            print(f"[WARN] failed to load {p}: {e}")

    raw_rows = sorted(raw_rows, key=lambda r: (r.get("degree", 9999), str(r.get("model", "")), r.get("seed", -1)))
    save_csv(out_dir / "inference_time_all_raw.csv", raw_rows)

    metrics = [
        "direct_mae",
        "direct_rmse",
        "direct_r2",
        "preprocessing_ms_per_sample",
        "forward_gpu_ms_per_sample_mean",
        "forward_with_copy_ms_per_sample_mean",
        "neural_direct_total_ms_per_sample",
        "heuristic_newton_ms_per_sample",
        "neural_newton_ms_per_sample",
        "neural_end_to_end_ms_per_sample",
        "heuristic_newton_iter_mean",
        "neural_newton_iter_mean",
        "heuristic_newton_converged_ratio",
        "neural_newton_converged_ratio",
    ]

    summary_rows = aggregate_mean_std(raw_rows, ["degree", "model"], metrics)
    save_csv(out_dir / "inference_time_degree_model_summary_raw.csv", summary_rows)

    paper_rows = make_paper_rows(summary_rows)
    paper_columns = [
        "Degree",
        "Model",
        "Direct RMSE",
        "Forward GPU ms/sample",
        "Forward+Copy ms/sample",
        "Direct Total ms/sample",
        "Neural+Newton ms/sample",
        "Neural Iter.",
        "Neural Conv.",
    ]

    save_csv(out_dir / "inference_time_paper_table.csv", paper_rows)
    (out_dir / "inference_time_paper_table.md").write_text(
        markdown_table(paper_rows, paper_columns),
        encoding="utf-8",
    )
    (out_dir / "inference_time_paper_table.tex").write_text(
        latex_table(
            paper_rows,
            paper_columns,
            caption=(
                "Inference-time comparison of the hybrid neural correction initializer "
                "across Taylor degrees and model backbones."
            ),
            label="tab:multidim_inference_time",
        ),
        encoding="utf-8",
    )

    best_degree_raw = choose_best_by_degree(summary_rows, best_criterion)
    best_degree_paper = make_paper_rows(best_degree_raw)

    save_csv(out_dir / "inference_time_best_by_degree_raw.csv", best_degree_raw)
    save_csv(out_dir / "inference_time_best_by_degree.csv", best_degree_paper)
    (out_dir / "inference_time_best_by_degree.md").write_text(
        markdown_table(best_degree_paper, paper_columns),
        encoding="utf-8",
    )
    (out_dir / "inference_time_best_by_degree.tex").write_text(
        latex_table(
            best_degree_paper,
            paper_columns,
            caption=(
                "Best inference-time result for each Taylor degree selected by "
                + best_criterion.replace("_", r"\_")
                + "."
            ),
            label="tab:multidim_inference_time_best_by_degree",
        ),
        encoding="utf-8",
    )

    model_avg_paper = make_model_average_rows(summary_rows)
    model_columns = [
        "Model",
        "N Degree-Model Rows",
        "Avg Direct RMSE",
        "Avg Forward GPU ms/sample",
        "Avg Forward+Copy ms/sample",
        "Avg Direct Total ms/sample",
        "Avg Neural+Newton ms/sample",
        "Avg Iter.",
        "Avg Conv.",
    ]

    save_csv(out_dir / "inference_time_model_average.csv", model_avg_paper)
    (out_dir / "inference_time_model_average.md").write_text(
        markdown_table(model_avg_paper, model_columns),
        encoding="utf-8",
    )
    (out_dir / "inference_time_model_average.tex").write_text(
        latex_table(
            model_avg_paper,
            model_columns,
            caption="Model-wise average inference time of the hybrid neural correction initializer.",
            label="tab:multidim_inference_time_model_average",
        ),
        encoding="utf-8",
    )

    print("\n[COLLECT DONE]")
    print("Saved:")
    for name in [
        "inference_time_all_raw.csv",
        "inference_time_degree_model_summary_raw.csv",
        "inference_time_paper_table.csv",
        "inference_time_paper_table.md",
        "inference_time_paper_table.tex",
        "inference_time_best_by_degree.csv",
        "inference_time_best_by_degree.md",
        "inference_time_best_by_degree.tex",
        "inference_time_model_average.csv",
        "inference_time_model_average.md",
        "inference_time_model_average.tex",
    ]:
        print(" -", out_dir / name)


# =========================================================
# parser
# =========================================================
def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--mode", choices=["launch", "worker", "collect"], required=True)

    # 공통
    p.add_argument("--hybrid_script", type=str, default="/home/seokjun/math_03_14/repeat_experiments_multidim_allinone_v2.py")
    p.add_argument("--out_dir", type=str, default="/home/seokjun/math_03_14/hybrid_inference_time_parallel_results")

    # launch / collect
    p.add_argument("--models_root", type=str, default="/home/seokjun/math_03_14/hybrid_params2_runs")
    p.add_argument("--data_root", type=str, default="/home/seokjun/math_03_14")
    p.add_argument("--degrees", nargs="+", type=int, default=[10, 15, 20, 25, 30, 35])
    p.add_argument("--models", nargs="+", default=["mlp", "lstm", "gru", "transformer"])
    p.add_argument("--max_parallel", type=int, default=4)
    p.add_argument("--gpu_ids", type=str, default="0")
    p.add_argument("--python_bin", type=str, default="python")
    p.add_argument("--cpu_threads_per_job", type=int, default=4)
    p.add_argument("--sleep_sec", type=float, default=5.0)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--fail_on_error", action="store_true")
    p.add_argument("--best_criterion", type=str, default="neural_direct_total_ms_per_sample_mean")

    # worker
    p.add_argument("--model_path", type=str)
    p.add_argument("--test_npz", type=str)
    p.add_argument("--out_json", type=str)

    # timing
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)

    p.add_argument("--include_newton", action="store_true")
    p.add_argument("--newton_repeats", type=int, default=1)
    p.add_argument("--tol", type=float, default=1e-12)
    p.add_argument("--max_newton_iter", type=int, default=20)

    return p


def main():
    args = build_parser().parse_args()

    if args.mode == "worker":
        if not args.model_path or not args.test_npz or not args.out_json:
            raise ValueError("--mode worker requires --model_path --test_npz --out_json")
        worker_main(args)

    elif args.mode == "launch":
        launcher_main(args)

    elif args.mode == "collect":
        collect_results(Path(args.out_dir), args.best_criterion)

    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()