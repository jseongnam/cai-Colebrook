#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_parallel_multidim_best_only.py

Multi-dimensional Colebrook training parallel launcher.

역할:
- degree 10/15/20/25/30/35/40 각각에 대해
- mlp/lstm/gru/transformer 모델을
- 별도 Python 프로세스로 병렬 실행한다.

중요:
- CUDA multiprocessing을 직접 쓰지 않는다.
- subprocess.Popen으로 독립 프로세스를 띄운다.
- H100처럼 GPU 메모리가 큰 환경에서 여러 학습 job을 동시에 돌리기 좋다.

사용 예:
python run_parallel_multidim_best_only.py \
  --script /home/seokjun/math_03_14/grid_search_multidim_best_only.py \
  --data_root /home/seokjun/math_03_14 \
  --out_root /home/seokjun/math_03_14/best_only_runs_multi_parallel \
  --log_root /home/seokjun/math_03_14/logs_best_only_multi_parallel \
  --degrees 10 15 20 25 30 35 \
  --models mlp lstm gru transformer \
  --max_parallel 4 \
  --epochs 5000 \
  --batch_size 1024 \
  --patience 200 \
  --device cuda \
  --rank_metric direct_rmse
"""

import argparse
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import List, Dict


def build_command(
    python_bin: str,
    script: str,
    data_root: str,
    out_root: str,
    degree: int,
    model: str,
    epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
    device: str,
    rank_metric: str,
    tol: float,
    max_newton_iter: int,
):
    train_npz = f"{data_root}/multi_colebrook_data_deg{degree}/parallel2_colebrook_deg{degree}_train.npz"
    val_npz = f"{data_root}/multi_colebrook_data_deg{degree}/parallel2_colebrook_deg{degree}_val.npz"
    test_npz = f"{data_root}/multi_colebrook_data_deg{degree}/parallel2_colebrook_deg{degree}_test.npz"

    out_dir = f"{out_root}/deg{degree}/{model}"

    cmd = [
        python_bin,
        "-u",
        script,
        "--train_npz", train_npz,
        "--val_npz", val_npz,
        "--test_npz", test_npz,
        "--out_dir", out_dir,
        "--models", model,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--patience", str(patience),
        "--seed", str(seed),
        "--device", device,
        "--rank_metric", rank_metric,
        "--tol", str(tol),
        "--max_newton_iter", str(max_newton_iter),
    ]

    return cmd, out_dir


def check_input_files(data_root: str, degrees: List[int]):
    missing = []

    for degree in degrees:
        base = Path(data_root) / f"multi_colebrook_data_deg{degree}"
        for split in ["train", "val", "test"]:
            path = base / f"parallel2_colebrook_deg{degree}_{split}.npz"
            if not path.exists():
                missing.append(str(path))

    if missing:
        print("\n[ERROR] Missing dataset files:")
        for p in missing:
            print("  ", p)
        raise FileNotFoundError("Some dataset files are missing.")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--script", type=str, required=True)
    ap.add_argument("--python_bin", type=str, default="python")

    ap.add_argument("--data_root", type=str, default="/home/seokjun/math_03_14")
    ap.add_argument("--out_root", type=str, default="/home/seokjun/math_03_14/best_only_runs_multi_parallel")
    ap.add_argument("--log_root", type=str, default="/home/seokjun/math_03_14/logs_best_only_multi_parallel")

    ap.add_argument("--degrees", nargs="+", type=int, default=[10, 15, 20, 25, 30, 35])
    ap.add_argument("--models", nargs="+", default=["mlp", "lstm", "gru", "transformer"])

    ap.add_argument("--max_parallel", type=int, default=4)
    ap.add_argument("--gpu_id", type=str, default="0")

    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--rank_metric", type=str, default="direct_rmse")

    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--max_newton_iter", type=int, default=20)

    ap.add_argument("--sleep_sec", type=float, default=5.0)

    args = ap.parse_args()

    script_path = Path(args.script)
    if not script_path.exists():
        raise FileNotFoundError(f"script not found: {script_path}")

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    Path(args.log_root).mkdir(parents=True, exist_ok=True)

    check_input_files(args.data_root, args.degrees)

    jobs = []
    for degree in args.degrees:
        for model in args.models:
            cmd, out_dir = build_command(
                python_bin=args.python_bin,
                script=args.script,
                data_root=args.data_root,
                out_root=args.out_root,
                degree=degree,
                model=model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                seed=args.seed,
                device=args.device,
                rank_metric=args.rank_metric,
                tol=args.tol,
                max_newton_iter=args.max_newton_iter,
            )

            log_path = Path(args.log_root) / f"train_multi_deg{degree}_{model}.log"

            jobs.append({
                "degree": degree,
                "model": model,
                "cmd": cmd,
                "out_dir": out_dir,
                "log_path": log_path,
            })

    print(f"[INFO] total jobs = {len(jobs)}")
    print(f"[INFO] max_parallel = {args.max_parallel}")
    print(f"[INFO] batch_size = {args.batch_size}")
    print(f"[INFO] gpu_id = {args.gpu_id}")

    running: List[Dict] = []
    finished = []
    failed = []

    env_base = os.environ.copy()
    env_base["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    env_base["PYTHONUNBUFFERED"] = "1"

    # CPU thread oversubscription 방지
    env_base.setdefault("OMP_NUM_THREADS", "4")
    env_base.setdefault("MKL_NUM_THREADS", "4")
    env_base.setdefault("OPENBLAS_NUM_THREADS", "4")
    env_base.setdefault("NUMEXPR_NUM_THREADS", "4")

    job_idx = 0

    while job_idx < len(jobs) or running:
        # 새 job 시작
        while job_idx < len(jobs) and len(running) < args.max_parallel:
            job = jobs[job_idx]
            job_idx += 1

            Path(job["out_dir"]).mkdir(parents=True, exist_ok=True)
            log_f = open(job["log_path"], "w", encoding="utf-8")

            print("\n[START]")
            print(f"degree={job['degree']} model={job['model']}")
            print("cmd =", " ".join(shlex.quote(x) for x in job["cmd"]))
            print("log =", job["log_path"])

            proc = subprocess.Popen(
                job["cmd"],
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env_base,
            )

            job["proc"] = proc
            job["log_f"] = log_f
            job["start_time"] = time.time()

            running.append(job)

        # 실행 중인 job 확인
        still_running = []

        for job in running:
            proc = job["proc"]
            ret = proc.poll()

            if ret is None:
                still_running.append(job)
                continue

            job["log_f"].close()
            elapsed = time.time() - job["start_time"]

            if ret == 0:
                print(
                    f"[DONE] degree={job['degree']} model={job['model']} "
                    f"time={elapsed:.1f}s"
                )
                finished.append(job)
            else:
                print(
                    f"[FAIL] degree={job['degree']} model={job['model']} "
                    f"ret={ret} time={elapsed:.1f}s log={job['log_path']}"
                )
                failed.append(job)

        running = still_running

        # 상태 출력
        print(
            f"[STATUS] launched={job_idx}/{len(jobs)} "
            f"running={len(running)} finished={len(finished)} failed={len(failed)}"
        )

        time.sleep(args.sleep_sec)

    print("\n================ SUMMARY ================")
    print(f"finished = {len(finished)}")
    print(f"failed   = {len(failed)}")

    if failed:
        print("\nFailed jobs:")
        for job in failed:
            print(f"degree={job['degree']} model={job['model']} log={job['log_path']}")
        raise SystemExit(1)

    print("\nAll jobs completed successfully.")


if __name__ == "__main__":
    main()