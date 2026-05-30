#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_parallel_hybrid_with_params2.py

목적
----
1번 코드의 hybrid correction 로직을 사용하고,
2번 코드에서 사용한 모델별 hyperparameter를 CLI 인자로 넘겨서
degree 10/15/20/25/30/35를 병렬로 학습 + 평가한다.

전제
----
1번 코드 파일이 다음처럼 실행 가능해야 한다.

python hybrid_multidim_correction.py \
  --mode repeat \
  --model lstm \
  --train_npz ... \
  --val_npz ... \
  --test_npz ... \
  --output_root ... \
  --use_log_features \
  --optimizer adamw \
  --loss_name mse \
  --dropout 0.0 \
  --lr 0.001 \
  ...

즉, 1번 코드의 argparse에 아래 인자들이 있어야 한다.
--mode, --model, --train_npz, --val_npz, --test_npz, --output_root,
--use_log_features, --optimizer, --loss_name, --dropout, --lr,
--weight_decay, --hidden_dims, --hidden_size, --num_layers,
--head_hidden, --head_layers, --d_model, --nhead, --ff_dim,
--use_cls_token, --batch_size, --epochs, --patience, --device,
--num_runs, --seed_start, --tol, --max_newton_iter
"""

import argparse
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, List


# =========================================================
# 2번 코드 기반 모델별 파라미터
# =========================================================
PARAMS_FROM_CODE2 = {
    "mlp": {
        "model": "mlp",
        "use_log_features": True,
        "optimizer": "adamw",
        "loss_name": "smoothl1",
        "dropout": 0.0,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "hidden_dims": [256, 256, 128],
        "hidden_size": 128,
        "num_layers": 2,
        "head_hidden": 128,
        "head_layers": 2,
        "d_model": 96,
        "nhead": 4,
        "ff_dim": 128,
        "use_cls_token": True,
        "source_trial": "trial_013_mlp",
    },

    "lstm": {
        "model": "lstm",
        "use_log_features": True,
        "optimizer": "adamw",
        "loss_name": "mse",
        "dropout": 0.0,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dims": [256, 256, 128],
        "hidden_size": 128,
        "num_layers": 1,
        "head_hidden": 128,
        "head_layers": 2,
        "d_model": 96,
        "nhead": 4,
        "ff_dim": 128,
        "use_cls_token": True,
        "source_trial": "trial_030_lstm",
    },

    "gru": {
        "model": "gru",
        "use_log_features": True,
        "optimizer": "adamw",
        "loss_name": "mse",
        "dropout": 0.0,
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "hidden_dims": [256, 256, 128],
        "hidden_size": 96,
        "num_layers": 1,
        "head_hidden": 128,
        "head_layers": 2,
        "d_model": 96,
        "nhead": 4,
        "ff_dim": 128,
        "use_cls_token": True,
        "source_trial": "trial_042_gru",
    },

    "transformer": {
        "model": "transformer",
        "use_log_features": True,
        "optimizer": "adamw",
        "loss_name": "mse",
        "dropout": 0.0,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dims": [256, 256, 128],
        "hidden_size": 128,
        "num_layers": 2,
        "head_hidden": 128,
        "head_layers": 2,
        "d_model": 96,
        "nhead": 4,
        "ff_dim": 192,
        "use_cls_token": True,
        "source_trial": "trial_064_transformer",
    },
}


def build_command(
    python_bin: str,
    train_script: str,
    data_root: str,
    out_root: str,
    degree: int,
    hp: Dict,
    epochs: int,
    batch_size: int,
    patience: int,
    num_runs: int,
    seed_start: int,
    device: str,
    tol: float,
    max_newton_iter: int,
):
    model = hp["model"]

    train_npz = (
        f"{data_root}/multi_colebrook_data_deg{degree}/"
        f"parallel2_colebrook_deg{degree}_train.npz"
    )
    val_npz = (
        f"{data_root}/multi_colebrook_data_deg{degree}/"
        f"parallel2_colebrook_deg{degree}_val.npz"
    )
    test_npz = (
        f"{data_root}/multi_colebrook_data_deg{degree}/"
        f"parallel2_colebrook_deg{degree}_test.npz"
    )

    output_root = (
        f"{out_root}/deg{degree}/{model}_{hp.get('source_trial', 'params2')}"
    )

    cmd = [
        python_bin,
        "-u",
        train_script,

        "--mode", "repeat",
        "--model", model,

        "--train_npz", train_npz,
        "--val_npz", val_npz,
        "--test_npz", test_npz,
        "--output_root", output_root,

        "--optimizer", hp["optimizer"],
        "--loss_name", hp["loss_name"],
        "--dropout", str(hp["dropout"]),
        "--lr", str(hp["lr"]),
        "--weight_decay", str(hp["weight_decay"]),

        "--hidden_dims", *[str(x) for x in hp["hidden_dims"]],
        "--hidden_size", str(hp["hidden_size"]),
        "--num_layers", str(hp["num_layers"]),
        "--head_hidden", str(hp["head_hidden"]),
        "--head_layers", str(hp["head_layers"]),
        "--d_model", str(hp["d_model"]),
        "--nhead", str(hp["nhead"]),
        "--ff_dim", str(hp["ff_dim"]),

        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--device", device,

        "--num_runs", str(num_runs),
        "--seed_start", str(seed_start),

        "--tol", str(tol),
        "--max_newton_iter", str(max_newton_iter),
    ]

    if hp.get("use_log_features", False):
        cmd.append("--use_log_features")

    if hp.get("use_cls_token", False):
        cmd.append("--use_cls_token")

    return cmd, output_root


def check_files(data_root: str, degrees: List[int]):
    missing = []

    for degree in degrees:
        base = Path(data_root) / f"multi_colebrook_data_deg{degree}"
        for split in ["train", "val", "test"]:
            p = base / f"parallel2_colebrook_deg{degree}_{split}.npz"
            if not p.exists():
                missing.append(str(p))

    if missing:
        print("[ERROR] Missing files:")
        for p in missing:
            print("  ", p)
        raise FileNotFoundError("Dataset files are missing.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_script",
        type=str,
        required=True,
        help="1번 hybrid correction 코드 경로",
    )
    parser.add_argument("--python_bin", type=str, default="python")

    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/seokjun/math_03_14",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="/home/seokjun/math_03_14/hybrid_params2_runs",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default="/home/seokjun/math_03_14/logs_hybrid_params2_runs",
    )

    parser.add_argument(
        "--degrees",
        nargs="+",
        type=int,
        default=[10, 15, 20, 25, 30, 35],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mlp", "lstm", "gru", "transformer"],
    )

    parser.add_argument("--max_parallel", type=int, default=4)
    parser.add_argument("--gpu_id", type=str, default="0")

    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=200)

    # repeat 실험 횟수
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--seed_start", type=int, default=42)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max_newton_iter", type=int, default=20)

    parser.add_argument("--sleep_sec", type=float, default=5.0)

    args = parser.parse_args()

    train_script = Path(args.train_script)
    if not train_script.exists():
        raise FileNotFoundError(f"train_script not found: {train_script}")

    args.models = [m.lower() for m in args.models]

    for m in args.models:
        if m not in PARAMS_FROM_CODE2:
            raise ValueError(
                f"Unknown model={m}. "
                f"Available: {list(PARAMS_FROM_CODE2.keys())}"
            )

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    Path(args.log_root).mkdir(parents=True, exist_ok=True)

    check_files(args.data_root, args.degrees)

    jobs = []

    for degree in args.degrees:
        for model in args.models:
            hp = PARAMS_FROM_CODE2[model]

            cmd, output_root = build_command(
                python_bin=args.python_bin,
                train_script=str(train_script),
                data_root=args.data_root,
                out_root=args.out_root,
                degree=degree,
                hp=hp,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                num_runs=args.num_runs,
                seed_start=args.seed_start,
                device=args.device,
                tol=args.tol,
                max_newton_iter=args.max_newton_iter,
            )

            log_path = (
                Path(args.log_root)
                / f"hybrid_params2_deg{degree}_{model}.log"
            )

            jobs.append({
                "degree": degree,
                "model": model,
                "cmd": cmd,
                "output_root": output_root,
                "log_path": log_path,
            })

    print(f"[INFO] total jobs    = {len(jobs)}")
    print(f"[INFO] degrees       = {args.degrees}")
    print(f"[INFO] models        = {args.models}")
    print(f"[INFO] max_parallel  = {args.max_parallel}")
    print(f"[INFO] batch_size    = {args.batch_size}")
    print(f"[INFO] num_runs      = {args.num_runs}")
    print(f"[INFO] gpu_id        = {args.gpu_id}")

    env_base = os.environ.copy()
    env_base["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    env_base["PYTHONUNBUFFERED"] = "1"

    # CPU oversubscription 방지
    env_base.setdefault("OMP_NUM_THREADS", "4")
    env_base.setdefault("MKL_NUM_THREADS", "4")
    env_base.setdefault("OPENBLAS_NUM_THREADS", "4")
    env_base.setdefault("NUMEXPR_NUM_THREADS", "4")

    running = []
    finished = []
    failed = []

    job_idx = 0

    while job_idx < len(jobs) or running:
        while job_idx < len(jobs) and len(running) < args.max_parallel:
            job = jobs[job_idx]
            job_idx += 1

            Path(job["output_root"]).mkdir(parents=True, exist_ok=True)

            log_f = open(job["log_path"], "w", encoding="utf-8")

            print("\n[START]")
            print(f"degree = {job['degree']}")
            print(f"model  = {job['model']}")
            print(f"log    = {job['log_path']}")
            print("cmd    =", " ".join(shlex.quote(x) for x in job["cmd"]))

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

        still_running = []

        for job in running:
            ret = job["proc"].poll()

            if ret is None:
                still_running.append(job)
                continue

            job["log_f"].close()
            elapsed = time.time() - job["start_time"]

            if ret == 0:
                print(
                    f"[DONE] degree={job['degree']} "
                    f"model={job['model']} "
                    f"time={elapsed:.1f}s"
                )
                finished.append(job)
            else:
                print(
                    f"[FAIL] degree={job['degree']} "
                    f"model={job['model']} "
                    f"ret={ret} "
                    f"time={elapsed:.1f}s "
                    f"log={job['log_path']}"
                )
                failed.append(job)

        running = still_running

        print(
            f"[STATUS] launched={job_idx}/{len(jobs)} "
            f"running={len(running)} "
            f"finished={len(finished)} "
            f"failed={len(failed)}"
        )

        time.sleep(args.sleep_sec)

    print("\n================ SUMMARY ================")
    print(f"finished = {len(finished)}")
    print(f"failed   = {len(failed)}")

    if failed:
        print("\nFailed jobs:")
        for job in failed:
            print(
                f"degree={job['degree']} "
                f"model={job['model']} "
                f"log={job['log_path']}"
            )
        raise SystemExit(1)

    print("\nAll jobs completed successfully.")


if __name__ == "__main__":
    main()