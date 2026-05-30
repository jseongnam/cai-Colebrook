#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_no_taylor_degree0_experiment.py

목적
----
기존 degree10 coupled Colebrook dataset을 이용해서
Taylor coefficient 없이 학습하는 degree0 ablation 실험을 수행한다.

핵심 아이디어
------------
기존 학습 코드는 coeffs key와 shape을 기대할 수 있으므로,
coeffs를 완전히 제거하지 않고 dummy zero coefficient로 대체한다.

기존:
  coeffs.shape = (N, 3, degree+1)

변경:
  coeffs.shape = (N, 3, 1)
  coeffs[:] = 0
  degree = 0

즉, 모델 입장에서는 degree0 Taylor coefficient를 받지만,
실제 Taylor 정보는 전혀 없는 상태다.

center 처리 옵션
----------------
--center_mode keep
  기존 center 유지.
  "Taylor coefficients만 제거한 ablation"으로 해석.

--center_mode zero
  center도 0으로 제거.
  "Taylor coefficients + Taylor expansion center 제거"로 해석.
  가장 엄격한 no-Taylor setting.

학습 방식
---------
기존 hybrid correction all-in-one 코드:
  repeat_experiments_multidim_allinone_v2.py

를 그대로 호출한다.

모델별 hyperparameter는 기존 최종 params2 설정을 사용한다.
- MLP: trial_013_mlp 기반
- LSTM: trial_030_lstm 기반
- GRU: trial_042_gru 기반
- Transformer: trial_064_transformer 기반

출력 구조
---------
data_root/
  multi_colebrook_data_deg0/
    parallel2_colebrook_deg0_train.npz
    parallel2_colebrook_deg0_val.npz
    parallel2_colebrook_deg0_test.npz

out_root/
  deg0/
    mlp_no_taylor_degree0/
    lstm_no_taylor_degree0/
    gru_no_taylor_degree0/
    transformer_no_taylor_degree0/
"""

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np


# =========================================================
# 기존 2번 코드 기반 최종 hyperparameter
# =========================================================
PARAMS = {
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
        "tag": "trial_013_mlp",
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
        "tag": "trial_030_lstm",
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
        "tag": "trial_042_gru",
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
        "tag": "trial_064_transformer",
    },
}


# =========================================================
# JSON safe
# =========================================================
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
        return float(obj)
    return obj


def save_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, ensure_ascii=False, indent=2)


# =========================================================
# Dataset creation
# =========================================================
def source_npz_path(data_root: Path, source_degree: int, split: str):
    return (
        data_root
        / f"multi_colebrook_data_deg{source_degree}"
        / f"parallel2_colebrook_deg{source_degree}_{split}.npz"
    )


def target_npz_path(data_root: Path, target_degree: int, split: str, suffix: str = ""):
    if suffix:
        folder = data_root / f"multi_colebrook_data_deg{target_degree}_{suffix}"
        name = f"parallel2_colebrook_deg{target_degree}_{suffix}_{split}.npz"
    else:
        folder = data_root / f"multi_colebrook_data_deg{target_degree}"
        name = f"parallel2_colebrook_deg{target_degree}_{split}.npz"

    return folder / name


def detect_coeff_shape(data: Dict[str, np.ndarray]):
    coeffs = np.asarray(data["coeffs"])
    if coeffs.ndim != 3:
        raise ValueError(f"Expected coeffs.ndim == 3, got shape={coeffs.shape}")
    return coeffs.shape


def make_degree0_npz(
    src_path: Path,
    dst_path: Path,
    center_mode: str,
    dtype: str = "float32",
):
    """
    기존 degree10 NPZ를 degree0 no-Taylor NPZ로 변환한다.
    """
    if not src_path.exists():
        raise FileNotFoundError(f"Source NPZ not found: {src_path}")

    src = np.load(src_path, allow_pickle=True)
    out = {}

    for k in src.files:
        arr = src[k]

        if k == "coeffs":
            coeffs = np.asarray(arr)
            n = coeffs.shape[0]
            eq_dim = coeffs.shape[1]

            # degree0 dummy coefficient.
            # 값은 0이므로 Taylor coefficient 정보는 없음.
            out[k] = np.zeros((n, eq_dim, 1), dtype=dtype)

        elif k == "center":
            center = np.asarray(arr)
            if center_mode == "keep":
                out[k] = center.astype(dtype) if np.issubdtype(center.dtype, np.number) else center
            elif center_mode == "zero":
                out[k] = np.zeros_like(center, dtype=dtype)
            else:
                raise ValueError(f"Unknown center_mode={center_mode}")

        elif k == "degree":
            old = np.asarray(arr)
            if old.shape == ():
                out[k] = np.array(0, dtype=old.dtype)
            else:
                out[k] = np.zeros_like(old)

        elif k in ["feature_desc", "expr_str"]:
            # object/string metadata는 유지하되 no Taylor ablation 표시를 추가하기 어렵기 때문에 그대로 둠.
            out[k] = arr

        else:
            # numeric arrays는 그대로 복사.
            # target, Q_total, D1, D2, eps1, eps2, L1, L2, rho, mu, g 등.
            if np.issubdtype(np.asarray(arr).dtype, np.number):
                out[k] = np.asarray(arr)
            else:
                out[k] = arr

    # metadata 추가
    out["no_taylor_ablation"] = np.array(True)
    out["source_npz"] = np.array(str(src_path))
    out["source_degree"] = np.array(10)
    out["target_degree"] = np.array(0)
    out["center_mode"] = np.array(center_mode)
    out["ablation_desc"] = np.array(
        "Degree-0 no-Taylor ablation: coeffs replaced by zeros with shape (N,3,1)."
    )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst_path, **out)

    # quick check
    chk = np.load(dst_path, allow_pickle=True)
    print(f"[CREATE] {dst_path}")
    print(f"         coeffs shape = {chk['coeffs'].shape}")
    if "center" in chk.files:
        print(f"         center shape = {chk['center'].shape}, center_mode={center_mode}")
    print(f"         keys = {chk.files}")


def create_degree0_dataset(
    data_root: Path,
    source_degree: int,
    center_mode: str,
    suffix: str,
    overwrite: bool,
):
    """
    train/val/test degree0 dataset 생성.
    """
    target_degree = 0

    created = []

    for split in ["train", "val", "test"]:
        src = source_npz_path(data_root, source_degree, split)
        dst = target_npz_path(data_root, target_degree, split, suffix=suffix)

        if dst.exists() and not overwrite:
            print(f"[SKIP] already exists: {dst}")
        else:
            make_degree0_npz(
                src_path=src,
                dst_path=dst,
                center_mode=center_mode,
            )

        created.append(dst)

    return created


# =========================================================
# Command construction
# =========================================================
def build_train_command(
    python_bin: str,
    train_script: Path,
    train_npz: Path,
    val_npz: Path,
    test_npz: Path,
    output_root: Path,
    hp: Dict[str, Any],
    epochs: int,
    batch_size: int,
    patience: int,
    num_runs: int,
    seed_start: int,
    device: str,
    tol: float,
    max_newton_iter: int,
):
    cmd = [
        python_bin,
        "-u",
        str(train_script),

        "--mode", "repeat",
        "--model", hp["model"],

        "--train_npz", str(train_npz),
        "--val_npz", str(val_npz),
        "--test_npz", str(test_npz),
        "--output_root", str(output_root),

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

    return cmd


# =========================================================
# Launcher
# =========================================================
def run_jobs_parallel(jobs: List[Dict[str, Any]], max_parallel: int, sleep_sec: float):
    running = []
    finished = []
    failed = []
    job_idx = 0

    while job_idx < len(jobs) or running:
        while job_idx < len(jobs) and len(running) < max_parallel:
            job = jobs[job_idx]
            job_idx += 1

            job["log_path"].parent.mkdir(parents=True, exist_ok=True)
            log_f = open(job["log_path"], "w", encoding="utf-8")

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            print("\n[START]")
            print(f"model = {job['model']}")
            print(f"log   = {job['log_path']}")
            print("cmd   =", " ".join(shlex.quote(x) for x in job["cmd"]))

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

            if ret == 0:
                print(f"[DONE] model={job['model']} time={elapsed:.1f}s")
                finished.append(job)
            else:
                print(
                    f"[FAIL] model={job['model']} ret={ret} "
                    f"time={elapsed:.1f}s log={job['log_path']}"
                )
                failed.append(job)

        running = still_running

        print(
            f"[STATUS] launched={job_idx}/{len(jobs)} "
            f"running={len(running)} finished={len(finished)} failed={len(failed)}"
        )

        time.sleep(sleep_sec)

    print("\n================ SUMMARY ================")
    print(f"finished = {len(finished)}")
    print(f"failed   = {len(failed)}")

    if failed:
        print("\nFailed jobs:")
        for job in failed:
            print(f"model={job['model']} log={job['log_path']}")
        raise SystemExit(1)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_script",
        type=str,
        default="/root/project/dataset/math_03_14/repeat_experiments_multidim_allinone_v2.py",
        help="기존 hybrid correction all-in-one 학습 코드",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/project/dataset/math_03_14",
    )
    parser.add_argument(
        "--source_degree",
        type=int,
        default=10,
        help="degree0 ablation을 만들 원본 dataset degree. 기본 degree10.",
    )
    parser.add_argument(
        "--center_mode",
        choices=["keep", "zero"],
        default="zero",
        help="center를 유지할지 제거할지. 엄격한 no-Taylor는 zero 권장.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="no_taylor_zero_center",
        help="degree0 dataset folder suffix.",
    )
    parser.add_argument(
        "--overwrite_dataset",
        action="store_true",
    )

    parser.add_argument(
        "--out_root",
        type=str,
        default="/root/project/dataset/math_03_14/no_taylor_degree0_runs",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default="/root/project/dataset/math_03_14/logs_no_taylor_degree0",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["mlp", "lstm", "gru", "transformer"],
    )

    parser.add_argument("--python_bin", type=str, default="python")
    parser.add_argument("--max_parallel", type=int, default=4)
    parser.add_argument("--sleep_sec", type=float, default=5.0)

    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--seed_start", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max_newton_iter", type=int, default=20)

    parser.add_argument(
        "--create_only",
        action="store_true",
        help="degree0 dataset만 만들고 학습은 실행하지 않음.",
    )
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="dataset 생성은 건너뛰고 기존 degree0 dataset으로 학습만 실행.",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_script = Path(args.train_script)
    out_root = Path(args.out_root)
    log_root = Path(args.log_root)

    if not train_script.exists():
        raise FileNotFoundError(f"train_script not found: {train_script}")

    args.models = [m.lower() for m in args.models]

    for m in args.models:
        if m not in PARAMS:
            raise ValueError(f"Unknown model={m}. Available={list(PARAMS.keys())}")

    # suffix 자동 보정
    suffix = args.suffix
    if not suffix:
        suffix = f"no_taylor_{args.center_mode}_center"

    # -----------------------------------------------------
    # 1. degree0 no-Taylor dataset 생성
    # -----------------------------------------------------
    if not args.train_only:
        print("\n============================================================")
        print("[STEP 1] Create degree0 no-Taylor dataset")
        print("============================================================")
        created = create_degree0_dataset(
            data_root=data_root,
            source_degree=args.source_degree,
            center_mode=args.center_mode,
            suffix=suffix,
            overwrite=args.overwrite_dataset,
        )

        manifest = {
            "source_degree": args.source_degree,
            "target_degree": 0,
            "center_mode": args.center_mode,
            "suffix": suffix,
            "created_files": [str(p) for p in created],
            "description": (
                "No-Taylor degree0 ablation dataset. "
                "coeffs are replaced by zeros with shape (N,3,1)."
            ),
        }
        save_json(data_root / f"multi_colebrook_data_deg0_{suffix}" / "manifest.json", manifest)

    if args.create_only:
        print("[DONE] create_only enabled. Stop before training.")
        return

    # -----------------------------------------------------
    # 2. train command 생성
    # -----------------------------------------------------
    print("\n============================================================")
    print("[STEP 2] Launch degree0 no-Taylor training")
    print("============================================================")

    train_npz = target_npz_path(data_root, 0, "train", suffix=suffix)
    val_npz = target_npz_path(data_root, 0, "val", suffix=suffix)
    test_npz = target_npz_path(data_root, 0, "test", suffix=suffix)

    for p in [train_npz, val_npz, test_npz]:
        if not p.exists():
            raise FileNotFoundError(f"Missing degree0 NPZ: {p}")

    out_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    jobs = []

    for model in args.models:
        hp = PARAMS[model]

        output_root = (
            out_root
            / "deg0"
            / f"{model}_no_taylor_{args.center_mode}_center_{hp['tag']}"
        )

        cmd = build_train_command(
            python_bin=args.python_bin,
            train_script=train_script,
            train_npz=train_npz,
            val_npz=val_npz,
            test_npz=test_npz,
            output_root=output_root,
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

        log_path = log_root / f"train_deg0_no_taylor_{args.center_mode}_{model}.log"

        jobs.append({
            "model": model,
            "cmd": cmd,
            "log_path": log_path,
            "output_root": output_root,
        })

    print(f"[INFO] total jobs   = {len(jobs)}")
    print(f"[INFO] models       = {args.models}")
    print(f"[INFO] max_parallel = {args.max_parallel}")
    print(f"[INFO] train_npz    = {train_npz}")
    print(f"[INFO] val_npz      = {val_npz}")
    print(f"[INFO] test_npz     = {test_npz}")
    print(f"[INFO] out_root     = {out_root}")
    print(f"[INFO] log_root     = {log_root}")

    run_jobs_parallel(
        jobs=jobs,
        max_parallel=args.max_parallel,
        sleep_sec=args.sleep_sec,
    )

    print("\n[DONE]")
    print("Degree0 no-Taylor experiment completed.")
    print("Results root:")
    print(out_root / "deg0")


if __name__ == "__main__":
    main()