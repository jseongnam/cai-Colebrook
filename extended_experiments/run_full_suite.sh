#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
DATA_ROOT="${DATA_ROOT:-data_extended}"
RESULT_ROOT="${RESULT_ROOT:-results_extended}"
EPOCHS="${EPOCHS:-300}"
TRAIN_N="${TRAIN_N:-20000}"
VAL_N="${VAL_N:-4000}"
TEST_N="${TEST_N:-4000}"
SEEDS="${SEEDS:-42 43 44}"
TRAIN_MODES="${TRAIN_MODES:-full no_state physics_only}"
OOD_REGIMES="${OOD_REGIMES:-high_flow high_roughness extreme_diameter_ratio combined}"
BRANCH_COUNTS="${BRANCH_COUNTS:-3 5}"
RUNTIME_DEVICES="${RUNTIME_DEVICES:-cpu $DEVICE}"
RUNTIME_SAMPLES="${RUNTIME_SAMPLES:-500}"
RUNTIME_REPEATS="${RUNTIME_REPEATS:-10}"
FORCE="${FORCE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

generate_split() {
  local branches="$1" regime="$2" split="$3" samples="$4" seed="$5"
  local path="${DATA_ROOT}/b${branches}_${regime}_${split}.npz"
  if [[ -f "$path" && "$FORCE" != "1" ]]; then
    "$PYTHON_BIN" - "$path" "$samples" "$branches" "$regime" <<'PY'
import sys
import numpy as np
path, expected_n, expected_b, expected_regime = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
with np.load(path, allow_pickle=False) as z:
    actual = (len(z["Q_total"]), int(z["branches"]), str(z["regime"]))
expected = (expected_n, expected_b, expected_regime)
if actual != expected:
    raise SystemExit(f"Existing dataset mismatch: {path}: actual={actual}, expected={expected}. Use a new DATA_ROOT or FORCE=1.")
PY
    echo "[resume] dataset exists: $path"
  else
    "$PYTHON_BIN" run_experiment.py generate --branches "$branches" --regime "$regime" \
      --samples "$samples" --seed "$seed" --output "$path"
  fi
}

train_one() {
  local branches="$1" mode="$2" seed="$3"
  local checkpoint="${RESULT_ROOT}/b${branches}/${mode}/seed_${seed}/model.pt"
  mkdir -p "$(dirname "$checkpoint")"
  if [[ -f "$checkpoint" && "$FORCE" != "1" ]]; then
    echo "[resume] checkpoint exists: $checkpoint"
    return
  fi
  "$PYTHON_BIN" run_experiment.py train \
    --train "${DATA_ROOT}/b${branches}_iid_train.npz" \
    --val "${DATA_ROOT}/b${branches}_iid_val.npz" \
    --input-mode "$mode" --seed "$seed" --device "$DEVICE" --epochs "$EPOCHS" \
    --output "$checkpoint"
}

evaluate_one() {
  local branches="$1" mode="$2" seed="$3" regime="$4"
  local base="${RESULT_ROOT}/b${branches}/${mode}/seed_${seed}"
  local output="${base}/eval_${regime}.csv"
  if [[ -f "$output" && "$FORCE" != "1" ]]; then
    echo "[resume] evaluation exists: $output"
    return
  fi
  "$PYTHON_BIN" run_experiment.py evaluate --checkpoint "${base}/model.pt" \
    --data "${DATA_ROOT}/b${branches}_${regime}_test.npz" --device "$DEVICE" \
    --output "$output"
}

# Two-branch IID and deliberately shifted test regimes.
generate_split 2 iid train "$TRAIN_N" 42
generate_split 2 iid val "$VAL_N" 43
generate_split 2 iid test "$TEST_N" 44
ood_seed=144
for regime in $OOD_REGIMES; do
  generate_split 2 "$regime" test "$TEST_N" "$ood_seed"
  ood_seed="$((ood_seed + 1))"
done

# Baseline-input ablation with independent training seeds.
for mode in $TRAIN_MODES; do
  for seed in $SEEDS; do
    train_one 2 "$mode" "$seed"
    for regime in iid $OOD_REGIMES; do
      evaluate_one 2 "$mode" "$seed" "$regime"
    done
  done
done

# Coupled end-to-end runtime on the primary two-branch test set.
for seed in $SEEDS; do
  base="${RESULT_ROOT}/b2/full/seed_${seed}"
  previous=""
  for runtime_device in $RUNTIME_DEVICES; do
    if [[ "$runtime_device" == "$previous" ]]; then
      continue
    fi
    previous="$runtime_device"
    safe_device="${runtime_device//:/_}"
    output="${base}/runtime_iid_${safe_device}.csv"
    if [[ -f "$output" && "$FORCE" != "1" ]]; then
      echo "[resume] runtime exists: $output"
      continue
    fi
    "$PYTHON_BIN" run_experiment.py runtime --checkpoint "${base}/model.pt" \
      --data "${DATA_ROOT}/b2_iid_test.npz" --device "$runtime_device" --samples "$RUNTIME_SAMPLES" \
      --warmup 2 --repeats "$RUNTIME_REPEATS" --output "$output"
  done
done

# Generality: repeat the same leakage-safe method for larger parallel systems.
for branches in $BRANCH_COUNTS; do
  generate_split "$branches" iid train "$TRAIN_N" "$((40 + branches))"
  generate_split "$branches" iid val "$VAL_N" "$((50 + branches))"
  generate_split "$branches" iid test "$TEST_N" "$((60 + branches))"
  for seed in $SEEDS; do
    train_one "$branches" full "$seed"
    evaluate_one "$branches" full "$seed" iid
  done
done

"$PYTHON_BIN" summarize_results.py --results "$RESULT_ROOT"
"$PYTHON_BIN" - "$RESULT_ROOT/run_environment.json" "$DEVICE" <<'PY'
import json, os, platform, subprocess, sys
import numpy, scipy, torch
payload = {
    "python": sys.version,
    "platform": platform.platform(),
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "requested_device": sys.argv[2],
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "environment": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]},
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PY
echo "Completed. Paper-ready tables are under ${RESULT_ROOT}."
