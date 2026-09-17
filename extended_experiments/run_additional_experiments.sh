#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"
DATA_ROOT="${DATA_ROOT:-data_extended}"
RESULT_ROOT="${RESULT_ROOT:-results_extended}"
TRAIN_N="${TRAIN_N:-20000}"
VAL_N="${VAL_N:-4000}"
TEST_N="${TEST_N:-4000}"
EPOCHS="${EPOCHS:-300}"
SEEDS="${SEEDS:-42 43 44}"
OOD_REGIMES="${OOD_REGIMES:-high_flow high_roughness extreme_diameter_ratio combined}"
RUN_OOD="${RUN_OOD:-1}"
RUN_RUNTIME="${RUN_RUNTIME:-1}"
RUNTIME_SAMPLES="${RUNTIME_SAMPLES:-500}"
RUNTIME_REPEATS="${RUNTIME_REPEATS:-30}"
FORCE="${FORCE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$($PYTHON_BIN - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
)"
fi

generate_if_missing() {
  local regime="$1" split="$2" samples="$3" seed="$4"
  local path="${DATA_ROOT}/b2_${regime}_${split}.npz"
  if [[ -f "$path" && "$FORCE" != "1" ]]; then
    echo "[resume] dataset exists: $path"
  else
    "$PYTHON_BIN" run_experiment.py generate --branches 2 --regime "$regime" \
      --samples "$samples" --seed "$seed" --output "$path"
  fi
}

train_variant() {
  local variant="$1" target_mode="$2" flow_parameterization="$3" seed="$4"
  local checkpoint="${RESULT_ROOT}/b2/${variant}/seed_${seed}/model.pt"
  mkdir -p "$(dirname "$checkpoint")"
  if [[ -f "$checkpoint" && "$FORCE" != "1" ]]; then
    echo "[resume] checkpoint exists: $checkpoint"
  else
    "$PYTHON_BIN" run_experiment.py train \
      --train "${DATA_ROOT}/b2_iid_train.npz" \
      --val "${DATA_ROOT}/b2_iid_val.npz" \
      --input-mode full --target-mode "$target_mode" \
      --flow-parameterization "$flow_parameterization" \
      --seed "$seed" --device "$DEVICE" --epochs "$EPOCHS" \
      --output "$checkpoint"
  fi
}

evaluate_variant() {
  local variant="$1" seed="$2" regime="$3"
  local base="${RESULT_ROOT}/b2/${variant}/seed_${seed}"
  local output="${base}/eval_${regime}.csv"
  if [[ -f "$output" && "$FORCE" != "1" ]]; then
    echo "[resume] evaluation exists: $output"
  else
    "$PYTHON_BIN" run_experiment.py evaluate --checkpoint "${base}/model.pt" \
      --data "${DATA_ROOT}/b2_${regime}_test.npz" --device "$DEVICE" \
      --output "$output"
  fi
}

runtime_one() {
  local seed="$1" runtime_device="$2" batch_size="$3"
  local base="${RESULT_ROOT}/b2/correction_logit/seed_${seed}"
  local safe_device="${runtime_device//:/_}"
  local output="${base}/runtime_iid_${safe_device}_batch${batch_size}.csv"
  if [[ -f "$output" && "$FORCE" != "1" ]]; then
    echo "[resume] runtime exists: $output"
  else
    "$PYTHON_BIN" run_experiment.py runtime --checkpoint "${base}/model.pt" \
      --data "${DATA_ROOT}/b2_iid_test.npz" --device "$runtime_device" \
      --samples "$RUNTIME_SAMPLES" --inference-batch-size "$batch_size" \
      --warmup 3 --repeats "$RUNTIME_REPEATS" --output "$output"
  fi
}

# The same leakage-safe train/validation/test sets are used for every factorial cell.
generate_if_missing iid train "$TRAIN_N" 42
generate_if_missing iid val "$VAL_N" 43
generate_if_missing iid test "$TEST_N" 44
if [[ "$RUN_OOD" == "1" ]]; then
  ood_seed=144
  for regime in $OOD_REGIMES; do
    generate_if_missing "$regime" test "$TEST_N" "$ood_seed"
    ood_seed="$((ood_seed + 1))"
  done
fi

# Complete 2x2 ablation: target formulation x flow parameterization.
variants=(
  "correction_logit correction logit"
  "direct_logit direct logit"
  "correction_raw_ratio correction raw_ratio"
  "direct_raw_ratio direct raw_ratio"
)
for spec in "${variants[@]}"; do
  read -r variant target_mode flow_parameterization <<<"$spec"
  for seed in $SEEDS; do
    train_variant "$variant" "$target_mode" "$flow_parameterization" "$seed"
    evaluate_variant "$variant" "$seed" iid
    if [[ "$RUN_OOD" == "1" ]]; then
      for regime in $OOD_REGIMES; do
        evaluate_variant "$variant" "$seed" "$regime"
      done
    fi
  done
done

# End-to-end coupled runtime, with online (batch 1) and throughput conditions.
if [[ "$RUN_RUNTIME" == "1" ]]; then
  runtime_devices=(cpu)
  if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then
    runtime_devices+=(cuda)
  fi
  for seed in $SEEDS; do
    for runtime_device in "${runtime_devices[@]}"; do
      runtime_one "$seed" "$runtime_device" 1
      runtime_one "$seed" "$runtime_device" "$RUNTIME_SAMPLES"
    done
  done
fi

"$PYTHON_BIN" summarize_results.py --results "$RESULT_ROOT"
"$PYTHON_BIN" compare_factorial.py --results "$RESULT_ROOT"
echo "Completed factorial ablation and latency/throughput runtime experiments."
