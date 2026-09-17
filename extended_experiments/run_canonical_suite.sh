#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"
DATA_ROOT="${DATA_ROOT:-data_canonical_v1}"
RESULT_ROOT="${RESULT_ROOT:-results_canonical_v1}"
TRAIN_N="${TRAIN_N:-20000}"
VAL_N="${VAL_N:-4000}"
TEST_N="${TEST_N:-4000}"
EPOCHS="${EPOCHS:-300}"
SEEDS="${SEEDS:-42 43 44}"
RUNTIME_SAMPLES="${RUNTIME_SAMPLES:-500}"
RUNTIME_REPEATS="${RUNTIME_REPEATS:-30}"
FORCE_RESULTS="${FORCE_RESULTS:-0}"
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

# Create once, then verify content fingerprints on every subsequent run.
"$PYTHON_BIN" canonical_data.py prepare --root "$DATA_ROOT" \
  --train-n "$TRAIN_N" --val-n "$VAL_N" --test-n "$TEST_N"
SUITE_ID="$($PYTHON_BIN canonical_data.py verify --root "$DATA_ROOT" --print-suite-id)"
echo "[canonical] suite_id=$SUITE_ID"

train_one() {
  local branches="$1" variant="$2" input_mode="$3" target_mode="$4" flow_mode="$5" seed="$6"
  local train="${DATA_ROOT}/b${branches}_iid_train.npz"
  local val="${DATA_ROOT}/b${branches}_iid_val.npz"
  local checkpoint="${RESULT_ROOT}/b${branches}/${variant}/seed_${seed}/model.pt"
  mkdir -p "$(dirname "$checkpoint")"
  if [[ -f "$checkpoint" && "$FORCE_RESULTS" != "1" ]]; then
    "$PYTHON_BIN" run_experiment.py verify-checkpoint --checkpoint "$checkpoint" \
      --train "$train" --val "$val" --suite-id "$SUITE_ID" \
      --input-mode "$input_mode" --target-mode "$target_mode" \
      --flow-parameterization "$flow_mode" --seed "$seed"
    echo "[resume] verified checkpoint: $checkpoint"
    return
  fi
  "$PYTHON_BIN" run_experiment.py train --train "$train" --val "$val" \
    --input-mode "$input_mode" --target-mode "$target_mode" \
    --flow-parameterization "$flow_mode" --suite-id "$SUITE_ID" \
    --seed "$seed" --device "$DEVICE" --epochs "$EPOCHS" --output "$checkpoint"
}

evaluate_one() {
  local branches="$1" variant="$2" seed="$3" regime="$4"
  local checkpoint="${RESULT_ROOT}/b${branches}/${variant}/seed_${seed}/model.pt"
  local data="${DATA_ROOT}/b${branches}_${regime}_test.npz"
  local output="${RESULT_ROOT}/b${branches}/${variant}/seed_${seed}/eval_${regime}.csv"
  if [[ -f "$output" && "$FORCE_RESULTS" != "1" ]]; then
    "$PYTHON_BIN" canonical_data.py verify-result --result "$output" --data "$data" \
      --checkpoint "$checkpoint" --suite-id "$SUITE_ID"
    echo "[resume] verified evaluation: $output"
    return
  fi
  "$PYTHON_BIN" run_experiment.py evaluate --checkpoint "$checkpoint" --data "$data" \
    --device "$DEVICE" --output "$output"
}

runtime_one() {
  local seed="$1" runtime_device="$2" batch_size="$3"
  local checkpoint="${RESULT_ROOT}/b2/correction_logit/seed_${seed}/model.pt"
  local data="${DATA_ROOT}/b2_iid_test.npz"
  local safe_device="${runtime_device//:/_}"
  local output="${RESULT_ROOT}/b2/correction_logit/seed_${seed}/runtime_iid_${safe_device}_batch${batch_size}.csv"
  if [[ -f "$output" && "$FORCE_RESULTS" != "1" ]]; then
    "$PYTHON_BIN" canonical_data.py verify-result --result "$output" --data "$data" \
      --checkpoint "$checkpoint" --suite-id "$SUITE_ID"
    echo "[resume] verified runtime: $output"
    return
  fi
  "$PYTHON_BIN" run_experiment.py runtime --checkpoint "$checkpoint" --data "$data" \
    --device "$runtime_device" --samples "$RUNTIME_SAMPLES" \
    --inference-batch-size "$batch_size" --warmup 3 --repeats "$RUNTIME_REPEATS" \
    --output "$output"
}

# Two-branch experiments all share the exact same frozen train/val/test files.
variants=(
  "correction_logit full correction logit"
  "no_state no_state correction logit"
  "physics_only physics_only correction logit"
  "direct_logit full direct logit"
  "correction_raw_ratio full correction raw_ratio"
  "direct_raw_ratio full direct raw_ratio"
)
regimes=(iid high_flow high_roughness extreme_diameter_ratio combined)
for spec in "${variants[@]}"; do
  read -r variant input_mode target_mode flow_mode <<<"$spec"
  for seed in $SEEDS; do
    train_one 2 "$variant" "$input_mode" "$target_mode" "$flow_mode" "$seed"
    for regime in "${regimes[@]}"; do
      evaluate_one 2 "$variant" "$seed" "$regime"
    done
  done
done

# Larger parallel systems use their own frozen branch-count-specific split.
for branches in 3 5; do
  for seed in $SEEDS; do
    train_one "$branches" correction_logit full correction logit "$seed"
    evaluate_one "$branches" correction_logit "$seed" iid
  done
done

# Online latency and batched throughput on exactly the same b2 IID test file.
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

"$PYTHON_BIN" canonical_data.py verify --root "$DATA_ROOT"
"$PYTHON_BIN" summarize_results.py --results "$RESULT_ROOT"
"$PYTHON_BIN" compare_factorial.py --results "$RESULT_ROOT"
echo "Completed immutable canonical suite: $SUITE_ID"
