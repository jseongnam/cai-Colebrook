#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROFILE="${1:-paper}"
BASE_PYTHON="${BASE_PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_DEPS="${INSTALL_DEPS:-auto}"

if [[ "$PROFILE" != "paper" && "$PROFILE" != "smoke" ]]; then
  echo "Usage: bash run_from_zero.sh [paper|smoke]" >&2
  exit 2
fi

if ! command -v "$BASE_PYTHON" >/dev/null 2>&1; then
  echo "Python executable not found: $BASE_PYTHON" >&2
  exit 2
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "[setup] creating virtual environment: $VENV_DIR"
  "$BASE_PYTHON" -m venv "$VENV_DIR"
fi

PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"

needs_install=0
if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import numpy, pandas, scipy, torch, yaml
PY
then
  needs_install=1
fi

if [[ "$INSTALL_DEPS" == "1" || ( "$INSTALL_DEPS" == "auto" && "$needs_install" == "1" ) ]]; then
  echo "[setup] installing Python dependencies"
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PIP_BIN" install -r requirements.txt
elif [[ "$needs_install" == "1" ]]; then
  echo "Required Python packages are missing. Run with INSTALL_DEPS=1." >&2
  exit 2
fi

if [[ "${DEVICE:-auto}" == "auto" ]]; then
  DEVICE="$($PYTHON_BIN - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
)"
else
  DEVICE="${DEVICE}"
fi

if [[ "$DEVICE" == cuda* ]]; then
  "$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("DEVICE requests CUDA, but torch.cuda.is_available() is false")
print("[setup] CUDA device:", torch.cuda.get_device_name(0))
PY
fi

echo "[check] running physics/data smoke tests"
PYTHON="$PYTHON_BIN" "$PYTHON_BIN" -m unittest discover -s tests -v
"$PYTHON_BIN" -m py_compile pipeflow_ext/*.py run_experiment.py summarize_results.py

if [[ "$PROFILE" == "smoke" ]]; then
  export TRAIN_N="${TRAIN_N:-200}"
  export VAL_N="${VAL_N:-50}"
  export TEST_N="${TEST_N:-50}"
  export EPOCHS="${EPOCHS:-3}"
  export DATA_ROOT="${DATA_ROOT:-data_smoke}"
  export RESULT_ROOT="${RESULT_ROOT:-results_smoke}"
  export RUNTIME_SAMPLES="${RUNTIME_SAMPLES:-30}"
  export RUNTIME_REPEATS="${RUNTIME_REPEATS:-2}"
else
  export TRAIN_N="${TRAIN_N:-20000}"
  export VAL_N="${VAL_N:-4000}"
  export TEST_N="${TEST_N:-4000}"
  export EPOCHS="${EPOCHS:-300}"
  export DATA_ROOT="${DATA_ROOT:-data_extended}"
  export RESULT_ROOT="${RESULT_ROOT:-results_extended}"
  export RUNTIME_SAMPLES="${RUNTIME_SAMPLES:-500}"
  export RUNTIME_REPEATS="${RUNTIME_REPEATS:-10}"
fi

export PYTHON_BIN DEVICE
mkdir -p "$RESULT_ROOT/logs"
LOG_FILE="$RESULT_ROOT/logs/run_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "[run] profile=$PROFILE device=$DEVICE log=$LOG_FILE"
bash run_full_suite.sh 2>&1 | tee "$LOG_FILE"

echo "[done] checkpoints"
find "$RESULT_ROOT" -type f -name model.pt -print | sort
echo "[done] summary tables"
find "$RESULT_ROOT" -maxdepth 1 -type f \( -name 'paper_*.csv' -o -name 'paper_*.tex' \) -print | sort
