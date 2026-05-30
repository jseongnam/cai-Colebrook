#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/project/dataset/math_03_14"

echo "============================================================"
echo "[1/2] Baseline CPU ms/sample measurement"
echo "============================================================"

mkdir -p "${ROOT}/baseline_cpu_ms_per_sample_results"

python "${ROOT}/measure_baseline_cpu_ms_per_sample.py" \
  --test_npz "${ROOT}/multi_colebrook_data_deg25/parallel2_colebrook_deg25_test.npz" \
  --out_dir "${ROOT}/baseline_cpu_ms_per_sample_results" \
  --initializers heuristic cond_haaland cond_swamee_jain cond_serghides \
  --repeats 30 \
  --warmup 1 \
  --tol 1e-12 \
  --max_newton_iter 20 \
  --cpu_threads 1

echo "============================================================"
echo "[1/2 DONE] Baseline CPU measurement finished"
echo "============================================================"

echo "============================================================"
echo "[2/2] Hybrid neural CPU-only inference measurement"
echo "============================================================"

mkdir -p "${ROOT}/hybrid_cpu_inference_time_results"

python "${ROOT}/measure_hybrid_inference_time_cpu_only.py" \
  --hybrid_script "${ROOT}/repeat_experiments_multidim_allinone_v2.py" \
  --models_root "${ROOT}/hybrid_params2_runs" \
  --data_root "${ROOT}" \
  --out_dir "${ROOT}/hybrid_cpu_inference_time_results" \
  --degrees 10 15 20 25 30 35 \
  --models mlp lstm gru transformer \
  --batch_size 4096 \
  --repeats 30 \
  --warmup 5 \
  --include_newton \
  --newton_repeats 1 \
  --tol 1e-12 \
  --max_newton_iter 20 \
  --cpu_threads 1

echo "============================================================"
echo "[2/2 DONE] Hybrid CPU-only measurement finished"
echo "============================================================"

echo "[ALL DONE]"
echo "Baseline result:"
echo "${ROOT}/baseline_cpu_ms_per_sample_results/baseline_cpu_paper_table.md"
echo
echo "Hybrid result:"
echo "${ROOT}/hybrid_cpu_inference_time_results/cpu_inference_time_paper_table.md"
echo "${ROOT}/hybrid_cpu_inference_time_results/cpu_inference_time_model_average.md"