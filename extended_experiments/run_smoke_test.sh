#!/usr/bin/env bash
set -euo pipefail
python -m unittest discover -s tests -v
python -m py_compile pipeflow_ext/*.py run_experiment.py summarize_results.py
