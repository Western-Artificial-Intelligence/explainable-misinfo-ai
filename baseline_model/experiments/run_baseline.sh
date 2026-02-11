#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${BASELINE_DIR}/.." && pwd)"

# Prefer repo-local venv if present, otherwise fall back to active python.
if [ -x "${PROJECT_ROOT}/venv/bin/python" ]; then
  PYTHON_BIN="${PROJECT_ROOT}/venv/bin/python"
else
  PYTHON_BIN="${PYTHON:-python3}"
fi

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m baseline_model.training.train_baseline \
  --config "${BASELINE_DIR}/configs/baseline.yaml" \
  --run_name "${RUN_NAME:-baseline_run_local}" \
  "$@"
