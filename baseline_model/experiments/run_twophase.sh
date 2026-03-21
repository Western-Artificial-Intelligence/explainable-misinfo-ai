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

# Default config and run name
CONFIG="${CONFIG:-twophase.yaml}"
RUN_NAME="${RUN_NAME:-twophase_run_local}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m baseline_model.training.train_twophase \
  --config "${BASELINE_DIR}/configs/${CONFIG}" \
  --run_name "${RUN_NAME}" \
  "$@"
