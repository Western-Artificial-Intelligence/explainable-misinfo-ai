#!/usr/bin/env bash
# ===== baseline_model/experiments/run_baseline.sh =====
# Convenience script to run baseline training.
# Make executable: chmod +x baseline_model/experiments/run_baseline.sh
#
# The script:
#  - finds the repo root (two levels above this file),
#  - cd's there and runs the training module via `python -m` so package relative imports work.
set -euo pipefail

# Config (path relative to repo root)
CONFIG="baseline_model/configs/baseline.yaml"
RUN_NAME="baseline_run_local"
RESUME_FROM=""  # e.g. baseline_model/baseline_outputs/<run>/checkpoints/best_model.pt

# Compute repo root (two directories up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../" && pwd)"
cd "${REPO_ROOT}"

echo "[run_baseline] Repo root: ${REPO_ROOT}"
echo "[run_baseline] Using config: ${CONFIG}"
echo "[run_baseline] Run name: ${RUN_NAME}"

if [ -n "${RESUME_FROM}" ]; then
  echo "[run_baseline] Resuming from checkpoint: ${RESUME_FROM}"
  python3 -m baseline_model.training.train_baseline --config "${CONFIG}" --run_name "${RUN_NAME}" --resume_from "${RESUME_FROM}"
else
  python3 -m baseline_model.training.train_baseline --config "${CONFIG}" --run_name "${RUN_NAME}"
fi