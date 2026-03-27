#!/bin/bash
set -e

echo "$(date): Starting training queue"

echo "$(date): [1/5] Baseline RoBERTa 384"
python -m baseline_model.training.train_baseline --config baseline_model/configs/baseline.yaml --run_name baseline_roberta_384 2>&1 | tee baseline_roberta_384.log
echo "$(date): [1/5] Done"

echo "$(date): [2/5] Two-Phase RoBERTa 384"
python -m baseline_model.training.train_twophase --config baseline_model/configs/twophase.yaml --run_name twophase_roberta_384 2>&1 | tee twophase_roberta_384.log
echo "$(date): [2/5] Done"

echo "$(date): [3/5] Two-Phase DistilRoBERTa 384"
python -m baseline_model.training.train_twophase --config baseline_model/configs/twophase_distil.yaml --run_name twophase_distil_384 2>&1 | tee twophase_distil_384.log
echo "$(date): [3/5] Done"

echo "$(date): [4/5] Baseline MLM 384"
python -m baseline_model.training.train_baseline --config baseline_model/configs/baseline.yaml --backbone ./models/mlm/roberta-misinfo-mlm --run_name baseline_mlm_384 2>&1 | tee baseline_mlm_384.log
echo "$(date): [4/5] Done"

echo "$(date): [5/5] Two-Phase MLM 384"
python -m baseline_model.training.train_twophase --config baseline_model/configs/twophase.yaml --backbone ./models/mlm/roberta-misinfo-mlm --run_name twophase_mlm_384 2>&1 | tee twophase_mlm_384.log
echo "$(date): [5/5] Done"

echo "$(date): All 5 models complete!"