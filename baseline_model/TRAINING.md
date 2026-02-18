# Training & Evaluation Pipeline

## Prerequisites

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

Requires Python 3.10+ (recommended 3.11). GPU with CUDA is recommended but not required.

---

## Overview

There are two model variants to train and compare:

1. **Baseline** — RoBERTa-base encoder with LoRA + domain-adversarial classifier
2. **+MLM** — Same architecture but encoder is initialized from a RoBERTa checkpoint that was further pre-trained on a misinformation corpus

The pipeline is:

```
[Step 1] (optional) MLM pre-training on misinfo corpus
                        |
                        v
[Step 2] Train classifier (baseline or +MLM)
                        |
                        v
[Step 3] Evaluate on test set
                        |
                        v
[Step 4] Run inference on new claims
```

---

## Step 1: MLM Pre-training (only for +MLM variant)

This adapts RoBERTa to misinformation language by masked language modeling on the `ioverho/misinfo-general` dataset from HuggingFace.

```bash
python mlm/train_mlm.py
```

**What it does:**
- Downloads the misinfo-general dataset (~multiple year splits)
- Wraps text with `<CLAIM>` / `<ARTICLE>` special tokens
- Trains RoBERTa MLM head for 3 epochs
- Saves the adapted encoder to `./models/mlm/roberta-misinfo-mlm`

**Key hyperparameters** (configured at top of `mlm/train_mlm.py`):

| Parameter | Value |
|---|---|
| Base model | `roberta-base` |
| Epochs | 3 |
| Batch size | 32 |
| Learning rate | 2e-5 |
| MLM probability | 0.15 |
| Max sequence length | 256 |
| Warmup ratio | 0.06 |
| Mixed precision | fp16 |

**Output:** `./models/mlm/roberta-misinfo-mlm/` (model weights + tokenizer)

**Duration:** Several hours on GPU depending on dataset size. Can be run on CPU but will be very slow.

Skip this step if you only want the baseline model.

---

## Step 2: Train the Classifier

### Baseline (roberta-base encoder)

```bash
python -m baseline_model.training.train_baseline \
    --config baseline_model/configs/baseline.yaml \
    --run_name baseline_v1
```

### +MLM variant (misinfo-adapted encoder)

```bash
python -m baseline_model.training.train_baseline \
    --config baseline_model/configs/baseline.yaml \
    --run_name mlm_v1 \
    --backbone ./models/mlm/roberta-misinfo-mlm
```

The `--backbone` flag overrides the `model.backbone` field in the yaml config. If omitted, it defaults to `roberta-base`.

### What happens during training

1. Loads parquet data from `data/processed/final_datasets/` (39,648 train / 11,328 val / 5,665 test rows)
2. Builds `input_text` from `claim_text` + `article_text` using `<CLAIM>` / `<ARTICLE>` tokens
3. Tokenizes with RoBERTa tokenizer (max_length=256)
4. Initializes `DomainAdversarialClassifier` with LoRA adapters on encoder
5. Trains with combined loss: 3-way CE (LIAR) + binary coarse loss (other sources) + domain adversarial loss
6. Validates on val set after every epoch, saves best checkpoint by val macro-F1
7. Prints full metrics report (train/val loss, accuracy, macro-F1, ECE, per-class P/R/F1, confusion matrix) every 1,000 epochs
8. Early stops after 10 epochs with no val macro-F1 improvement

### Configuration

All hyperparameters are in `baseline_model/configs/baseline.yaml`:

```yaml
model:
  backbone: roberta-base       # or path to MLM checkpoint
  max_length: 256              # token sequence length
  lambda_adv: 0.5              # domain adversarial loss weight

lora:
  enabled: true
  r: 8                         # LoRA rank
  alpha: 16                    # LoRA scaling
  dropout: 0.05
  target_modules: ["query", "value"]

training:
  batch_size: 16
  lr: 1e-4
  epochs: 35000                # max epochs (early stopping will kick in)
  weight_decay: 0.01
  warmup_ratio: 0.06           # linear warmup over 6% of total steps
  grad_clip_norm: 1.0          # gradient clipping max norm
  seed: 42
  device: auto                 # auto-detects GPU/CPU
  grad_accum_steps: 1
  save_every_epoch: false
  early_stopping_patience: 10  # stop after N epochs with no improvement
  log_every: 1000              # print full metrics every N epochs
```

### Outputs

All outputs are saved to `baseline_outputs/<run_name>/`:

```
baseline_outputs/<run_name>/
├── checkpoints/
│   └── best.ckpt              # best model by val macro-F1
└── logs/
    └── metrics_log.jsonl      # per-epoch metrics (train_loss, val_loss, val_acc, val_macro_f1, val_ece, best_val_f1)
```

### Resume from checkpoint

If training is interrupted:

```bash
python -m baseline_model.training.train_baseline \
    --config baseline_model/configs/baseline.yaml \
    --run_name baseline_v1_resumed \
    --resume_from baseline_outputs/baseline_v1/checkpoints/best.ckpt
```

---

## Step 3: Evaluate on Test Set

```bash
python -m baseline_model.training.evaluate \
    --checkpoint baseline_outputs/baseline_v1/checkpoints/best.ckpt \
    --config baseline_model/configs/baseline.yaml
```

**Metrics reported:**
- Unified macro-F1 (primary metric, computed on 3-way labels)
- Accuracy
- Per-class precision, recall, F1 (false / mixed / true)
- Confusion matrix
- Expected Calibration Error (ECE)

Results are saved to `baseline_outputs/evaluation_results.json`.

### Comparing baseline vs +MLM

Run evaluation for both checkpoints and compare:

```bash
# Evaluate baseline
python -m baseline_model.training.evaluate \
    --checkpoint baseline_outputs/baseline_v1/checkpoints/best.ckpt \
    --config baseline_model/configs/baseline.yaml

# Evaluate +MLM
python -m baseline_model.training.evaluate \
    --checkpoint baseline_outputs/mlm_v1/checkpoints/best.ckpt \
    --config baseline_model/configs/baseline.yaml
```

Key comparisons per the architecture spec:
- Baseline vs +MLM on unified macro-F1
- Improvements in OOD and calibration (ECE)

---

## Step 4: Run Inference

Single claim/article prediction:

```bash
python baseline_model/training/infer_example.py \
    --checkpoint baseline_outputs/baseline_v1/checkpoints/best.ckpt \
    --backbone roberta-base \
    --claim "COVID vaccines contain microchips" \
    --article "Article text here..." \
    --device cpu
```

For the +MLM variant, use the MLM backbone:

```bash
python baseline_model/training/infer_example.py \
    --checkpoint baseline_outputs/mlm_v1/checkpoints/best.ckpt \
    --backbone ./models/mlm/roberta-misinfo-mlm \
    --claim "COVID vaccines contain microchips" \
    --article "Article text here..."
```

**Output:** JSON with 3-way probabilities and predicted label (0=false, 1=mixed, 2=true)

---

## Data

Training data lives in `data/processed/final_datasets/`:

| File | Rows | Description |
|---|---|---|
| `unified_train.parquet` | 39,648 | Training split |
| `unified_val.parquet` | 11,328 | Validation split |
| `unified_test.parquet` | 5,665 | Test split |

**Sources:** LIAR (3-way labels), CoAID, FakeHealth, FakeNewsNet, HoVER (binary labels)

**Key columns:**
- `claim_text` — the claim to classify
- `article_text` — supporting/refuting article (may be null)
- `content_status` — `title_only`, `full_article`, or `partial_article`
- `label_3way` — 0 (false), 1 (mixed), 2 (true), or -100 (not applicable)
- `label_bin` — 0 (false), 1 (true), or -100 (not applicable)
- `source_id` — 0-4 (LIAR, CoAID, FakeHealth, FakeNewsNet, HoVER)

---

## Hardware Notes

- **CPU only:** Works but slow. Expect ~minutes per epoch with batch_size=16 and ~40k rows.
- **Single GPU (4-6GB VRAM):** Sufficient for roberta-base with LoRA at max_length=256 and batch_size=16.
- **Multi-GPU:** Not currently supported (single device training).
- **MLM pre-training** is more compute-intensive than classifier training — GPU strongly recommended.

---

## Quick Reference

```bash
# Full pipeline: MLM + baseline + +MLM + evaluate both

# 1. MLM pre-training
python mlm/train_mlm.py

# 2. Train baseline
python -m baseline_model.training.train_baseline \
    --config baseline_model/configs/baseline.yaml \
    --run_name baseline_v1

# 3. Train +MLM
python -m baseline_model.training.train_baseline \
    --config baseline_model/configs/baseline.yaml \
    --run_name mlm_v1 \
    --backbone ./models/mlm/roberta-misinfo-mlm

# 4. Evaluate both
python -m baseline_model.training.evaluate \
    --checkpoint baseline_outputs/baseline_v1/checkpoints/best.ckpt \
    --config baseline_model/configs/baseline.yaml

python -m baseline_model.training.evaluate \
    --checkpoint baseline_outputs/mlm_v1/checkpoints/best.ckpt \
    --config baseline_model/configs/baseline.yaml

# 5. Inference
python baseline_model/training/infer_example.py \
    --checkpoint baseline_outputs/baseline_v1/checkpoints/best.ckpt \
    --backbone roberta-base \
    --claim "Your claim" --article "Your article"
```