Here’s a clean, polished **README.md–ready** version with proper Markdown structure and clarity:

---

# Baseline Misinformation Classification Training

**RoBERTa + LoRA + Domain-Adversarial Learning**

This folder contains a complete, runnable baseline pipeline for training misinformation classification models with optional domain-adversarial learning and parameter-efficient fine-tuning.

The pipeline is designed to work **out of the box**:

* Loads a merged HuggingFace dataset from disk, **or**
* Automatically creates a small synthetic dataset if none is found—allowing local end-to-end testing.

---

## 📦 Features

* **Dataset Loading**

  * Reads a merged dataset saved at `data/merged_dataset/`
  * Falls back to generating a synthetic dataset with the same schema if missing

* **Tokenization**

  * Uses a RoBERTa tokenizer
  * Adds special tokens:

    * `<CLAIM>`, `</CLAIM>`
    * `<ARTICLE>`, `</ARTICLE>`

* **Model Architecture**

  * `DomainAdversarialClassifier`

    * RoBERTa backbone
    * `[CLS]` representation
    * Label prediction head
    * Domain prediction head

* **LoRA Support**

  * Uses PEFT LoRA adapters if the `peft` package is installed
  * Automatically falls back to full fine-tuning otherwise

* **Training Pipeline**

  * Progress bars and batch-level logging
  * Metric reporting
  * Checkpoint saving
  * Resume-from-checkpoint support

* **Evaluation & Inference**

  * Evaluation script
  * Simple inference script

---

## ⚠️ Assumptions

### Dataset Location

If you already produced a merged HuggingFace dataset using the scripts in `preprocess_scripts/`, save it to:

```
data/merged_dataset/
```

using:

```python
datasets.Dataset.save_to_disk(...)
```

The training script will automatically detect and load it.

---

### Synthetic Dataset Fallback

If no dataset is present at that path, the pipeline will:

* Generate a small synthetic dataset
* Match the expected schema
* Allow full local testing without external downloads

---

## 📊 Dataset Schema

Each example in the dataset contains:

| Field        | Type   | Description                                                                                |
| ------------ | ------ | ------------------------------------------------------------------------------------------ |
| `input_text` | string | Formatted as `<CLAIM> ... </CLAIM> <ARTICLE> ... </ARTICLE>`                               |
| `label_3way` | int    | `{0,1,2}` for LIAR-style rows (`0=false`, `1=mixed`, `2=true`) or `-100` if not applicable |
| `label_bin`  | int    | `{0,1}` for binary rows (`0=false`, `1=true-ish`) or `-100` if not applicable              |
| `source_id`  | int    | Dataset/source identifier (`0..N-1`)                                                       |
| `split`      | string | One of `train`, `val`, `test`                                                              |

---

## 🚀 How to Use

### 1️⃣ Configure (Optional)

Edit the configuration file:

```
configs/baseline.yaml
```

to adjust hyperparameters.

---

### 2️⃣ Train the Model

Run:

```bash
./baseline_model/experiments/run_baseline.sh
```

---

## 📁 Outputs

All trained artifacts and checkpoints are saved under:

```
outputs/checkpoints/<run_name>/
```
Here’s a polished **`README.md`** version for your project with proper formatting, headings, and code blocks for clarity:

````markdown
# Misinformation Classifier Baseline (RoBERTa + LoRA + Domain-Adversarial)

## What the Model Does

This project trains a text classifier to detect misinformation. The model outputs a probability distribution over three classes:

- `false` (index 0)  
- `mixed` (index 1)  
- `true` (index 2)  

**Model architecture:**

- **Encoder:** RoBERTa-base  
- **Label head:** small MLP mapping CLS representation to 3 logits  
- **Domain head:** adversarial MLP (with gradient reversal) mapping CLS to source ID logits (encourages domain-invariant features)  
- **Optional LoRA adapters** (via PEFT) to reduce full fine-tuning cost  

---

## Dataset Preparation

The training pipeline expects a merged dataset saved as a HuggingFace `DatasetDict` at `data/merged_dataset/`. Required columns:

- `input_text`: string like `<CLAIM> ... </CLAIM> <ARTICLE> ... </ARTICLE>`  
- `label_3way`: integer `{0,1,2}` or `-100` if not applicable  
- `label_bin`: integer `{0,1}` or `-100` if not applicable  
- `source_id`: integer indicating data source  
- `split`: string in `{"train","val","test"}`  

> If no dataset is present at `data/merged_dataset/`, a small synthetic dataset will be generated automatically for testing and development.

---

## Installation

Create and activate a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
````

Upgrade pip and install requirements:

```bash
pip install --upgrade pip
pip install -r baseline_model/requirements_extra.txt
```

If your environment already has NumPy 2.x installed, run:

```bash
pip install "numpy<2"
```

Optional (for LoRA support):

```bash
pip install peft
```

---

## Training

Run the training script:

```bash
RUN_NAME=my_baseline_run ./baseline_model/experiments/run_baseline.sh
```

**Outputs:**

* Checkpoints: `baseline_outputs/<run_name>/checkpoints/`
* Logs (epoch-level metrics): `baseline_outputs/<run_name>/logs/metrics_log.jsonl`

---

## Resume Training

To resume from a saved checkpoint:

```bash
python3 -m baseline_model.training.train_baseline \
    --config baseline_model/configs/baseline.yaml \
    --run_name resumed_run \
    --resume_from baseline_outputs/<run_name>/checkpoints/checkpoint_epochX.pt
```

The script will load the model and optimizer states and continue training from the checkpoint epoch + 1.

---

## Evaluation

Run evaluation on the saved "best" checkpoint:

```bash
python3 -c "
from baseline_model.training.evaluate import evaluate_checkpoint, load_config
cfg = load_config('configs/baseline.yaml')
evaluate_checkpoint('baseline_outputs/<run_name>/checkpoints/best_model.pt', cfg, device='cpu')
"
```

Results are saved to:
`baseline_outputs/<run_name>/evaluation_results.json`

---

## Inference

A minimal inference example:

```bash
python3 baseline_model/training/infer_example.py
```

> For a REST API, see `api/` directory. Integrate `best_model.pt` and tokenizer for serving predictions.

---

## Hardware Requirements

* CPU is sufficient for small datasets and testing
* GPU (NVIDIA CUDA) recommended for training RoBERTa-base:

  * Batch size 16, max_len 256 → 8–16GB VRAM
* LoRA reduces resource usage by training only adapter weights

---

## Output Directory Structure

By default, outputs are saved under `baseline_outputs/<run_name>/`:

```
checkpoints/           # checkpoint files including best_model.pt
logs/metrics_log.jsonl  # epoch-level metrics
```

```

This version:  
- Uses **Markdown headings** for clarity  
- Formats **commands** and **paths** as code blocks  
- Breaks sections into **logical subsections** for easy reading  

---

If you want, I can also add a **“Quickstart” section** at the top with **one-liner commands** for running training, evaluation, and inference to make it super beginner-friendly.  

Do you want me to do that?
```
source ../venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_extra.txt
python -m pip install --upgrade --force-reinstall "numpy<2"
python -c "import torch, numpy; print('torch', torch.__version__); print('numpy', numpy.__version__)"
python training/train_baseline.py --config configs/baseline.yaml --run_name my_baseline_run

cd ..
RUN_NAME=my_baseline_run ./baseline_model/experiments/run_baseline.sh

cd /Users/aryankhimani/Downloads/WAI_Project/explainable-misinfo-ai

conda deactivate || true
conda deactivate || true

python -m pip install --upgrade pip
python -m pip install --upgrade --force-reinstall -r baseline_model/requirements_extra.txt
python -m pip install --upgrade --force-reinstall "numpy<2"

python -c "import sys, torch, numpy; print(sys.executable); print('torch', torch.__version__); print('numpy', numpy.__version__)"

export KMP_USE_SHM=0
export OMP_NUM_THREADS=1

RUN_NAME=my_baseline_run ./baseline_model/experiments/run_baseline.sh
