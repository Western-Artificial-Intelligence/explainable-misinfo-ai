# Baseline Model Training for Misinformation Classification


This folder contains a full, runnable **baseline training pipeline** using **RoBERTa + LoRA + Domain-Adversarial** methods.

It is designed to work out-of-the-box with a real or synthetic dataset.

---

## Features

* Loads a **merged dataset** from HuggingFace saved to disk at `data/merged_dataset/`
* Creates a **small synthetic dataset** if none exists (so you can run locally)
* Tokenizes text using a **RoBERTa tokenizer** with special tokens:

  ```
  <CLAIM>, </CLAIM>, <ARTICLE>, </ARTICLE>
  ```
* Builds a **DomainAdversarialClassifier**:

  * RoBERTa backbone → CLS token → label head + domain head
* Supports **LoRA adapters** via PEFT (optional, falls back to full fine-tuning if not installed)
* Training loop with:

  * Progress bars
  * Batch-level metrics
  * Checkpoint saving
  * Resume support
* **Evaluation script**
* **Simple inference script**

---

## Dataset Schema

The training pipeline expects the following schema for the dataset:

| Column       | Type   | Description                                             |
| ------------ | ------ | ------------------------------------------------------- |
| `input_text` | string | Format: `<CLAIM> ... </CLAIM> <ARTICLE> ... </ARTICLE>` |
| `label_3way` | int    | 0=false, 1=mixed, 2=true (LIAR-style), or -100 if N/A   |
| `label_bin`  | int    | 0=false, 1=true-ish (binary), or -100 if N/A            |
| `source_id`  | int    | Integer representing dataset/source (0..N-1)            |
| `split`      | string | One of `"train"`, `"val"`, `"test"`                     |

**Notes:**

* If you already have a merged HF dataset, save it to:

  ```
  data/merged_dataset/
  ```

  and the training script will load it automatically.
* If no dataset exists at that path, the pipeline will **generate a small synthetic dataset** so you can run everything end-to-end.

---

## Usage

1. **Edit the config** (optional)

   ```
   configs/baseline.yaml
   ```

   to adjust hyperparameters.

2. **Run training**

   ```bash
   python training/train_baseline.py --config configs/baseline.yaml
   ```

3. **Outputs**
   Trained artifacts are saved under:

   ```
   outputs/checkpoints/<run_name>/
   ```

---

## Additional Notes

* The training pipeline supports **checkpointing and resuming** from intermediate states.
* The evaluation script can be used to calculate **accuracy, loss, and other metrics**.
* The inference script can process single examples or batches for quick testing.
