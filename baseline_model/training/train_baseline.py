# ===== baseline_model/training/train_baseline.py =====
"""
Main training script for the baseline misinformation classifier.

Saves checkpoints, logs per-epoch JSON-lines logs, shows tqdm progress bars,
supports gradient accumulation, mixed precision (optional), LoRA via PEFT (optional),
and evaluates on validation set each epoch using label_3way metrics.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data_utils.load_dataset import load_merged_dataset
from ..data_utils.tokenize import build_tokenizer, tokenize_batch
from ..models.domain_adversarial import DomainAdversarialClassifier
from ..models.lora_utils import try_apply_peft_lora
from ..utils import load_config, save_checkpoint, set_seed
from .metrics import compute_basic_metrics, expected_calibration_error


def evaluate_on_loader(model: nn.Module, loader: DataLoader, device: str):
    """
    Run model in eval mode on loader and return accuracy, macro_f1, and ece computed on label_3way.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_3 = batch["label_3way"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out["logits_label"]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1)

            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels_3.cpu().numpy())

    if len(all_preds) == 0:
        model.train()
        return {"accuracy": None, "macro_f1": None, "ece": None}

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = compute_basic_metrics(all_preds, all_labels)
    ece = expected_calibration_error(np.concatenate(all_probs, axis=0), all_labels)
    model.train()
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "ece": ece,
    }


def train(
    config_path: str, run_name: Optional[str] = None, resume_from: Optional[str] = None
):
    # Load config
    config = load_config(config_path)

    # Seed and device
    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)
    device_cfg = config.get("training", {}).get("device", "auto")
    device = "cuda" if device_cfg == "auto" and torch.cuda.is_available() else "cpu"
    print(f"[train] Using device: {device}")

    # Directories / run naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name or f"baseline_{timestamp}"
    output_base = Path(config["output"]["base_dir"])
    run_dir = output_base / run_name
    ckpt_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset (creates synthetic if missing when enabled)
    ds_path = config["data"]["dataset_path"]
    ds = load_merged_dataset(
        ds_path,
        create_synthetic_if_missing=config["data"].get(
            "create_synthetic_if_missing", True
        ),
    )

    # Tokenizer and tokenization
    tokenizer = build_tokenizer(config["model"]["backbone"])
    max_len = config["model"].get("max_length", 256)
    ds = ds.map(
        lambda b: tokenize_batch(b, tokenizer=tokenizer, max_len=max_len), batched=True
    )
    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label_3way", "label_bin", "source_id"],
    )

    # DataLoaders
    batch_size = config["training"].get("batch_size", 16)
    grad_accum_steps = config["training"].get("grad_accum_steps", 1)
    train_loader = DataLoader(ds["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds["val"], batch_size=batch_size)

    # Model initialization
    num_sources = len(set(ds["train"]["source_id"])) if len(ds["train"]) > 0 else 3
    model = DomainAdversarialClassifier(
        backbone_name=config["model"]["backbone"],
        num_sources=num_sources,
        lambda_adv=config["model"].get("lambda_adv", 0.5),
    )

    # Ensure encoder embeddings match tokenizer size (special tokens may have been added)
    try:
        model.encoder.resize_token_embeddings(len(tokenizer))
        print(f"[train] Resized encoder token embeddings to {len(tokenizer)}")
    except Exception as e:
        print(f"[train] Warning: failed to resize token embeddings: {e}")

    # Optional LoRA via PEFT
    lora_cfg = config.get("lora", {})
    if lora_cfg.get("enabled", False):
        model, peft_applied = try_apply_peft_lora(
            model,
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("alpha", 16),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules", None),
        )
        print(f"[train] LoRA applied: {peft_applied}")
    else:
        peft_applied = False

    model.to(device)

    # Optimizer (only parameters that require grad)
    # ===== Robust parsing of training hyperparameters =====
    training_cfg = config.get("training", {})

    # Defensive casting of types from config
    batch_size = int(training_cfg.get("batch_size", 16))
    lr = float(training_cfg.get("lr", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    epochs = int(training_cfg.get("epochs", 1))
    seed = int(training_cfg.get("seed", 42))
    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    use_amp = bool(training_cfg.get("use_amp", False))

    print(
        f"[train] Parsed training settings: batch_size={batch_size}, lr={lr}, weight_decay={weight_decay}, epochs={epochs}, grad_accum_steps={grad_accum_steps}, use_amp={use_amp}"
    )

    # Ensure model params that require grad are passed as an iterable (filter returns generator; list is fine)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]

    # Create optimizer using numeric types
    optimizer = AdamW(trainable_parameters, lr=lr, weight_decay=weight_decay)
    # Mixed precision (optional)
    use_amp = config["training"].get("use_amp", False)
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device == "cuda") else None

    # Resume support
    start_epoch = 0
    best_metric = -float("inf")
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt.get("optimizer_state", {}))
        start_epoch = ckpt.get("epoch", 0) + 1
        best_metric = ckpt.get("best_metric", best_metric)
        print(f"[train] Resumed from {resume_from}, starting epoch {start_epoch}")

    epochs = config["training"].get("epochs", 1)

    # Logging file (JSON-lines)
    log_path = logs_dir / "train_log.jsonl"
    f_log = open(log_path, "a", encoding="utf-8")

    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        batch_idx = -1

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{epochs}",
        )
        for batch_idx, batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_3way = batch["label_3way"].to(device)
            label_bin = batch["label_bin"].to(device)
            source_id = batch["source_id"].to(device)

            optimizer.zero_grad() if grad_accum_steps == 1 else None

            # Mixed precision context if using AMP
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        label_3way=label_3way,
                        label_bin=label_bin,
                        source_id=source_id,
                    )
                    loss = outputs["loss"] / grad_accum_steps
                scaler.scale(loss).backward()
                if (batch_idx + 1) % grad_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    label_3way=label_3way,
                    label_bin=label_bin,
                    source_id=source_id,
                )
                loss = outputs["loss"] / grad_accum_steps
                loss.backward()
                if (batch_idx + 1) % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # Stats for display: compute mini-batch accuracy using 3-way labels only when available
            with torch.no_grad():
                logits = outputs["logits_label"]
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1).cpu()
                mask_eval = (label_3way != -100).cpu()
                if mask_eval.any():
                    correct = (
                        (preds[mask_eval] == label_3way.cpu()[mask_eval]).sum().item()
                    )
                    total = mask_eval.sum().item()
                else:
                    # fallback crude signal using label_bin if present
                    lb = label_bin.cpu()
                    if (lb != -100).any():
                        # crude: count predictions that are in {1,2} as 'true-ish'
                        pred_trueish = ((preds == 2) | (preds == 1)).sum().item()
                        total = lb.shape[0]
                        correct = pred_trueish
                    else:
                        correct = 0
                        total = 0

            batch_loss = (
                outputs["loss"].item() * grad_accum_steps
            )  # recover un-normalized batch loss approx.
            running_loss += batch_loss
            running_correct += correct
            running_total += total

            avg_loss = running_loss / (batch_idx + 1)
            avg_acc = (
                (running_correct / running_total * 100.0) if running_total > 0 else 0.0
            )
            pbar.set_postfix(
                {"Loss": f"{avg_loss:.4f}", "Train Acc": f"{avg_acc:.2f}%"}
            )

        # End epoch: validation
        val_res = evaluate_on_loader(model, val_loader, device)
        val_acc = val_res.get("accuracy")
        val_macro_f1 = val_res.get("macro_f1")
        print()
        print(f"Epoch {epoch+1}/{epochs} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"Train Loss: {avg_loss:.4f} | Train Acc (approx): {avg_acc:.2f}%")
        print(f"Val Macro-F1: {val_macro_f1:.4f} | Val Acc: {val_acc:.4f}")

        # Save checkpoint for this epoch
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_metric": best_metric,
            "config": config,
        }
        ckpt_path = ckpt_dir / f"checkpoint_epoch{epoch+1}.pt"
        save_checkpoint(ckpt, ckpt_path)

        # Update best model based on val_macro_f1
        if val_macro_f1 is not None and val_macro_f1 > best_metric:
            best_metric = val_macro_f1
            best_path = ckpt_dir / "best_model.pt"
            save_checkpoint(ckpt, best_path)
            print("[train] Saved best checkpoint.")

        # Log epoch metrics as JSON line
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": float(avg_loss),
            "train_acc_est_pct": float(avg_acc),
            "val_macro_f1": None if val_macro_f1 is None else float(val_macro_f1),
            "val_acc": None if val_acc is None else float(val_acc),
            "best_metric": float(best_metric) if best_metric != -float("inf") else None,
        }
        f_log.write(json.dumps(log_entry) + "\n")
        f_log.flush()

    f_log.close()
    print(f"[train] Finished training. Outputs saved to {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Run name to save outputs"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()
    train(args.config, args.run_name, resume_from=args.resume_from)
