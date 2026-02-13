"""
Main training script. Loads config, dataset, tokenizer, model, optionally applies LoRA (PEFT),
runs training loop with tqdm progress bars, prints epoch/batch/loss/accuracy/val accuracy,
saves checkpoints, supports resume, logs metrics to CSV.
"""

import os
import yaml
import argparse
import random
import sys
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

if __package__ is None or __package__ == "":
    # Support direct script execution: python training/train_baseline.py ...
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from baseline_model.data_utils.load_dataset import load_merged_dataset
    from baseline_model.data_utils.tokenize import build_tokenizer, tokenize_batch
    from baseline_model.models.domain_adversarial import DomainAdversarialClassifier
    from baseline_model.models.lora_utils import try_apply_peft_lora
    from baseline_model.training.metrics import compute_basic_metrics, expected_calibration_error
else:
    from ..data_utils.load_dataset import load_merged_dataset
    from ..data_utils.tokenize import build_tokenizer, tokenize_batch
    from ..models.domain_adversarial import DomainAdversarialClassifier
    from ..models.lora_utils import try_apply_peft_lora
    from .metrics import compute_basic_metrics, expected_calibration_error


# ========== Utilities ==========

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (base_dir / path).resolve()


def save_checkpoint(state: dict, path: str):
    torch.save(state, path)
    print(f"[train] Saved checkpoint to {path}")


# ========== Training ==========

def train(config_path: str, run_name: str, resume_from: str = None):
    config_path_obj = Path(config_path).resolve()
    config = load_config(str(config_path_obj))
    config_dir = config_path_obj.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name or f"baseline_{timestamp}"
    output_base = resolve_path(config["output"]["base_dir"], config_dir)
    run_dir = output_base / run_name
    ckpt_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if config["training"]["device"] == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config["training"]["device"]
    print(f"[train] Using device: {device}")

    # Data
    ds_path = resolve_path(config["data"]["dataset_path"], config_dir)
    ds = load_merged_dataset(str(ds_path), create_synthetic_if_missing=config["data"].get("create_synthetic_if_missing", True))

    # Tokenizer
    backbone = config["model"]["backbone"]
    max_len = config["model"]["max_length"]
    tokenizer = build_tokenizer(backbone)
    ds = ds.map(lambda b: tokenize_batch(b, tokenizer=tokenizer, max_len=max_len), batched=True)
    # Keep required columns for set_format
    set_cols = ["input_ids", "attention_mask", "label_3way", "label_bin", "source_id"]
    ds.set_format(type="torch", columns=set_cols)

    train_ds = ds["train"]
    val_ds = ds["val"]
    test_ds = ds["test"]

    # DataLoaders
    batch_size = int(config["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    num_sources = len(set(train_ds["source_id"])) if len(train_ds) > 0 else 3
    model = DomainAdversarialClassifier(backbone_name=backbone, num_sources=num_sources, lambda_adv=config["model"]["lambda_adv"])

    # Apply LoRA if desired
    lora_cfg = config.get("lora", {})
    peft_applied = False
    if lora_cfg.get("enabled", True):
        model, peft_applied = try_apply_peft_lora(model,
                                                 r=lora_cfg.get("r", 8),
                                                 lora_alpha=lora_cfg.get("alpha", 16),
                                                 lora_dropout=lora_cfg.get("dropout", 0.05),
                                                 target_modules=lora_cfg.get("target_modules", ["query", "value"]))
    # If tokenizer added tokens, resize embeddings
    try:
        model.encoder.resize_token_embeddings(len(tokenizer))
    except Exception:
        # Some models may not support resize_token_embeddings via encoder attribute depending on PEFT
        pass

    model.to(device)

    # Seed
    set_seed(int(config["training"]["seed"]))

    # Optimizer - only parameters that require grad (important for PEFT)
    lr = float(config["training"]["lr"])
    weight_decay = float(config["training"]["weight_decay"])
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Learning rate scheduler with warmup
    epochs = int(config["training"]["epochs"])
    total_steps = len(train_loader) * epochs
    warmup_ratio = float(config["training"].get("warmup_ratio", 0.06))
    warmup_steps = int(warmup_ratio * total_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    best_metric = -1.0
    # Resume support
    if resume_from:
        resume_from_path = resolve_path(resume_from, config_dir)
        ckpt = torch.load(str(resume_from_path), map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_metric = ckpt.get("best_metric", -1.0)
        print(f"[train] Resumed from {resume_from}, starting epoch {start_epoch}")

    # Training loop
    grad_accum_steps = int(config["training"].get("grad_accum_steps", 1))
    scaler = torch.cuda.amp.GradScaler() if isinstance(device, str) and device.startswith("cuda") else (torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None)

    # Logging file
    metrics_log = logs_dir / "metrics_log.jsonl"
    f_log = open(metrics_log, "a")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        # epoch progress bar
        print(f"Epoch {epoch+1}/{epochs}")
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_3way = batch["label_3way"].to(device)
            label_bin = batch["label_bin"].to(device)
            source_id = batch["source_id"].to(device)

            # Mixed precision if available
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                    label_3way=label_3way, label_bin=label_bin, source_id=source_id)
                    loss = outputs["loss"] / grad_accum_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                label_3way=label_3way, label_bin=label_bin, source_id=source_id)
                loss = outputs["loss"] / grad_accum_steps
                loss.backward()

            # Stats
            with torch.no_grad():
                logits = outputs["logits_label"]
                preds = torch.argmax(logits, dim=-1)
                mask = (label_3way != -100)
                if mask.any():
                    running_correct += (preds[mask] == label_3way[mask]).sum().item()
                    running_total += mask.sum().item()
            running_loss += loss.item() * grad_accum_steps

            # Optimizer step
            is_update_step = ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx + 1 == len(train_loader))
            if is_update_step:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                grad_clip_norm = float(config["training"].get("grad_clip_norm", 1.0))
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), max_norm=grad_clip_norm
                )
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Progress bar update
            avg_loss = running_loss / max(1, (batch_idx + 1))
            train_acc = (running_correct / running_total * 100.0) if running_total > 0 else 0.0
            pbar.set_postfix({"Batch": batch_idx + 1, "Loss": f"{avg_loss:.4f}", "Train Acc": f"{train_acc:.2f}%"})

        # Validation
        val_metrics = evaluate_on_loader(model, val_loader, device)
        val_acc = val_metrics.get("accuracy")
        val_f1 = val_metrics.get("macro_f1")

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {running_loss / max(1, len(train_loader)):.4f} | Train Acc (approx): {((running_correct / running_total) * 100.0) if running_total > 0 else 0.0:.2f}%")
        print(f"Val Macro-F1: {val_f1 if val_f1 is not None else 'N/A'} | Val Acc: {val_acc if val_acc is not None else 'N/A'}")

        # Save checkpoints
        if (val_f1 is not None) and (val_f1 > best_metric):
            best_metric = val_f1
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_metric": best_metric,
                "config": config,
            }, ckpt_dir / "best.ckpt")
            print("Saved best checkpoint.")

        if config["training"].get("save_every_epoch", False):
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_metric": best_metric,
                "config": config,
            }, ckpt_dir / f"epoch_{epoch+1}.ckpt")

        # Log metrics
        log_row = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(1, len(train_loader)),
            "train_acc": (running_correct / running_total) if running_total > 0 else None,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
        }
        f_log.write(json.dumps(log_row) + "\n")
        f_log.flush()

    # Close log file at end of training loop (if reached)
    f_log.close()
    print(f"Finished training. Outputs saved to {run_dir}")


def evaluate_on_loader(model, loader, device):
    """
    Run model in eval mode on loader and return macro-f1 and accuracy computed on label_3way
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
        return {"accuracy": None, "macro_f1": None}
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = compute_basic_metrics(all_preds, all_labels)
    ece = expected_calibration_error(np.concatenate(all_probs, axis=0), all_labels)
    model.train()
    return {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "ece": ece}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config yaml")
    parser.add_argument("--run_name", type=str, default=None, help="Run name to save outputs")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(args.config, args.run_name, resume_from=args.resume_from)
