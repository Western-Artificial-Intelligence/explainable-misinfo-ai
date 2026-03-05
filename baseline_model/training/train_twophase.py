"""
Two-phase training with layer freezing and layer-wise learning rate decay (LLRD).

Phase 1: Freeze encoder, train classification heads only (feature extraction).
Phase 2: Unfreeze LoRA adapters (or full encoder for DistilBERT), apply LLRD.

Usage:
    python -m baseline_model.training.train_twophase \
        --config baseline_model/configs/twophase.yaml \
        --run_name roberta_twophase

    python -m baseline_model.training.train_twophase \
        --config baseline_model/configs/twophase_distil.yaml \
        --run_name distil_twophase
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from baseline_model.data_utils.load_dataset import load_merged_dataset
    from baseline_model.data_utils.tokenize import build_tokenizer, tokenize_batch
    from baseline_model.models.domain_adversarial import DomainAdversarialClassifier
    from baseline_model.models.freezing_utils import (
        build_phase1_param_groups,
        build_phase2_param_groups_distil,
        build_phase2_param_groups_lora,
        count_parameters,
        detect_num_layers,
        freeze_encoder_all,
        unfreeze_all_encoder_layers,
        unfreeze_lora_only,
    )
    from baseline_model.models.lora_utils import try_apply_peft_lora
    from baseline_model.training.metrics import (
        compute_basic_metrics,
        expected_calibration_error,
    )
else:
    from ..data_utils.load_dataset import load_merged_dataset
    from ..data_utils.tokenize import build_tokenizer, tokenize_batch
    from ..models.domain_adversarial import DomainAdversarialClassifier
    from ..models.freezing_utils import (
        build_phase1_param_groups,
        build_phase2_param_groups_distil,
        build_phase2_param_groups_lora,
        count_parameters,
        detect_num_layers,
        freeze_encoder_all,
        unfreeze_all_encoder_layers,
        unfreeze_lora_only,
    )
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
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (base_dir / path).resolve()


def save_checkpoint(state: dict, path: str):
    torch.save(state, path)
    print(f"[train] Saved checkpoint to {path}")


# ========== Evaluation ==========


def evaluate_on_loader_extended(model, loader, device, ece_bins=15):
    """
    Evaluate model on a data loader.

    Returns dict with: val_loss, accuracy, macro_f1, per_class, confusion_matrix,
    ece, domain_accuracy.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_domain_preds = []
    all_domain_labels = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_3way = batch["label_3way"].to(device)
            label_bin = batch["label_bin"].to(device)
            source_id = batch["source_id"].to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                label_3way=label_3way,
                label_bin=label_bin,
                source_id=source_id,
            )
            logits = out["logits_label"]
            if out.get("loss") is not None:
                total_loss += out["loss"].item()
                num_batches += 1

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(label_3way.cpu().numpy())

            # Domain accuracy
            if out.get("logits_domain") is not None:
                domain_preds = out["logits_domain"].argmax(dim=-1).cpu().numpy()
                all_domain_preds.append(domain_preds)
                all_domain_labels.append(source_id.cpu().numpy())

    if len(all_preds) == 0:
        return {
            "accuracy": None,
            "macro_f1": None,
            "val_loss": None,
            "ece": None,
            "per_class": None,
            "confusion_matrix": None,
            "domain_accuracy": None,
        }

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = compute_basic_metrics(all_preds, all_labels)
    ece = expected_calibration_error(
        np.concatenate(all_probs, axis=0), all_labels, n_bins=ece_bins
    )
    val_loss = total_loss / max(1, num_batches)

    domain_accuracy = None
    if all_domain_preds:
        dp = np.concatenate(all_domain_preds, axis=0)
        dl = np.concatenate(all_domain_labels, axis=0)
        domain_accuracy = float((dp == dl).mean())

    model.train()
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "ece": ece,
        "val_loss": val_loss,
        "per_class": metrics.get("per_class"),
        "confusion_matrix": metrics.get("confusion_matrix"),
        "domain_accuracy": domain_accuracy,
    }


# ========== Phase Training Loop ==========


def train_phase(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    phase_name,
    epochs,
    grad_accum_steps,
    grad_clip_norm,
    early_stopping_patience,
    ece_bins,
    best_metric,
    ckpt_dir,
    f_log,
    eval_every_steps=None,
    scaler=None,
):
    """
    Generic training loop for one phase.

    Args:
        eval_every_steps: If set, run validation every N optimizer steps (sub-epoch eval).
        best_metric: Current best macro-F1 across phases.

    Returns:
        best_metric: Updated best macro-F1.
        epoch_times: List of per-epoch wall-clock times (seconds).
    """
    epochs_without_improvement = 0
    global_step = 0
    epoch_times = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        epoch_start = time.time()

        pbar = tqdm(
            train_loader,
            desc=f"{phase_name} Epoch {epoch + 1}/{epochs}",
            leave=True,
        )

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_3way = batch["label_3way"].to(device)
            label_bin = batch["label_bin"].to(device)
            source_id = batch["source_id"].to(device)

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

            # Stats
            with torch.no_grad():
                logits = outputs["logits_label"]
                preds = torch.argmax(logits, dim=-1)
                mask = label_3way != -100
                if mask.any():
                    running_correct += (preds[mask] == label_3way[mask]).sum().item()
                    running_total += mask.sum().item()
            running_loss += loss.item() * grad_accum_steps

            # Optimizer step
            is_update_step = ((batch_idx + 1) % grad_accum_steps == 0) or (
                batch_idx + 1 == len(train_loader)
            )
            if is_update_step:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=grad_clip_norm,
                )
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Sub-epoch evaluation
                if (
                    eval_every_steps
                    and global_step % eval_every_steps == 0
                ):
                    val_metrics = evaluate_on_loader_extended(
                        model, val_loader, device, ece_bins
                    )
                    val_f1 = val_metrics.get("macro_f1")
                    print(
                        f"\n  [{phase_name}] Step {global_step}: "
                        f"val_macro_f1={val_f1:.4f}" if val_f1 else f"\n  [{phase_name}] Step {global_step}: val_macro_f1=N/A"
                    )
                    if val_f1 is not None and val_f1 > best_metric:
                        best_metric = val_f1
                        save_checkpoint(
                            {
                                "phase": phase_name,
                                "step": global_step,
                                "model_state": model.state_dict(),
                                "best_metric": best_metric,
                            },
                            str(ckpt_dir / "best.ckpt"),
                        )
                    model.train()

            # Progress bar
            avg_loss = running_loss / max(1, batch_idx + 1)
            train_acc = (
                (running_correct / running_total * 100.0) if running_total > 0 else 0.0
            )
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "Acc": f"{train_acc:.2f}%"})

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # End-of-epoch evaluation
        train_loss = running_loss / max(1, len(train_loader))
        train_acc = (
            (running_correct / running_total * 100.0) if running_total > 0 else 0.0
        )
        val_metrics = evaluate_on_loader_extended(model, val_loader, device, ece_bins)
        val_loss = val_metrics.get("val_loss")
        val_acc = val_metrics.get("accuracy")
        val_f1 = val_metrics.get("macro_f1")
        val_ece = val_metrics.get("ece")
        domain_acc = val_metrics.get("domain_accuracy")

        # Check for improvement
        if val_f1 is not None and val_f1 > best_metric:
            best_metric = val_f1
            epochs_without_improvement = 0
            save_checkpoint(
                {
                    "phase": phase_name,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "best_metric": best_metric,
                },
                str(ckpt_dir / "best.ckpt"),
            )
        else:
            epochs_without_improvement += 1

        # Log
        per_class = val_metrics.get("per_class")
        log_row = {
            "phase": phase_name,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "val_ece": val_ece,
            "domain_accuracy": domain_acc,
            "best_val_f1": best_metric,
            "epoch_time_sec": epoch_time,
        }
        if per_class:
            log_row["per_class_f1"] = per_class["f1"]
        f_log.write(json.dumps(log_row) + "\n")
        f_log.flush()

        # Print metrics
        print(f"\n{'=' * 60}")
        print(f"  {phase_name} — Epoch {epoch + 1}/{epochs}  ({epoch_time:.1f}s)")
        print(f"{'=' * 60}")
        print(f"  Train Loss:       {train_loss:.4f}")
        print(f"  Train Acc:        {train_acc:.2f}%")
        print(
            f"  Val Loss:         {val_loss:.4f}"
            if val_loss is not None
            else "  Val Loss:         N/A"
        )
        print(
            f"  Val Acc:          {val_acc * 100:.2f}%"
            if val_acc is not None
            else "  Val Acc:          N/A"
        )
        print(
            f"  Val Macro-F1:     {val_f1:.4f}"
            if val_f1 is not None
            else "  Val Macro-F1:     N/A"
        )
        print(
            f"  Val ECE ({ece_bins} bins): {val_ece:.4f}"
            if val_ece is not None
            else f"  Val ECE ({ece_bins} bins): N/A"
        )
        print(
            f"  Domain Accuracy:  {domain_acc * 100:.2f}%"
            if domain_acc is not None
            else "  Domain Accuracy:  N/A"
        )
        print(f"  Best Val F1:      {best_metric:.4f}")
        if per_class:
            labels = ["false", "mixed", "true"]
            print(f"  Per-class:")
            print(f"    {'Class':<8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
            for i, lbl in enumerate(labels):
                print(
                    f"    {lbl:<8} {per_class['precision'][i]:>8.4f} "
                    f"{per_class['recall'][i]:>8.4f} {per_class['f1'][i]:>8.4f}"
                )
        print(f"{'=' * 60}\n")

        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"\n[{phase_name}] Early stopping at epoch {epoch + 1} — "
                f"no improvement for {early_stopping_patience} epochs."
            )
            break

    return best_metric, epoch_times


# ========== Efficiency Metrics ==========


def measure_efficiency(model, test_loader, device):
    """Measure inference latency and throughput on the test set."""
    model.eval()
    total_samples = 0
    start = time.time()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Measuring inference", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)
            total_samples += input_ids.size(0)

    elapsed = time.time() - start
    latency_ms = (elapsed / total_samples * 1000) if total_samples > 0 else 0
    throughput = total_samples / elapsed if elapsed > 0 else 0
    return {
        "total_samples": total_samples,
        "total_time_sec": elapsed,
        "latency_ms_per_sample": latency_ms,
        "throughput_samples_per_sec": throughput,
    }


# ========== Main Entry Point ==========


def train_twophase(config_path: str, run_name: str = None, backbone: str = None):
    config_path_obj = Path(config_path).resolve()
    config = load_config(str(config_path_obj))
    config_dir = config_path_obj.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name or f"twophase_{timestamp}"
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

    # Seed
    set_seed(int(config["training"]["seed"]))

    # Data
    ds_path = resolve_path(config["data"]["dataset_path"], config_dir)
    ds = load_merged_dataset(
        str(ds_path),
        create_synthetic_if_missing=config["data"].get("create_synthetic_if_missing", True),
    )

    backbone = backbone or config["model"]["backbone"]
    print(f"[train] Using backbone: {backbone}")
    max_len = config["model"]["max_length"]
    tokenizer = build_tokenizer(backbone)
    ds = ds.map(
        lambda b: tokenize_batch(b, tokenizer=tokenizer, max_len=max_len), batched=True
    )
    set_cols = ["input_ids", "attention_mask", "label_3way", "label_bin", "source_id"]
    ds.set_format(type="torch", columns=set_cols)

    train_ds = ds["train"]
    val_ds = ds["val"]
    test_ds = ds["test"]

    # Model
    num_sources = len(set(int(x) for x in train_ds["source_id"])) if len(train_ds) > 0 else 3
    model = DomainAdversarialClassifier(
        backbone_name=backbone,
        num_sources=num_sources,
        lambda_adv=config["model"]["lambda_adv"],
    )

    # Apply LoRA if configured
    lora_cfg = config.get("lora", {})
    use_lora = lora_cfg.get("enabled", False)
    peft_applied = False
    if use_lora:
        model, peft_applied = try_apply_peft_lora(
            model,
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("alpha", 16),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules", ["query", "value"]),
        )

    # Resize embeddings if tokenizer added special tokens
    try:
        model.encoder.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass

    model.to(device)

    # Detect encoder layers
    num_layers = detect_num_layers(model)
    print(f"[train] Detected {num_layers} transformer layers")

    # AMP scaler
    use_amp = device == "cuda" or (
        isinstance(device, str) and device.startswith("cuda")
    )
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Shared config
    grad_accum_steps = int(config["training"].get("grad_accum_steps", 1))
    ece_bins = int(config["training"].get("ece_bins", 15))

    # Open log file
    metrics_log = logs_dir / "metrics_log.jsonl"
    f_log = open(metrics_log, "a")

    best_metric = -1.0
    all_epoch_times = []

    # ========================
    # Phase 1: Head-only training
    # ========================
    phase1_cfg = config["phase1"]
    phase1_batch_size = int(phase1_cfg["batch_size"])

    train_loader = DataLoader(
        train_ds,
        batch_size=phase1_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=phase1_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Freeze encoder
    freeze_encoder_all(model)
    param_counts = count_parameters(model)
    print(f"\n[Phase 1] Head-only training")
    print(
        f"  Trainable: {param_counts['trainable']:,} | "
        f"Frozen: {param_counts['frozen']:,} | "
        f"Total: {param_counts['total']:,}"
    )

    # Phase 1 optimizer + scheduler
    phase1_lr = float(phase1_cfg["lr"])
    phase1_wd = float(phase1_cfg["weight_decay"])
    param_groups = build_phase1_param_groups(model, phase1_lr, phase1_wd)
    optimizer1 = AdamW(param_groups)

    phase1_epochs = int(phase1_cfg["epochs"])
    total_steps1 = len(train_loader) * phase1_epochs
    warmup_steps1 = int(float(phase1_cfg.get("warmup_ratio", 0.06)) * total_steps1)

    def lr_lambda1(step):
        if step < warmup_steps1:
            return step / max(1, warmup_steps1)
        return 1.0

    scheduler1 = LambdaLR(optimizer1, lr_lambda1)

    best_metric, epoch_times1 = train_phase(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer1,
        scheduler=scheduler1,
        device=device,
        phase_name="Phase1",
        epochs=phase1_epochs,
        grad_accum_steps=grad_accum_steps,
        grad_clip_norm=float(phase1_cfg.get("grad_clip_norm", 1.0)),
        early_stopping_patience=int(phase1_cfg.get("early_stopping_patience", 2)),
        ece_bins=ece_bins,
        best_metric=best_metric,
        ckpt_dir=ckpt_dir,
        f_log=f_log,
        eval_every_steps=int(phase1_cfg.get("eval_every_steps", 0)) or None,
        scaler=scaler,
    )
    all_epoch_times.extend(epoch_times1)

    # ========================
    # Phase 2: Encoder fine-tuning with LLRD
    # ========================
    phase2_cfg = config["phase2"]
    phase2_batch_size = int(phase2_cfg["batch_size"])

    # Rebuild loaders if batch size changed
    if phase2_batch_size != phase1_batch_size:
        train_loader = DataLoader(
            train_ds,
            batch_size=phase2_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=phase2_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    # Unfreeze appropriate parameters
    if use_lora and peft_applied:
        unfreeze_lora_only(model)
        print(f"\n[Phase 2] LoRA + heads training with LLRD")
        param_groups2 = build_phase2_param_groups_lora(
            model,
            head_lr=float(phase2_cfg["head_lr"]),
            lora_top_lr=float(phase2_cfg["lora_top_lr"]),
            decay=float(phase2_cfg["lora_lr_decay"]),
            weight_decay=float(phase2_cfg["weight_decay"]),
            num_layers=num_layers,
        )
    else:
        unfreeze_all_encoder_layers(model)
        print(f"\n[Phase 2] Full encoder + heads training with LLRD")
        param_groups2 = build_phase2_param_groups_distil(
            model,
            head_lr=float(phase2_cfg["head_lr"]),
            encoder_top_lr=float(phase2_cfg["encoder_top_lr"]),
            decay=float(phase2_cfg["encoder_lr_decay"]),
            weight_decay=float(phase2_cfg["weight_decay"]),
            num_layers=num_layers,
        )

    param_counts = count_parameters(model)
    print(
        f"  Trainable: {param_counts['trainable']:,} | "
        f"Frozen: {param_counts['frozen']:,} | "
        f"Total: {param_counts['total']:,}"
    )

    # New optimizer for Phase 2 (no momentum carryover)
    optimizer2 = AdamW(param_groups2)

    phase2_epochs = int(phase2_cfg["epochs"])
    total_steps2 = len(train_loader) * phase2_epochs
    warmup_steps2 = int(float(phase2_cfg.get("warmup_ratio", 0.06)) * total_steps2)

    def lr_lambda2(step):
        if step < warmup_steps2:
            return step / max(1, warmup_steps2)
        return 1.0

    scheduler2 = LambdaLR(optimizer2, lr_lambda2)

    best_metric, epoch_times2 = train_phase(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer2,
        scheduler=scheduler2,
        device=device,
        phase_name="Phase2",
        epochs=phase2_epochs,
        grad_accum_steps=grad_accum_steps,
        grad_clip_norm=float(phase2_cfg.get("grad_clip_norm", 1.0)),
        early_stopping_patience=int(phase2_cfg.get("early_stopping_patience", 2)),
        ece_bins=ece_bins,
        best_metric=best_metric,
        ckpt_dir=ckpt_dir,
        f_log=f_log,
        scaler=scaler,
    )
    all_epoch_times.extend(epoch_times2)

    # ========================
    # Efficiency Metrics
    # ========================
    print(f"\n{'=' * 60}")
    print("  Efficiency Metrics")
    print(f"{'=' * 60}")

    # Parameter counts
    param_counts = count_parameters(model)
    print(f"  Total parameters:     {param_counts['total']:,}")
    print(f"  Trainable parameters: {param_counts['trainable']:,}")
    print(f"  Frozen parameters:    {param_counts['frozen']:,}")

    # Training time
    if all_epoch_times:
        avg_epoch_time = np.mean(all_epoch_times)
        print(f"  Avg epoch time:       {avg_epoch_time:.1f}s")

    # Inference latency / throughput on test set
    test_loader = DataLoader(
        test_ds,
        batch_size=phase2_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    eff = measure_efficiency(model, test_loader, device)
    print(f"  Inference latency:    {eff['latency_ms_per_sample']:.2f} ms/sample")
    print(f"  Throughput:           {eff['throughput_samples_per_sec']:.1f} samples/sec")
    print(f"  Best val macro-F1:    {best_metric:.4f}")
    print(f"{'=' * 60}\n")

    # Log efficiency
    eff_log = {
        "type": "efficiency",
        "params": param_counts,
        "avg_epoch_time_sec": float(np.mean(all_epoch_times)) if all_epoch_times else None,
        "inference_latency_ms": eff["latency_ms_per_sample"],
        "throughput_samples_sec": eff["throughput_samples_per_sec"],
        "best_val_f1": best_metric,
    }
    f_log.write(json.dumps(eff_log) + "\n")
    f_log.flush()
    f_log.close()

    print(f"Finished two-phase training. Outputs saved to {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-phase training with LLRD")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/twophase.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Run name for output directory"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Override encoder backbone (e.g. distilroberta-base)",
    )
    args = parser.parse_args()
    train_twophase(args.config, args.run_name, backbone=args.backbone)
