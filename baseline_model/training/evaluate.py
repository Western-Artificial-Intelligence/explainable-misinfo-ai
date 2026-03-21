"""
Evaluate a saved checkpoint on the test set and print/save metrics.

Usage:
    python -m baseline_model.training.evaluate \
        --checkpoint baseline_model/baseline_outputs/distil_twophase/checkpoints/best.ckpt \
        --config baseline_model/configs/twophase_distil.yaml

    python -m baseline_model.training.evaluate \
        --checkpoint baseline_model/baseline_outputs/roberta_twophase/checkpoints/best.ckpt \
        --config baseline_model/configs/twophase.yaml

    # With backbone override (e.g. MLM encoder):
    python -m baseline_model.training.evaluate \
        --checkpoint baseline_model/baseline_outputs/roberta_mlm_twophase_v4/checkpoints/best.ckpt \
        --config baseline_model/configs/twophase.yaml \
        --backbone ./models/mlm/roberta-misinfo-mlm
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
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


def load_config(path: str):
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (base_dir / path).resolve()


def evaluate_checkpoint(checkpoint_path: str, config_path: str, backbone: str = None, device: str = None):
    config_path_obj = Path(config_path).resolve()
    config = load_config(str(config_path_obj))
    config_dir = config_path_obj.parent

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[evaluate] Using device: {device}")

    # Load dataset
    ds_path = resolve_path(config["data"]["dataset_path"], config_dir)
    ds = load_merged_dataset(
        str(ds_path),
        create_synthetic_if_missing=config["data"].get("create_synthetic_if_missing", True),
    )

    # Tokenizer
    backbone = backbone or config["model"]["backbone"]
    print(f"[evaluate] Using backbone: {backbone}")
    max_len = config["model"]["max_length"]
    tokenizer = build_tokenizer(backbone)
    ds = ds.map(lambda b: tokenize_batch(b, tokenizer=tokenizer, max_len=max_len), batched=True)
    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label", "source_id"],
    )

    test_ds = ds["test"]
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Build model
    num_sources = len(set(int(x) for x in test_ds["source_id"])) if len(test_ds) > 0 else 3
    model = DomainAdversarialClassifier(
        backbone_name=backbone,
        num_sources=num_sources,
        lambda_adv=config["model"]["lambda_adv"],
    )

    # Apply LoRA if config says so (needed to match checkpoint shapes)
    lora_cfg = config.get("lora", {})
    if lora_cfg.get("enabled", False):
        model, _ = try_apply_peft_lora(
            model,
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("alpha", 16),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules", ["query", "value"]),
        )

    # Resize embeddings to match tokenizer
    try:
        model.encoder.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass

    # Load checkpoint
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()

    print(f"[evaluate] Loaded checkpoint from {checkpoint_path}")

    # Run evaluation
    all_probs = []
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on test set"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            source_id = batch["source_id"].to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                label=label,
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
            all_labels.append(label.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Compute metrics
    metrics = compute_basic_metrics(all_preds, all_labels)
    ece = expected_calibration_error(all_probs, all_labels)
    test_loss = total_loss / max(1, num_batches)

    # Print results
    labels = ["false", "true"]
    per_class = metrics.get("per_class")
    cm = metrics.get("confusion_matrix")

    print(f"\n{'=' * 60}")
    print(f"  TEST SET EVALUATION")
    print(f"{'=' * 60}")
    print(f"  Checkpoint:      {checkpoint_path}")
    print(f"  Backbone:        {backbone}")
    print(f"  Test samples:    {len(all_labels)}")
    print(f"  Test Loss:       {test_loss:.4f}")
    print(f"  Test Accuracy:   {metrics['accuracy'] * 100:.2f}%")
    print(f"  Test Macro-F1:   {metrics['macro_f1']:.4f}")
    print(f"  Test ECE:        {ece:.4f}")
    if per_class:
        print(f"  Per-class:")
        print(f"    {'Class':<8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
        for i, lbl in enumerate(labels):
            print(f"    {lbl:<8} {per_class['precision'][i]:>8.4f} {per_class['recall'][i]:>8.4f} {per_class['f1'][i]:>8.4f}")
    if cm:
        print(f"  Confusion Matrix (rows=true, cols=pred):")
        print(f"    {'':>8} {'false':>8} {'true':>8}")
        for i, lbl in enumerate(labels):
            print(f"    {lbl:>8} {cm[i][0]:>8} {cm[i][1]:>8}")
    print(f"{'=' * 60}\n")

    # Save results
    ckpt_dir = Path(checkpoint_path).parent.parent
    result_path = ckpt_dir / "test_results.json"
    results = {
        "test_loss": test_loss,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "ece": ece,
        "per_class": per_class,
        "confusion_matrix": cm,
    }
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[evaluate] Saved test results to {result_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on test set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.ckpt")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--backbone", type=str, default=None, help="Override backbone (e.g. ./models/mlm/roberta-misinfo-mlm)")
    parser.add_argument("--device", type=str, default=None, help="Device (auto if not set)")
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.config, backbone=args.backbone, device=args.device)