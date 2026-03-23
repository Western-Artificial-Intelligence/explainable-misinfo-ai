"""
Sweep classification thresholds on val/test set to find optimal threshold for recall.

Usage:
    python -m baseline_model.training.threshold_sweep \
        --checkpoint baseline_outputs/baseline_roberta/checkpoints/best.ckpt \
        --config baseline_model/configs/baseline.yaml

    # With backbone override:
    python -m baseline_model.training.threshold_sweep \
        --checkpoint baseline_outputs/baseline_mlm/checkpoints/best.ckpt \
        --config baseline_model/configs/baseline.yaml \
        --backbone ./models/mlm/roberta-misinfo-mlm
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from baseline_model.data_utils.load_dataset import load_merged_dataset
    from baseline_model.data_utils.tokenize import build_tokenizer, tokenize_batch
    from baseline_model.models.domain_adversarial import DomainAdversarialClassifier
    from baseline_model.models.lora_utils import try_apply_peft_lora
else:
    from ..data_utils.load_dataset import load_merged_dataset
    from ..data_utils.tokenize import build_tokenizer, tokenize_batch
    from ..models.domain_adversarial import DomainAdversarialClassifier
    from ..models.lora_utils import try_apply_peft_lora


def load_config(path: str):
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (base_dir / path).resolve()


def get_predictions(checkpoint_path, config_path, backbone=None, split="val", device=None):
    config_path_obj = Path(config_path).resolve()
    config = load_config(str(config_path_obj))
    config_dir = config_path_obj.parent

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ds_path = resolve_path(config["data"]["dataset_path"], config_dir)
    ds = load_merged_dataset(
        str(ds_path),
        create_synthetic_if_missing=config["data"].get("create_synthetic_if_missing", True),
    )

    backbone = backbone or config["model"]["backbone"]
    max_len = config["model"]["max_length"]
    tokenizer = build_tokenizer(backbone)
    ds = ds.map(lambda b: tokenize_batch(b, tokenizer=tokenizer, max_len=max_len), batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label", "source_id"])

    split_ds = ds[split]
    loader = DataLoader(split_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    num_sources = len(set(int(x) for x in split_ds["source_id"])) if len(split_ds) > 0 else 3
    model = DomainAdversarialClassifier(
        backbone_name=backbone,
        num_sources=num_sources,
        lambda_adv=config["model"]["lambda_adv"],
    )

    lora_cfg = config.get("lora", {})
    if lora_cfg.get("enabled", False):
        model, _ = try_apply_peft_lora(
            model,
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("alpha", 16),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules", ["query", "value"]),
        )

    try:
        model.encoder.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Getting {split} predictions"):
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
            probs = torch.softmax(out["logits_label"], dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(label.cpu().numpy())

    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


def sweep_thresholds(probs, labels):
    # probs[:, 1] is P(true)
    true_probs = probs[:, 1]
    thresholds = np.arange(0.10, 0.91, 0.05)

    print(f"\n{'Threshold':>10} {'Macro-F1':>10} {'Acc':>8} {'true_P':>8} {'true_R':>8} {'true_F1':>8} {'false_P':>8} {'false_R':>8} {'false_F1':>8}")
    print("-" * 90)

    best_f1 = 0
    best_threshold = 0.5
    best_recall_threshold = 0.5
    best_recall_f1 = 0

    for t in thresholds:
        preds = (true_probs >= t).astype(int)
        macro_f1 = f1_score(labels, preds, average="macro")
        acc = (preds == labels).mean()
        true_p = precision_score(labels, preds, pos_label=1, zero_division=0)
        true_r = recall_score(labels, preds, pos_label=1, zero_division=0)
        true_f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
        false_p = precision_score(labels, preds, pos_label=0, zero_division=0)
        false_r = recall_score(labels, preds, pos_label=0, zero_division=0)
        false_f1 = f1_score(labels, preds, pos_label=0, zero_division=0)

        marker = ""
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_threshold = t
            marker = " <-- best F1"
        if true_r >= 0.80 and macro_f1 > best_recall_f1:
            best_recall_f1 = macro_f1
            best_recall_threshold = t

        print(f"{t:>10.2f} {macro_f1:>10.4f} {acc:>8.4f} {true_p:>8.4f} {true_r:>8.4f} {true_f1:>8.4f} {false_p:>8.4f} {false_r:>8.4f} {false_f1:>8.4f}{marker}")

    print(f"\nBest macro-F1 threshold: {best_threshold:.2f} (F1={best_f1:.4f})")
    if best_recall_f1 > 0:
        print(f"Best threshold with true recall >= 0.80: {best_recall_threshold:.2f} (F1={best_recall_f1:.4f})")
    else:
        print("No threshold achieved true recall >= 0.80")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep thresholds on val/test set")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    probs, labels = get_predictions(args.checkpoint, args.config, args.backbone, args.split, args.device)
    sweep_thresholds(probs, labels)