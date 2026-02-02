# ===== baseline_model/training/evaluate.py =====
"""
Evaluate a checkpoint on the test set and save metrics to JSON.
"""

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ..data_utils.load_dataset import load_merged_dataset
from ..data_utils.tokenize import build_tokenizer, tokenize_batch
from ..models.domain_adversarial import DomainAdversarialClassifier
from .metrics import compute_basic_metrics, expected_calibration_error


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax along last dimension.
    """
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def evaluate_checkpoint(
    checkpoint_path: str, config: dict, device: str = "cpu"
) -> dict:
    """
    Loads model checkpoint and runs evaluation on the 'test' split.

    Args:
        checkpoint_path: Path to .pt file saved by train_baseline.py
        config: parsed YAML configuration
        device: 'cpu' or 'cuda'

    Returns:
        dict with evaluation results (basic metrics + ECE)
    """
    # Load dataset
    ds_path = config["data"]["dataset_path"]
    ds = load_merged_dataset(
        ds_path,
        create_synthetic_if_missing=config["data"].get(
            "create_synthetic_if_missing", True
        ),
    )

    # Build tokenizer and tokenize dataset
    tokenizer = build_tokenizer(config["model"]["backbone"])
    max_len = config["model"]["max_length"]
    ds = ds.map(
        lambda b: tokenize_batch(b, tokenizer=tokenizer, max_len=max_len), batched=True
    )
    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label_3way", "label_bin", "source_id"],
    )

    test_ds = ds["test"]
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=16)

    # Build model and load checkpoint
    num_sources = len(set(test_ds["source_id"])) if len(test_ds) > 0 else 3
    model = DomainAdversarialClassifier(
        backbone_name=config["model"]["backbone"],
        num_sources=num_sources,
        lambda_adv=config["model"]["lambda_adv"],
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_3 = batch["label_3way"].numpy()

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out["logits_label"].cpu().numpy()
            probs = softmax(logits)
            preds = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels_3)

    # Concatenate all batches
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Compute metrics
    basic = compute_basic_metrics(all_preds, all_labels)
    ece = expected_calibration_error(all_probs, all_labels)
    results = {"basic": basic, "ece": ece}

    # Save to JSON
    out_dir = Path(config["output"]["base_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "evaluation_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[evaluate] Saved evaluation results to {result_path}")
    return results
