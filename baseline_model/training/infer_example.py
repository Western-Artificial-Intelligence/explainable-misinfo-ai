"""
Simple inference example: loads the best checkpoint and tokenizer, runs a prediction on a short input.

Assumptions:
- Best checkpoint is at baseline_outputs/<run_name>/checkpoints/best_model.pt
- Config from configs/baseline.yaml indicates model backbone and dataset path (for tokenizer)
"""

import torch
import argparse
import json
from pathlib import Path
from ..models.domain_adversarial import DomainAdversarialClassifier
from ..data_utils.tokenize import build_tokenizer


def load_checkpoint_model(
    checkpoint_path: str,
    backbone: str,
    num_sources: int = 3,
    lambda_adv: float = 0.5,
    device: str = "cpu"
):
    """Load model checkpoint and return ready-to-use model."""
    state = torch.load(checkpoint_path, map_location=device)
    model = DomainAdversarialClassifier(
        backbone_name=backbone,
        num_sources=num_sources,
        lambda_adv=lambda_adv
    )
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()
    return model


def predict(model, tokenizer, claim: str, article: str, max_len: int = 256, device: str = "cpu"):
    """Run a single inference and return probabilities for binary labels (false/true)."""
    text = f"<CLAIM> {claim} </CLAIM> <ARTICLE> {article} </ARTICLE>"
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(out["logits_label"], dim=-1).cpu().numpy()[0].tolist()
    return probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--backbone", type=str, default="roberta-base")
    parser.add_argument("--claim", type=str, default="Eating carrots fixes eyesight", help="Claim text")
    parser.add_argument("--article", type=str, default="Some article text contradicting the claim", help="Article text")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    tokenizer = build_tokenizer(args.backbone)
    model = load_checkpoint_model(
        args.checkpoint,
        backbone=args.backbone,
        num_sources=3,
        lambda_adv=0.5,
        device=args.device
    )

    probs = predict(model, tokenizer, args.claim, args.article, max_len=256, device=args.device)
    pred_label = int(probs.index(max(probs)))
    label_names = ["false", "true"]

    print(json.dumps({
        "probs": {label_names[i]: round(p, 4) for i, p in enumerate(probs)},
        "pred_label": pred_label,
        "pred_name": label_names[pred_label],
    }, indent=2))