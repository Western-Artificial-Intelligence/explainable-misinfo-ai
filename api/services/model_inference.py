"""Real RoBERTa model inference service.

Loads the trained DomainAdversarialClassifier checkpoint and runs inference.
Model is lazily loaded on first call and cached for subsequent calls.

Config via env vars:
    TRUTHLENS_CHECKPOINT_PATH — path to best.ckpt (required for real predictions)
    TRUTHLENS_BACKBONE — HuggingFace model name (default: roberta-base)
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

LABELS_3WAY = ["false", "mixed", "true"]

# Map model labels to pipeline-friendly prediction names
LABEL_TO_PREDICTION = {
    "false": "FAKE",
    "mixed": "REQUIRES_VERIFICATION",
    "true": "REAL",
}


def _get_checkpoint_path() -> Optional[str]:
    path = os.getenv("TRUTHLENS_CHECKPOINT_PATH", "").strip()
    return path or None


def _get_backbone() -> str:
    return os.getenv("TRUTHLENS_BACKBONE", "roberta-base").strip()


@lru_cache(maxsize=1)
def _load_model_and_tokenizer() -> Tuple[Any, Any]:
    """Load the RoBERTa model and tokenizer. Cached after first call."""
    import torch
    from baseline_model.models.domain_adversarial import DomainAdversarialClassifier
    from baseline_model.data_utils.tokenize import build_tokenizer

    checkpoint_path = _get_checkpoint_path()
    backbone = _get_backbone()

    if not checkpoint_path:
        raise FileNotFoundError(
            "TRUTHLENS_CHECKPOINT_PATH not set. Cannot load model."
        )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}"
        )

    logger.info("Loading RoBERTa model from %s (backbone=%s)", checkpoint_path, backbone)

    device = "cpu"
    state = torch.load(checkpoint_path, map_location=device)
    model = DomainAdversarialClassifier(
        backbone_name=backbone,
        num_sources=state.get("num_sources", 3),
        lambda_adv=state.get("lambda_adv", 0.5),
    )
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()

    tokenizer = build_tokenizer(backbone)

    logger.info("RoBERTa model loaded successfully")
    return model, tokenizer


def is_model_available() -> bool:
    """Check whether a checkpoint path is configured and exists."""
    path = _get_checkpoint_path()
    return path is not None and os.path.isfile(path)


def predict(claim: str, article: str = "", max_len: int = 256) -> Dict[str, Any]:
    """Run inference on a claim (and optional article) using the real RoBERTa model.

    Returns:
        dict with keys: label, confidence, explanation, probs, prediction
    """
    import torch

    model, tokenizer = _load_model_and_tokenizer()

    text = f"<CLAIM> {claim} </CLAIM> <ARTICLE> {article} </ARTICLE>"
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out["logits_label"]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0].tolist()
        logits_list = logits.cpu().numpy()[0].tolist()

    class_id = int(probs.index(max(probs)))
    class_name = LABELS_3WAY[class_id]
    confidence = float(probs[class_id])
    prediction = LABEL_TO_PREDICTION[class_name]

    return {
        "label": class_name,
        "confidence": round(confidence, 4),
        "explanation": f"RoBERTa classified this claim as '{class_name}' with {confidence:.1%} confidence.",
        "probs": probs,
        "logits": logits_list,
        "class_id": class_id,
        "prediction": prediction,
    }
