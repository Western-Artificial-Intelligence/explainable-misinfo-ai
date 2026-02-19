"""Stage 2: RoBERTa inference."""

from pipeline.blackboxes.roberta import roberta_infer
from pipeline.config import (
    LABELS_3WAY,
    ROBERTA_BACKBONE,
    ROBERTA_CHECKPOINT,
    T_CONTEXT,
)


def run(state: dict) -> dict:
    normalized = state["normalized_claim"]
    pred = roberta_infer(
        claim=normalized,
        article=None,
        checkpoint=ROBERTA_CHECKPOINT,
        backbone=ROBERTA_BACKBONE,
        max_len=T_CONTEXT,
    )
    probs = pred["label_probs"]
    class_id = int(max(range(3), key=lambda i: probs[i]))
    class_name = LABELS_3WAY[class_id]
    confidence = probs[class_id]

    state["roberta"] = {
        "label": {
            "logits": pred["label_logits"],
            "probs": probs,
            "class_id": class_id,
            "class_name": class_name,
        },
        "confidence": confidence,
        "model": {
            "name": "roberta-base",
            "revision": "prod",
            "ctx_tokens": 512,
        },
        "tokenization": {
            "truncated": False,
            "input_tokens": len(normalized.split()),
        },
    }
    return state
