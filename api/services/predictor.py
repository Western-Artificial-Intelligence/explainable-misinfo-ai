from __future__ import annotations

import logging
import random
from typing import Any

from api.services.model_inference import is_model_available, predict as model_predict

logger = logging.getLogger(__name__)


def _fallback_prediction() -> dict[str, Any]:
    label = random.choice(["factual", "mixed", "false"])
    confidence = round(random.uniform(0.6, 0.98), 2)
    explanation = f"This is a placeholder explanation for '{label}' classification."
    return {"label": label, "confidence": confidence, "explanation": explanation}


def _normalize_output(raw_output: Any) -> dict[str, Any]:
    if not isinstance(raw_output, dict):
        raise ValueError("Model output must be a dict with label/confidence/explanation.")

    label = str(raw_output.get("label", "mixed")).strip() or "mixed"

    try:
        confidence = float(raw_output.get("confidence", 0.65))
    except (TypeError, ValueError):
        confidence = 0.65

    # Support both 0..1 and 0..100 confidence formats.
    if confidence > 1:
        confidence = confidence / 100.0
    confidence = max(0.0, min(1.0, confidence))

    explanation = raw_output.get(
        "explanation",
        f"Model output for '{label}' classification.",
    )

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "explanation": str(explanation),
    }


def predict_text(text: str) -> dict[str, Any]:
    if not is_model_available():
        logger.warning("No RoBERTa checkpoint available; using random fallback predictor.")
        return _fallback_prediction()

    try:
        raw_output = model_predict(claim=text)
        return _normalize_output(raw_output)
    except Exception:  # noqa: BLE001
        logger.exception("Model inference failed; returning fallback prediction.")
        return _fallback_prediction()

