from __future__ import annotations

import importlib
import logging
import os
import random
from typing import Any, Callable

logger = logging.getLogger(__name__)

PredictFn = Callable[[str], dict[str, Any]]

_model_predict_fn: PredictFn | None = None
_model_load_attempted = False


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


def _load_model_predictor() -> PredictFn | None:
    global _model_load_attempted, _model_predict_fn

    if _model_load_attempted:
        return _model_predict_fn

    _model_load_attempted = True

    module_name = os.getenv("TRUTHLENS_MODEL_MODULE", "").strip()
    function_name = os.getenv("TRUTHLENS_MODEL_FUNCTION", "predict").strip() or "predict"

    if not module_name:
        logger.info(
            "No TRUTHLENS_MODEL_MODULE configured; using random fallback predictor.",
        )
        return None

    try:
        module = importlib.import_module(module_name)
        fn = getattr(module, function_name)
        if not callable(fn):
            raise TypeError(f"{module_name}.{function_name} is not callable")
        _model_predict_fn = fn
        logger.info(
            "Loaded model predictor from %s.%s",
            module_name,
            function_name,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not load model predictor from %s.%s (%s). Falling back to random predictor.",
            module_name,
            function_name,
            exc,
        )
        _model_predict_fn = None

    return _model_predict_fn


def predict_text(text: str) -> dict[str, Any]:
    predictor_fn = _load_model_predictor()
    if predictor_fn is None:
        return _fallback_prediction()

    try:
        raw_output = predictor_fn(text)
        return _normalize_output(raw_output)
    except Exception:  # noqa: BLE001
        logger.exception("Model inference failed; returning fallback prediction.")
        return _fallback_prediction()

