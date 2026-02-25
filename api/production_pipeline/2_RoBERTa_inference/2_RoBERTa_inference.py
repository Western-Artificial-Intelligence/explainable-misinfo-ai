# 2_roberta_inference.py
from __future__ import annotations

import logging
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LABELS_3WAY = ["false", "mixed", "true"]
CTX_TOKENS_MAX = 512

# paper default is 256; can be configured to 512
T_CONFIG = int(os.getenv("ROBERTA_T_CONFIG", "256"))
if T_CONFIG < 1:
    T_CONFIG = 1
if T_CONFIG > CTX_TOKENS_MAX:
    T_CONFIG = CTX_TOKENS_MAX

MODEL_NAME = os.getenv("ROBERTA_MODEL_NAME", "roberta-base(+lora)")
MODEL_REVISION = os.getenv("ROBERTA_MODEL_REVISION", "dev")

EPS = 1e-9

# Try to import real model inference
try:
    from api.services.model_inference import is_model_available, predict as _real_predict
    _HAS_REAL_MODEL = True
except ImportError:
    _HAS_REAL_MODEL = False


class RobertaInferenceError(Exception):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def _now_iso8601_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    if s == 0:
        return [1.0 / len(logits)] * len(logits)
    return [e / s for e in exps]


def _argmax_eps(vals: List[float]) -> int:
    m = max(vals)
    for i, v in enumerate(vals):
        if abs(v - m) <= EPS:
            return i
    return 0


def _build_model_text(normalized_claim: str) -> str:
    return f"<CLAIM> {normalized_claim} </CLAIM>"


def roberta_infer(payload: Any) -> Dict[str, Any]:
    """Run real RoBERTa inference if a checkpoint is available, otherwise raise."""
    if not _HAS_REAL_MODEL or not is_model_available():
        raise RobertaInferenceError(
            code="MODEL_NOT_AVAILABLE",
            message="Real RoBERTa model not available. Set TRUTHLENS_CHECKPOINT_PATH.",
        )

    claim_text = payload.get("x", "") if isinstance(payload, dict) else str(payload)
    # Strip the <CLAIM> wrapper if present (model_inference adds its own)
    claim_text = claim_text.replace("<CLAIM>", "").replace("</CLAIM>", "").strip()

    pred = _real_predict(claim=claim_text, max_len=T_CONFIG)
    return {
        "h": None,
        "label_logits": pred["logits"],
        "label_probs": pred["probs"],
    }


def run_2_roberta_inference(step1_out: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input: Step 1 output (passthrough contract)
    Output: Step 2 output_schema.json
    """
    # Validate minimal required keys (Step1 should guarantee these)
    try:
        request_id = step1_out["request_id"]
        claim_id = step1_out["claim_id"]
        user_claim = step1_out["user_claim"]
        normalized_claim = step1_out["normalized_claim"]
        meta_in = step1_out["meta"]
        received_at = meta_in["received_at"]
        version = meta_in["version"]
    except Exception as e:
        raise RobertaInferenceError(
            code="INVALID_INPUT",
            message="Step2 expected Step1 output shape.",
            details={"error": str(e)},
        )

    x = _build_model_text(normalized_claim)

    # Estimate token count for metadata
    input_tokens = len(x.split())
    truncated = input_tokens > T_CONFIG

    t0 = time.perf_counter()
    pred = roberta_infer({"x": x, "T_CONFIG": T_CONFIG})
    latency_ms = int(round((time.perf_counter() - t0) * 1000))

    probs3 = list(map(float, pred["label_probs"]))
    logits3 = list(map(float, pred["label_logits"]))

    class_id = _argmax_eps(probs3)
    class_name = LABELS_3WAY[class_id]
    confidence = float(probs3[class_id])

    out: Dict[str, Any] = {
        "request_id": request_id,
        "claim_id": claim_id,
        "user_claim": user_claim,
        "normalized_claim": normalized_claim,
        "roberta": {
            "label": {
                "logits": logits3,
                "probs": probs3,
                "class_id": class_id,
                "class_name": class_name,
            },
            "confidence": confidence,
            "model": {
                "name": MODEL_NAME,
                "revision": MODEL_REVISION,
                "ctx_tokens": CTX_TOKENS_MAX,
            },
            "tokenization": {
                "truncated": truncated,
                "input_tokens": min(input_tokens, T_CONFIG),
            },
        },
        "meta": {
            "received_at": received_at,
            "version": version,
            "latency_ms": max(0, latency_ms),
        },
    }

    # passthrough warnings if present (schema allows it)
    if isinstance(meta_in, dict) and "warnings" in meta_in:
        out["meta"]["warnings"] = meta_in["warnings"]

    return out


def process_step1_output(step1_out: Dict[str, Any]) -> Dict[str, Any]:
    """
    Public entrypoint (same style as Step1's process_user_claim).
    """
    return run_2_roberta_inference(step1_out)