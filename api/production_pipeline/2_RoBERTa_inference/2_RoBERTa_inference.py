# 2_roberta_inference.py
from __future__ import annotations

import math
import os
import time
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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
        # defensive (should never happen)
        return [1.0 / len(logits)] * len(logits)
    return [e / s for e in exps]


def _argmax_eps(vals: List[float]) -> int:
    # deterministic tie-break: earliest index within EPS of max
    m = max(vals)
    for i, v in enumerate(vals):
        if abs(v - m) <= EPS:
            return i
    return 0


def _build_model_text(normalized_claim: str) -> str:
    # x = "<CLAIM> " + normalized_claim + " </CLAIM>"
    return f"<CLAIM> {normalized_claim} </CLAIM>"


def _pretend_tokenize(x: str) -> Dict[str, Any]:
    """
    Stand-in tokenizer:
    - counts whitespace-separated tokens deterministically
    - applies truncation against T_CONFIG
    """
    # crude token proxy (good enough for now)
    original_tokens = len(x.split())
    truncated = original_tokens > T_CONFIG
    input_tokens = min(original_tokens, T_CONFIG)

    return {
        "input_tokens": max(1, input_tokens),
        "truncated": truncated,
    }


def roberta_infer(_: Any) -> Dict[str, Any]:
    """
    Pretend this exists in production and returns:
      { "h": optional embedding, "label_logits": [3 floats], "label_probs": [3 floats] }
    For now: deterministic mock based on hash.
    """
    # Deterministic logits from sha256
    digest = hashlib.sha256(str(_).encode("utf-8")).digest()
    # map 3 bytes -> [-2, 2]
    logits = [((digest[i] / 255.0) * 4.0 - 2.0) for i in range(3)]
    probs = _softmax(logits)
    return {"h": None, "label_logits": logits, "label_probs": probs}


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

    tok = _pretend_tokenize(x)

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
                "truncated": bool(tok["truncated"]),
                "input_tokens": int(tok["input_tokens"]),
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