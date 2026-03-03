# 2_roberta_inference.py
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LABELS_3WAY = ["false", "mixed", "true"]
LABEL2ID = {name: i for i, name in enumerate(LABELS_3WAY)}
ID2LABEL = {i: name for i, name in enumerate(LABELS_3WAY)}

CTX_TOKENS_MAX = 512

# paper default is 256; can be configured to 512
T_CONFIG = int(os.getenv("ROBERTA_T_CONFIG", "256"))
T_CONFIG = max(1, min(T_CONFIG, CTX_TOKENS_MAX))

MODEL_NAME = os.getenv("ROBERTA_MODEL_NAME", "roberta-base(+lora)")
MODEL_REVISION = os.getenv("ROBERTA_MODEL_REVISION", "dev")

# When True, use LLM blackbox to mimic RoBERTa (for testing without a trained model).
# When False, use the real RoBERTa backend (currently mock; replace with trained model later).
USE_LLM = (os.getenv("ROBERTA_USE_LLM", "false") or "false").strip().lower() in ("1", "true", "yes")

EPS = 1e-9


# --- Mock-only label overrides (best for regression tests) ---
# Keys should be normalized strings (lowercase, collapsed whitespace) AFTER you strip wrapper tokens.
MOCK_LABEL_OVERRIDES: Dict[str, str] = {
    "kimjeongeun is dead": "false",
}

# Optional: allow overrides from env as JSON:
# ROBERTA_MOCK_OVERRIDES='{"kimjeongeun is dead":"false","the earth is flat":"false"}'
try:
    _ENV_OVERRIDES = json.loads(os.getenv("ROBERTA_MOCK_OVERRIDES", "{}"))
    if isinstance(_ENV_OVERRIDES, dict):
        for k, v in _ENV_OVERRIDES.items():
            if isinstance(k, str) and isinstance(v, str):
                MOCK_LABEL_OVERRIDES[k.strip().lower()] = v.strip().lower()
except Exception:
    # ignore malformed env var; keep defaults
    pass


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
    # deterministic tie-break: earliest index within EPS of max
    m = max(vals)
    for i, v in enumerate(vals):
        if abs(v - m) <= EPS:
            return i
    return 0


def _build_model_text(normalized_claim: str) -> str:
    # Keep this stable once you start training, because it affects learned behavior.
    return f"<CLAIM> {normalized_claim} </CLAIM>"


def _extract_claim_from_wrapped_text(text: str) -> str:
    """
    Best-effort extraction for the mock override layer.
    Keeps things simple: pulls content between <CLAIM> and </CLAIM> if present.
    """
    lower = text.lower()
    start_tag = "<claim>"
    end_tag = "</claim>"
    s = lower.find(start_tag)
    e = lower.find(end_tag)
    if s != -1 and e != -1 and s < e:
        inner = text[s + len(start_tag): e]
    else:
        inner = text
    # normalize for override matching
    inner = " ".join(inner.strip().lower().split())
    return inner


def _forced_logits_for_label(label: str) -> List[float]:
    """
    Produce logits that strongly favor the chosen label.
    (Softmax will make the chosen class ~1.0)
    """
    label = label.strip().lower()
    # strong separation; can tune
    hi = 6.0
    lo = -6.0
    if label == "false":
        return [hi, lo, lo]
    if label == "mixed":
        return [lo, hi, lo]
    if label == "true":
        return [lo, lo, hi]
    # unknown label -> neutral
    return [0.0, 0.0, 0.0]


# ----------------------------
# LLM-as-RoBERTa (when USE_LLM is True)
# ----------------------------

_LLM_BLACKBOX_CLASS: Optional[Any] = None


def _load_llm_blackbox_class() -> Any:
    """Load LLMBlackbox so step2 works when dynamically imported (e.g. importlib)."""
    global _LLM_BLACKBOX_CLASS
    if _LLM_BLACKBOX_CLASS is not None:
        return _LLM_BLACKBOX_CLASS
    try:
        from api.production_pipeline.middlewares.llm_blackbox import LLMBlackbox
        _LLM_BLACKBOX_CLASS = LLMBlackbox
        return _LLM_BLACKBOX_CLASS
    except Exception:
        pass
    here = Path(__file__).resolve()
    llm_path = here.parents[1] / "middlewares" / "llm_blackbox.py"
    spec = importlib.util.spec_from_file_location("step2_llm_blackbox", llm_path)
    if spec is None or spec.loader is None:
        raise RobertaInferenceError("IMPORT_ERROR", "Could not load llm_blackbox for ROBERTA_USE_LLM.", {"path": str(llm_path)})
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _LLM_BLACKBOX_CLASS = getattr(mod, "LLMBlackbox")
    return _LLM_BLACKBOX_CLASS


# Prompt for LLM to mimic 3-way classification (false / mixed / true)
_SYS_LLM_CLASSIFY = (
    "You are a fact-checking classifier. Your task is to classify the given claim as exactly one of: false, mixed, or true. "
    "Respond with only that one word, nothing else (no explanation, no punctuation)."
)


def _parse_llm_label(raw: str) -> str:
    """Extract 'false', 'mixed', or 'true' from LLM response; default 'mixed' if unclear."""
    if not raw or not isinstance(raw, str):
        return "mixed"
    raw = raw.strip().lower()
    # exact match first
    for label in LABELS_3WAY:
        if raw == label:
            return label
    # then first occurrence of the word
    for label in LABELS_3WAY:
        if label in raw:
            return label
    return "mixed"


def _infer_via_llm(text: str) -> Tuple[List[float], List[float]]:
    """
    Use LLM blackbox to classify claim as false/mixed/true; return (logits3, probs3).
    Uses sync generate() so it's safe when step2 is run in a threadpool.
    """
    claim = _extract_claim_from_wrapped_text(text)
    if not claim:
        claim = text.strip() or "(empty)"
    LLMBlackbox = _load_llm_blackbox_class()
    llm = LLMBlackbox()
    response = llm.generate(
        system_context=_SYS_LLM_CLASSIFY,
        user_context=claim,
        temperature=0.0,
        num_predict=8,
    )
    label = _parse_llm_label(response)
    logits = _forced_logits_for_label(label)
    probs = _softmax(logits)
    return logits, probs


@dataclass(frozen=True)
class TokenizationInfo:
    # Keep these fields compatible with a real tokenizer later
    original_tokens: int
    input_tokens: int
    truncated: bool
    max_length: int


class RobertaBackend:
    """
    Backend abstraction so you can swap:
      - mock backend (today)
      - HF transformers backend (later)
    without changing pipeline code.
    """

    def __init__(self, *, t_config: int):
        self.t_config = t_config

    def tokenize(self, text: str) -> TokenizationInfo:
        # Placeholder token proxy; keep same keys as "real" later.
        original_tokens = max(1, len(text.split()))
        input_tokens = min(original_tokens, self.t_config)
        return TokenizationInfo(
            original_tokens=original_tokens,
            input_tokens=max(1, input_tokens),
            truncated=original_tokens > self.t_config,
            max_length=self.t_config,
        )

    def infer(self, text: str) -> Tuple[List[float], List[float]]:
        """
        Returns (logits3, probs3).
        If USE_LLM (ROBERTA_USE_LLM=true): LLM blackbox mimics RoBERTa.
        Else: mock overrides, then deterministic sha256 mock (or real RoBERTa later).
        """
        # 1) When enabled, use LLM to classify (for testing without a trained model)
        if USE_LLM:
            return _infer_via_llm(text)

        # 2) Mock override (for known regression-test strings)
        claim_norm = _extract_claim_from_wrapped_text(text)
        forced_label = MOCK_LABEL_OVERRIDES.get(claim_norm)
        if forced_label is not None:
            logits = _forced_logits_for_label(forced_label)
            probs = _softmax(logits)
            return logits, probs

        # 3) Deterministic mock logits from sha256 (or replace with real RoBERTa later)
        digest = hashlib.sha256(f"{self.t_config}\n{text}".encode("utf-8")).digest()
        logits = [((digest[i] / 255.0) * 4.0 - 2.0) for i in range(3)]
        probs = _softmax(logits)
        return logits, probs


# Single backend instance (safe: deterministic, no GPU resources)
_BACKEND = RobertaBackend(t_config=T_CONFIG)


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

    tok = _BACKEND.tokenize(x)

    t0 = time.perf_counter()
    logits3, probs3 = _BACKEND.infer(x)
    latency_ms = int(round((time.perf_counter() - t0) * 1000))

    # Convert to JSON-safe primitives
    logits3 = list(map(float, logits3))
    probs3 = list(map(float, probs3))

    if len(probs3) != 3 or len(logits3) != 3:
        raise RobertaInferenceError(
            code="INFERENCE_SHAPE_ERROR",
            message="Expected 3-way logits/probs.",
            details={"len_logits": len(logits3), "len_probs": len(probs3)},
        )

    class_id = _argmax_eps(probs3)
    class_name = ID2LABEL.get(class_id, "unknown")
    confidence = float(probs3[class_id])

    # When using LLM to mimic RoBERTa, report it in model.name for transparency
    _model_display_name = "llm_proxy" if USE_LLM else MODEL_NAME

    out: Dict[str, Any] = {
        "request_id": request_id,
        "claim_id": claim_id,
        "user_claim": user_claim,
        "normalized_claim": normalized_claim,
        "roberta": {
            "label": {
                "logits": logits3,
                "probs": probs3,
                "class_id": int(class_id),
                "class_name": class_name,
            },
            "confidence": confidence,
            "model": {
                "name": _model_display_name,
                "revision": MODEL_REVISION,
                # report both: model hard limit and configured truncation window
                "ctx_tokens_max": CTX_TOKENS_MAX,
                "t_config": T_CONFIG,
                "labels": LABELS_3WAY,
            },
            "tokenization": {
                "truncated": bool(tok.truncated),
                "input_tokens": int(tok.input_tokens),
                "original_tokens": int(tok.original_tokens),
                "max_length": int(tok.max_length),
            },
        },
        "meta": {
            "received_at": received_at,
            "version": version,
            "latency_ms": max(0, int(latency_ms)),
        },
    }

    # passthrough warnings if present (schema allows it)
    if isinstance(meta_in, dict) and "warnings" in meta_in:
        out["meta"]["warnings"] = meta_in["warnings"]

    # Debug: print step 2 result to console
    print("[Step 2 RoBERTa] result:", json.dumps(out, indent=2))

    return out


def process_step1_output(step1_out: Dict[str, Any]) -> Dict[str, Any]:
    """Public entrypoint."""
    return run_2_roberta_inference(step1_out)