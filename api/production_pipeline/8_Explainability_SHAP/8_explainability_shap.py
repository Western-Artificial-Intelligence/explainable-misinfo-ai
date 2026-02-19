# 8_explainability_shap.py
from __future__ import annotations

import math
import os
import sys
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Errors
# ----------------------------


class ExplainabilitySHAPError(Exception):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


# ----------------------------
# Constants (tunable)
# ----------------------------


METHOD = os.getenv("SHAP_METHOD", "shap_partition")  # enum: shap_partition|shap_kernel
if METHOD not in ("shap_partition", "shap_kernel"):
    METHOD = "shap_partition"

OUTPUT_SPACE = os.getenv("SHAP_OUTPUT_SPACE", "probability")  # enum: logit|probability
if OUTPUT_SPACE not in ("logit", "probability"):
    OUTPUT_SPACE = "probability"

NSAMPLES = int(os.getenv("SHAP_NSAMPLES", "256") or "256")
if NSAMPLES < 1:
    NSAMPLES = 1

TOPK_TOKENS = int(os.getenv("SHAP_TOPK_TOKENS", "8") or "8")
if TOPK_TOKENS < 1:
    TOPK_TOKENS = 1

BACKGROUND_STRATEGY = os.getenv("SHAP_BACKGROUND", "masker_default")

EPS = 1e-9


# ----------------------------
# Cache (process-wide)
# ----------------------------


EXPLAINER_CACHE: Dict[str, Dict[str, Any]] = {}


# ----------------------------
# Meta helpers (match prior stages)
# ----------------------------


def _copy_meta(meta_in: Dict[str, Any]) -> Dict[str, Any]:
    meta_out: Dict[str, Any] = {
        "received_at": meta_in.get("received_at"),
        "version": meta_in.get("version"),
    }
    for k in (
        "warnings",
        "latency_ms",
        "stage_latencies_ms",
        "total_latency_ms",
        "stage_errors",
        "embed_model",
        "embedding_cache_enabled",
        "embedding_cache_namespace",
        "embedding_cache_ttl_s",
    ):
        if k in meta_in:
            meta_out[k] = meta_in[k]
    return meta_out


def _ensure_warnings(meta: Dict[str, Any]) -> List[Dict[str, str]]:
    w = meta.get("warnings")
    if isinstance(w, list):
        return w
    meta["warnings"] = []
    return meta["warnings"]


def _ensure_stage_errors(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    se = meta.get("stage_errors")
    if isinstance(se, list):
        return se
    meta["stage_errors"] = []
    return meta["stage_errors"]


def _mark_latency(meta: Dict[str, Any], stage_name: str, ms: int) -> None:
    meta["latency_ms"] = max(0, int(ms))
    sl = meta.get("stage_latencies_ms")
    if not isinstance(sl, dict):
        sl = {}
        meta["stage_latencies_ms"] = sl
    sl[stage_name] = max(0, int(ms))
    try:
        meta["total_latency_ms"] = int(sum(int(v) for v in sl.values() if isinstance(v, (int, float))))
    except Exception:
        pass


# ----------------------------
# Helpers (per pseudo)
# ----------------------------


def now_ms() -> int:
    return int(time.time() * 1000)


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def argsort_desc(values: List[float]) -> List[int]:
    # deterministic: primary -value, tie -> index asc
    return [i for i, _v in sorted(enumerate(values), key=lambda t: (-float(t[1]), int(t[0])))]


def argsort_asc(values: List[float]) -> List[int]:
    # deterministic: primary +value, tie -> index asc
    return [i for i, _v in sorted(enumerate(values), key=lambda t: (float(t[1]), int(t[0])))]


def _softmax_row(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(float(x) - float(m)) for x in logits]
    s = float(sum(exps))
    if s <= 0:
        return [1.0 / max(1, len(logits))] * max(1, len(logits))
    return [float(e) / s for e in exps]


def _build_model_text(normalized_claim: str) -> str:
    return f"<CLAIM> {normalized_claim} </CLAIM>"


def _get_step2_backend():
    # When invoked via routes/production.py, Step2 is loaded as module name 'step2_roberta_inference'.
    m = sys.modules.get("step2_roberta_inference")
    if not m:
        return None
    if hasattr(m, "roberta_infer"):
        return m
    return None


def _roberta_infer_fallback(payload: Any) -> Dict[str, Any]:
    """Deterministic mock identical in spirit to Step2."""
    digest = hashlib.sha256(str(payload).encode("utf-8")).digest()
    logits = [((digest[i] / 255.0) * 4.0 - 2.0) for i in range(3)]
    probs = _softmax_row(logits)
    return {"label_logits": logits, "label_probs": probs}


def roberta_predict(texts: List[str]) -> List[List[float]]:
    """Predict for a batch of texts.

    Returns rows in the requested output space (probabilities by default).
    """
    out: List[List[float]] = []

    m = _get_step2_backend()
    T_CONFIG = int(os.getenv("ROBERTA_T_CONFIG", "256") or "256")

    for t in texts:
        x = _build_model_text(str(t))
        if m is not None:
            pred = m.roberta_infer({"x": x, "T_CONFIG": T_CONFIG})  # type: ignore[attr-defined]
            logits = list(map(float, pred.get("label_logits") or [0.0, 0.0, 0.0]))
            probs = list(map(float, pred.get("label_probs") or _softmax_row(logits)))
        else:
            pred = _roberta_infer_fallback({"x": x, "T_CONFIG": T_CONFIG})
            logits = list(map(float, pred.get("label_logits") or [0.0, 0.0, 0.0]))
            probs = list(map(float, pred.get("label_probs") or _softmax_row(logits)))

        if OUTPUT_SPACE == "logit":
            out.append(logits)
        else:
            out.append(probs)

    return out


def _simple_tokens(text: str) -> List[str]:
    # Stable, UI-friendly tokenization.
    # Keep punctuation attached (matches Step2's coarse token proxy style).
    toks = [t for t in (text or "").split() if t.strip()]
    return toks if toks else [text.strip() or ""]


def safe_token_spans(token_strings: List[str], text: str) -> Optional[List[Dict[str, int]]]:
    """Best-effort char spans for tokens in text.

    Works reliably for simple whitespace tokens.
    If we cannot align, return None.
    """
    if not text:
        return None

    spans: List[Dict[str, int]] = []
    cursor = 0
    for tok in token_strings:
        if tok == "":
            spans.append({"start": cursor, "end": cursor})
            continue
        idx = text.find(tok, cursor)
        if idx < 0:
            # try from 0 (tokens may repeat)
            idx = text.find(tok)
        if idx < 0:
            return None
        start = idx
        end = idx + len(tok)
        spans.append({"start": int(start), "end": int(end)})
        cursor = end
    return spans


# ----------------------------
# Main stage
# ----------------------------


async def process_step7_output(step7_out: Dict[str, Any]) -> Dict[str, Any]:
    """Step7 -> Step8: SHAP token-level explainability for claim-only RoBERTa."""

    t0 = time.perf_counter()
    meta_out: Dict[str, Any] = _copy_meta(step7_out.get("meta") or {})
    warnings = _ensure_warnings(meta_out)
    stage_errors = _ensure_stage_errors(meta_out)

    try:
        request_id = step7_out["request_id"]
        claim_id = step7_out["claim_id"]
        user_claim = step7_out["user_claim"]
        normalized_claim = step7_out["normalized_claim"]
        roberta = step7_out.get("roberta") or {}
        routing = step7_out.get("routing")
        query_plan = step7_out.get("query_plan")
        rag_candidates = step7_out.get("rag_candidates")
        mmr_selected = step7_out.get("mmr_selected")
        evidence_topk = step7_out.get("evidence_topk")
    except Exception as e:
        raise ExplainabilitySHAPError("INVALID_INPUT", "Step8 expected Step7 output shape.", {"error": str(e)})

    # 1) explained class from roberta predicted label
    try:
        explained_class_id = int((((roberta or {}).get("label") or {}).get("class_id")))
        explained_class_name = str((((roberta or {}).get("label") or {}).get("class_name")))
    except Exception:
        explained_class_id = 0
        explained_class_name = "false"

    if explained_class_id not in (0, 1, 2):
        explained_class_id = 0
    if explained_class_name not in ("false", "mixed", "true"):
        explained_class_name = "false"

    # 2) cache key
    model = (roberta or {}).get("model") or {}
    model_name = str(model.get("name") or "unknown")
    revision = str(model.get("revision") or "unknown")
    cache_key = f"{model_name}:{revision}:{METHOD}:{OUTPUT_SPACE}:{NSAMPLES}"

    # Defaults for output (in case SHAP isn't available)
    probs = (((roberta or {}).get("label") or {}).get("probs") or [0.0, 0.0, 0.0])
    try:
        probs = list(map(float, probs))
    except Exception:
        probs = [0.0, 0.0, 0.0]

    output_value_fallback = float(probs[explained_class_id]) if len(probs) >= 3 else 0.0

    shap_values: List[float] = []
    token_strings: List[str] = []
    base_value: float = 0.0
    output_value: float = output_value_fallback
    time_ms: int = 0

    # 3-6) SHAP computation (best-effort)
    try:
        try:
            import shap  # type: ignore
        except Exception:
            shap = None

        if shap is not None:
            # Get or create cached masker + explainer
            cached = EXPLAINER_CACHE.get(cache_key)
            if cached and "explainer" in cached:
                explainer = cached["explainer"]
            else:
                def f(text_list: List[str]):
                    # SHAP expects numpy-ish; list-of-lists is okay for most explainers
                    return roberta_predict(list(text_list))

                # If tokenizer is unavailable, Text masker defaults to whitespace.
                masker = shap.maskers.Text(None)
                explainer = shap.Explainer(f, masker, algorithm=METHOD)
                EXPLAINER_CACHE[cache_key] = {"masker": masker, "explainer": explainer}

            start = now_ms()
            explanation = explainer([normalized_claim], nsamples=NSAMPLES)
            time_ms = max(0, now_ms() - start)

            # Extract token strings + shap values
            token_strings = list(explanation.data[0])
            vals = explanation.values[0]

            # vals: [num_tokens, num_classes]
            shap_values = [float(v[explained_class_id]) for v in vals]

            # base_values often [1, num_classes] or [num_classes]
            bv0 = explanation.base_values[0]
            base_value = float(bv0[explained_class_id]) if isinstance(bv0, (list, tuple)) else float(bv0)

            # output_values may exist
            if hasattr(explanation, "output_values") and explanation.output_values is not None:
                ov0 = explanation.output_values[0]
                try:
                    output_value = float(ov0[explained_class_id])
                except Exception:
                    output_value = output_value_fallback
            else:
                output_value = output_value_fallback

        else:
            raise RuntimeError("shap_not_installed")

    except Exception as e:
        # Fallback: leave-one-out probability deltas (deterministic)
        stage_errors.append(
            {
                "stage": "8_Explainability_SHAP",
                "code": "SHAP_FALLBACK",
                "message": "SHAP unavailable or failed; used leave-one-out token attribution fallback.",
                "details": {"error": str(e)},
            }
        )

        token_strings = _simple_tokens(normalized_claim)

        # background: empty claim baseline
        base_pred = roberta_predict([""])[0]
        base_value = float(base_pred[explained_class_id]) if len(base_pred) >= 3 else 0.0

        full_pred = roberta_predict([normalized_claim])[0]
        output_value = float(full_pred[explained_class_id]) if len(full_pred) >= 3 else output_value_fallback

        # LOO deltas: output(full) - output(with token i removed)
        shap_values = []
        for i in range(len(token_strings)):
            toks = token_strings[:i] + token_strings[i + 1 :]
            masked = " ".join(toks).strip()
            p = roberta_predict([masked])[0]
            pv = float(p[explained_class_id]) if len(p) >= 3 else 0.0
            shap_values.append(float(output_value - pv))

        time_ms = 0

        # If everything collapsed, ensure at least 1 token record
        if len(token_strings) == 0:
            token_strings = [normalized_claim.strip() or ""]
            shap_values = [0.0]

        warnings.append(
            {
                "code": "SHAP_FALLBACK_USED",
                "message": "Explainability used a simple token-occlusion fallback; results may be less stable than SHAP.",
            }
        )

    # 7) optional spans
    spans = safe_token_spans(token_strings, normalized_claim)

    # 8) token records
    tokens_out: List[Dict[str, Any]] = []
    for i, tok in enumerate(token_strings):
        rec: Dict[str, Any] = {"i": int(i), "token": str(tok), "shap_value": float(shap_values[i] if i < len(shap_values) else 0.0)}
        if spans is not None and i < len(spans):
            rec["char_span"] = {"start": int(spans[i]["start"]), "end": int(spans[i]["end"])}
        tokens_out.append(rec)

    # Ensure schema minItems=1
    if len(tokens_out) == 0:
        tokens_out = [{"i": 0, "token": normalized_claim.strip() or "", "shap_value": 0.0}]

    # 9) top tokens indices (positive / negative), deterministic tie-breaks
    k = clamp_int(TOPK_TOKENS, 1, len(tokens_out))

    # Positive = largest shap_value
    idx_pos = argsort_desc([float(t["shap_value"]) for t in tokens_out])[:k]
    # Negative = smallest shap_value
    idx_neg = argsort_asc([float(t["shap_value"]) for t in tokens_out])[:k]

    shap_explainability: Dict[str, Any] = {
        "method": METHOD,
        "explained_class": {"class_id": int(explained_class_id), "class_name": explained_class_name},
        "base_value": float(base_value),
        "output_value": float(output_value),
        "output_space": OUTPUT_SPACE,
        "tokens": tokens_out,
        "top_tokens": {"k": int(k), "positive": list(map(int, idx_pos)), "negative": list(map(int, idx_neg))},
        "meta": {"background": BACKGROUND_STRATEGY, "nsamples": int(NSAMPLES), "time_ms": int(max(0, time_ms))},
    }

    ms = int(round((time.perf_counter() - t0) * 1000))
    _mark_latency(meta_out, "8_Explainability_SHAP", ms)

    return {
        "request_id": request_id,
        "claim_id": claim_id,
        "user_claim": user_claim,
        "normalized_claim": normalized_claim,
        "roberta": roberta,
        "routing": routing,
        "query_plan": query_plan,
        "rag_candidates": rag_candidates,
        "mmr_selected": mmr_selected,
        "evidence_topk": evidence_topk,
        "shap_explainability": shap_explainability,
        "meta": meta_out,
    }
