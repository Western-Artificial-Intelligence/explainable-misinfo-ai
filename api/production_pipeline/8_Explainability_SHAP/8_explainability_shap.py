# 8_explainability_shap.py
from __future__ import annotations

import hashlib
import inspect
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional


class ExplainabilitySHAPError(Exception):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


# ----------------------------
# Config
# ----------------------------
SHAP_ENABLED = (os.getenv("EXPLAIN_USE_SHAP", "false") or "false").strip().lower() in ("1", "true", "yes")

SHAP_ALGO = (os.getenv("SHAP_ALGO", "partition") or "partition").strip().lower()
if SHAP_ALGO not in ("partition", "permutation", "auto"):
    SHAP_ALGO = "partition"

SHAP_OUTPUT_SPACE = (os.getenv("SHAP_OUTPUT_SPACE", "probability") or "probability").strip().lower()
if SHAP_OUTPUT_SPACE not in ("probability", "logit"):
    SHAP_OUTPUT_SPACE = "probability"

SHAP_NSAMPLES = int((os.getenv("SHAP_NSAMPLES", "256") or "256").strip() or "256")
SHAP_NSAMPLES = max(1, min(4096, SHAP_NSAMPLES))

TOPK_TOKENS = int((os.getenv("SHAP_TOPK_TOKENS", "8") or "8").strip() or "8")
TOPK_TOKENS = max(1, min(64, TOPK_TOKENS))

FALLBACK_MAX_SEGMENTS = int((os.getenv("EXPLAIN_MAX_SEGMENTS", "8") or "8").strip() or "8")
FALLBACK_MAX_SEGMENTS = max(1, min(24, FALLBACK_MAX_SEGMENTS))


# ----------------------------
# Meta helpers
# ----------------------------
def _copy_meta(meta_in: Dict[str, Any]) -> Dict[str, Any]:
    meta_out: Dict[str, Any] = {"received_at": meta_in.get("received_at"), "version": meta_in.get("version")}
    for k in ("warnings", "latency_ms", "stage_latencies_ms", "total_latency_ms", "stage_errors"):
        if isinstance(meta_in, dict) and k in meta_in:
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
# Model inference (prefer Step2 backend)
# ----------------------------
def _softmax(logits: List[float]) -> List[float]:
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(float(x) - float(m)) for x in logits]
    s = float(sum(exps))
    if s <= 0:
        return [1.0 / max(1, len(logits))] * max(1, len(logits))
    return [float(e) / s for e in exps]


def _get_step2_module():
    return sys.modules.get("step2_roberta_inference")


def _infer_logits_probs(text: str) -> Dict[str, List[float]]:
    m = _get_step2_module()
    if m is not None:
        backend = getattr(m, "_BACKEND", None)
        build = getattr(m, "_build_model_text", None)
        if backend is not None and hasattr(backend, "infer") and callable(build):
            x = build(str(text))
            logits, probs = backend.infer(x)
            return {"logits": list(map(float, logits)), "probs": list(map(float, probs))}

    digest = hashlib.sha256(str(text).encode("utf-8")).digest()
    logits = [((digest[i] / 255.0) * 4.0 - 2.0) for i in range(3)]
    probs = _softmax(logits)
    return {"logits": list(map(float, logits)), "probs": list(map(float, probs))}


def _predict_batch(texts: List[str]) -> List[List[float]]:
    out: List[List[float]] = []
    for t in texts:
        pred = _infer_logits_probs(t)
        out.append(pred["probs"] if SHAP_OUTPUT_SPACE == "probability" else pred["logits"])
    return out


# ----------------------------
# Token span mapping (best-effort)
# ----------------------------
def _safe_token_spans(tokens: List[str], text: str) -> Optional[List[Dict[str, int]]]:
    if not text:
        return None
    spans: List[Dict[str, int]] = []
    cursor = 0
    for tok in tokens:
        if tok is None:
            return None
        s = str(tok)
        if s == "":
            spans.append({"start": cursor, "end": cursor})
            continue
        idx = text.find(s, cursor)
        if idx < 0:
            idx = text.find(s)
        if idx < 0:
            return None
        spans.append({"start": int(idx), "end": int(idx + len(s))})
        cursor = idx + len(s)
    return spans


def _argsort_desc(vals: List[float]) -> List[int]:
    return [i for i, _ in sorted(enumerate(vals), key=lambda t: (-float(t[1]), int(t[0])))]


def _argsort_asc(vals: List[float]) -> List[int]:
    return [i for i, _ in sorted(enumerate(vals), key=lambda t: (float(t[1]), int(t[0])))]


# ----------------------------
# Fallback: word-span occlusion
# ----------------------------
_WORD_RE = re.compile(r"\S+")


def _word_spans(text: str) -> List[Dict[str, int]]:
    return [{"start": m.start(), "end": m.end()} for m in _WORD_RE.finditer(text or "")]


def _segment_word_spans(text: str, max_segments: int) -> List[Dict[str, int]]:
    spans = _word_spans(text)
    if not spans:
        return [{"start": 0, "end": 0}]
    k = max(1, min(max_segments, len(spans)))
    # group contiguous words into k buckets
    step = max(1, len(spans) // k)
    out: List[Dict[str, int]] = []
    i = 0
    while i < len(spans):
        j = min(len(spans), i + step)
        out.append({"start": spans[i]["start"], "end": spans[j - 1]["end"]})
        i = j
    # merge if too many (due to rounding)
    while len(out) > k:
        a = out[-2]
        b = out[-1]
        out[-2] = {"start": a["start"], "end": b["end"]}
        out.pop()
    return out


def _mask_span(text: str, start: int, end: int) -> str:
    if not text:
        return ""
    start = max(0, min(len(text), start))
    end = max(0, min(len(text), end))
    if end <= start:
        return text
    return text[:start] + (" " * (end - start)) + text[end:]


def _call_explainer(explainer: Any, inputs: List[str]) -> Any:
    """
    SHAP versions differ:
      - some accept nsamples=
      - some accept max_evals=
      - some accept neither
    We pick what exists in the signature to avoid runtime crashes.
    """
    try:
        sig = inspect.signature(explainer.__call__)
        params = set(sig.parameters.keys())
    except Exception:
        params = set()

    kwargs: Dict[str, Any] = {}
    if "nsamples" in params:
        kwargs["nsamples"] = int(SHAP_NSAMPLES)
    elif "max_evals" in params:
        # A common substitute in newer SHAP builds
        kwargs["max_evals"] = int(SHAP_NSAMPLES)

    return explainer(inputs, **kwargs)


# ----------------------------
# Public entrypoint
# ----------------------------
async def process_step7_output(step7_out: Dict[str, Any]) -> Dict[str, Any]:
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

    # class to explain
    class_id = 0
    class_name = "false"
    try:
        class_id = int((((roberta or {}).get("label") or {}).get("class_id")))
        class_name = str((((roberta or {}).get("label") or {}).get("class_name")))
    except Exception:
        pass
    if class_id not in (0, 1, 2):
        class_id = 0

    # base/output in probability space
    base_probs = _infer_logits_probs("")["probs"]
    out_probs = _infer_logits_probs(normalized_claim)["probs"]
    base_value = float(base_probs[class_id]) if len(base_probs) >= 3 else 0.0
    output_value = float(out_probs[class_id]) if len(out_probs) >= 3 else 0.0

    shap_explainability: Dict[str, Any]

    if SHAP_ENABLED:
        # Try SHAP first – caller has explicitly enabled it via EXPLAIN_USE_SHAP.
        try:
            import shap  # type: ignore

            def f(text_list: List[str]) -> List[List[float]]:
                return _predict_batch(list(text_list))

            masker = shap.maskers.Text(None)
            explainer = shap.Explainer(f, masker, algorithm=SHAP_ALGO)

            start_ms = time.perf_counter()
            exp = _call_explainer(explainer, [normalized_claim])
            shap_ms = int(round((time.perf_counter() - start_ms) * 1000))

            tokens = list(exp.data[0])
            vals = exp.values[0]
            shap_vals = [float(v[class_id]) for v in vals]

            base_v = exp.base_values[0]
            if isinstance(base_v, (list, tuple)):
                base_out = float(base_v[class_id])
            else:
                base_out = float(base_v)

            out_v = None
            if hasattr(exp, "output_values") and exp.output_values is not None:
                ov = exp.output_values[0]
                try:
                    out_v = float(ov[class_id])
                except Exception:
                    out_v = None

            spans = _safe_token_spans([str(t) for t in tokens], normalized_claim)

            tok_out: List[Dict[str, Any]] = []
            for i, tok in enumerate(tokens):
                rec: Dict[str, Any] = {"i": int(i), "token": str(tok), "shap_value": float(shap_vals[i])}
                if spans is not None and i < len(spans):
                    rec["char_span"] = {"start": int(spans[i]["start"]), "end": int(spans[i]["end"])}
                tok_out.append(rec)

            if not tok_out:
                tok_out = [{"i": 0, "token": normalized_claim, "shap_value": 0.0}]

            k = max(1, min(TOPK_TOKENS, len(tok_out)))
            raw_vals = [float(x["shap_value"]) for x in tok_out]
            top_pos = _argsort_desc(raw_vals)[:k]
            top_neg = _argsort_asc(raw_vals)[:k]

            shap_explainability = {
                "method": f"shap_{SHAP_ALGO}",
                "explained_class": {"class_id": int(class_id), "class_name": class_name},
                "base_value": float(base_out),
                "output_value": float(out_v) if out_v is not None else (float(output_value) if SHAP_OUTPUT_SPACE == "probability" else None),
                "output_space": SHAP_OUTPUT_SPACE,
                "tokens": tok_out,
                "top_tokens": {"k": int(k), "positive": list(map(int, top_pos)), "negative": list(map(int, top_neg))},
                "meta": {"requested_samples": int(SHAP_NSAMPLES), "time_ms": int(shap_ms)},
            }
        except Exception as e:
            stage_errors.append(
                {
                    "stage": "8_Explainability_SHAP",
                    "code": "SHAP_FALLBACK",
                    "message": "SHAP unavailable or failed; used word-span occlusion fallback.",
                    "details": {"error": str(e)},
                }
            )
            warnings.append({"code": "SHAP_FALLBACK_USED", "message": "Explainability used word-span occlusion fallback."})
            SHAP_ENABLED_LOCAL = False
        else:
            SHAP_ENABLED_LOCAL = True
    else:
        SHAP_ENABLED_LOCAL = False

    if not SHAP_ENABLED_LOCAL:
        spans = _segment_word_spans(normalized_claim, max_segments=FALLBACK_MAX_SEGMENTS)
        contribs: List[float] = []
        seg_out: List[Dict[str, Any]] = []

        full = _infer_logits_probs(normalized_claim)["probs"]
        full_v = float(full[class_id]) if len(full) >= 3 else 0.0

        for i, sp in enumerate(spans):
            masked = _mask_span(normalized_claim, sp["start"], sp["end"])
            p = _infer_logits_probs(masked)["probs"]
            pv = float(p[class_id]) if len(p) >= 3 else 0.0
            val = float(full_v - pv)
            contribs.append(val)
            seg_out.append(
                {
                    "i": int(i),
                    "span": {"start": int(sp["start"]), "end": int(sp["end"])},
                    "text": normalized_claim[sp["start"] : sp["end"]],
                    "value": float(val),
                }
            )

        k = max(1, min(TOPK_TOKENS, len(seg_out)))
        top_pos = _argsort_desc(contribs)[:k]
        top_neg = _argsort_asc(contribs)[:k]

        shap_explainability = {
            "method": "word_span_occlusion",
            "explained_class": {"class_id": int(class_id), "class_name": class_name},
            "base_value": float(base_value),
            "output_value": float(output_value),
            "output_space": "probability",
            "segments": seg_out,
            "top_segments": {"k": int(k), "positive": list(map(int, top_pos)), "negative": list(map(int, top_neg))},
            "meta": {"max_segments": int(FALLBACK_MAX_SEGMENTS)},
        }

    _mark_latency(meta_out, "8_Explainability_SHAP", int(round((time.perf_counter() - t0) * 1000)))

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