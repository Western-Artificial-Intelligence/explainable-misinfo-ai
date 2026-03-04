# 4_query_building.py
from __future__ import annotations

import importlib.util
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class QueryBuildingError(Exception):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


# ----------------------------
# Config
# ----------------------------

# Default paraphrase count (override via routing.rag.query_variants or env)
DEFAULT_N_VARIANTS = int((os.getenv("QUERY_EXPAND_N", "4") or "4").strip() or "4")
DEFAULT_N_VARIANTS = max(0, min(12, DEFAULT_N_VARIANTS))

TEMP_BASE = float((os.getenv("QUERY_EXPAND_TEMP", "0.7") or "0.7").strip() or "0.7")
TEMP_BASE = max(0.0, min(2.0, TEMP_BASE))

NUM_PREDICT = int((os.getenv("QUERY_EXPAND_NUM_PREDICT", "64") or "64").strip() or "64")
NUM_PREDICT = max(8, min(512, NUM_PREDICT))

MAX_ATTEMPTS = int((os.getenv("QUERY_EXPAND_MAX_ATTEMPTS", "12") or "12").strip() or "12")
MAX_ATTEMPTS = max(1, min(50, MAX_ATTEMPTS))

MAX_QUERY_CHARS = int((os.getenv("QUERY_MAX_CHARS", "200") or "200").strip() or "200")
MAX_QUERY_CHARS = max(40, min(500, MAX_QUERY_CHARS))


SYS_PARAPHRASE = (
    "/no_think\n"
    "Return ONE paraphrase suitable as a web search query.\n"
    "Prefer ONE LINE.\n"
    "Do NOT include reasoning, bullets, numbering, quotes, or multiple options.\n"
    "Preserve entities and names exactly as in the input.\n"
    "Do NOT add suffixes like 'news' or 'fact check'.\n"
)


# ----------------------------
# Meta helpers (schema-safe)
# ----------------------------

def _copy_meta(meta_in: Dict[str, Any]) -> Dict[str, Any]:
    meta_out: Dict[str, Any] = {
        "received_at": meta_in.get("received_at"),
        "version": meta_in.get("version"),
    }
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


def _ensure_stage_latencies(meta: Dict[str, Any]) -> Dict[str, int]:
    sl = meta.get("stage_latencies_ms")
    if isinstance(sl, dict):
        return sl  # type: ignore[return-value]
    meta["stage_latencies_ms"] = {}
    return meta["stage_latencies_ms"]  # type: ignore[return-value]


def _mark_latency(meta: Dict[str, Any], stage_name: str, ms: int) -> None:
    sl = _ensure_stage_latencies(meta)
    sl[stage_name] = max(0, int(ms))
    try:
        meta["total_latency_ms"] = int(sum(int(v) for v in sl.values() if isinstance(v, (int, float))))
    except Exception:
        pass


# ----------------------------
# Local dependency loader (robust under importlib.exec_module)
# ----------------------------

def _load_llm_blackbox_class():
    """Load LLMBlackbox via file path so this module works even when dynamically imported."""
    here = Path(__file__).resolve()
    llm_path = here.parents[1] / "middlewares" / "llm_blackbox.py"  # production_pipeline/middlewares/llm_blackbox.py
    spec = importlib.util.spec_from_file_location("pipeline_llm_blackbox", llm_path)
    if spec is None or spec.loader is None:
        raise QueryBuildingError("IMPORT_ERROR", "Failed to create spec for llm_blackbox.", {"path": str(llm_path)})
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        return getattr(mod, "LLMBlackbox")
    except Exception as e:
        raise QueryBuildingError("IMPORT_ERROR", "llm_blackbox.py did not expose LLMBlackbox.", {"error": str(e)})


# ----------------------------
# Query cleaning
# ----------------------------

def _clean_query(q: str) -> str:
    if not isinstance(q, str):
        return ""
    q = q.strip()

    # Keep only the first line if the model returns multi-line by accident.
    q = q.splitlines()[0].strip()

    # Strip wrapping quotes/backticks.
    q = q.strip(" \t\r\n\"'`")

    # Remove common bullet prefixes.
    q = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", q).strip()

    # Collapse whitespace.
    q = re.sub(r"\s+", " ", q).strip()

    if len(q) > MAX_QUERY_CHARS:
        q = q[:MAX_QUERY_CHARS].rstrip()

    return q


# ----------------------------
# Simple non-LLM fallback paraphraser
# ----------------------------

def _fallback_paraphrases(claim: str, k: int) -> List[str]:
    """
    Deterministic, cheap paraphrases used when LLM is unavailable.
    This keeps the rest of the pipeline working even without Ollama.
    """
    base = _clean_query(claim)
    if not base or k <= 0:
        return []

    templates = [
        "What is the truth about {c}?",
        "Fact-check the claim: {c}",
        "Is it accurate that {c}?",
        "Latest news about whether {c}",
        "Background information on the claim that {c}",
    ]

    out: List[str] = []
    for t in templates:
        if len(out) >= k:
            break
        q = _clean_query(t.format(c=base))
        if q and q.lower() != base.lower() and all(q.lower() != x.lower() for x in out):
            out.append(q)

    return out[:k]


# ----------------------------
# Paraphrase generation
# ----------------------------

async def _paraphrase_once(
    llm,
    claim: str,
    *,
    temperature: float,
    avoid: Optional[List[str]] = None,
) -> str:
    extra = ""
    if avoid:
        # Encourage non-duplicate wording without changing entities.
        joined = " | ".join(x[:80] for x in avoid[:6])
        extra = f"Avoid repeating these phrasings: {joined}"

    system = SYS_PARAPHRASE + ("\n" + extra if extra else "")
    text = await llm.generate_async(
        system_context=system,
        user_context=claim,
        temperature=temperature,
        num_predict=NUM_PREDICT,
    )
    return _clean_query(text)


async def _generate_paraphrases(
    claim: str,
    claim_id: str,
    *,
    k: int,
) -> List[str]:
    if k <= 0:
        return []

    kept: List[str] = []

    # Try LLM-based expansion first.
    try:
        LLMBlackbox = _load_llm_blackbox_class()
        llm = LLMBlackbox()

        # We can't seed the backend from here, but we can vary temperature a bit.
        for attempt in range(MAX_ATTEMPTS):
            if len(kept) >= k:
                break

            # Mild temperature jitter to reduce duplicates.
            temp = float(max(0.0, min(2.0, TEMP_BASE + (0.15 * (attempt % 3)))))

            try:
                cand = await _paraphrase_once(llm, claim, temperature=temp, avoid=kept)
            except Exception:
                # Give up on further LLM attempts; we'll fall back below.
                break

            if not cand:
                continue

            # Do not include the original claim verbatim as a "paraphrase".
            if _clean_query(claim).lower() == cand.lower():
                continue

            # Deduplicate (case-insensitive).
            low = cand.lower()
            if all(low != x.lower() for x in kept):
                kept.append(cand)

    except Exception:
        # Any error loading/constructing the LLM will be handled by fallback.
        kept = []

    # If LLM produced nothing, use deterministic non-LLM fallbacks.
    if not kept:
        kept = _fallback_paraphrases(claim, k)

    return kept[:k]


# ----------------------------
# Main
# ----------------------------

def _resolve_k(step3_out: Dict[str, Any]) -> int:
    routing = step3_out.get("routing")
    rag = (routing or {}).get("rag") if isinstance(routing, dict) else {}

    k = rag.get("query_variants")
    if k is None:
        k = DEFAULT_N_VARIANTS
    try:
        k = int(k)
    except Exception:
        k = DEFAULT_N_VARIANTS

    return max(0, min(12, k))


async def run_4_query_building(step3_out: Dict[str, Any]) -> Dict[str, Any]:
    try:
        request_id = step3_out["request_id"]
        claim_id = step3_out["claim_id"]
        user_claim = step3_out["user_claim"]
        normalized_claim = step3_out["normalized_claim"]
        roberta = step3_out.get("roberta")
        routing = step3_out.get("routing") or {}
        meta_in = step3_out.get("meta") or {}
    except Exception as e:
        raise QueryBuildingError("INVALID_INPUT", "Step4 expected Step3 output shape.", {"error": str(e)})

    meta_out = _copy_meta(meta_in)
    warnings = _ensure_warnings(meta_out)

    rag = (routing.get("rag") or {}) if isinstance(routing, dict) else {}
    use_qexp = bool(rag.get("use_query_expansion", True))

    primary = _clean_query(str(normalized_claim or user_claim or ""))
    if not primary:
        raise QueryBuildingError("EMPTY_CLAIM", "No claim text available for query building.")

    # Step5 expects query_plan["queries"] to be a list of objects
    # with at least {"q": <query_text>, "variant": <label>}.
    queries: List[Dict[str, Any]] = [
        {"q": primary, "variant": "primary"},
    ]
    paraphrases: List[str] = []

    if use_qexp:
        k = _resolve_k(step3_out)
        try:
            paraphrases = await _generate_paraphrases(primary, claim_id, k=k)
        except Exception as e:
            warnings.append({"code": "query_expansion_failed", "message": str(e)})
            paraphrases = []

        for p in paraphrases:
            if not p:
                continue
            # Deduplicate case-insensitively against existing queries.
            if any(p.lower() == str((qobj or {}).get("q") or "").lower() for qobj in queries):
                continue
            queries.append({"q": p, "variant": "paraphrase"})

    query_plan: Dict[str, Any] = {
        "primary_query": primary,
        "queries": queries,
        "expansion": {
            "enabled": bool(use_qexp),
            "method": "llm_paraphrase_v1" if use_qexp else "none",
            "requested": int(_resolve_k(step3_out)) if use_qexp else 0,
            "kept": int(len(paraphrases)) if use_qexp else 0,
        },
    }

    return {
        "request_id": request_id,
        "claim_id": claim_id,
        "user_claim": user_claim,
        "normalized_claim": normalized_claim,
        "roberta": roberta,
        "routing": routing,
        "query_plan": query_plan,
        "meta": meta_out,
    }


async def process_step3_output(step3_out: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    out = await run_4_query_building(step3_out)
    dt_ms = int(round((time.perf_counter() - t0) * 1000))
    _mark_latency(out["meta"], "step4_query_building", dt_ms)
    return out
