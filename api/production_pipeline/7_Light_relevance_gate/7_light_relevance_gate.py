# 7_light_relevance_gate.py
from __future__ import annotations

import math
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Errors
# ----------------------------


class LightRelevanceGateError(Exception):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


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
# Embedding + cache (reuse Step6 if loaded)
# ----------------------------


DEFAULT_EMBED_MODEL = "hashing/384"  # must match Step6 default for cache reuse
DEFAULT_EMBED_CACHE_ENABLED = True
DEFAULT_EMBED_CACHE_NAMESPACE = "emb"
DEFAULT_EMBED_CACHE_TTL_S = 0

GATE_METHOD = "cosine_threshold"  # schema enum
EPS = 1e-9


def _get_step6_backend():
    """Attempt to reuse the exact Step6 embedding + cache backend.

    This enables Stage7 to read embeddings cached during Stage6, as recommended.
    """
    m = sys.modules.get("step6_mmr_selection")
    if not m:
        return None
    needed = ("_cache_get", "_cache_set", "_cache_key", "_embed_one", "_cosine")
    if all(hasattr(m, k) for k in needed):
        return m
    return None


def _cache_key(namespace: str, embed_model: str, chunk_id: str) -> str:
    # If Step6 is present, prefer its exact key format.
    m = _get_step6_backend()
    if m is not None:
        return m._cache_key(namespace, embed_model, chunk_id)  # type: ignore[attr-defined]
    return f"{namespace}:{embed_model}:{chunk_id}"


def _cache_get(key: str) -> Optional[List[float]]:
    m = _get_step6_backend()
    if m is not None:
        return m._cache_get(key)  # type: ignore[attr-defined]
    return None


def _cache_set(key: str, vec: List[float], ttl_s: int) -> None:
    m = _get_step6_backend()
    if m is not None:
        m._cache_set(key, vec, ttl_s)  # type: ignore[attr-defined]


def _embed_one(text: str, embed_model: str) -> List[float]:
    m = _get_step6_backend()
    if m is not None:
        return m._embed_one(text, embed_model)  # type: ignore[attr-defined]

    # Fallback: minimal deterministic hashing embedding (same spirit as Step6)
    # NOTE: We only hit this if Stage7 is used standalone.
    import hashlib

    dims = 384
    if embed_model.startswith("hashing/"):
        try:
            dims = max(64, min(4096, int(embed_model.split("/", 1)[1])))
        except Exception:
            dims = 384

    toks = [t.strip().lower() for t in (text or "").split() if t.strip()]
    vec = [0.0] * dims
    for tok in toks:
        h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        x = int.from_bytes(h, "big", signed=False)
        idx = x % dims
        sign = -1.0 if (x & 1) else 1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        inv = 1.0 / norm
        vec = [v * inv for v in vec]
    return vec


def _cosine(a: List[float], b: List[float]) -> float:
    m = _get_step6_backend()
    if m is not None:
        return float(m._cosine(a, b))  # type: ignore[attr-defined]

    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        va = float(a[i])
        vb = float(b[i])
        dot += va * vb
        na += va * va
        nb += vb * vb
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (math.sqrt(na) * math.sqrt(nb))))


def _cosine_01(a: List[float], b: List[float]) -> float:
    """Map cosine [-1,1] to [0,1] for schema compatibility."""
    c = _cosine(a, b)
    s = 0.5 * (c + 1.0)
    return max(0.0, min(1.0, float(s)))


# ----------------------------
# Helpers
# ----------------------------


def _index_candidates_by_chunk_id(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        cid = str(it.get("chunk_id") or "").strip()
        if not cid:
            continue
        by_id[cid] = it
    return by_id


def _doc_for_output(doc: Any) -> Dict[str, Any]:
    """EvidenceTopK requires doc:{source,title,url} only (no doc_id)."""
    if not isinstance(doc, dict):
        return {"source": "unknown", "title": "Unknown", "url": "about:blank"}
    out = {
        "source": str(doc.get("source") or "unknown").strip() or "unknown",
        "title": str(doc.get("title") or "Unknown").strip() or "Unknown",
        "url": str(doc.get("url") or "about:blank").strip() or "about:blank",
    }
    if doc.get("published_at"):
        out["published_at"] = doc.get("published_at")
    return out


def _doc_id(doc: Any, fallback: str) -> str:
    if isinstance(doc, dict):
        d = str(doc.get("doc_id") or "").strip()
        if d:
            return d
    return fallback


# ----------------------------
# Public entrypoint
# ----------------------------


async def process_step6_output(step6_out: Dict[str, Any]) -> Dict[str, Any]:
    """Step6 -> Step7: lightweight relevance gate on MMR-selected chunks."""

    t0 = time.perf_counter()
    meta_out: Dict[str, Any] = _copy_meta(step6_out.get("meta") or {})
    warnings = _ensure_warnings(meta_out)
    stage_errors = _ensure_stage_errors(meta_out)

    try:
        request_id = step6_out["request_id"]
        claim_id = step6_out["claim_id"]
        user_claim = step6_out["user_claim"]
        normalized_claim = step6_out["normalized_claim"]
        roberta = step6_out.get("roberta")
        routing = step6_out.get("routing") or {}
        query_plan = step6_out.get("query_plan")
        rag_candidates = step6_out.get("rag_candidates") or {}
        mmr_selected = step6_out.get("mmr_selected") or {}
    except Exception as e:
        raise LightRelevanceGateError("INVALID_INPUT", "Step7 expected Step6 output shape.", {"error": str(e)})

    rag = ((routing or {}).get("rag") or {}) if isinstance(routing, dict) else {}
    K = int(rag.get("final_top_k") or 5)
    T = float(rag.get("min_relevance") or 0.0)
    K = max(1, K)
    T = max(0.0, min(1.0, T))

    # Meta embedding knobs (reuse Stage6 defaults)
    embed_model = str(meta_out.get("embed_model") or DEFAULT_EMBED_MODEL)
    cache_enabled = meta_out.get("embedding_cache_enabled")
    if cache_enabled is None:
        cache_enabled = DEFAULT_EMBED_CACHE_ENABLED
    cache_enabled = bool(cache_enabled)
    namespace = str(meta_out.get("embedding_cache_namespace") or DEFAULT_EMBED_CACHE_NAMESPACE)
    ttl_s = meta_out.get("embedding_cache_ttl_s")
    ttl_s = int(ttl_s) if isinstance(ttl_s, (int, float, str)) and str(ttl_s).strip() != "" else DEFAULT_EMBED_CACHE_TTL_S
    ttl_s = max(0, int(ttl_s))

    meta_out["embed_model"] = embed_model
    meta_out["embedding_cache_enabled"] = cache_enabled
    meta_out["embedding_cache_namespace"] = namespace
    meta_out["embedding_cache_ttl_s"] = ttl_s

    items = rag_candidates.get("items")
    if not isinstance(items, list):
        items = []
    by_id = _index_candidates_by_chunk_id([it for it in items if isinstance(it, dict)])

    sel_items = mmr_selected.get("items")
    if not isinstance(sel_items, list):
        sel_items = []

    if len(sel_items) == 0:
        warnings.append({"code": "EMPTY_MMR_SELECTION", "message": "No mmr_selected.items to gate."})
        evidence_topk = {"top_k": K, "min_relevance": T, "method": GATE_METHOD, "items": []}
        ms = int(round((time.perf_counter() - t0) * 1000))
        _mark_latency(meta_out, "7_Light_relevance_gate", ms)
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
            "meta": meta_out,
        }

    # 3) Load / compute claim embedding (reuse Stage6 cache if available)
    try:
        q_key = _cache_key(namespace, embed_model, f"claim:{claim_id}")
        q_vec = _cache_get(q_key) if cache_enabled else None
        if q_vec is None:
            q_vec = _embed_one(normalized_claim, embed_model)
            if cache_enabled:
                _cache_set(q_key, q_vec, ttl_s)

        scored: List[Tuple[str, float, float, Dict[str, Any], str]] = []
        # tuple: (chunk_id, relevance_score, retrieval_score, doc, text)

        for sel in sel_items:
            if not isinstance(sel, dict):
                continue
            cid = str(sel.get("chunk_id") or "").strip()
            if not cid:
                continue
            base = by_id.get(cid)
            if not base:
                continue

            text = str(base.get("text") or "")
            if not text.strip():
                continue
            doc = base.get("doc")
            retrieval_score = float(base.get("score") or 0.0)

            k = _cache_key(namespace, embed_model, cid)
            v = _cache_get(k) if cache_enabled else None
            if v is None:
                v = _embed_one(text, embed_model)
                if cache_enabled:
                    _cache_set(k, v, ttl_s)

            rel = _cosine_01(v, q_vec)

            if rel + EPS < T:
                continue

            scored.append((cid, float(rel), float(retrieval_score), doc if isinstance(doc, dict) else {}, text))

    except Exception as e:
        stage_errors.append(
            {
                "stage": "7_Light_relevance_gate",
                "code": "RELEVANCE_SCORING_FAILED",
                "message": "Failed to compute relevance scores; returning empty evidence_topk.",
                "details": {"error": str(e)},
            }
        )
        scored = []

    # 5) Select top-K with deterministic tie-breakers
    #   primary: relevance_score DESC
    #   tie 1: retrieval_score DESC
    #   tie 2: chunk_id ASC
    scored.sort(key=lambda x: (-x[1], -x[2], x[0]))
    kept = scored[:K]

    if len(kept) == 0:
        warnings.append(
            {
                "code": "NO_EVIDENCE_AFTER_GATE",
                "message": "No MMR-selected chunks passed the relevance threshold; evidence_topk is empty.",
            }
        )

    out_items: List[Dict[str, Any]] = []
    for i, (cid, rel, _rs, doc, text) in enumerate(kept):
        did = _doc_id(doc, fallback="doc_unknown")
        out_items.append(
            {
                "rank": i + 1,
                "chunk_id": cid,
                "doc_id": did,
                "text": text,
                "relevance_score": float(rel),
                "doc": _doc_for_output(doc),
            }
        )

    evidence_topk = {
        "top_k": K,
        "min_relevance": T,
        "method": GATE_METHOD,
        "items": out_items,
    }

    ms = int(round((time.perf_counter() - t0) * 1000))
    _mark_latency(meta_out, "7_Light_relevance_gate", ms)

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
        "meta": meta_out,
    }
