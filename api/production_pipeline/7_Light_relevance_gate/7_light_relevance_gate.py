from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional


class LightRelevanceGateError(Exception):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


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


def _domain_blocklist() -> set[str]:
    # User-controlled only. Default empty.
    raw = os.getenv("EVIDENCE_DOMAIN_BLOCKLIST", "") or ""
    out = set()
    for x in raw.split(","):
        x = x.strip().lower()
        if x:
            out.add(x)
    return out


def _blocked(doc: Any, block: set[str]) -> bool:
    if not block or not isinstance(doc, dict):
        return False
    url = str(doc.get("url") or "").lower()
    src = str(doc.get("source") or "").lower()
    for d in block:
        if d and (d in url or src == d or src.endswith("." + d)):
            return True
    return False


def _doc_for_output(doc: Any) -> Dict[str, Any]:
    if not isinstance(doc, dict):
        return {"source": "unknown", "title": "Unknown", "url": "about:blank"}
    return {
        "source": str(doc.get("source") or "unknown").strip() or "unknown",
        "title": str(doc.get("title") or "Unknown").strip() or "Unknown",
        "url": str(doc.get("url") or "about:blank").strip() or "about:blank",
    }


async def process_step6_output(step6_out: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    meta_out = _copy_meta(step6_out.get("meta") or {})
    warnings = _ensure_warnings(meta_out)

    try:
        request_id = step6_out["request_id"]
        claim_id = step6_out["claim_id"]
        user_claim = step6_out["user_claim"]
        normalized_claim = step6_out["normalized_claim"]
        roberta = step6_out.get("roberta")
        routing = step6_out.get("routing") or {}
        query_plan = step6_out.get("query_plan") or {}
        rag_candidates = step6_out.get("rag_candidates") or {}
        mmr_selected = step6_out.get("mmr_selected") or {}
    except Exception as e:
        raise LightRelevanceGateError("INVALID_INPUT", "Step7 expected Step6 output shape.", {"error": str(e)})

    rag = ((routing or {}).get("rag") or {}) if isinstance(routing, dict) else {}
    K = max(1, int(rag.get("final_top_k") or 6))

    # threshold = routing min_relevance; optional env override ONLY if explicitly set
    min_rel = float(rag.get("min_relevance") or 0.0)
    env_thr = os.getenv("EVIDENCE_MIN_RELEVANCE")
    if env_thr is not None and env_thr.strip() != "":
        try:
            min_rel = max(min_rel, float(env_thr))
        except Exception:
            warnings.append({"code": "BAD_EVIDENCE_MIN_RELEVANCE", "message": "EVIDENCE_MIN_RELEVANCE set but not float; ignored."})
    min_rel = max(0.0, min(1.0, min_rel))

    block = _domain_blocklist()
    if block:
        meta_out["evidence_domain_blocklist"] = sorted(list(block))

    sel_items = mmr_selected.get("items")
    if not isinstance(sel_items, list) or not sel_items:
        warnings.append({"code": "EMPTY_MMR_SELECTION", "message": "No mmr_selected.items to gate."})
        evidence_topk = {"top_k": K, "min_relevance": min_rel, "method": "mmr_relevance_threshold", "items": []}
        _mark_latency(meta_out, "7_Light_relevance_gate", int(round((time.perf_counter() - t0) * 1000)))
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

    pool: List[Dict[str, Any]] = []
    for it in sel_items:
        if not isinstance(it, dict):
            continue
        rel = float(it.get("mmr_relevance") or 0.0)
        if rel + 1e-9 < min_rel:
            continue
        doc = it.get("doc")
        if _blocked(doc, block):
            continue
        pool.append(it)

    pool.sort(key=lambda x: (-float(x.get("mmr_relevance") or 0.0),
                             -float(x.get("retrieval_score") or x.get("score") or 0.0),
                             int(x.get("rank") or 10**9)))
    pool = pool[:K]

    out_items: List[Dict[str, Any]] = []
    for i, it in enumerate(pool, start=1):
        doc = it.get("doc") if isinstance(it.get("doc"), dict) else {}
        out_items.append({
            "rank": i,
            "chunk_id": str(it.get("chunk_id") or ""),
            "doc_id": str(doc.get("doc_id") or "doc_unknown"),
            "text": str(it.get("text") or ""),
            "relevance_score": float(it.get("mmr_relevance") or 0.0),
            "doc": _doc_for_output(doc),
        })

    if not out_items:
        warnings.append({"code": "NO_EVIDENCE_AFTER_GATE", "message": "No chunks passed evidence gating."})

    evidence_topk = {"top_k": K, "min_relevance": float(min_rel), "method": "mmr_relevance_threshold", "items": out_items}
    _mark_latency(meta_out, "7_Light_relevance_gate", int(round((time.perf_counter() - t0) * 1000)))

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