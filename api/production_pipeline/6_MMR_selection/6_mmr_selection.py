# 6_mmr_selection.py
from __future__ import annotations

import hashlib
import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Errors
# ----------------------------


class MMRSelectionError(Exception):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


# ----------------------------
# Meta helpers (match Step4/5 pattern)
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
        # leave as-is
        pass


# ----------------------------
# Embedding backend (MVP)
# ----------------------------


DEFAULT_EMBED_MODEL = "hashing/384"  # offline-safe MVP
DEFAULT_EMBED_CACHE_ENABLED = True
DEFAULT_EMBED_CACHE_NAMESPACE = "emb"
DEFAULT_EMBED_CACHE_TTL_S = 0

EPS = 1e-9


def _parse_hashing_dims(embed_model: str, default: int = 384) -> int:
    if not embed_model.startswith("hashing/"):
        return default
    try:
        return max(64, min(4096, int(embed_model.split("/", 1)[1])))
    except Exception:
        return default


def _hashing_embed(text: str, dims: int) -> List[float]:
    """Deterministic, offline-safe hashed bag-of-words embedding."""
    # Simple tokenization: split on whitespace; keep alnum-ish chunks.
    toks = [t.strip().lower() for t in (text or "").split() if t.strip()]
    vec = [0.0] * dims
    if not toks:
        return vec

    for tok in toks:
        h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        x = int.from_bytes(h, "big", signed=False)
        idx = x % dims
        sign = -1.0 if (x & 1) else 1.0
        vec[idx] += sign

    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        inv = 1.0 / norm
        vec = [v * inv for v in vec]
    return vec


_ST_MODEL = None
_ST_LOCK = threading.Lock()


def _sbert_embed(texts: List[str], model_id: str) -> Optional[List[List[float]]]:
    """Optional: sentence-transformers if available locally.

    NOTE: If the model is not present in local cache, SentenceTransformer will try
    to download it; in restricted/offline envs this can fail. We catch and return None.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return None

    global _ST_MODEL
    try:
        with _ST_LOCK:
            if _ST_MODEL is None or getattr(_ST_MODEL, "_model_id", None) != model_id:
                m = SentenceTransformer(model_id)
                setattr(m, "_model_id", model_id)
                _ST_MODEL = m
        embs = _ST_MODEL.encode(texts, normalize_embeddings=True)
        return [list(map(float, row)) for row in embs]
    except Exception:
        return None


def _embed_one(text: str, embed_model: str) -> List[float]:
    # Prefer SBERT if user explicitly requested a non-hashing model and it works.
    if not embed_model.startswith("hashing/"):
        out = _sbert_embed([text], embed_model)
        if out and len(out) == 1:
            return out[0]

    dims = _parse_hashing_dims(embed_model if embed_model.startswith("hashing/") else DEFAULT_EMBED_MODEL)
    return _hashing_embed(text, dims)


def _cosine(a: List[float], b: List[float]) -> float:
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


# ----------------------------
# Embedding cache (in-memory, per-process)
# ----------------------------


_EMB_CACHE: Dict[str, Tuple[List[float], Optional[float]]] = {}
_EMB_CACHE_LOCK = threading.Lock()


def _cache_get(key: str) -> Optional[List[float]]:
    now = time.time()
    with _EMB_CACHE_LOCK:
        hit = _EMB_CACHE.get(key)
        if not hit:
            return None
        vec, exp = hit
        if exp is not None and now >= exp:
            _EMB_CACHE.pop(key, None)
            return None
        return vec


def _cache_set(key: str, vec: List[float], ttl_s: int) -> None:
    exp = None
    if isinstance(ttl_s, int) and ttl_s > 0:
        exp = time.time() + float(ttl_s)
    with _EMB_CACHE_LOCK:
        _EMB_CACHE[key] = (vec, exp)


def _cache_key(namespace: str, embed_model: str, chunk_id: str) -> str:
    return f"{namespace}:{embed_model}:{chunk_id}"


# ----------------------------
# Input adapters (tolerate current Step5 shape)
# ----------------------------


def _stable_chunk_id(*parts: str) -> str:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"chunk_{h}"


def _coerce_rag_candidates(step5_out: Dict[str, Any], warnings: List[Dict[str, str]]) -> Dict[str, Any]:
    """Accept either:
      - new shape: step5_out['rag_candidates']
      - legacy shape: step5_out['retrieval'] with 'candidates'
    Returns canonical rag_candidates with an 'items' list.
    """
    if isinstance(step5_out.get("rag_candidates"), dict):
        rc = step5_out["rag_candidates"]
        if isinstance(rc.get("items"), list):
            return rc

    # Legacy adapter: Step5 currently returns web SERP docs under retrieval.candidates
    if isinstance(step5_out.get("retrieval"), dict):
        retrieval = step5_out["retrieval"]
        cands = retrieval.get("candidates")
        if isinstance(cands, list):
            warnings.append(
                {
                    "code": "LEGACY_STEP5_SHAPE",
                    "message": "Step6 adapted legacy Step5 'retrieval.candidates' into rag_candidates.items.",
                }
            )
            items: List[Dict[str, Any]] = []
            for it in cands:
                if not isinstance(it, dict):
                    continue
                url = str(it.get("url") or "").strip()
                doc_id = str(it.get("doc_id") or "").strip() or (
                    f"doc_{hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]}" if url else "doc_unknown"
                )
                title = str(it.get("title") or "")
                snippet = str(it.get("snippet") or "")
                text = (title + " " + snippet).strip()
                chunk_id = _stable_chunk_id(doc_id, title, snippet)
                score = float(it.get("relevance") or it.get("score") or 0.0)
                items.append(
                    {
                        "chunk_id": chunk_id,
                        "text": text,
                        "score": score,
                        "doc": {
                            "doc_id": doc_id,
                            "url": url,
                            "title": title,
                            "source": str(it.get("displayLink") or ""),
                        },
                    }
                )
            return {
                "provider": retrieval.get("provider"),
                "params": retrieval.get("params"),
                "items": items,
                "stats": retrieval.get("stats"),
            }

    raise MMRSelectionError("INVALID_INPUT", "Step6 expected Step5 output with rag_candidates or retrieval.")


# ----------------------------
# Core MMR
# ----------------------------


def _mmr_select(
    candidates: List[Dict[str, Any]],
    *,
    q_vec: List[float],
    cand_vecs: Dict[str, List[float]],
    top_n: int,
    mmr_lambda: float,
) -> List[Dict[str, Any]]:
    lam = max(0.0, min(1.0, float(mmr_lambda)))
    N = max(1, int(top_n))

    # precompute relevance sims
    rel_sim: Dict[str, float] = {}
    for it in candidates:
        cid = str(it.get("chunk_id") or "")
        if not cid:
            continue
        rel_sim[cid] = _cosine(cand_vecs[cid], q_vec)

    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()

    # cache pairwise sims computed on demand
    pair_sim: Dict[Tuple[str, str], float] = {}

    def sim(a: str, b: str) -> float:
        if a == b:
            return 1.0
        k = (a, b) if a < b else (b, a)
        v = pair_sim.get(k)
        if v is not None:
            return v
        v = _cosine(cand_vecs[a], cand_vecs[b])
        pair_sim[k] = v
        return v

    for step in range(1, N + 1):
        best: Optional[Dict[str, Any]] = None
        best_mmr = -1e18
        best_red = 0.0
        best_rel = 0.0

        for it in candidates:
            cid = str(it.get("chunk_id") or "")
            if not cid or cid in selected_ids:
                continue

            r = float(rel_sim.get(cid, 0.0))
            if not selected_ids:
                red = 0.0
            else:
                red = max(sim(cid, sid) for sid in selected_ids)

            mmr = lam * r - (1.0 - lam) * red

            if best is None or (mmr > best_mmr + EPS):
                best = it
                best_mmr = mmr
                best_red = red
                best_rel = r
            elif abs(mmr - best_mmr) <= EPS:
                # tie-break 1: original score desc
                s_it = float(it.get("score") or 0.0)
                s_best = float(best.get("score") or 0.0)
                if s_it > s_best + EPS:
                    best = it
                    best_mmr = mmr
                    best_red = red
                    best_rel = r
                elif abs(s_it - s_best) <= EPS:
                    # tie-break 2: chunk_id asc
                    if cid < str(best.get("chunk_id") or ""):
                        best = it
                        best_mmr = mmr
                        best_red = red
                        best_rel = r

        if best is None:
            break

        best_out = dict(best)
        best_out["_mmr_score"] = float(best_mmr)
        best_out["_relevance_sim"] = float(best_rel)
        best_out["_max_redundancy_sim"] = float(best_red)

        selected.append(best_out)
        selected_ids.add(str(best.get("chunk_id")))

    return selected


# ----------------------------
# Public entrypoint
# ----------------------------


async def process_step5_output(step5_out: Dict[str, Any]) -> Dict[str, Any]:
    """Step5 -> Step6: MMR selection over candidate chunks."""

    t0 = time.perf_counter()
    meta_out: Dict[str, Any] = _copy_meta(step5_out.get("meta") or {})
    warnings = _ensure_warnings(meta_out)
    stage_errors = _ensure_stage_errors(meta_out)

    try:
        request_id = step5_out["request_id"]
        claim_id = step5_out["claim_id"]
        user_claim = step5_out["user_claim"]
        normalized_claim = step5_out["normalized_claim"]
        roberta = step5_out.get("roberta")
        routing = step5_out.get("routing") or {}
        query_plan = step5_out.get("query_plan")
    except Exception as e:
        raise MMRSelectionError("INVALID_INPUT", "Step6 expected Step5 output shape.", {"error": str(e)})

    rag = ((routing or {}).get("rag") or {}) if isinstance(routing, dict) else {}
    N = int(rag.get("mmr_top_n") or 5)
    lam = float(rag.get("mmr_lambda") or 0.7)

    N = max(1, N)
    lam = max(0.0, min(1.0, lam))

    # Meta embedding knobs
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

    # Candidate pool
    rag_candidates = _coerce_rag_candidates(step5_out, warnings)
    items = rag_candidates.get("items")
    if not isinstance(items, list):
        items = []

    if len(items) == 0:
        warnings.append({"code": "EMPTY_CANDIDATE_POOL", "message": "No rag_candidates.items to select from."})
        ms = int(round((time.perf_counter() - t0) * 1000))
        _mark_latency(meta_out, "6_MMR_selection", ms)
        return {
            "request_id": request_id,
            "claim_id": claim_id,
            "user_claim": user_claim,
            "normalized_claim": normalized_claim,
            "roberta": roberta,
            "routing": routing,
            "query_plan": query_plan,
            "rag_candidates": rag_candidates,
            "mmr_selected": {
                "top_n": N,
                "lambda": lam,
                "similarity": {"metric": "cosine", "embed_model": embed_model},
                "items": [],
            },
            "meta": meta_out,
        }

    # Build embeddings
    try:
        # Claim embedding (cache under claim:{claim_id})
        q_key = _cache_key(namespace, embed_model, f"claim:{claim_id}")
        q_vec = _cache_get(q_key) if cache_enabled else None
        if q_vec is None:
            q_vec = _embed_one(normalized_claim, embed_model)
            if cache_enabled:
                _cache_set(q_key, q_vec, ttl_s)

        cand_vecs: Dict[str, List[float]] = {}
        for it in items:
            if not isinstance(it, dict):
                continue
            cid = str(it.get("chunk_id") or "").strip()
            text = str(it.get("text") or "")
            if not cid:
                # deterministically derive one (won't match schema but prevents crash)
                cid = _stable_chunk_id(claim_id, text[:200])
                it["chunk_id"] = cid

            k = _cache_key(namespace, embed_model, cid)
            v = _cache_get(k) if cache_enabled else None
            if v is None:
                v = _embed_one(text, embed_model)
                if cache_enabled:
                    _cache_set(k, v, ttl_s)
            cand_vecs[cid] = v

    except Exception as e:
        stage_errors.append(
            {
                "stage": "6_MMR_selection",
                "code": "EMBEDDING_FAILED",
                "message": "Embedding computation failed; returning empty mmr_selected.",
                "details": {"error": str(e)},
            }
        )
        ms = int(round((time.perf_counter() - t0) * 1000))
        _mark_latency(meta_out, "6_MMR_selection", ms)
        return {
            "request_id": request_id,
            "claim_id": claim_id,
            "user_claim": user_claim,
            "normalized_claim": normalized_claim,
            "roberta": roberta,
            "routing": routing,
            "query_plan": query_plan,
            "rag_candidates": rag_candidates,
            "mmr_selected": {
                "top_n": N,
                "lambda": lam,
                "similarity": {"metric": "cosine", "embed_model": embed_model},
                "items": [],
            },
            "meta": meta_out,
        }

    # Run MMR
    selected = _mmr_select(
        [it for it in items if isinstance(it, dict)],
        q_vec=q_vec,
        cand_vecs=cand_vecs,
        top_n=N,
        mmr_lambda=lam,
    )

    # Emit selection trace (ranked + diagnostics)
    out_items: List[Dict[str, Any]] = []
    for idx, it in enumerate(selected, start=1):
        cid = str(it.get("chunk_id") or "")
        doc_id = ""
        if isinstance(it.get("doc"), dict):
            doc_id = str(it["doc"].get("doc_id") or "")
        if not doc_id:
            doc_id = str(it.get("doc_id") or "")

        out_items.append(
            {
                "rank": idx,
                "chunk_id": cid,
                "doc_id": doc_id,
                "mmr_score": float(it.get("_mmr_score") or 0.0),
                "relevance_sim": float(it.get("_relevance_sim") or 0.0),
                "max_redundancy_sim": float(it.get("_max_redundancy_sim") or 0.0),
            }
        )

    ms = int(round((time.perf_counter() - t0) * 1000))
    _mark_latency(meta_out, "6_MMR_selection", ms)

    return {
        "request_id": request_id,
        "claim_id": claim_id,
        "user_claim": user_claim,
        "normalized_claim": normalized_claim,
        "roberta": roberta,
        "routing": routing,
        "query_plan": query_plan,
        "rag_candidates": rag_candidates,
        "mmr_selected": {
            "top_n": N,
            "lambda": lam,
            "similarity": {"metric": "cosine", "embed_model": embed_model},
            "items": out_items,
        },
        "meta": meta_out,
    }
