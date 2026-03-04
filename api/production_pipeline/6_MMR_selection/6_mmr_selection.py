# 6_mmr_selection.py
from __future__ import annotations

import hashlib
import math
import os
import secrets
import threading
import time
from typing import Any, Dict, List, Optional, Tuple


class MMRSelectionError(Exception):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def _copy_meta(meta_in: Dict[str, Any]) -> Dict[str, Any]:
    meta_out: Dict[str, Any] = {"received_at": meta_in.get("received_at"), "version": meta_in.get("version")}
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
        "embedding_nondeterministic",
    ):
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
# Char-ngram hashing embedding (same family as Step5)
# ----------------------------
DEFAULT_DIMS = int(os.getenv("MMR_EMB_DIMS", "1024") or "1024")
DEFAULT_NGRAM = int(os.getenv("MMR_EMB_NGRAM", "3") or "3")

DEFAULT_CACHE_ENABLED = (os.getenv("EMBED_CACHE_ENABLED", "true").strip().lower() in ("1", "true", "yes"))
DEFAULT_CACHE_NAMESPACE = os.getenv("EMBED_CACHE_NAMESPACE", "emb")
DEFAULT_CACHE_TTL_S = int(os.getenv("EMBED_CACHE_TTL_S", "0") or "0")
DEFAULT_NONDETERMINISTIC = (os.getenv("EMBED_NONDETERMINISTIC", "false").strip().lower() in ("1", "true", "yes"))

_PROCESS_SALT = secrets.token_hex(16)
EPS = 1e-9


def _compact(s: str) -> str:
    out = []
    for ch in (s or ""):
        if ch.isalnum():
            out.append(ch.lower())
    return "".join(out)


def _char_ngrams(s: str, n: int) -> List[str]:
    if not s:
        return []
    if len(s) <= n:
        return [s]
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def _hash_embed(text: str, dims: int, n: int, *, salt: str) -> List[float]:
    s = _compact(text)
    grams = _char_ngrams(s, n)
    vec = [0.0] * dims
    if not grams:
        return vec
    for g in grams:
        h = hashlib.blake2b((salt + "\n" + g).encode("utf-8"), digest_size=8).digest()
        x = int.from_bytes(h, "big", signed=False)
        idx = x % dims
        sign = -1.0 if (x & 1) else 1.0
        vec[idx] += sign
    norm = 0.0
    for v in vec:
        norm += v * v
    if norm > 0:
        inv = 1.0 / (norm ** 0.5)
        vec = [v * inv for v in vec]
    return vec


def _embed_one(text: str, *, dims: int, n: int, nondeterministic: bool) -> List[float]:
    salt = _PROCESS_SALT if nondeterministic else "det"
    return _hash_embed(text, dims=dims, n=n, salt=salt)


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    m = min(len(a), len(b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(m):
        va = float(a[i])
        vb = float(b[i])
        dot += va * vb
        na += va * va
        nb += vb * vb
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (math.sqrt(na) * math.sqrt(nb))))


def _cos01(x: float) -> float:
    return max(0.0, min(1.0, (float(x) + 1.0) * 0.5))


# ----------------------------
# Embedding cache
# ----------------------------
_EMB_CACHE: Dict[str, Tuple[List[float], Optional[float]]] = {}
_EMB_LOCK = threading.Lock()


def _cache_key(namespace: str, dims: int, n: int, chunk_id: str, *, nondeterministic: bool) -> str:
    salt_part = _PROCESS_SALT if nondeterministic else "det"
    return f"{namespace}:{salt_part}:charhash/{dims}/{n}:{chunk_id}"


def _cache_get(key: str) -> Optional[List[float]]:
    now = time.time()
    with _EMB_LOCK:
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
    with _EMB_LOCK:
        _EMB_CACHE[key] = (vec, exp)


# ----------------------------
# Coerce Step5 shape
# ----------------------------
def _coerce_rag_candidates(step5_out: Dict[str, Any], warnings: List[Dict[str, str]]) -> Dict[str, Any]:
    rc = step5_out.get("rag_candidates")
    if isinstance(rc, dict) and isinstance(rc.get("items"), list):
        return rc

    retrieval = step5_out.get("retrieval")
    if isinstance(retrieval, dict) and isinstance(retrieval.get("candidates"), list):
        warnings.append({"code": "LEGACY_STEP5_SHAPE", "message": "Adapted retrieval.candidates -> rag_candidates.items."})
        items: List[Dict[str, Any]] = []
        for it in retrieval["candidates"]:
            if not isinstance(it, dict):
                continue
            url = str(it.get("url") or "").strip()
            doc_id = str(it.get("doc_id") or "").strip() or (
                f"doc_{hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]}" if url else "doc_unknown"
            )
            title = str(it.get("title") or "")
            snippet = str(it.get("snippet") or "")
            text = (title + " " + snippet).strip()
            score = float(it.get("score") or 0.0)

            items.append(
                {
                    "chunk_id": f"chunk_{doc_id}",
                    "text": text,
                    "score": score,
                    "doc": {"doc_id": doc_id, "url": url, "title": title, "source": str(it.get("displayLink") or "")},
                }
            )
        return {"provider": retrieval.get("provider"), "params": retrieval.get("params"), "items": items, "stats": retrieval.get("stats")}

    raise MMRSelectionError("INVALID_INPUT", "Step6 expected Step5 output with rag_candidates.items (or legacy retrieval.candidates).")


# ----------------------------
# MMR selection
# ----------------------------
def _mmr_select(
    candidates: List[Dict[str, Any]],
    *,
    cand_vecs: Dict[str, List[float]],
    rel_scores: Dict[str, float],
    top_n: int,
    mmr_lambda: float,
) -> List[Dict[str, Any]]:
    lam = max(0.0, min(1.0, float(mmr_lambda)))
    N = max(1, int(top_n))

    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    pair_sim: Dict[Tuple[str, str], float] = {}

    def sim01(a: str, b: str) -> float:
        if a == b:
            return 1.0
        k = (a, b) if a < b else (b, a)
        v = pair_sim.get(k)
        if v is not None:
            return v
        v = _cos01(_cosine(cand_vecs[a], cand_vecs[b]))
        pair_sim[k] = v
        return v

    order_index = {str(it.get("chunk_id") or ""): i for i, it in enumerate(candidates)}

    for _ in range(N):
        best = None
        best_mmr = -1e18
        best_rel = 0.0

        for it in candidates:
            cid = str(it.get("chunk_id") or "")
            if not cid or cid in selected_ids or cid not in cand_vecs:
                continue
            r = float(rel_scores.get(cid, 0.0))
            red = 0.0 if not selected_ids else max(sim01(cid, sid) for sid in selected_ids)
            mmr = lam * r - (1.0 - lam) * red

            if best is None or mmr > best_mmr + EPS:
                best = it
                best_mmr = mmr
                best_rel = r
            elif abs(mmr - best_mmr) <= EPS:
                if r > best_rel + EPS:
                    best = it
                    best_mmr = mmr
                    best_rel = r
                elif abs(r - best_rel) <= EPS:
                    if order_index.get(cid, 10**9) < order_index.get(str(best.get("chunk_id") or ""), 10**9):
                        best = it
                        best_mmr = mmr
                        best_rel = r

        if best is None:
            break

        out = dict(best)
        out["_mmr_score"] = float(best_mmr)
        selected.append(out)
        selected_ids.add(str(best.get("chunk_id") or ""))

    return selected


async def process_step5_output(step5_out: Dict[str, Any]) -> Dict[str, Any]:
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
        query_plan = step5_out.get("query_plan") or {}
    except Exception as e:
        raise MMRSelectionError("INVALID_INPUT", "Step6 expected Step5 output shape.", {"error": str(e)})

    rag = ((routing or {}).get("rag") or {}) if isinstance(routing, dict) else {}
    N = max(1, int(rag.get("mmr_top_n") or 12))
    K = max(1, int(rag.get("final_top_k") or 6))
    K = min(K, N)
    lam = max(0.0, min(1.0, float(rag.get("mmr_lambda") or 0.70)))

    alpha = float(os.getenv("MMR_REL_ALPHA", "0.65") or "0.65")
    alpha = max(0.0, min(1.0, alpha))

    dims = int(os.getenv("MMR_EMB_DIMS", str(DEFAULT_DIMS)) or str(DEFAULT_DIMS))
    dims = max(256, min(4096, dims))
    ngram = int(os.getenv("MMR_EMB_NGRAM", str(DEFAULT_NGRAM)) or str(DEFAULT_NGRAM))
    ngram = max(2, min(6, ngram))

    cache_enabled = meta_out.get("embedding_cache_enabled")
    if cache_enabled is None:
        cache_enabled = DEFAULT_CACHE_ENABLED
    cache_enabled = bool(cache_enabled)

    namespace = str(meta_out.get("embedding_cache_namespace") or DEFAULT_CACHE_NAMESPACE)
    ttl_s = meta_out.get("embedding_cache_ttl_s")
    ttl_s = int(ttl_s) if isinstance(ttl_s, (int, float, str)) and str(ttl_s).strip() != "" else DEFAULT_CACHE_TTL_S
    ttl_s = max(0, int(ttl_s))

    nondet = meta_out.get("embedding_nondeterministic")
    if nondet is None:
        nondet = DEFAULT_NONDETERMINISTIC
    nondet = bool(nondet)

    meta_out["embed_model"] = f"charhash/{dims}/{ngram}"
    meta_out["embedding_cache_enabled"] = cache_enabled
    meta_out["embedding_cache_namespace"] = namespace
    meta_out["embedding_cache_ttl_s"] = ttl_s
    meta_out["embedding_nondeterministic"] = nondet

    rag_candidates = _coerce_rag_candidates(step5_out, warnings)
    items = rag_candidates.get("items")
    if not isinstance(items, list):
        items = []

    if not items:
        warnings.append({"code": "EMPTY_CANDIDATE_POOL", "message": "No rag_candidates.items to select from."})
        _mark_latency(meta_out, "step6_mmr_selection", int(round((time.perf_counter() - t0) * 1000)))
        return {
            "request_id": request_id,
            "claim_id": claim_id,
            "user_claim": user_claim,
            "normalized_claim": normalized_claim,
            "roberta": roberta,
            "routing": routing,
            "query_plan": query_plan,
            "rag_candidates": rag_candidates,
            "mmr_selected": {"top_n": N, "lambda": lam, "items": []},
            "final": {"top_k": K, "items": []},
            "meta": meta_out,
        }

    # Prefer Step4 primary_query for the query embedding (matches retrieval behavior)
    q_text = str(query_plan.get("primary_query") or "").strip() or normalized_claim

    try:
        q_key = _cache_key(namespace, dims, ngram, f"query:{claim_id}", nondeterministic=nondet)
        q_vec = _cache_get(q_key) if cache_enabled else None
        if q_vec is None:
            q_vec = _embed_one(q_text, dims=dims, n=ngram, nondeterministic=nondet)
            if cache_enabled:
                _cache_set(q_key, q_vec, ttl_s)

        cand_vecs: Dict[str, List[float]] = {}
        for it in items:
            if not isinstance(it, dict):
                continue
            cid = str(it.get("chunk_id") or "").strip()
            text = str(it.get("text") or "").strip()
            if not cid:
                cid = f"chunk_{secrets.token_hex(12)}"
                it["chunk_id"] = cid

            k = _cache_key(namespace, dims, ngram, cid, nondeterministic=nondet)
            v = _cache_get(k) if cache_enabled else None
            if v is None:
                v = _embed_one(text, dims=dims, n=ngram, nondeterministic=nondet)
                if cache_enabled:
                    _cache_set(k, v, ttl_s)
            cand_vecs[cid] = v

    except Exception as e:
        stage_errors.append(
            {"stage": "step6_mmr_selection", "code": "EMBEDDING_FAILED", "message": "Embedding failed.", "details": {"error": str(e)}}
        )
        _mark_latency(meta_out, "step6_mmr_selection", int(round((time.perf_counter() - t0) * 1000)))
        return {
            "request_id": request_id,
            "claim_id": claim_id,
            "user_claim": user_claim,
            "normalized_claim": normalized_claim,
            "roberta": roberta,
            "routing": routing,
            "query_plan": query_plan,
            "rag_candidates": rag_candidates,
            "mmr_selected": {"top_n": N, "lambda": lam, "items": []},
            "final": {"top_k": K, "items": []},
            "meta": meta_out,
        }

    rel_scores: Dict[str, float] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        cid = str(it.get("chunk_id") or "")
        if cid not in cand_vecs:
            continue
        s5 = float(it.get("score") or 0.0)
        s5 = max(0.0, min(1.0, s5))
        sem = _cos01(_cosine(q_vec, cand_vecs[cid]))
        rel_scores[cid] = max(0.0, min(1.0, alpha * s5 + (1.0 - alpha) * sem))

    selected = _mmr_select(
        [it for it in items if isinstance(it, dict)],
        cand_vecs=cand_vecs,
        rel_scores=rel_scores,
        top_n=min(N, len(cand_vecs)),
        mmr_lambda=lam,
    )

    mmr_items: List[Dict[str, Any]] = []
    for idx, it in enumerate(selected, start=1):
        out = dict(it)
        out["rank"] = idx
        out["mmr_score"] = float(out.pop("_mmr_score", 0.0))
        out["mmr_relevance"] = float(rel_scores.get(str(out.get("chunk_id") or ""), 0.0))
        out["retrieval_score"] = float(out.get("score") or 0.0)
        mmr_items.append(out)

    final_items = mmr_items[:K]
    _mark_latency(meta_out, "step6_mmr_selection", int(round((time.perf_counter() - t0) * 1000)))

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
            "relevance_blend_alpha": alpha,
            "similarity": {"metric": "cosine", "embed_model": meta_out["embed_model"]},
            "items": mmr_items,
        },
        "final": {"top_k": K, "items": final_items},
        "meta": meta_out,
    }