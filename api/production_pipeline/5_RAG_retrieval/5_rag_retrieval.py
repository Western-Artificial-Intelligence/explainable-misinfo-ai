# 5_rag_retrieval.py
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
from dotenv import load_dotenv

load_dotenv(override=False)


class RAGRetrievalError(Exception):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


# ----------------------------
# Env helpers
# ----------------------------
def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v if v != "" else default


def _as_int(v: Optional[str], default: int, lo: int, hi: int) -> int:
    try:
        x = int(str(v))
        return max(lo, min(hi, x))
    except Exception:
        return default


def _as_float(v: Optional[str], default: float, lo: float, hi: float) -> float:
    try:
        x = float(str(v))
        if x != x:  # NaN
            return default
        return max(lo, min(hi, x))
    except Exception:
        return default


def _timeout_s() -> Optional[float]:
    raw = (_env("RAG_HTTP_TIMEOUT_S", "20") or "20").strip().lower()
    if raw in ("0", "none", "null", "inf", "infinite", "unlimited"):
        return None
    try:
        return float(raw)
    except Exception:
        return 20.0


# ----------------------------
# Brave Search ONLY
# ----------------------------
DEFAULT_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


def _assert_brave_only() -> None:
    """
    Hard-disable Google. If someone sets RAG_WEB_PROVIDER to google_*,
    fail fast so behavior is obvious.
    """
    p = (_env("RAG_WEB_PROVIDER", "brave") or "brave").strip().lower()
    if p not in ("brave", "brave_search"):
        raise RAGRetrievalError(
            "UNSUPPORTED_PROVIDER",
            f"Only Brave is supported. Got RAG_WEB_PROVIDER={p!r}.",
        )


def _get_brave_key() -> str:
    key = _env("BRAVE_API_KEY") or _env("BRAVE_SEARCH_API_KEY")
    if not key:
        raise RAGRetrievalError(
            "MISSING_BRAVE_API_KEY",
            "Missing BRAVE_API_KEY (or BRAVE_SEARCH_API_KEY) in environment.",
        )
    return key


def _doc_id_from_url(url: str) -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"doc_{h}"


def _domain_from_url(url: str) -> str:
    try:
        u = httpx.URL(url)
        return u.host or ""
    except Exception:
        return ""


_TRACKING_KEYS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "mc_cid", "mc_eid", "igshid",
}


def _canonicalize_url(url: str) -> str:
    try:
        u = httpx.URL(url)
        u = u.copy_with(fragment="")
        if u.params:
            kept = [(k, v) for (k, v) in u.params.multi_items() if k.lower() not in _TRACKING_KEYS]
            u = u.copy_with(params=kept)
        return str(u)
    except Exception:
        return (url or "").strip()


async def _brave_web_search(
    client: httpx.AsyncClient,
    *,
    q: str,
    offset: int,
    count: int,
    safesearch: Optional[str],
    country: Optional[str],
    search_lang: Optional[str],
    ui_lang: Optional[str],
    freshness: Optional[str],
    extra_snippets: Optional[bool],
) -> Dict[str, Any]:
    key = _get_brave_key()
    endpoint = _env("BRAVE_ENDPOINT", DEFAULT_BRAVE_ENDPOINT) or DEFAULT_BRAVE_ENDPOINT

    params: Dict[str, Any] = {"q": q, "count": int(count), "offset": int(offset)}
    if safesearch:
        params["safesearch"] = safesearch
    if country:
        params["country"] = country
    if search_lang:
        params["search_lang"] = search_lang
    if ui_lang:
        params["ui_lang"] = ui_lang
    if freshness:
        params["freshness"] = freshness
    if extra_snippets is True:
        params["extra_snippets"] = "true"

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": key,
        "Cache-Control": "no-cache",
    }

    retry_429_max = _as_int(_env("BRAVE_RETRY_429_MAX"), 2, 0, 10)
    retry_429_base_delay_s = _as_float(_env("BRAVE_RETRY_429_BASE_DELAY_S"), 1.0, 0.0, 60.0)
    attempt = 0

    while True:
        try:
            r = await client.get(endpoint, params=params, headers=headers)
        except httpx.TimeoutException as e:
            raise RAGRetrievalError("BRAVE_TIMEOUT", "Brave Search request timed out.", {"error": str(e)})
        except httpx.RequestError as e:
            raise RAGRetrievalError("BRAVE_CONNECTION", "Could not reach Brave Search endpoint.", {"error": str(e)})

        if r.status_code == 200:
            return r.json()

        if r.status_code == 429 and attempt < retry_429_max and retry_429_base_delay_s > 0:
            retry_after = (r.headers.get("Retry-After") or "").strip()
            delay_s = retry_429_base_delay_s * (2.0 ** float(attempt))
            try:
                if retry_after:
                    delay_s = max(delay_s, float(retry_after))
            except Exception:
                pass
            delay_s = max(0.0, min(60.0, float(delay_s)))
            attempt += 1
            await asyncio.sleep(delay_s)
            continue

        raise RAGRetrievalError(
            "BRAVE_BAD_STATUS",
            f"Brave Search returned HTTP {r.status_code}.",
            {"status_code": r.status_code, "body": r.text[:2000]},
        )


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


def _ensure_stage_latencies(meta: Dict[str, Any]) -> Dict[str, int]:
    v = meta.get("stage_latencies_ms")
    if isinstance(v, dict):
        return v
    meta["stage_latencies_ms"] = {}
    return meta["stage_latencies_ms"]


def _ensure_stage_errors(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    v = meta.get("stage_errors")
    if isinstance(v, list):
        return v
    meta["stage_errors"] = []
    return meta["stage_errors"]


def _fail_open_brave_errors() -> bool:
    return ((_env("RAG_FAIL_OPEN", "true") or "true").strip().lower() in ("1", "true", "yes"))


def _brave_error_reason_from_details(details: Dict[str, Any]) -> Optional[str]:
    """
    Parse Brave API error body from details (e.g. from RAGRetrievalError.details)
    and return a short user-facing reason, or None if unparseable.
    """
    body = details.get("body")
    if not isinstance(body, str) or not body.strip():
        return None
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return None
    err = (data or {}).get("error")
    if not isinstance(err, dict):
        return None
    detail = (err.get("detail") or "").strip()
    code = (err.get("code") or "").strip().upper()
    meta = err.get("meta") if isinstance(err.get("meta"), dict) else {}
    limit_type = (meta.get("usage_limit_type") or "").strip().lower()
    if code == "USAGE_LIMIT_EXCEEDED" and detail:
        if limit_type == "monthly":
            return "usage limit exceeded (monthly cap reached)."
        if limit_type:
            return f"usage limit exceeded ({limit_type})."
        return detail.rstrip(".") + "." if detail else "usage limit exceeded."
    if detail:
        return detail.rstrip(".") + "." if not detail.endswith(".") else detail
    if code:
        return code.replace("_", " ").lower() + "."
    return None


# ----------------------------
# Char-ngram semantic hashing (deterministic, no LLM)
# ----------------------------
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


def _hash_embed_char_ngrams(text: str, *, dims: int = 1024, n: int = 3, salt: str = "det") -> List[float]:
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
    if norm > 0.0:
        inv = 1.0 / (norm ** 0.5)
        vec = [v * inv for v in vec]
    return vec


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
    return max(-1.0, min(1.0, dot / ((na ** 0.5) * (nb ** 0.5))))


def _cos01(x: float) -> float:
    return max(0.0, min(1.0, (float(x) + 1.0) * 0.5))


def _serp_prior(rank: int) -> float:
    r = max(1, int(rank))
    return 1.0 / (1.0 + 0.10 * float(r - 1))


_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4, "april": 4, "may": 5,
    "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}


def _age_to_days(age: Any) -> Optional[float]:
    if age is None:
        return None
    s = str(age).strip().lower()
    if not s:
        return None

    # "2 weeks ago"
    parts = s.split()
    if len(parts) >= 3 and parts[1].startswith(("day", "week", "month", "year")) and parts[2] == "ago":
        try:
            x = float(parts[0])
        except Exception:
            return None
        unit = parts[1]
        if unit.startswith("day"):
            return x
        if unit.startswith("week"):
            return x * 7.0
        if unit.startswith("month"):
            return x * 30.0
        if unit.startswith("year"):
            return x * 365.0

    # "April 27, 2020"
    try:
        m, dyy = s.split(" ", 1)
        mon = _MONTHS.get(m)
        if mon and "," in dyy:
            day_str, year_str = dyy.split(",", 1)
            day = int(day_str.strip())
            yr = int(year_str.strip())
            dt = datetime(yr, mon, day, tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
    except Exception:
        pass

    return None


def _freshness_score(age: Any) -> float:
    d = _age_to_days(age)
    if d is None:
        return 0.35
    if d <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 / (1.0 + (d / 180.0))))


@dataclass(frozen=True)
class _Score:
    score: float
    sim: float
    serp: float
    fresh: float


async def process_step4_output(step4_out: Dict[str, Any]) -> Dict[str, Any]:
    try:
        request_id = step4_out["request_id"]
        claim_id = step4_out["claim_id"]
        user_claim = step4_out["user_claim"]
        normalized_claim = step4_out["normalized_claim"]
        roberta = step4_out.get("roberta")
        routing = step4_out.get("routing") or {}
        query_plan = step4_out.get("query_plan") or {}
        meta_in = step4_out.get("meta") or {}

        rag = (routing.get("rag") or {}) if isinstance(routing, dict) else {}
        M = int(rag.get("candidate_pool_size_m") or 100)
        N = int(rag.get("mmr_top_n") or 12)
        K = int(rag.get("final_top_k") or 6)
        min_rel = float(rag.get("min_relevance") or 0.25)
        allow_web = (_env("RAG_ALLOW_WEB_SEARCH", "1") or "1").strip().lower() in ("1", "true", "yes")

        queries = list(query_plan.get("queries") or [])
        primary_q = str(query_plan.get("primary_query") or normalized_claim).strip() or normalized_claim
    except Exception as e:
        raise RAGRetrievalError("INVALID_INPUT", "Step5 expected Step4 output shape.", {"error": str(e)})

    t0 = time.perf_counter()
    meta_out = _copy_meta(meta_in)
    warnings = _ensure_warnings(meta_out)
    stage_lat = _ensure_stage_latencies(meta_out)

    # Clamp knobs
    M = max(1, min(200, int(M)))
    N = max(1, min(int(N), M))
    K = max(1, min(int(K), N))
    min_rel = max(0.0, min(1.0, float(min_rel)))

    if not allow_web:
        warnings.append({"code": "WEB_SEARCH_DISABLED", "message": "Web search disabled (RAG_ALLOW_WEB_SEARCH=0); using LLM-only RAG."})
        stage_lat["step5_rag_retrieval"] = int(round((time.perf_counter() - t0) * 1000))
        return {
            "request_id": request_id,
            "claim_id": claim_id,
            "user_claim": user_claim,
            "normalized_claim": normalized_claim,
            "roberta": roberta,
            "routing": routing,
            "query_plan": query_plan,
            "rag_candidates": {"provider": "brave", "params": {}, "items": [], "stats": {"fetched": 0, "kept": 0}},
            "meta": meta_out,
        }

    # Embedding settings
    emb_dims = _as_int(_env("RAG_EMB_DIMS"), 1024, 256, 4096)
    emb_ngram = _as_int(_env("RAG_EMB_NGRAM"), 3, 2, 6)

    # Query vector: average across planned queries (more robust than single string)
    q_texts: List[str] = []
    for qobj in queries:
        q = str((qobj or {}).get("q") or "").strip()
        if q:
            q_texts.append(q)
    if not q_texts:
        q_texts = [primary_q]

    q_vecs = [_hash_embed_char_ngrams(q, dims=emb_dims, n=emb_ngram, salt="det") for q in q_texts]
    q_vec = [0.0] * emb_dims
    for v in q_vecs:
        for i in range(min(len(q_vec), len(v))):
            q_vec[i] += float(v[i])
    # normalize avg
    norm = sum(x * x for x in q_vec) ** 0.5
    if norm > 0:
        q_vec = [x / norm for x in q_vec]

    # Brave paging knobs
    brave_count = _as_int(_env("BRAVE_COUNT"), 10, 1, 20)
    brave_max_pages = _as_int(_env("BRAVE_MAX_PAGES"), 6, 1, 10)

    timeout = _timeout_s()
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
    items_raw: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    duplicates_dropped = 0
    brave_error: Optional[RAGRetrievalError] = None

    async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True) as client:
        async def fetch_page(q_text: str, variant: str, offset: int) -> bool:
            nonlocal duplicates_dropped
            nonlocal brave_error
            try:
                data = await _brave_web_search(
                    client,
                    q=q_text,
                    offset=offset,
                    count=brave_count,
                    safesearch=_env("BRAVE_SAFESEARCH"),
                    country=_env("BRAVE_COUNTRY"),
                    search_lang=_env("BRAVE_SEARCH_LANG"),
                    ui_lang=_env("BRAVE_UI_LANG"),
                    freshness=_env("BRAVE_FRESHNESS"),
                    extra_snippets=((_env("BRAVE_EXTRA_SNIPPETS") or "").strip().lower() in ("1", "true", "yes")),
                )
            except RAGRetrievalError as e:
                if _fail_open_brave_errors():
                    brave_error = e
                    return False
                raise
            got = (data.get("web") or {}).get("results") or []
            for idx, it in enumerate(got):
                url = str(it.get("url") or "").strip()
                if not url:
                    continue
                canon = _canonicalize_url(url)
                if canon in seen:
                    duplicates_dropped += 1
                    continue
                seen.add(canon)
                rank = int(offset) * int(brave_count) + idx + 1
                items_raw.append(
                    {
                        "doc_id": _doc_id_from_url(canon),
                        "url": canon,
                        "title": str(it.get("title") or ""),
                        "snippet": str(it.get("description") or ""),
                        "domain": _domain_from_url(canon),
                        "serp_rank": rank,
                        "age": it.get("age"),
                        "source_query": q_text,
                        "source_variant": variant,
                    }
                )
                if len(items_raw) >= M:
                    break

            more = (data.get("query") or {}).get("more_results_available")
            if more is not None:
                return bool(more)
            return len(got) >= brave_count

        # 1) One page for each planned query
        for qobj in queries:
            if len(items_raw) >= M:
                break
            if brave_error is not None:
                break
            q_text = str((qobj or {}).get("q") or "").strip()
            if not q_text:
                continue
            await fetch_page(q_text, str((qobj or {}).get("variant") or ""), offset=0)

        # 2) Additional pages for primary query (fill up to M)
        offset = 1
        while len(items_raw) < M and offset < brave_max_pages and brave_error is None:
            more = await fetch_page(primary_q, "original", offset=offset)
            if not more:
                break
            offset += 1

    if brave_error is not None:
        stage_errors = _ensure_stage_errors(meta_out)
        stage_errors.append(
            {
                "stage": "step5_rag_retrieval",
                "code": getattr(brave_error, "code", "BRAVE_ERROR"),
                "message": getattr(brave_error, "message", str(brave_error)),
                "details": getattr(brave_error, "details", None),
            }
        )

        details = getattr(brave_error, "details", None) or {}
        status = details.get("status_code")
        brave_reason = _brave_error_reason_from_details(details)
        if brave_reason:
            msg = f"Web retrieval skipped: Brave Search — {brave_reason}"
        else:
            hint = ""
            if status == 402:
                hint = " (likely out of credits / plan issue)"
            elif status == 429:
                hint = " (rate limited)"
            msg = f"Web retrieval skipped because Brave Search failed{hint}."

        warnings.append(
            {
                "code": f"BRAVE_HTTP_{status}" if status is not None else getattr(brave_error, "code", "BRAVE_ERROR"),
                "message": msg,
            }
        )

        stage_lat["step5_rag_retrieval"] = int(round((time.perf_counter() - t0) * 1000))
        params: Dict[str, Any] = {
            "candidate_pool_size_m": M,
            "mmr_top_n": N,
            "final_top_k": K,
            "min_relevance": min_rel,
            "provider": "brave",
            "embedding": {"type": "char_ngram_hash", "dims": emb_dims, "ngram": emb_ngram},
            "brave": {"count": brave_count, "max_pages": brave_max_pages},
        }
        stats = {"fetched": 0, "kept": 0, "duplicates_dropped": 0, "filtered": 0}

        return {
            "request_id": request_id,
            "claim_id": claim_id,
            "user_claim": user_claim,
            "normalized_claim": normalized_claim,
            "roberta": roberta,
            "routing": routing,
            "query_plan": query_plan,
            "rag_candidates": {"provider": "brave", "params": params, "items": [], "stats": stats},
            # legacy passthrough (kept for compatibility if anything still reads it)
            "retrieval": {
                "provider": "brave",
                "skipped": True,
                "params": params,
                "candidates": [],
                "mmr_selected": [],
                "final": [],
                "stats": stats,
            },
            "meta": meta_out,
        }

    if not items_raw:
        warnings.append({"code": "NO_SEARCH_RESULTS", "message": "Brave search returned no results."})

    # Scoring weights
    w_sim = _as_float(_env("RAG_W_SIM"), 0.80, 0.0, 1.0)
    w_serp = _as_float(_env("RAG_W_SERP"), 0.12, 0.0, 1.0)
    w_fsh = _as_float(_env("RAG_W_FRESH"), 0.08, 0.0, 1.0)
    sw = max(1e-9, w_sim + w_serp + w_fsh)
    w_sim, w_serp, w_fsh = (w_sim / sw), (w_serp / sw), (w_fsh / sw)

    # Score each candidate; use max(title_sim, snippet_sim) for robustness
    scores: Dict[str, _Score] = {}
    for it in items_raw:
        did = str(it.get("doc_id") or "")
        title = str(it.get("title") or "")
        snippet = str(it.get("snippet") or "")

        t_vec = _hash_embed_char_ngrams(title, dims=emb_dims, n=emb_ngram, salt="det") if title else [0.0] * emb_dims
        s_vec = _hash_embed_char_ngrams(snippet, dims=emb_dims, n=emb_ngram, salt="det") if snippet else [0.0] * emb_dims
        sim = max(_cos01(_cosine(q_vec, t_vec)), _cos01(_cosine(q_vec, s_vec)))

        serp = _serp_prior(int(it.get("serp_rank") or 10**9))
        fresh = _freshness_score(it.get("age"))
        score = max(0.0, min(1.0, (w_sim * sim) + (w_serp * serp) + (w_fsh * fresh)))
        scores[did] = _Score(score=score, sim=sim, serp=serp, fresh=fresh)

    # Threshold + adaptive keep (for MMR efficiency)
    filtered = [it for it in items_raw if scores[str(it.get("doc_id") or "")].score >= min_rel]
    target = min(len(items_raw), max(N * 3, K * 5, N))

    if len(filtered) < min(N, len(items_raw)) and items_raw:
        ranked = sorted(items_raw, key=lambda it: scores[str(it.get("doc_id") or "")].score, reverse=True)
        filtered = ranked[:target]
        warnings.append({"code": "MIN_SCORE_ADAPTIVE", "message": f"Adaptive cutoff used; kept top {len(filtered)} by score."})
    elif len(filtered) > target:
        filtered = sorted(filtered, key=lambda it: scores[str(it.get("doc_id") or "")].score, reverse=True)[:target]
        warnings.append({"code": "POOL_TRIMMED", "message": f"Trimmed pool to {len(filtered)} for MMR efficiency."})

    filtered.sort(key=lambda it: (-scores[str(it.get("doc_id") or "")].score, int(it.get("serp_rank") or 10**9)))

    rag_items: List[Dict[str, Any]] = []
    legacy_candidates: List[Dict[str, Any]] = []
    for it in filtered:
        did = str(it.get("doc_id") or "")
        s = scores[did]
        title = str(it.get("title") or "")
        snippet = str(it.get("snippet") or "")
        url = str(it.get("url") or "")
        text = (title + " " + snippet).strip()

        chunk_id = f"chunk_{did}"
        breakdown = {"sim": float(s.sim), "serp": float(s.serp), "fresh": float(s.fresh)}

        rag_items.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "score": float(s.score),
                "score_breakdown": breakdown,
                "doc": {
                    "doc_id": did,
                    "url": url,
                    "title": title,
                    "source": str(it.get("domain") or ""),
                    "serp_rank": int(it.get("serp_rank") or 0),
                    "age": it.get("age"),
                    "source_query": str(it.get("source_query") or ""),
                    "source_variant": str(it.get("source_variant") or ""),
                },
            }
        )

        legacy_candidates.append(
            {
                "doc_id": did,
                "url": url,
                "title": title,
                "snippet": snippet,
                "displayLink": str(it.get("domain") or ""),
                "source_query": str(it.get("source_query") or ""),
                "source_variant": str(it.get("source_variant") or ""),
                "serp_rank": int(it.get("serp_rank") or 0),
                "age": it.get("age"),
                "score": float(s.score),
                "score_breakdown": breakdown,
            }
        )

    stage_lat["step5_rag_retrieval"] = int(round((time.perf_counter() - t0) * 1000))

    params: Dict[str, Any] = {
        "candidate_pool_size_m": M,
        "mmr_top_n": N,
        "final_top_k": K,
        "min_relevance": min_rel,
        "provider": "brave",
        "scoring": {"weights": {"sim": w_sim, "serp": w_serp, "fresh": w_fsh}},
        "embedding": {"type": "char_ngram_hash", "dims": emb_dims, "ngram": emb_ngram},
        "brave": {"count": brave_count, "max_pages": brave_max_pages},
    }

    stats = {
        "fetched": len(items_raw) + duplicates_dropped,
        "kept": len(items_raw),
        "duplicates_dropped": duplicates_dropped,
        "filtered": len(filtered),
    }

    return {
        "request_id": request_id,
        "claim_id": claim_id,
        "user_claim": user_claim,
        "normalized_claim": normalized_claim,
        "roberta": roberta,
        "routing": routing,
        "query_plan": query_plan,
        "rag_candidates": {"provider": "brave", "params": params, "items": rag_items, "stats": stats},
        # legacy passthrough (kept for compatibility if anything still reads it)
        "retrieval": {
            "provider": "brave",
            "skipped": False,
            "params": params,
            "candidates": legacy_candidates,
            "mmr_selected": [],
            "final": [],
            "stats": stats,
        },
        "meta": meta_out,
    }