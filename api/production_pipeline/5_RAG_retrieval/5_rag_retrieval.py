# 5_rag_retrieval.py
from __future__ import annotations

import hashlib
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv


# ----------------------------
# Errors
# ----------------------------
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
    # Load .env at call-time to match the project's pattern
    load_dotenv(override=False)
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v if v != "" else default


def _get_timeout_s() -> Optional[float]:
    raw = (_env("RAG_HTTP_TIMEOUT_S", "20") or "20").strip().lower()
    if raw in ("0", "none", "null", "inf", "infinite", "unlimited"):
        return None
    try:
        return float(raw)
    except Exception:
        return 20.0


def _doc_id_from_url(url: str) -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"doc_{h}"


def _provider() -> str:
    # Default to brave (since Google CSE JSON API is often unavailable)
    p = (_env("RAG_WEB_PROVIDER", "brave") or "brave").strip().lower()
    if p in ("brave", "brave_search"):
        return "brave"
    if p in ("google", "google_cse"):
        return "google_cse"
    raise RAGRetrievalError(
        "UNSUPPORTED_PROVIDER",
        f"Unsupported RAG_WEB_PROVIDER={p!r}. Use 'brave' or 'google_cse'.",
    )


def _coerce_int(v: Optional[str], default: int, lo: int, hi: int) -> int:
    try:
        x = int(str(v))
        return max(lo, min(hi, x))
    except Exception:
        return default


# ----------------------------
# Google CSE settings (optional)
# ----------------------------
DEFAULT_GOOGLE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
DEFAULT_GOOGLE_NUM = 10  # Google CSE max is 10


def _get_google_key_and_cx() -> Tuple[str, str]:
    key = _env("GOOGLE_CSE_API_KEY") or _env("GOOGLE_API_KEY")
    cx = _env("GOOGLE_CSE_ID") or _env("GOOGLE_CSE_CX")
    if not key:
        raise RAGRetrievalError(
            "MISSING_GOOGLE_API_KEY",
            "Missing GOOGLE_CSE_API_KEY (or GOOGLE_API_KEY) in environment.",
        )
    if not cx:
        raise RAGRetrievalError(
            "MISSING_GOOGLE_CSE_ID",
            "Missing GOOGLE_CSE_ID (cx) in environment.",
        )
    return key, cx


async def _google_cse_list(
    *,
    q: str,
    start: int,
    num: int,
    safe: Optional[str],
    lr: Optional[str],
    gl: Optional[str],
    date_restrict: Optional[str],
) -> Dict[str, Any]:
    key, cx = _get_google_key_and_cx()
    endpoint = _env("GOOGLE_CSE_ENDPOINT", DEFAULT_GOOGLE_ENDPOINT) or DEFAULT_GOOGLE_ENDPOINT

    params: Dict[str, Any] = {
        "key": key,
        "cx": cx,
        "q": q,
        "start": int(start),
        "num": int(num),
    }
    if safe:
        params["safe"] = safe
    if lr:
        params["lr"] = lr
    if gl:
        params["gl"] = gl
    if date_restrict:
        params["dateRestrict"] = date_restrict

    timeout = _get_timeout_s()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(endpoint, params=params)
    except httpx.TimeoutException as e:
        raise RAGRetrievalError("GOOGLE_TIMEOUT", "Google CSE request timed out.", {"error": str(e)})
    except httpx.RequestError as e:
        raise RAGRetrievalError("GOOGLE_CONNECTION", "Could not reach Google CSE endpoint.", {"error": str(e)})

    if r.status_code != 200:
        details = {"status_code": r.status_code, "body": r.text[:2000]}
        raise RAGRetrievalError("GOOGLE_BAD_STATUS", f"Google CSE returned HTTP {r.status_code}.", details)

    return r.json()


# ----------------------------
# Brave Search settings
# ----------------------------
DEFAULT_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
DEFAULT_BRAVE_COUNT = 10  # max 20
DEFAULT_BRAVE_OFFSET_MAX = 9  # 0..9 (10 pages)


def _get_brave_key() -> str:
    key = _env("BRAVE_API_KEY") or _env("BRAVE_SEARCH_API_KEY")
    if not key:
        raise RAGRetrievalError(
            "MISSING_BRAVE_API_KEY",
            "Missing BRAVE_API_KEY (or BRAVE_SEARCH_API_KEY) in environment.",
        )
    return key


def _domain_from_url(url: str) -> str:
    try:
        u = httpx.URL(url)
        return (u.host or "")
    except Exception:
        return ""


async def _brave_web_search(
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

    params: Dict[str, Any] = {
        "q": q,
        "count": int(count),
        "offset": int(offset),
    }
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
        # Brave recommends/accepts Cache-Control: no-cache to avoid cached content.
        # Also prevents edge cases where proxies inject other Cache-Control values.
        "Cache-Control": "no-cache",
    }

    timeout = _get_timeout_s()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(endpoint, params=params, headers=headers)
    except httpx.TimeoutException as e:
        raise RAGRetrievalError("BRAVE_TIMEOUT", "Brave Search request timed out.", {"error": str(e)})
    except httpx.RequestError as e:
        raise RAGRetrievalError("BRAVE_CONNECTION", "Could not reach Brave Search endpoint.", {"error": str(e)})

    if r.status_code != 200:
        details = {"status_code": r.status_code, "body": r.text[:2000]}
        raise RAGRetrievalError("BRAVE_BAD_STATUS", f"Brave Search returned HTTP {r.status_code}.", details)

    return r.json()


# ----------------------------
# Meta helpers (consistent with earlier steps)
# ----------------------------

def _copy_meta(meta_in: Dict[str, Any]) -> Dict[str, Any]:
    meta_out: Dict[str, Any] = {
        "received_at": meta_in.get("received_at"),
        "version": meta_in.get("version"),
    }
    for k in ("warnings", "latency_ms", "stage_latencies_ms", "total_latency_ms", "stage_errors"):
        if k in meta_in:
            meta_out[k] = meta_in[k]
    return meta_out


def _ensure_warnings(meta: Dict[str, Any]) -> List[Dict[str, str]]:
    w = meta.get("warnings")
    if isinstance(w, list):
        return w
    meta["warnings"] = []
    return meta["warnings"]


# ----------------------------
# Similarity + MMR (no extra deps)
# ----------------------------

_WORD_RE = re.compile(r"[\w]+", re.UNICODE)

_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "as",
    "at",
    "from",
    "that",
    "this",
    "it",
}


def _tokens(text: str) -> List[str]:
    toks = [t.lower() for t in _WORD_RE.findall(text or "")]
    out = []
    for t in toks:
        if len(t) <= 1:
            continue
        if t in _STOP:
            continue
        out.append(t)
    return out


def _tf(tokens: List[str]) -> Dict[str, float]:
    d: Dict[str, float] = {}
    for t in tokens:
        d[t] = d.get(t, 0.0) + 1.0
    return d


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    if len(a) > len(b):
        a, b = b, a
    for k, va in a.items():
        vb = b.get(k)
        if vb is not None:
            dot += va * vb
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def _mmr_select(
    query_tf: Dict[str, float],
    doc_tfs: Dict[str, Dict[str, float]],
    rel: Dict[str, float],
    *,
    top_n: int,
    mmr_lambda: float,
) -> List[str]:
    """Return doc_ids in selected order."""

    lam = max(0.0, min(1.0, float(mmr_lambda)))
    candidates = list(doc_tfs.keys())
    if not candidates:
        return []

    selected: List[str] = []

    first = max(candidates, key=lambda d: (rel.get(d, 0.0), d))
    selected.append(first)
    candidates.remove(first)

    def sim(d1: str, d2: str) -> float:
        return _cosine(doc_tfs.get(d1, {}), doc_tfs.get(d2, {}))

    while candidates and len(selected) < top_n:
        best_d = None
        best_score = -1e9
        for d in candidates:
            r = rel.get(d, 0.0)
            max_sim_to_selected = 0.0
            for s in selected:
                max_sim_to_selected = max(max_sim_to_selected, sim(d, s))
            score = lam * r - (1.0 - lam) * max_sim_to_selected
            if score > best_score or (
                abs(score - best_score) < 1e-12 and (best_d is None or d < best_d)
            ):
                best_score = score
                best_d = d
        if best_d is None:
            break
        selected.append(best_d)
        candidates.remove(best_d)

    return selected


# ----------------------------
# Public entrypoint
# ----------------------------


async def process_step4_output(step4_out: Dict[str, Any]) -> Dict[str, Any]:
    """Step4 -> Step5: web search retrieval + simple MMR selection."""

    try:
        request_id = step4_out["request_id"]
        claim_id = step4_out["claim_id"]
        user_claim = step4_out["user_claim"]
        normalized_claim = step4_out["normalized_claim"]
        roberta = step4_out["roberta"]
        routing = step4_out["routing"]
        query_plan = step4_out["query_plan"]
        meta_in = step4_out["meta"]

        rag = routing["rag"]
        M = int(rag["candidate_pool_size_m"])
        N = int(rag["mmr_top_n"])
        K = int(rag["final_top_k"])
        mmr_lambda = float(rag.get("mmr_lambda", 0.7))
        min_rel = float(rag.get("min_relevance", 0.25))
        allow_web = bool(rag.get("allow_web_search", True))

        queries = list(query_plan["queries"])
    except Exception as e:
        raise RAGRetrievalError("INVALID_INPUT", "Step5 expected Step4 output shape.", {"error": str(e)})

    t0 = time.perf_counter()
    meta_out = _copy_meta(meta_in)
    warnings = _ensure_warnings(meta_out)

    prov = _provider()

    if not allow_web:
        warnings.append({"code": "WEB_SEARCH_DISABLED", "message": "Routing disabled web search for this request."})
        meta_out["latency_ms"] = int(round((time.perf_counter() - t0) * 1000))
        return {
            "request_id": request_id,
            "claim_id": claim_id,
            "user_claim": user_claim,
            "normalized_claim": normalized_claim,
            "roberta": roberta,
            "routing": routing,
            "query_plan": query_plan,
            "retrieval": {
                "provider": prov,
                "skipped": True,
                "candidates": [],
                "mmr_selected": [],
                "final": [],
                "params": {
                    "candidate_pool_size_m": M,
                    "mmr_top_n": N,
                    "final_top_k": K,
                    "mmr_lambda": mmr_lambda,
                    "min_relevance": min_rel,
                },
            },
            "meta": meta_out,
        }

    # clamps
    M = max(1, M)
    N = max(1, min(N, M))
    K = max(1, min(K, N))
    min_rel = max(0.0, min(1.0, min_rel))
    mmr_lambda = max(0.0, min(1.0, mmr_lambda))

    items: List[Dict[str, Any]] = []
    seen_url: set[str] = set()

    primary_q = str(query_plan.get("primary_query") or normalized_claim)

    try:
        if prov == "brave":
            count = _coerce_int(_env("BRAVE_COUNT"), DEFAULT_BRAVE_COUNT, 1, 20)
            safesearch = _env("BRAVE_SAFESEARCH")  # off|moderate|strict
            country = _env("BRAVE_COUNTRY")  # e.g., CA
            search_lang = _env("BRAVE_SEARCH_LANG")  # e.g., en
            ui_lang = _env("BRAVE_UI_LANG")  # e.g., en-US
            freshness = _env("BRAVE_FRESHNESS")  # pd|pw|pm|py|YYYY-MM-DDtoYYYY-MM-DD
            extra_snippets = (_env("BRAVE_EXTRA_SNIPPETS") or "").strip().lower() in ("1", "true", "yes")

            async def fetch_one(q_text: str, variant: str, offset: int) -> bool:
                nonlocal items, seen_url
                data = await _brave_web_search(
                    q=q_text,
                    offset=offset,
                    count=count,
                    safesearch=safesearch,
                    country=country,
                    search_lang=search_lang,
                    ui_lang=ui_lang,
                    freshness=freshness,
                    extra_snippets=extra_snippets,
                )
                got = (data.get("web") or {}).get("results") or []
                for idx, it in enumerate(got):
                    link = str(it.get("url") or "").strip()
                    if not link or link in seen_url:
                        continue
                    seen_url.add(link)
                    items.append(
                        {
                            "doc_id": _doc_id_from_url(link),
                            "url": link,
                            "title": str(it.get("title") or ""),
                            "snippet": str(it.get("description") or ""),
                            "displayLink": _domain_from_url(link),
                            "source_query": q_text,
                            "source_variant": variant,
                            "serp_rank": int(offset) * int(count) + idx + 1,
                            "age": it.get("age"),
                        }
                    )
                    if len(items) >= M:
                        break

                # use more_results_available if present
                more = (data.get("query") or {}).get("more_results_available")
                return bool(more) if more is not None else (len(got) >= count)

            # 1) first page (offset=0) for each query
            for qobj in queries:
                if len(items) >= M:
                    break
                q_text = str(qobj.get("q") or "").strip()
                variant = str(qobj.get("variant") or "")
                if not q_text:
                    continue
                await fetch_one(q_text, variant, offset=0)

            # 2) if still short, paginate primary query only (offset 1..9)
            offset = 1
            while len(items) < M and offset <= DEFAULT_BRAVE_OFFSET_MAX:
                more = await fetch_one(primary_q, "original", offset=offset)
                if not more:
                    break
                offset += 1

        else:  # google_cse
            num = _coerce_int(_env("GOOGLE_CSE_NUM"), DEFAULT_GOOGLE_NUM, 1, 10)
            safe = _env("GOOGLE_CSE_SAFE")  # e.g., 'active'
            lr = _env("GOOGLE_CSE_LR")  # e.g., 'lang_en'
            gl = _env("GOOGLE_CSE_GL")  # e.g., 'ca'
            date_restrict = _env("GOOGLE_CSE_DATE_RESTRICT")  # e.g., 'd30'

            async def fetch_one(q_text: str, variant: str, start: int) -> None:
                nonlocal items, seen_url
                data = await _google_cse_list(
                    q=q_text,
                    start=start,
                    num=num,
                    safe=safe,
                    lr=lr,
                    gl=gl,
                    date_restrict=date_restrict,
                )
                got = data.get("items") or []
                for idx, it in enumerate(got):
                    link = str(it.get("link") or "").strip()
                    if not link or link in seen_url:
                        continue
                    seen_url.add(link)
                    items.append(
                        {
                            "doc_id": _doc_id_from_url(link),
                            "url": link,
                            "title": str(it.get("title") or ""),
                            "snippet": str(it.get("snippet") or ""),
                            "displayLink": str(it.get("displayLink") or ""),
                            "source_query": q_text,
                            "source_variant": variant,
                            "serp_rank": int(start) + idx,
                        }
                    )
                    if len(items) >= M:
                        break

            for qobj in queries:
                if len(items) >= M:
                    break
                q_text = str(qobj.get("q") or "").strip()
                variant = str(qobj.get("variant") or "")
                if not q_text:
                    continue
                await fetch_one(q_text, variant, start=1)

            start = 1 + num
            while len(items) < M and start <= 91:  # start+num must be <= 100
                await fetch_one(primary_q, "original", start=start)
                start += num

    except RAGRetrievalError:
        raise
    except Exception as e:
        raise RAGRetrievalError("RETRIEVAL_UNEXPECTED", "Unexpected error during retrieval.", {"error": str(e)})

    if not items:
        warnings.append({"code": "NO_SEARCH_RESULTS", "message": "Web search returned no results."})

    # Relevance scoring uses claim vs (title+snippet)
    q_tf = _tf(_tokens(normalized_claim))

    doc_tfs: Dict[str, Dict[str, float]] = {}
    rel: Dict[str, float] = {}

    for it in items:
        doc_id = it["doc_id"]
        text = f"{it.get('title','')} {it.get('snippet','')}"
        tf = _tf(_tokens(text))
        doc_tfs[doc_id] = tf
        rel[doc_id] = _cosine(q_tf, tf)

    filtered = [it for it in items if rel.get(it["doc_id"], 0.0) >= min_rel]
    if not filtered and items:
        filtered = items[:]
        warnings.append(
            {"code": "MIN_RELEVANCE_TOO_HIGH", "message": "No candidates met min_relevance; kept unfiltered pool."}
        )

    f_ids = {it["doc_id"] for it in filtered}
    doc_tfs_f = {k: v for k, v in doc_tfs.items() if k in f_ids}
    rel_f = {k: v for k, v in rel.items() if k in f_ids}

    selected_ids = _mmr_select(q_tf, doc_tfs_f, rel_f, top_n=min(N, len(doc_tfs_f)), mmr_lambda=mmr_lambda)
    final_ids = selected_ids[: min(K, len(selected_ids))]

    by_id = {it["doc_id"]: it for it in filtered}

    mmr_selected = []
    for d in selected_ids:
        it = by_id.get(d)
        if it:
            mmr_selected.append({**it, "relevance": rel_f.get(d, 0.0)})

    final = []
    for d in final_ids:
        it = by_id.get(d)
        if it:
            final.append({**it, "relevance": rel_f.get(d, 0.0)})

    step5_ms = int(round((time.perf_counter() - t0) * 1000))
    meta_out["latency_ms"] = max(0, step5_ms)

    provider_params: Dict[str, Any] = {
        "candidate_pool_size_m": M,
        "mmr_top_n": N,
        "final_top_k": K,
        "mmr_lambda": mmr_lambda,
        "min_relevance": min_rel,
    }

    if prov == "brave":
        provider_params["brave"] = {
            "endpoint": _env("BRAVE_ENDPOINT", DEFAULT_BRAVE_ENDPOINT) or DEFAULT_BRAVE_ENDPOINT,
            "count": _coerce_int(_env("BRAVE_COUNT"), DEFAULT_BRAVE_COUNT, 1, 20),
            "safesearch": _env("BRAVE_SAFESEARCH"),
            "country": _env("BRAVE_COUNTRY"),
            "search_lang": _env("BRAVE_SEARCH_LANG"),
            "ui_lang": _env("BRAVE_UI_LANG"),
            "freshness": _env("BRAVE_FRESHNESS"),
            "extra_snippets": (_env("BRAVE_EXTRA_SNIPPETS") or "").strip().lower() in ("1", "true", "yes"),
        }
    else:
        provider_params["google"] = {
            "endpoint": _env("GOOGLE_CSE_ENDPOINT", DEFAULT_GOOGLE_ENDPOINT) or DEFAULT_GOOGLE_ENDPOINT,
            "num": _coerce_int(_env("GOOGLE_CSE_NUM"), DEFAULT_GOOGLE_NUM, 1, 10),
            "safe": _env("GOOGLE_CSE_SAFE"),
            "lr": _env("GOOGLE_CSE_LR"),
            "gl": _env("GOOGLE_CSE_GL"),
            "dateRestrict": _env("GOOGLE_CSE_DATE_RESTRICT"),
        }

    return {
        "request_id": request_id,
        "claim_id": claim_id,
        "user_claim": user_claim,
        "normalized_claim": normalized_claim,
        "roberta": roberta,
        "routing": routing,
        "query_plan": query_plan,
        "retrieval": {
            "provider": prov,
            "skipped": False,
            "params": provider_params,
            "candidates": [{**it, "relevance": rel.get(it["doc_id"], 0.0)} for it in filtered],
            "mmr_selected": mmr_selected,
            "final": final,
            "stats": {
                "fetched": len(items),
                "deduped": len(items),
                "filtered": len(filtered),
                "mmr_selected": len(selected_ids),
                "final": len(final_ids),
            },
        },
        "meta": meta_out,
    }
