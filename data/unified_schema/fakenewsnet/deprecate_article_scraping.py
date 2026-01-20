# article_scraping.py
"""
Article hydration module for fetching and parsing news articles.

This module provides functions to:
1. Fetch article content from URLs
2. Parse HTML to extract article text
3. Handle dead URLs via Wayback Machine
4. Respect robots.txt
5. Parallel processing with rate limiting
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse, quote
from datetime import datetime, timezone
import hashlib
import re
import time
import logging
from collections import defaultdict
import threading
import numpy as np

_ST_MODEL = None

# Parsing libs - try newspaper first, fallback to readability + bs4
try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except Exception:
    NEWSPAPER_AVAILABLE = False

# readability (optional)
try:
    from readability import Document
    READABILITY_AVAILABLE = True
except Exception:
    READABILITY_AVAILABLE = False

# bs4 (optional, but used even without readability)
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except Exception:
    BS4_AVAILABLE = False


# language detection (optional and light)
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except Exception:
    LANGDETECT_AVAILABLE = False

logger = logging.getLogger("hydrator")
logger.setLevel(logging.INFO)

def _get_st_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _ST_MODEL

def _cosine_sim_text(a: str, b: str) -> float:
    """
    Semantic cosine similarity between two texts using sentence-transformers.
    Returns in [0,1] typically when normalize_embeddings=True.
    """
    if not a or not b:
        return 0.0
    a = a.strip()
    b = b.strip()
    if not a or not b:
        return 0.0

    # Avoid wasting compute on huge bodies
    b_short = b[:2000]
    model = _get_st_model()
    emb = model.encode([a, b_short], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))

def _requests_session_with_retries(total_retries=3, backoff_factor=0.3,
                                  allowed_methods=frozenset(['GET', 'HEAD'])):
    s = requests.Session()
    retries = Retry(total=total_retries, backoff_factor=backoff_factor,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=allowed_methods)
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({
        "User-Agent": "hydrator-bot/1.0 (+https://example.com/)",
        "Accept-Language": "en-US,en;q=0.9"
    })
    return s


def _get_domain(url):
    try:
        parsed = urlparse(url)
        if parsed.netloc:
            return parsed.netloc.lower()
        # Handle URLs missing scheme (e.g., "example.com/path")
        parsed2 = urlparse("http://" + str(url))
        return (parsed2.netloc or "").lower()
    except Exception:
        return ""


def _normalize_url(url: str) -> str:
    """Normalize URLs to include a scheme so requests can fetch them.

    FakeNewsNet often includes values like 'example.com/path' or 'www.site.com/...'.
    This function prepends 'http://' when the scheme is missing and handles
    protocol-relative URLs ('//example.com').
    """
    if url is None:
        return url
    u = str(url).strip()
    if not u:
        return u
    if u.startswith('//'):
        return 'http:' + u
    from urllib.parse import urlparse
    parsed = urlparse(u)
    if parsed.scheme:
        return u
    return 'http://' + u




def _claim_norm_hash(title):
    if title is None:
        return None
    s = re.sub(r'\s+', ' ', str(title).strip().lower())
    return hashlib.sha1(s.encode('utf-8')).hexdigest()


_robots_cache = {}
_robots_cache_lock = threading.Lock()


def _respect_robots(url, session, timeout=5.0):
    """
    Best-effort robots.txt check. If robots.txt cannot be fetched or parsed,
    this function returns True (permissive) so that the hydrator can still try.
    """
    from urllib.robotparser import RobotFileParser
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return True

        domain = parsed.netloc.lower()
        with _robots_cache_lock:
            cached = _robots_cache.get(domain)
        if cached is not None:
            return cached

        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        r = session.get(robots_url, timeout=timeout)
        if r.status_code == 200 and r.text:
            rp = RobotFileParser()
            rp.parse(r.text.splitlines())

            allowed = rp.can_fetch(session.headers.get("User-Agent"), url)
            with _robots_cache_lock:
                _robots_cache[domain] = allowed
            return allowed

        with _robots_cache_lock:
            _robots_cache[domain] = True
        return True
    except Exception:
        # Be permissive on parser/network failure (so we don't block hydration)
        return True


def _parse_with_newspaper(url, html_text=None):
    if not NEWSPAPER_AVAILABLE:
        return None
    try:
        art = Article(url)
        if html_text:
            art.download(input_html=html_text)
        else:
            art.download()
        art.parse()
        txt = art.text.strip()
        return txt if txt else None
    except Exception as e:
        logger.debug("newspaper parse failed: %s", e)
        return None

def _parse_with_readability(html_text):
    if not READABILITY_AVAILABLE or not BS4_AVAILABLE or not html_text:
        return None
    try:
        doc = Document(html_text)
        content_html = doc.summary()
        soup = BeautifulSoup(content_html, "html.parser")
        txt = soup.get_text(separator="\n").strip()
        return txt if txt else None
    except Exception as e:
        logger.debug("readability parse failed: %s", e)
        return None

def _wayback_latest_snapshot_url(url, session, timeout=10):
    """
    Query Wayback CDX API for the most recent snapshot and return an archive URL
    or None if not found.
    """
    try:
        cdx = ("https://web.archive.org/cdx/search/cdx?url={url}"
            "&output=json&fl=timestamp,original"
            "&filter=statuscode:200"
            "&collapse=timestamp:8"
            "&sort=descending"
            "&limit=1")

        q = cdx.format(url=quote(url, safe=''))
        r = session.get(q, timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.json()
        if isinstance(data, list) and len(data) >= 2:
            row = data[1]
            timestamp = row[0]
            orig = row[1]
            archive_url = f"https://web.archive.org/web/{timestamp}/{orig}"
            return archive_url
        return None
    except Exception as e:
        logger.debug("wayback lookup failed: %s", e)
        return None


def _fetch_url(url, session, timeout=10):
    """Return (status_code_or_str, html_text_or_None, final_url)"""
    try:
        r = session.get(url, timeout=timeout, allow_redirects=True)
        return r.status_code, r.text, r.url
    except requests.exceptions.Timeout:
        return "timeout", None, url
    except Exception as e:
        logger.debug("fetch error: %s", e)
        return "error", None, url

def hydrate_claim(title, news_url, dataset=None, id=None, split=None,
                  max_attempts=2, timeout=10, session=None,
                  relevance_threshold=0.15):
    """
    Hydrate a single claim row using the tiered strategy:
      1) try live news_url -> parse (newspaper/readability/bs4)
      2) if fail -> try Wayback snapshot and parse
      3) fallback -> keep title only

    Adds semantic relevance cosine similarity:
      - relevance_cosine < 0.15 => ads/nav/irrelevant
      - else => relevant
    """
    owns_session = False
    if session is None:
        session = _requests_session_with_retries(total_retries=2, backoff_factor=0.5)
        owns_session = True

    now = datetime.now(timezone.utc).isoformat()

    # Normalize URLs (FakeNewsNet often omits scheme like 'example.com/...')
    news_url = _normalize_url(news_url)

    out = {
        "article_text": None,
        "content_status": "title_only",
        "news_url": news_url,
        "archive_url": None,
        "is_archived": False,
        "source_domain": None,
        "is_hydrated": False,
        "fetch_status": None,
        "lang": None,
        "content_char_len": 0,
        "claim_norm_hash": _claim_norm_hash(title),
        "ingested_at": now,
        "fetch_attempts": 0,
        "last_fetch_at": None,

        # NEW: relevance signals
        "relevance_cosine": None,
        "relevance_label": None,
    }

    if not news_url:
        out["fetch_status"] = "no_url"
        out["last_fetch_at"] = datetime.now(timezone.utc).isoformat()
        if owns_session:
            try: session.close()
            except Exception: pass
        return out

    out["news_url"] = news_url
    out["source_domain"] = _get_domain(news_url)

    # robots.txt best-effort
    if not _respect_robots(news_url, session):
        out["fetch_status"] = "robots_blocked"
        out["fetch_attempts"] = 0
        out["last_fetch_at"] = datetime.now(timezone.utc).isoformat()
        if owns_session:
            try: session.close()
            except Exception: pass
        return out

    parsed_text = None
    final_url = news_url

    # -----------------
    # Live fetch + parse
    # -----------------
    for attempt in range(max_attempts):
        out["fetch_attempts"] = attempt + 1
        status_code, html_text, final_url = _fetch_url(news_url, session, timeout=timeout)
        out["last_fetch_at"] = datetime.now(timezone.utc).isoformat()

        if status_code == 200 and html_text:
            parsed_text = None

            # try newspaper first
            if NEWSPAPER_AVAILABLE:
                parsed_text = _parse_with_newspaper(final_url, html_text)

            # readability fallback
            if not parsed_text and READABILITY_AVAILABLE and BS4_AVAILABLE:
                parsed_text = _parse_with_readability(html_text)

            # bs4 visible-text fallback
            if not parsed_text:
                try:
                    if BS4_AVAILABLE and html_text:
                        soup = BeautifulSoup(html_text, "html.parser")
                        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
                            tag.decompose()

                        candidates = soup.select("article, main, [role='main'], .article, .post, .entry-content")
                        texts = [c.get_text(separator="\n", strip=True) for c in candidates]
                        best = max(texts, key=len, default="")

                        if not best:
                            best = soup.get_text(separator="\n", strip=True)

                        if best and len(best) > 50:
                            parsed_text = best
                except Exception as e:
                    logger.debug("bs4 fallback failed: %s", e)
                    parsed_text = None

            # paywall heuristic
            paywall_indicators = [
                "subscription", "subscribe", "paywall", "sign in to continue",
                "you must login", "enable javascript", "disable your ad blocker"
            ]
            paywall_found = any(w in (html_text or "").lower() for w in paywall_indicators)

            if parsed_text:
                parsed_text = re.sub(r"\n{3,}", "\n\n", parsed_text)
                parsed_text = re.sub(r"[ \t]{2,}", " ", parsed_text)
                parsed_text = parsed_text.strip()

                out["article_text"] = parsed_text
                out["is_hydrated"] = True
                out["content_char_len"] = len(parsed_text)
                out["content_status"] = "partial" if paywall_found or len(parsed_text) < 200 else "full_article"
                out["fetch_status"] = "success"
                out["news_url"] = final_url
                break
            else:
                out["fetch_status"] = "200_no_parse"
                break

        elif status_code == "timeout":
            out["fetch_status"] = "timeout"
            time.sleep(1.0 * (attempt + 1))
            continue
        elif status_code == "error":
            out["fetch_status"] = "error"
            break
        else:
            out["fetch_status"] = f"http_{status_code}"
            break

    # -----------------
    # Wayback fallback
    # -----------------
    if (not out["article_text"]) and (out.get("fetch_status") in {
        "200_no_parse",
        "timeout", "error", "http_404", "http_410", "http_451", "http_403", "http_429"
    }):
        archive_url = _wayback_latest_snapshot_url(news_url, session, timeout=timeout)
        if archive_url:
            status_code, html_text, final_url = _fetch_url(archive_url, session, timeout=timeout)
            out["fetch_attempts"] = out.get("fetch_attempts", 0) + 1
            out["last_fetch_at"] = datetime.now(timezone.utc).isoformat()

            if status_code == 200 and html_text:
                parsed_text = None

                if NEWSPAPER_AVAILABLE:
                    parsed_text = _parse_with_newspaper(final_url, html_text)

                if not parsed_text and READABILITY_AVAILABLE and BS4_AVAILABLE:
                    parsed_text = _parse_with_readability(html_text)

                if not parsed_text:
                    try:
                        if BS4_AVAILABLE and html_text:
                            soup = BeautifulSoup(html_text, "html.parser")
                            for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
                                tag.decompose()

                            candidates = soup.select("article, main, [role='main'], .article, .post, .entry-content")
                            texts = [c.get_text(separator="\n", strip=True) for c in candidates]
                            best = max(texts, key=len, default="")

                            if not best:
                                best = soup.get_text(separator="\n", strip=True)

                            if best and len(best) > 50:
                                parsed_text = best
                    except Exception as e:
                        logger.debug("bs4 fallback failed: %s", e)
                        parsed_text = None

                if parsed_text:
                    parsed_text = re.sub(r"\n{3,}", "\n\n", parsed_text)
                    parsed_text = re.sub(r"[ \t]{2,}", " ", parsed_text)
                    parsed_text = parsed_text.strip()

                    out["article_text"] = parsed_text
                    out["is_archived"] = True
                    out["archive_url"] = archive_url
                    out["is_hydrated"] = True
                    out["content_char_len"] = len(parsed_text)
                    out["content_status"] = "full_article" if len(parsed_text) > 200 else "partial"
                    out["fetch_status"] = "archived_success"
                else:
                    out["fetch_status"] = "archived_no_parse"
            else:
                out["fetch_status"] = f"archived_http_{status_code}"
        else:
            if not out["fetch_status"]:
                out["fetch_status"] = "no_wayback_snapshot"

    if not out["article_text"]:
        out["content_status"] = "title_only"
        out["content_char_len"] = 0

    # cosine similarity check
    if out["article_text"]:
        try:
            cos = _cosine_sim_text(title or "", out["article_text"])
        except Exception as e:
            logger.debug("cosine sim failed: %s", e)
            cos = None

        out["relevance_cosine"] = cos

        if cos is None:
            out["relevance_label"] = None
        elif cos < relevance_threshold:
            out["relevance_label"] = "irrelevant_ads_nav"
            if out["fetch_status"] in ("success", "archived_success"):
                out["fetch_status"] = out["fetch_status"] + "_irrelevant"
        else:
            out["relevance_label"] = "relevant"

    if LANGDETECT_AVAILABLE and out["article_text"]:
        try:
            out["lang"] = detect(out["article_text"])
        except Exception:
            out["lang"] = None

    if not out["last_fetch_at"]:
        out["last_fetch_at"] = datetime.now(timezone.utc).isoformat()

    if owns_session:
        try:
            session.close()
        except Exception:
            pass

    return out


# ---------------- threaded batch helper ----------------
_last_call_time = defaultdict(lambda: 0.0)
_throttle_lock = threading.Lock()

def _domain_throttle(url, min_interval_secs=1.0):
    if not url:
        return
    try:
        domain = urlparse(url).netloc.lower() or _get_domain(url)
        now = time.time()

        with _throttle_lock:
            next_allowed = _last_call_time[domain]
            if now < next_allowed:
                sleep_for = next_allowed - now
                _last_call_time[domain] = next_allowed + min_interval_secs
            else:
                sleep_for = 0.0
                _last_call_time[domain] = now + min_interval_secs

        if sleep_for > 0:
            time.sleep(sleep_for)
    except Exception:
        return



def threaded_hydrate(df, title_col="title", url_col="news_url", hydrate_fn=None,
                     dataset=None, max_workers=12, batch_size=None, save_batches_dir=None,
                     throttle_seconds=1.0, show_progress=True):
    """
    Run hydrate_fn(title, news_url, dataset=..., id=idx) in threads over df rows.
    Returns a DataFrame indexed by the original df index containing the returned dicts flattened.
    If save_batches_dir provided, writes per-batch parquet for resumability.
    """
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm.auto import tqdm
    import os

    if hydrate_fn is None:
        hydrate_fn = hydrate_claim

    os.makedirs(save_batches_dir, exist_ok=True) if save_batches_dir else None

    tasks = [(idx, row.get(title_col), row.get(url_col)) for idx, row in df[[title_col, url_col]].iterrows()]

    # Thread-local session reuse for connection pooling and faster hydration
    _tls = threading.local()

    def _get_thread_session():
        if getattr(_tls, "session", None) is None:
            _tls.session = _requests_session_with_retries(total_retries=2, backoff_factor=0.5)
        return _tls.session

    def _worker(task):
        idx, title, url = task
        _domain_throttle(url, min_interval_secs=throttle_seconds)
        try:
            sess = _get_thread_session()
            out = hydrate_fn(title=title, news_url=url, dataset=dataset, id=idx, session=sess)
            if not isinstance(out, dict):
                out = {"result": out}
            out["_row_idx"] = idx
            out["_error"] = None
        except Exception as e:
            out = {
                "_row_idx": idx,
                "_error": str(e),
                "claim_text": title if title is not None else "",
                "article_text": None,
                "content_status": "title_only",
                "news_url": url
            }
        return out

    def _run_chunk(chunk_tasks):
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_worker, t): t for t in chunk_tasks}
            if show_progress:
                for fut in tqdm(as_completed(futures), total=len(futures)):
                    results.append(fut.result())
            else:
                for fut in as_completed(futures):
                    results.append(fut.result())
        return results

    all_results = []
    if batch_size is None:
        all_results = _run_chunk(tasks)
    else:
        n = len(tasks)
        for i in range(0, n, batch_size):
            chunk = tasks[i: i + batch_size]
            chunk_results = _run_chunk(chunk)
            if save_batches_dir:
                batch_df = pd.json_normalize(chunk_results).set_index("_row_idx")
                fname = os.path.join(save_batches_dir, f"hydrated_batch_{i}_{i+len(chunk)-1}.parquet")
                tmp = fname + ".part"
                batch_df.to_parquet(tmp)
                os.replace(tmp, fname)
            all_results.extend(chunk_results)

    res_df = pd.json_normalize(all_results)
    if "_row_idx" not in res_df.columns:
        raise RuntimeError("Worker did not return _row_idx for tasks.")
    res_df = res_df.set_index("_row_idx")
    res_df = res_df.reindex(df.index)
    return res_df