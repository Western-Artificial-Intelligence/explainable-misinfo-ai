# article_scraping.py
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse, quote
import tldextract
from datetime import datetime, timezone
import hashlib
import re
import time
import logging
from collections import defaultdict

# Parsing libs - try newspaper first, fallback to readability + bs4
try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except Exception:
    NEWSPAPER_AVAILABLE = False

try:
    from readability import Document
    from bs4 import BeautifulSoup
    READABILITY_AVAILABLE = True
except Exception:
    READABILITY_AVAILABLE = False

# language detection (optional and light)
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except Exception:
    LANGDETECT_AVAILABLE = False

logger = logging.getLogger("hydrator")
logger.setLevel(logging.INFO)


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
        return (parsed.netloc or "").lower()
    except Exception:
        return ""


def _claim_norm_hash(title):
    if title is None:
        return None
    s = re.sub(r'\s+', ' ', str(title).strip().lower())
    return hashlib.sha1(s.encode('utf-8')).hexdigest()


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
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        r = session.get(robots_url, timeout=timeout)
        if r.status_code == 200 and r.text:
            rp = RobotFileParser()
            rp.parse(r.text.splitlines())
            return rp.can_fetch(session.headers.get("User-Agent"), url)
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
    if not READABILITY_AVAILABLE or not html_text:
        return None
    try:
        doc = Document(html_text)
        content_html = doc.summary()
        soup = BeautifulSoup(content_html, "lxml")
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
        cdx = ("http://web.archive.org/cdx/search/cdx?url={url}"
               "&output=json&fl=timestamp,original&filter=statuscode:200&limit=1&collapse=timestamp:8")
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
                  max_attempts=2, timeout=10):
    """
    Hydrate a single claim row using the tiered strategy:
      1) try live news_url -> parse (newspaper/readability/bs4)
      2) if fail -> try Wayback snapshot and parse
      3) fallback -> keep title only

    Returns a dict matching the unified schema keys relevant to hydration.
    """
    session = _requests_session_with_retries(total_retries=2, backoff_factor=0.5)
    now = datetime.now(timezone.utc).isoformat()

    # baseline output dict (keys you asked for)
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
        "last_fetch_at": None
    }

    if not news_url:
        out["fetch_status"] = "no_url"
        out["last_fetch_at"] = datetime.now(timezone.utc).isoformat()
        return out

    out["source_domain"] = _get_domain(news_url)

    # robots.txt best-effort
    if not _respect_robots(news_url, session):
        out["fetch_status"] = "robots_blocked"
        out["fetch_attempts"] = 0
        out["last_fetch_at"] = datetime.now(timezone.utc).isoformat()
        return out

    parsed_text = None
    final_url = news_url
    fetch_status = None

    for attempt in range(max_attempts):
        out["fetch_attempts"] = attempt + 1
        status_code, html_text, final_url = _fetch_url(news_url, session, timeout=timeout)
        out["last_fetch_at"] = datetime.now(timezone.utc).isoformat()

        if status_code == 200 and html_text:
            # try newspaper first
            parsed_text = None
            if NEWSPAPER_AVAILABLE:
                parsed_text = _parse_with_newspaper(final_url, html_text)
            if not parsed_text and READABILITY_AVAILABLE:
                parsed_text = _parse_with_readability(html_text)

            # fallback to visible-text extraction
            if not parsed_text:
                try:
                    soup = BeautifulSoup(html_text, "lxml")
                    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
                        tag.decompose()
                    page_text = soup.get_text(separator="\n").strip()
                    if page_text and len(page_text) > 50:
                        parsed_text = page_text
                except Exception:
                    parsed_text = None

            # paywall heuristic
            paywall_indicators = ["subscription", "subscribe", "paywall", "sign in to continue", "you must login"]
            paywall_found = any(w in (html_text or "").lower() for w in paywall_indicators)

            if parsed_text:
                out["article_text"] = parsed_text
                out["is_hydrated"] = True
                out["content_char_len"] = len(parsed_text)
                out["content_status"] = "partial" if paywall_found or len(parsed_text) < 200 else "full_article"
                fetch_status = "success"
                out["fetch_status"] = fetch_status
                out["news_url"] = final_url
                break
            else:
                out["fetch_status"] = "200_no_parse"
                # we'll try wayback after loop
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

    # Try Wayback if we couldn't parse live page
    if not out["article_text"]:
        archive_url = _wayback_latest_snapshot_url(news_url, session, timeout=timeout)
        if archive_url:
            status_code, html_text, final_url = _fetch_url(archive_url, session, timeout=timeout)
            out["fetch_attempts"] = out.get("fetch_attempts", 0) + 1
            out["last_fetch_at"] = datetime.now(timezone.utc).isoformat()
            if status_code == 200 and html_text:
                parsed_text = None
                if NEWSPAPER_AVAILABLE:
                    parsed_text = _parse_with_newspaper(final_url, html_text)
                if not parsed_text and READABILITY_AVAILABLE:
                    parsed_text = _parse_with_readability(html_text)
                if not parsed_text:
                    try:
                        soup = BeautifulSoup(html_text, "lxml")
                        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
                            tag.decompose()
                        page_text = soup.get_text(separator="\n").strip()
                        if page_text and len(page_text) > 50:
                            parsed_text = page_text
                    except Exception:
                        parsed_text = None
                if parsed_text:
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

    if LANGDETECT_AVAILABLE and out["article_text"]:
        try:
            out["lang"] = detect(out["article_text"])
        except Exception:
            out["lang"] = None

    if not out["last_fetch_at"]:
        out["last_fetch_at"] = datetime.now(timezone.utc).isoformat()

    return out


# ---------------- threaded batch helper ----------------
# simple per-domain throttle
_last_call_time = defaultdict(lambda: 0.0)


def _domain_throttle(url, min_interval_secs=1.0):
    if not url:
        return
    try:
        domain = urlparse(url).netloc.lower()
        now = time.time()
        elapsed = now - _last_call_time[domain]
        if elapsed < min_interval_secs:
            time.sleep(min_interval_secs - elapsed)
        _last_call_time[domain] = time.time()
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

    def _worker(task):
        idx, title, url = task
        _domain_throttle(url, min_interval_secs=throttle_seconds)
        try:
            out = hydrate_fn(title=title, news_url=url, dataset=dataset, id=idx)
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