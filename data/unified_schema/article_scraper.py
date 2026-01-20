from __future__ import annotations

import logging
import threading
from typing import Any, Optional, Tuple, Union
from urllib.parse import quote, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("article_scraper")
logger.setLevel(logging.INFO)

try:
    from newspaper import Article
    _HAS_NEWSPAPER = True
except Exception:
    _HAS_NEWSPAPER = False

try:
    from readability import Document
    _HAS_READABILITY = True
except Exception:
    _HAS_READABILITY = False

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except Exception:
    _HAS_BS4 = False

TimeoutT = Union[float, tuple[float, float]]

# Session + retries
def _requests_session_with_retries(
    total_retries: int = 2,
    backoff_factor: float = 0.5,
    allowed_methods=frozenset(["GET", "HEAD"]),
) -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=allowed_methods,
        raise_on_status=False,
    )
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({
        "User-Agent": "article-scraper/1.0 (+https://example.com/)",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    return s

# URL normalization
def _normalize_url(url: str) -> str:
    if url is None:
        return ""
    u = str(url).strip()
    if not u:
        return ""
    if u.startswith("//"):
        return "http:" + u
    parsed = urlparse(u)
    if parsed.scheme:
        return u
    return "http://" + u


# robots.txt best-effort
_robots_cache: dict[str, bool] = {}
_robots_lock = threading.Lock()

def _respect_robots(url: str, session: requests.Session, timeout: float = 5.0) -> bool:
    """Permissive on failure."""
    from urllib.robotparser import RobotFileParser
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return True
        domain = parsed.netloc.lower()
        with _robots_lock:
            if domain in _robots_cache:
                return _robots_cache[domain]

        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        r = session.get(robots_url, timeout=timeout)
        if r.status_code == 200 and r.text:
            rp = RobotFileParser()
            rp.parse(r.text.splitlines())
            allowed = rp.can_fetch(session.headers.get("User-Agent", "*"), url)
        else:
            allowed = True

        with _robots_lock:
            _robots_cache[domain] = allowed
        return allowed
    except Exception:
        return True


# Fetch + Wayback
def _fetch_url(
    url: str,
    session: requests.Session,
    timeout: TimeoutT,
) -> Tuple[int | str, Optional[str], str]:
    """Return (status, html_or_none, final_url)."""
    try:
        r = session.get(url, timeout=timeout, allow_redirects=True)
        return r.status_code, r.text, r.url
    except requests.exceptions.Timeout:
        return "timeout", None, url
    except Exception as e:
        logger.debug("fetch error: %s", e)
        return "error", None, url


def _wayback_latest_snapshot_url(
    url: str,
    session: requests.Session,
    timeout: float = 10.0,
) -> Optional[str]:
    try:
        cdx = (
            "https://web.archive.org/cdx/search/cdx?url={url}"
            "&output=json&fl=timestamp,original"
            "&filter=statuscode:200"
            "&collapse=timestamp:8"
            "&sort=descending"
            "&limit=1"
        )
        q = cdx.format(url=quote(url, safe=""))
        r = session.get(q, timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.json()
        if isinstance(data, list) and len(data) >= 2:
            ts, orig = data[1][0], data[1][1]
            return f"https://web.archive.org/web/{ts}/{orig}"
        return None
    except Exception:
        return None


# Parse HTML -> raw text
def _parse_html(url: str, html: str) -> Optional[str]:
    if not html:
        return None

    # newspaper3k
    if _HAS_NEWSPAPER:
        try:
            art = Article(url)
            art.download(input_html=html)
            art.parse()
            txt = (art.text or "").strip()
            if txt:
                return txt
        except Exception:
            pass

    # readability
    if _HAS_READABILITY and _HAS_BS4:
        try:
            doc = Document(html)
            soup = BeautifulSoup(doc.summary(), "html.parser")
            txt = soup.get_text(separator="\n").strip()
            if txt:
                return txt
        except Exception:
            pass

    # bs4 fallback
    if _HAS_BS4:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
                tag.decompose()

            candidates = soup.select("article, main, [role='main'], .article, .post, .entry-content")
            texts = [c.get_text(separator="\n", strip=True) for c in candidates]
            best = max(texts, key=len, default="") or soup.get_text(separator="\n", strip=True)
            best = (best or "").strip()
            return best if len(best) >= 50 else None
        except Exception:
            return None

    return html


# Public function
def scrap_from_web(
    url: str,
    *,
    timeout: TimeoutT = (5.0, 12.0),
    max_attempts: int = 2,
    use_wayback: bool = False,
    respect_robots: bool = True,
    session: Optional[requests.Session] = None,
    return_meta: bool = True,
):
    """
    General-purpose: input URL -> output raw extracted text (or None).

    Default (return_meta=False):
        -> Optional[str]

    If return_meta=True:
        -> tuple[Optional[str], dict]
           meta includes fetch_status/http_status/final_url/used_wayback/etc.
    """
    u = _normalize_url(url)

    meta: dict[str, Any] = {
        "input_url": url,
        "normalized_url": u,
        "final_url": None,
        "http_status": None,
        "fetch_status": None,
        "used_wayback": False,
        "wayback_url": None,
        "robots_allowed": None,
    }

    if not u:
        meta["fetch_status"] = "no_url"
        return (None, meta) if return_meta else None

    owns_session = session is None
    if session is None:
        session = _requests_session_with_retries(total_retries=0)

    try:
        # robots best-effort
        if respect_robots:
            allowed = _respect_robots(u, session)
            meta["robots_allowed"] = allowed
            if not allowed:
                meta["fetch_status"] = "robots_blocked"
                return (None, meta) if return_meta else None
        else:
            meta["robots_allowed"] = True

        last_status: int | str | None = None

        # Live attempts
        for _ in range(max_attempts):
            status, html, final_url = _fetch_url(u, session, timeout=timeout)
            last_status = status

            meta["final_url"] = final_url
            if isinstance(status, int):
                meta["http_status"] = status

            if status == 200 and html:
                txt = _parse_html(final_url, html)
                if txt and txt.strip():
                    meta["fetch_status"] = "success"
                    return (txt, meta) if return_meta else txt
                meta["fetch_status"] = "200_no_parse"
                return (None, meta) if return_meta else None

            if status == "timeout":
                meta["fetch_status"] = "timeout"
                continue
            if status == "error":
                meta["fetch_status"] = "error"
                break

            if isinstance(status, int):
                meta["fetch_status"] = f"http_{status}"
            else:
                meta["fetch_status"] = str(status)
            break

        # -----------------
        # Wayback fallback
        # -----------------
        if use_wayback and last_status in {"timeout", "error", 403, 404, 410, 429, 451}:
            wb_timeout = 10.0 if isinstance(timeout, tuple) else max(10.0, timeout)
            snap = _wayback_latest_snapshot_url(u, session, timeout=wb_timeout)

            if snap:
                meta["used_wayback"] = True
                meta["wayback_url"] = snap

                status, html, final_url = _fetch_url(snap, session, timeout=timeout)
                meta["final_url"] = final_url
                if isinstance(status, int):
                    meta["http_status"] = status

                if status == 200 and html:
                    txt = _parse_html(final_url, html)
                    if txt and txt.strip():
                        meta["fetch_status"] = "archived_success"
                        return (txt, meta) if return_meta else txt
                    meta["fetch_status"] = "archived_200_no_parse"
                    return (None, meta) if return_meta else None

                if status == "timeout":
                    meta["fetch_status"] = "archived_timeout"
                elif status == "error":
                    meta["fetch_status"] = "archived_error"
                else:
                    meta["fetch_status"] = f"archived_http_{status}"
            else:
                meta["fetch_status"] = meta["fetch_status"] or "no_wayback_snapshot"

        return (None, meta) if return_meta else None

    finally:
        if owns_session:
            try:
                session.close()
            except Exception:
                pass
