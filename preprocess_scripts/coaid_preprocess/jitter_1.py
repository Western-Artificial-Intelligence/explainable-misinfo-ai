#!/usr/bin/env python3
# jitter_1.py — fast, no-auth tweet text fetcher
# Tries fxTwitter + vxTwitter in parallel (race), then optional oEmbed/Nitter.
# Tunables via env:
#   JITTER_TIMEOUT=6            # per-request timeout (sec)
#   JITTER_FAST_ONLY=1          # only fx/vx (skip slower fallbacks)
#   JITTER_PAR_FAST=1           # race fx & vx concurrently
#   JITTER_ALLOW_NITTER=0       # include Nitter as last resort (can be slow)

import os, re, sys, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

UA = {"User-Agent": "Mozilla/5.0 (compatible; jitter_noauth/2.0)"}
TIMEOUT = float(os.getenv("JITTER_TIMEOUT", "6"))
FAST_ONLY = os.getenv("JITTER_FAST_ONLY", "1") == "1"
PAR_FAST = os.getenv("JITTER_PAR_FAST", "1") == "1"
ALLOW_NITTER = os.getenv("JITTER_ALLOW_NITTER", "0") == "1"

# Reuse connections across threads
SESSION = requests.Session()
ADAPTER = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=0)
SESSION.mount("https://", ADAPTER)
SESSION.headers.update(UA)

def extract_id(s: str) -> str:
    m = re.search(r"/status(?:es)?/(\d+)", s)
    if m: return m.group(1)
    return s if s.isdigit() else ""

def from_fxtwitter(tweet_id: str):
    r = SESSION.get(f"https://api.fxtwitter.com/status/{tweet_id}", timeout=TIMEOUT)
    if not r.ok: return None
    j = r.json()
    return (j.get("tweet") or {}).get("text")

def from_vxtwitter(tweet_id: str):
    r = SESSION.get(f"https://api.vxtwitter.com/status/{tweet_id}", timeout=TIMEOUT)
    if not r.ok: return None
    j = r.json()
    return (j.get("tweet") or {}).get("text")

def from_oembed(tweet_id: str):
    r = SESSION.get(
        "https://publish.twitter.com/oembed",
        params={"url": f"https://x.com/i/web/status/{tweet_id}", "omit_script":"true","hide_thread":"true"},
        timeout=TIMEOUT
    )
    if not r.ok: return None
    html = (r.json() or {}).get("html", "")
    if not html: return None
    if BeautifulSoup:
        return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    return html

def from_nitter(tweet_id: str):
    if not BeautifulSoup: return None
    for host in ["nitter.net","nitter.poast.org","nitter.fdn.fr","nitter.lacontrevoie.fr"]:
        try:
            r = SESSION.get(f"https://{host}/i/status/{tweet_id}", timeout=TIMEOUT)
            if not r.ok: continue
            soup = BeautifulSoup(r.text, "html.parser")
            og = soup.find("meta", attrs={"property":"og:description"})
            if og and og.get("content"):
                t = og["content"].strip()
                if t: return t
            main = soup.select_one("div.tweet-content") or soup.select_one("article")
            if main:
                t = main.get_text(" ", strip=True)
                if t: return t
        except Exception:
            continue
    return None

def _race_fast(tweet_id: str):
    # Race fx & vx; return first non-empty result
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = [
            ex.submit(from_fxtwitter, tweet_id),
            ex.submit(from_vxtwitter, tweet_id),
        ]
        for fut in as_completed(futs):
            try:
                txt = fut.result()
                if txt: return txt
            except Exception:
                pass
    return None

def fetch_text(id_or_url: str) -> str:
    tid = extract_id(id_or_url.strip())
    if not tid:
        raise SystemExit("[error] provide a numeric tweet ID or /status/<id> URL")

    # FAST path: fx/vx only
    txt = _race_fast(tid) if PAR_FAST else (from_fxtwitter(tid) or from_vxtwitter(tid))
    if txt: return txt.strip()
    if FAST_ONLY:
        raise SystemExit("[error] Fast-only failed.")

    # Slow fallbacks if allowed
    txt = from_oembed(tid)
    if txt: return txt.strip()
    if ALLOW_NITTER:
        txt = from_nitter(tid)
        if txt: return txt.strip()

    raise SystemExit("[error] No-auth sources failed (tweet protected/deleted or endpoints blocked).")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Print tweet text without API keys")
    ap.add_argument("id_or_url", help="Tweet ID or URL")
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    print(fetch_text(args.id_or_url))

if __name__ == "__main__":
    main()
