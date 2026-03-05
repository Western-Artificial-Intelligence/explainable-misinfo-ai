#!/usr/bin/env python3
"""
Test Brave Search API so step-2 classifier can use web search.
Run from repo root: python scripts/test_brave_search.py
Uses .env (load from cwd or parent dirs). Prints whether the key works and returns snippets.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Load .env from repo root (cwd or walk up)
def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        for d in [Path.cwd()] + list(Path.cwd().resolve().parents[:5]):
            f = d / ".env"
            if f.is_file():
                load_dotenv(f, override=False)
                print(f"Loaded .env from {f}", flush=True)
                return
        load_dotenv(override=False)
    except Exception as e:
        print(f"Could not load dotenv: {e}", flush=True)

_load_dotenv()

try:
    import httpx
except ImportError:
    print("Install httpx: pip install httpx")
    sys.exit(1)

def main() -> None:
    key = (os.getenv("BRAVE_API_KEY") or os.getenv("BRAVE_SEARCH_API_KEY") or "").strip()
    if not key:
        print("BRAVE_API_KEY (or BRAVE_SEARCH_API_KEY) not set in environment.")
        print("Add it to .env in the repo root and run this script from the repo root.")
        sys.exit(1)
    endpoint = (os.getenv("BRAVE_ENDPOINT") or "https://api.search.brave.com/res/v1/web/search").strip()
    q = "fact check: water boils at 100 degrees celsius"
    params = {"q": q, "count": 5}
    for env_key, param_key in [("BRAVE_SAFESEARCH", "safesearch"), ("BRAVE_COUNTRY", "country"), ("BRAVE_SEARCH_LANG", "search_lang")]:
        v = (os.getenv(env_key) or "").strip()
        if v:
            params[param_key] = v
    headers = {"Accept": "application/json", "X-Subscription-Token": key}
    print(f"Requesting: GET {endpoint} q={q!r} ...", flush=True)
    try:
        with httpx.Client(timeout=20) as client:
            r = client.get(endpoint, params=params, headers=headers)
    except Exception as e:
        print(f"Request failed: {e}")
        sys.exit(1)
    print(f"HTTP {r.status_code}", flush=True)
    if r.status_code != 200:
        print(r.text[:500])
        if r.status_code in (401, 403):
            print("Check BRAVE_API_KEY: invalid or expired.")
        sys.exit(1)
    data = r.json()
    results = (data.get("web") or {}).get("results") or []
    print(f"Got {len(results)} result(s). Snippets:", flush=True)
    for i, it in enumerate(results[:5], 1):
        title = (it.get("title") or "").strip() or "(no title)"
        desc = (it.get("description") or "").strip()
        desc_short = (desc[:80] + "...") if len(desc) > 80 else desc
        print(f"  {i}. {title} | {desc_short}")
    if results:
        print("\nBrave search works. Step-2 classifier can use ROBERTA_LLM_USE_WEB_SEARCH=true.")
    else:
        print("\nNo snippets returned (query or API limits). Check Brave dashboard.")

if __name__ == "__main__":
    main()
