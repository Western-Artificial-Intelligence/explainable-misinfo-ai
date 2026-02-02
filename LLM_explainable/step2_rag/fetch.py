# verifier_pipeline/rag/fetch.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup


@dataclass
class FetchedDoc:
    url: str
    title: str
    text: str


def fetch_url(
    url: str, timeout_s: int = 20, max_chars: int = 250_000
) -> Optional[FetchedDoc]:
    try:
        r = requests.get(
            url,
            timeout=timeout_s,
            headers={"User-Agent": "Mozilla/5.0 (compatible; verifier_pipeline/1.0)"},
        )
        if r.status_code != 200:
            return None

        html = (r.text or "")[:max_chars]
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(
            ["script", "style", "noscript", "header", "footer", "nav", "aside"]
        ):
            tag.decompose()

        title = (soup.title.get_text(" ", strip=True) if soup.title else "").strip()
        text = soup.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) < 400:
            return None
        return FetchedDoc(url=url, title=title, text=text)
    except Exception:
        return None
