# verifier_pipeline/rag/google_pse.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional

import requests


@dataclass
class SearchResult:
    title: str
    link: str
    snippet: str
    display_link: Optional[str] = None


class GooglePSEClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        cx: Optional[str] = None,
        sleep_s: float = 0.0,
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_PSE_API_KEY")
        self.cx = cx or os.environ.get("GOOGLE_PSE_CX")
        if not self.api_key or not self.cx:
            raise RuntimeError("Missing GOOGLE_PSE_API_KEY or GOOGLE_PSE_CX")
        self.sleep_s = sleep_s

    def search(self, q: str, num: int = 5) -> List[SearchResult]:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": q,
            "num": min(max(num, 1), 10),
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        out: List[SearchResult] = []
        for it in data.get("items") or []:
            link = (it.get("link") or "").strip()
            if not link:
                continue
            out.append(
                SearchResult(
                    title=(it.get("title") or "").strip(),
                    link=link,
                    snippet=(it.get("snippet") or "").strip(),
                    display_link=it.get("displayLink"),
                )
            )

        if self.sleep_s:
            time.sleep(self.sleep_s)
        return out
