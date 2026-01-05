# verifier_pipeline/rag/retrieve.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import re
import math
from collections import Counter

@dataclass
class RankedChunk:
    score: float
    url: str
    title: str
    chunk_id: str
    text: str

def _tok(s: str) -> List[str]:
    s = (s or "").casefold()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return [t for t in s.split(" ") if len(t) > 2]

def rank_chunks(claim: str, chunks: List[Tuple[str, str, str, str]], top_k: int = 8) -> List[RankedChunk]:
    q = _tok(claim)
    if not q:
        return []

    qset = set(q)
    dfs = Counter()
    chunk_terms = []

    for (_, _, _, txt) in chunks:
        toks = set(_tok(txt))
        chunk_terms.append(toks)
        for t in toks:
            dfs[t] += 1

    N = max(1, len(chunks))
    ranked: List[RankedChunk] = []

    for (url, title, cid, txt), toks in zip(chunks, chunk_terms):
        overlap = qset.intersection(toks)
        if not overlap:
            continue
        score = 0.0
        for t in overlap:
            df = dfs.get(t, 1)
            idf = math.log((N + 1) / (df + 1)) + 1.0
            score += idf
        ranked.append(RankedChunk(score=score, url=url, title=title, chunk_id=cid, text=txt))

    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked[:top_k]
