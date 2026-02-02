# verifier_pipeline/rag/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .chunking import chunk_text
from .fetch import fetch_url
from .google_pse import GooglePSEClient
from .retrieve import rank_chunks


def build_queries(claim: str) -> List[str]:
    claim = " ".join(str(claim).split()).strip()
    return [f'"{claim}"', f"{claim} fact check"]


@dataclass
class EvidenceSnippet:
    eid: str
    url: str
    title: str
    text: str
    score: float


def gather_evidence(
    claim: str,
    *,
    per_query: int = 5,  # search results per query
    max_urls: int = 8,  # total unique urls to consider
    max_fetch: int = 5,  # how many urls to fetch
    top_k_snippets: int = 8,  # E1..Ek
) -> Dict[str, Any]:
    client = GooglePSEClient()
    queries = build_queries(claim)

    # 1) search + dedupe
    results = []
    seen = set()
    for q in queries:
        for r in client.search(q, num=per_query):
            if r.link and r.link not in seen:
                results.append(r)
                seen.add(r.link)
            if len(results) >= max_urls:
                break
        if len(results) >= max_urls:
            break

    # 2) fetch top urls
    docs = []
    for r in results[:max_fetch]:
        doc = fetch_url(r.link)
        if doc:
            docs.append(doc)

    # 3) chunk
    flat_chunks = []
    for d in docs:
        for ch in chunk_text(d.url, d.title, d.text):
            flat_chunks.append((ch.url, ch.title, ch.chunk_id, ch.text))

    # 4) retrieve + rerank
    ranked = rank_chunks(claim, flat_chunks, top_k=top_k_snippets)

    snippets: List[EvidenceSnippet] = []
    for i, rc in enumerate(ranked, start=1):
        snippets.append(
            EvidenceSnippet(
                eid=f"E{i}",
                url=rc.url,
                title=rc.title,
                text=rc.text,
                score=float(rc.score),
            )
        )

    return {
        "queries": queries,
        "sources": [
            {"url": r.link, "title": r.title, "snippet": r.snippet} for r in results
        ],
        "snippets": [s.__dict__ for s in snippets],
        "stats": {
            "num_urls": len(results),
            "num_docs_fetched": len(docs),
            "num_chunks": len(flat_chunks),
            "num_snippets": len(snippets),
        },
    }
