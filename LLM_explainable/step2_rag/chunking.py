# verifier_pipeline/rag/chunking.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import re

@dataclass
class Chunk:
    url: str
    title: str
    chunk_id: str
    text: str

def chunk_text(url: str, title: str, text: str, max_words: int = 180, overlap: int = 40) -> List[Chunk]:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    words = text.split(" ")
    step = max(1, max_words - overlap)

    chunks: List[Chunk] = []
    i, k = 0, 0
    while i < len(words):
        window = words[i:i + max_words]
        chunk = " ".join(window).strip()
        if chunk:
            chunks.append(Chunk(url=url, title=title, chunk_id=str(k), text=chunk))
            k += 1
        i += step
    return chunks
