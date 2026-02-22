from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
import re
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass
class DocumentRecord:
    document_id: str
    title: str
    source: str
    text: str
    created_at: str


_DOCUMENTS: dict[str, DocumentRecord] = {}
_LATEST_DOCUMENT_ID: str | None = None


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _normalize_text(raw_text: str) -> str:
    text = re.sub(r"\s+", " ", raw_text).strip()
    return text


def _decode_body(body: bytes, content_type: str) -> str:
    charset_match = re.search(r"charset=([a-zA-Z0-9_-]+)", content_type or "", re.IGNORECASE)
    if charset_match:
        charset = charset_match.group(1)
        try:
            return body.decode(charset, errors="ignore")
        except LookupError:
            pass
    try:
        return body.decode("utf-8", errors="ignore")
    except Exception:
        return body.decode("latin-1", errors="ignore")


def _extract_title_from_html(html: str) -> str | None:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return _normalize_text(match.group(1))


def _extract_text_from_html(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return _normalize_text(parser.get_text())


def _guess_title_from_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.path and parsed.path.strip("/"):
        return parsed.path.rstrip("/").split("/")[-1]
    return parsed.netloc or "Web Document"


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def _store_document(*, title: str, source: str, text: str) -> dict[str, Any]:
    global _LATEST_DOCUMENT_ID

    cleaned = _normalize_text(text)
    if not cleaned:
        raise ValueError("Document has no readable text.")

    document_id = str(uuid4())
    record = DocumentRecord(
        document_id=document_id,
        title=title.strip() or "Untitled Document",
        source=source,
        text=cleaned,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _DOCUMENTS[document_id] = record
    _LATEST_DOCUMENT_ID = document_id

    return {
        "document_id": record.document_id,
        "title": record.title,
        "source": record.source,
        "char_count": len(record.text),
        "created_at": record.created_at,
    }


def ingest_document_text(*, filename: str, content: str) -> dict[str, Any]:
    title = filename.strip() or "Local Document"
    source = f"file:{title}"
    return _store_document(title=title, source=source, text=content)


def ingest_document_url(url: str) -> dict[str, Any]:
    target = url.strip()
    if not target:
        raise ValueError("URL cannot be empty.")
    if not (target.startswith("http://") or target.startswith("https://")):
        raise ValueError("URL must start with http:// or https://")

    request = Request(
        target,
        headers={
            "User-Agent": "TruthLensDocumentReader/1.0 (+https://localhost)",
            "Accept": "text/html, text/plain, application/xhtml+xml;q=0.9, */*;q=0.8",
        },
    )

    with urlopen(request, timeout=20) as response:  # noqa: S310
        body = response.read()
        content_type = response.headers.get("Content-Type", "")

    decoded = _decode_body(body, content_type)

    is_html = "html" in content_type.lower() or "<html" in decoded[:500].lower()
    if is_html:
        title = _extract_title_from_html(decoded) or _guess_title_from_url(target)
        text = _extract_text_from_html(decoded)
    else:
        title = _guess_title_from_url(target)
        text = _normalize_text(decoded)

    return _store_document(title=title, source=target, text=text)


def list_documents() -> list[dict[str, Any]]:
    records = sorted(_DOCUMENTS.values(), key=lambda r: r.created_at, reverse=True)
    return [
        {
            "document_id": r.document_id,
            "title": r.title,
            "source": r.source,
            "char_count": len(r.text),
            "created_at": r.created_at,
        }
        for r in records
    ]


def answer_question(*, question: str, document_id: str | None = None) -> dict[str, Any]:
    prompt = question.strip()
    if not prompt:
        raise ValueError("Question cannot be empty.")

    target_id = document_id or _LATEST_DOCUMENT_ID
    if not target_id or target_id not in _DOCUMENTS:
        raise ValueError("No document found. Upload a document or URL first.")

    doc = _DOCUMENTS[target_id]
    question_terms = _tokenize(prompt)
    sentences = _split_sentences(doc.text)

    if not sentences:
        raise ValueError("The selected document has no parsable sentences.")

    scored: list[tuple[int, int, str]] = []
    for idx, sentence in enumerate(sentences):
        sentence_terms = _tokenize(sentence)
        overlap = len(question_terms & sentence_terms) if question_terms else 0
        scored.append((overlap, -idx, sentence))

    scored.sort(reverse=True)

    top_sentences = [s for overlap, _, s in scored if overlap > 0][:3]
    if not top_sentences:
        top_sentences = sentences[:2]

    answer = " ".join(top_sentences)
    answer = answer[:1200]

    best_overlap = scored[0][0] if scored else 0
    confidence = 0.55 if best_overlap == 0 else min(0.95, 0.6 + (best_overlap * 0.08))

    return {
        "document_id": doc.document_id,
        "title": doc.title,
        "source": doc.source,
        "question": prompt,
        "answer": answer,
        "snippets": top_sentences,
        "confidence": round(confidence, 2),
    }

