"""LLM-based claim extraction service.

Uses Ollama to extract structured factual claims from raw text
(e.g., Whisper transcripts, OCR output) before classification.

Falls back to returning raw text if Ollama is unavailable.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_TRANSCRIPT = """\
You are a factual claim extractor. Given a transcript from audio or video,
extract the core factual claim(s) being made. Rules:
- Strip filler words, opinions, hedging, and non-factual content
- Return ONLY the concise factual claim(s), nothing else
- If multiple claims exist, return the most prominent one
- Do NOT add your own commentary or analysis
- Keep the claim in the original language
- If no factual claim is present, return the text as-is"""

_SYSTEM_PROMPT_OCR = """\
You are a factual claim extractor. Given text extracted from an image via OCR,
extract the core factual claim(s) being made. Rules:
- Fix obvious OCR errors if possible
- Strip non-claim content (headers, captions, watermarks, UI elements)
- Return ONLY the concise factual claim(s), nothing else
- Do NOT add your own commentary or analysis
- If no factual claim is present, return the cleaned text as-is"""


async def extract_claim_from_transcript(transcript_text: str) -> str:
    """Extract a structured claim from a Whisper transcript via Ollama LLM."""
    return await _extract(transcript_text, _SYSTEM_PROMPT_TRANSCRIPT)


async def extract_claim_from_ocr(ocr_text: str) -> str:
    """Extract a structured claim from OCR-extracted text via Ollama LLM."""
    return await _extract(ocr_text, _SYSTEM_PROMPT_OCR)


async def extract_claim(raw_text: str, source_type: Optional[str] = None) -> str:
    """Extract a claim from raw text, choosing the right prompt based on source type.

    Args:
        raw_text: The raw text to extract a claim from.
        source_type: One of "transcript", "ocr", or None (generic).

    Returns:
        The extracted claim text, or the original text if extraction fails.
    """
    if source_type == "ocr":
        return await extract_claim_from_ocr(raw_text)
    return await extract_claim_from_transcript(raw_text)


async def _extract(raw_text: str, system_prompt: str) -> str:
    """Core extraction logic: call Ollama, fall back to raw text on failure."""
    if not raw_text or not raw_text.strip():
        return raw_text

    try:
        from api.production_pipeline.middlewares.ollama_blackbox import ollama_chat

        result = await ollama_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_text},
            ],
            temperature=0.1,
            num_predict=256,
        )
        extracted = result.get("content", "").strip()
        if extracted:
            logger.info("LLM claim extraction succeeded (%d → %d chars)", len(raw_text), len(extracted))
            return extracted
        logger.warning("LLM returned empty claim; using raw text")
        return raw_text
    except Exception:
        logger.warning("Ollama unavailable for claim extraction; using raw text", exc_info=True)
        return raw_text
