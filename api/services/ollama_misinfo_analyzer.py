"""Ollama-based misinformation analyzer with web search reasoning.

Uses Ollama to classify text as REAL or FAKE and generate explanations
that reference internet articles from DuckDuckGo search results.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Optional web search
def _search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search DuckDuckGo for fact-check / verification articles. Returns list of {title, snippet, href}."""
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            # Search for fact-check and verification context
            search_query = f'"{query[:80]}" fact check OR verification OR debunk'
            results = list(ddgs.text(search_query, max_results=max_results))
            return [
                {"title": r.get("title", ""), "snippet": r.get("body", ""), "href": r.get("href", "")}
                for r in results
            ]
    except Exception as e:
        logger.warning("Web search failed: %s", e)
        return []


_SYSTEM_PROMPT = """You are a misinformation analyst. Your job is to:
1. Classify the given claim or text as either REAL (factual, verifiable) or FAKE (misinformation, false, misleading).
2. Provide a clear, concise explanation with reasoning.
3. If internet article snippets are provided below, reference them in your reasoning to support your classification. Cite specific findings from the snippets when relevant.
4. Respond in this exact JSON format only, no other text:
{"prediction":"REAL" or "FAKE","confidence":0.0 to 1.0,"explanation":"Your detailed reasoning here, referencing articles when available."}
"""


async def analyze_with_ollama(
    text: str,
    use_web_search: bool = True,
    max_search_results: int = 5,
) -> Dict[str, Any]:
    """
    Analyze text for misinformation using Ollama and optional web search.

    Returns:
        {"prediction": "REAL"|"FAKE", "confidence": float, "explanation": str}
    """
    from api.production_pipeline.middlewares.ollama_blackbox import ollama_chat
    from api.production_pipeline.middlewares.ollama_blackbox import OllamaBlackboxError

    search_snippets: List[Dict[str, str]] = []
    if use_web_search and text.strip():
        loop = asyncio.get_event_loop()
        search_snippets = await loop.run_in_executor(
            None, lambda: _search_web(text.strip()[:300], max_results=max_search_results)
        )

    articles_section = ""
    if search_snippets:
        articles_section = "\n\nRelevant internet articles (use these to support your reasoning):\n"
        for i, r in enumerate(search_snippets[:5], 1):
            articles_section += f"\n[{i}] {r.get('title', '')}\n  {r.get('snippet', '')[:200]}...\n  Source: {r.get('href', '')}\n"

    user_content = f"Analyze this claim/text for misinformation:\n\n\"{text.strip()}\"{articles_section}\n\nRespond with JSON only."
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        result = await ollama_chat(
            messages=messages,
            temperature=0.2,
            num_predict=512,
            stop=[],  # allow full JSON output
        )
        content = (result.get("content") or "").strip()
        if not content:
            raise ValueError("Ollama returned empty response")

        # Extract JSON from response (handle markdown code blocks)
        json_str = content
        if "```json" in content:
            m = re.search(r"```json\s*([\s\S]*?)```", content)
            if m:
                json_str = m.group(1).strip()
        elif "```" in content:
            m = re.search(r"```\s*([\s\S]*?)```", content)
            if m:
                json_str = m.group(1).strip()
        else:
            # Try to find JSON object
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = content[start:end]

        data = json.loads(json_str)
        prediction = str(data.get("prediction", "REAL")).strip().upper()
        if prediction not in ("REAL", "FAKE"):
            prediction = "FAKE" if "fake" in prediction.lower() or "misinformation" in prediction.lower() else "REAL"
        confidence = float(data.get("confidence", 0.7))
        confidence = max(0.0, min(1.0, confidence))
        explanation = str(data.get("explanation", "No explanation provided.")).strip()
        if not explanation:
            explanation = f"Classified as {prediction} based on analysis."

        return {
            "prediction": prediction,
            "confidence": confidence,
            "explanation": explanation,
        }
    except OllamaBlackboxError as e:
        logger.warning("Ollama unavailable for analyze: %s", e.message)
        raise
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse Ollama response: %s", e)
        raise ValueError("Analysis failed: could not parse response") from e
