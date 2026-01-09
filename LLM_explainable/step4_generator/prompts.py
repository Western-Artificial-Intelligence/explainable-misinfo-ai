from __future__ import annotations

from typing import Any, Dict, List
import json


SYSTEM_RULES = (
    "You are a verification assistant.\n"
    "You MUST use only the provided evidence snippets.\n"
    "If evidence is insufficient, set verdict='nei'.\n"
    "If evidence conflicts, set verdict='mixed'.\n"
    "Return ONLY valid JSON. No extra text.\n"
)

WHY_FALSE_SCHEMA: Dict[str, Any] = {
    "verdict": "false|mixed|nei",
    "explanation": "string",
    "citations": ["E1", "E2"],
}

WHAT_TRUE_SCHEMA: Dict[str, Any] = {
    "verdict": "true|mixed|nei",
    "correction": "string",
    "support": "string",
    "citations": ["E1", "E2"],
}


def format_evidence(snippets: List[Dict[str, Any]], max_chars_each: int = 900) -> str:
    lines: List[str] = []
    for s in snippets:
        eid = (s.get("eid") or "").strip()
        title = (s.get("title") or "").strip()
        url = (s.get("url") or "").strip()
        text = (s.get("text") or s.get("snippet") or "").strip()

        if max_chars_each and len(text) > max_chars_each:
            text = text[:max_chars_each].rstrip() + "…"

        lines.append(f"{eid}\nTITLE: {title}\nURL: {url}\nTEXT: {text}\n")
    return "\n".join(lines).strip()


def build_prompt(mode: str, claim: str, snippets: List[Dict[str, Any]]) -> str:
    evidence_block = format_evidence(snippets)

    if mode == "WHY_FALSE":
        schema = WHY_FALSE_SCHEMA
        task = (
            "Task: Explain why the claim is false OR mixed based on evidence.\n"
            "If evidence conflicts, verdict='mixed'. If insufficient, verdict='nei'.\n"
        )
    else:
        schema = WHAT_TRUE_SCHEMA
        task = (
            "Task: Provide the corrected true statement (or best-supported version), using only evidence.\n"
            "If evidence conflicts, verdict='mixed'. If insufficient, verdict='nei'.\n"
        )

    return (
        f"{SYSTEM_RULES}\n"
        f"{task}\n"
        f"CLAIM: {claim}\n\n"
        f"EVIDENCE SNIPPETS:\n{evidence_block}\n\n"
        f"Return JSON matching this schema (keys and types must match):\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )
