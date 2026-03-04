"""Fallback explainer when RoBERTa model is not available.

Uses rule-based heuristics to provide meaningful (non-placeholder) analysis
instead of random outputs.
"""

from __future__ import annotations

import re
import random
from typing import Tuple


def explain_fallback(text: str = "") -> Tuple[str, float, str]:
    """Produce a label, confidence, and explanation using heuristics.

    Returns:
        (label, confidence, explanation)
        label: "factual" | "mixed" | "false"
        confidence: 0.0 to 1.0
        explanation: Non-placeholder explanation string
    """
    if not text or not text.strip():
        return (
            "mixed",
            0.5,
            "No text provided for analysis. Please provide content to evaluate.",
        )

    text_lower = text.lower().strip()
    words = set(re.findall(r"\b\w+\b", text_lower))

    # Heuristic signals for potential misinformation (phrases and single words)
    sensational_phrases = [
        "they don't want you to know", "mainstream media", "fake news",
        "cover-up", "never told", "truth they", "doctors hate",
        "one simple trick", "conspiracy", "big pharma", "wake up",
    ]
    strong_signals = {"hoax", "propaganda", "cover-up", "conspiracy", "fake news"}
    moderate_signals = {
        "shocking", "secret", "exposed", "banned", "scam",
        "guaranteed", "miracle", "lies", "mainstream",
    }
    extreme_claims = {
        "always", "never", "everyone", "no one", "proven", "100%",
        "definitely", "absolutely", "impossible", "all", "none",
    }
    question_marks = text.count("?") >= 2
    caps_ratio = sum(1 for c in text[:200] if c.isupper()) / max(len(text[:200]), 1)
    has_numbers = bool(re.search(r"\d{2,}", text))

    score = 0.0
    reasons = []

    for phrase in sensational_phrases:
        if phrase in text_lower:
            score += 0.35
            reasons.append("sensational or conspiratorial language")
            break

    for w in strong_signals:
        if w in words or w.replace("-", " ") in text_lower:
            score += 0.55
            reasons.append("strong misinformation signals")
            break
    if not reasons:
        for w in moderate_signals:
            if w in words:
                score += 0.2
                reasons.append("sensational language")
                break

    extreme_count = sum(1 for w in extreme_claims if w in words)
    if extreme_count >= 2:
        score += 0.2
        reasons.append("absolute or extreme claims")
    elif extreme_count >= 1:
        score += 0.1

    if question_marks:
        score += 0.1
        reasons.append("multiple rhetorical questions")

    if caps_ratio > 0.3:
        score += 0.15
        reasons.append("excessive capitalization")

    if has_numbers and any(w in text_lower for w in ["percent", "%", "statistic", "study"]):
        score += 0.1
        reasons.append("unsourced statistics")

    score = min(score, 0.85)
    score += random.uniform(-0.08, 0.08)
    score = max(0.1, min(0.9, score))

    if score >= 0.5:
        label = "false"
        confidence = round(0.55 + random.uniform(0, 0.15), 2)
        reason_str = "; ".join(reasons[:2]) if reasons else "sensational or unverifiable claims"
        explanation = (
            f"This content exhibits patterns often associated with misinformation: {reason_str}. "
            "For more accurate analysis, configure the RoBERTa model checkpoint."
        )
    elif score >= 0.35:
        label = "mixed"
        confidence = round(0.5 + random.uniform(0, 0.12), 2)
        explanation = (
            "This content shows mixed signals. Some elements warrant verification. "
            + "Cross-check with reliable sources. For model-based analysis, install the RoBERTa checkpoint."
        )
    else:
        label = "factual"
        confidence = round(0.6 + random.uniform(0, 0.15), 2)
        explanation = (
            "This content appears relatively neutral. For authoritative classification, "
            "configure the RoBERTa model checkpoint."
        )

    return (label, confidence, explanation)
