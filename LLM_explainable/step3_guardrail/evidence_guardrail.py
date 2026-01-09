# verifier_pipeline/guardrail/evidence_guardrail.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import re

def _tok(s: str) -> List[str]:
    s = (s or "").casefold()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    toks = [t for t in s.split(" ") if len(t) > 2]
    return toks

STOPWORDS = {
    "the","and","for","that","with","this","from","are","was","were","has","have","had",
    "but","not","you","your","their","they","them","his","her","she","him","its","into",
    "about","over","under","after","before","than","then","also","when","where","what",
    "who","whom","which","why","how","can","could","should","would","will","just",
}

def _kw_set(s: str) -> set[str]:
    return {t for t in _tok(s) if t not in STOPWORDS}

@dataclass
class GuardrailDecision:
    status: str  # "OK" | "INSUFFICIENT_EVIDENCE"
    reason: str
    proceed_to_generator: bool
    contradiction: bool
    metrics: Dict[str, Any]
    used_eids: List[str]

def _distinct_sources(snippets: List[Dict[str, Any]]) -> int:
    return len({(s.get("url") or "").strip() for s in snippets if (s.get("url") or "").strip()})

def _keyword_coverage(claim: str, snippet_texts: List[str]) -> float:
    """
    Fraction of claim keywords that appear in the union of snippet keywords.
    """
    ck = _kw_set(claim)
    if not ck:
        return 0.0
    union = set()
    for t in snippet_texts:
        union |= _kw_set(t)
    return len(ck & union) / max(1, len(ck))

NEG_CUES = {"not", "no", "never", "false", "hoax", "debunk", "deny", "denies", "denied"}

def _negation_present(text: str) -> bool:
    toks = _tok(text)
    return any(t in NEG_CUES for t in toks)

def _contradiction_heuristic(claim: str, snippets: List[Dict[str, Any]]) -> bool:
    """
    Very light heuristic:
      - If we have multiple sources AND
      - some snippets look negating (debunk/false/not) and others don't
    """
    if _distinct_sources(snippets) < 2:
        return False
    neg = 0
    pos = 0
    for s in snippets:
        txt = (s.get("text") or "")
        if not txt:
            continue
        if _negation_present(txt):
            neg += 1
        else:
            pos += 1
    return neg > 0 and pos > 0

def evidence_guardrail(
    claim: str,
    snippets: List[Dict[str, Any]],
    *,
    min_snippets: int = 3,
    min_sources: int = 2,
    min_keyword_coverage: float = 0.35,
    use_top_n: int = 8,
) -> GuardrailDecision:
    """
    Input: claim + Step2 snippets list (each has eid,url,title,text,score)
    Output: decision whether evidence is strong enough to proceed.
    """
    claim = (claim or "").strip()
    snippets = (snippets or [])[:use_top_n]

    used_eids = [str(s.get("eid")) for s in snippets if s.get("eid")]

    if not claim or len(claim) < 4:
        return GuardrailDecision(
            status="INSUFFICIENT_EVIDENCE",
            reason="empty_or_too_short_claim",
            proceed_to_generator=False,
            contradiction=False,
            metrics={"claim_len": len(claim)},
            used_eids=used_eids,
        )

    num_snips = len([s for s in snippets if (s.get("text") or "").strip()])
    num_sources = _distinct_sources(snippets)
    coverage = _keyword_coverage(claim, [(s.get("text") or "") for s in snippets])

    contradiction = _contradiction_heuristic(claim, snippets)

    enough = (num_snips >= min_snippets) and (num_sources >= min_sources) and (coverage >= min_keyword_coverage)

    if not enough:
        reason_parts = []
        if num_snips < min_snippets:
            reason_parts.append(f"too_few_snippets({num_snips}<{min_snippets})")
        if num_sources < min_sources:
            reason_parts.append(f"too_few_sources({num_sources}<{min_sources})")
        if coverage < min_keyword_coverage:
            reason_parts.append(f"low_keyword_coverage({coverage:.2f}<{min_keyword_coverage:.2f})")

        return GuardrailDecision(
            status="INSUFFICIENT_EVIDENCE",
            reason=";".join(reason_parts) if reason_parts else "weak_evidence",
            proceed_to_generator=False,   # by your diagram, you stop here
            contradiction=contradiction,
            metrics={
                "num_snippets": num_snips,
                "num_sources": num_sources,
                "keyword_coverage": coverage,
                "min_snippets": min_snippets,
                "min_sources": min_sources,
                "min_keyword_coverage": min_keyword_coverage,
            },
            used_eids=used_eids,
        )

    return GuardrailDecision(
        status="OK",
        reason="evidence_sufficient",
        proceed_to_generator=True,
        contradiction=contradiction,
        metrics={
            "num_snippets": num_snips,
            "num_sources": num_sources,
            "keyword_coverage": coverage,
        },
        used_eids=used_eids,
    )

