from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class EvidenceSnippet:
    eid: str
    url: str
    title: str
    text: str
    score: float


@dataclass(frozen=True)
class VerifierResult:
    # canonical: false=0, mixed=1, true=2, nei=3
    label_id: int
    label: str
    conf: float
    score: float
    probs: Dict[str, float]


@dataclass(frozen=True)
class GuardrailResult:
    status: str  # "OK" | "INSUFFICIENT_EVIDENCE"
    has_contradiction: bool
    used_eids: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratorResult:
    mode: str  # "WHY_FALSE" | "WHAT_TRUE"
    content: Dict[str, Any]


@dataclass(frozen=True)
class FinalResult:
    claim: str
    label_id: int
    label: str
    conf: float
    score: float
    verdict_status: str  # "OK" | "INSUFFICIENT_EVIDENCE"

    explanation: Optional[str] = None
    correction: Optional[str] = None
    citations: List[str] = field(default_factory=list)

    evidence: List[EvidenceSnippet] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)

    debug: Dict[str, Any] = field(default_factory=dict)
