# verifier_pipeline/rag/query_builder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import re

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

@dataclass
class BuiltQueries:
    primary: str
    alternates: List[str]

def build_queries(claim: str) -> BuiltQueries:
    claim = _clean(claim)
    primary = f"\"{claim}\""
    alt1 = claim
    alt2 = re.sub(r"[^\w\s]", "", claim)
    alt3 = f"{alt2} fact check"
    return BuiltQueries(primary=primary, alternates=[alt1, alt2, alt3])
