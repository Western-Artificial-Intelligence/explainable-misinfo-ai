# verifier_pipeline/generator/mode.py
from __future__ import annotations
from typing import Optional

LABEL_FALSE = 0
LABEL_MIXED = 1
LABEL_TRUE = 2
LABEL_NEI = 3

def choose_mode(label_id: Optional[int], contradiction: bool) -> str:
    if label_id in (LABEL_FALSE, LABEL_MIXED):
        return "WHY_FALSE"
    if label_id == LABEL_TRUE:
        return "WHAT_TRUE"
    # NEI or unknown -> explanation mode (don’t “correct” without support)
    return "WHY_FALSE"

