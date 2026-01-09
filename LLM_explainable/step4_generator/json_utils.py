# verifier_pipeline/generator/json_utils.py
from __future__ import annotations
import json
import re
from typing import Any, Dict

def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Extract first {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output")
    return json.loads(m.group(0))
