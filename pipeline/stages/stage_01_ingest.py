"""Stage 1: Ingest and normalize user claim."""
import re
from datetime import datetime, timezone


def collapse_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def run(state: dict) -> dict:
    user_claim = state.get("user_claim", "").strip()
    if not user_claim:
        raise ValueError("INVALID_CLAIM_TEXT")

    raw = user_claim
    normalized = raw
    try:
        import unicodedata
        normalized = unicodedata.normalize("NFKC", raw)
    except Exception:
        pass
    normalized = collapse_whitespace(normalized)
    if not normalized:
        raise ValueError("INVALID_CLAIM_TEXT")

    warnings = []
    if raw != normalized:
        warnings.append({"code": "TEXT_NORMALIZED", "message": "Text was normalized"})
    if len(normalized) > 2000:
        warnings.append({"code": "VERY_LONG_CLAIM", "message": "Claim is very long"})

    state["normalized_claim"] = normalized
    state["meta"] = state.get("meta", {}) | {
        "received_at": datetime.now(timezone.utc).isoformat(),
        "version": state.get("PIPELINE_VERSION", "0.1.0"),
        "warnings": warnings if warnings else None,
    }
    return state
