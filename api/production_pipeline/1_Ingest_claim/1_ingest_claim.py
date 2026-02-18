# 1_ingest_claim.py
from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

# PIPELINE_VERSION required by output schema meta.version
PIPELINE_VERSION = os.getenv("PIPELINE_VERSION", "dev")

# Non-fatal warning threshold (chars). Not specified by schema; choose a stable constant.
VERY_LONG_CLAIM_CHARS = 5000

# Optional normalization helpers
_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")


class IngestClaimError(Exception):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def adapt_user_text_to_input_schema(user_text: Any) -> Dict[str, Any]:
    """
    Refactor external input 'user_text' -> step input schema:
      INPUT_SCHEMA = { "user_claim": <string> }
    """
    return {"user_claim": user_text}


def _now_iso8601_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_claim(raw: str) -> str:
    # 2) Normalize claim text (per pseudo)
    normalized = unicodedata.normalize("NFKC", raw)
    normalized = _ZERO_WIDTH_RE.sub("", normalized)  # optional
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")  # optional
    normalized = normalized.strip()
    normalized = " ".join(normalized.split())  # collapse internal whitespace
    return normalized


def run_1_ingest_claim(
    step_input: Dict[str, Any],
    *,
    request_id: str,
    claim_id: str,
) -> Dict[str, Any]:
    """
    Implements 1_Ingest_claim exactly per pseudo.
    request_id and claim_id are assumed to be assigned BEFORE this step runs.
    """

    # 1) Validate input
    if not isinstance(step_input, dict) or "user_claim" not in step_input:
        raise IngestClaimError(code="INVALID_CLAIM_TEXT", message="Missing user_claim.")

    user_claim = step_input.get("user_claim")
    if not isinstance(user_claim, str):
        raise IngestClaimError(code="INVALID_CLAIM_TEXT", message="user_claim must be a string.")

    if user_claim.strip() == "":
        raise IngestClaimError(code="INVALID_CLAIM_TEXT", message="Claim text is empty or whitespace-only.")

    # 2) Normalize claim text
    raw = user_claim
    normalized = _normalize_claim(raw)

    # 3) Build warnings (non-fatal) — deterministic order
    warnings: List[Dict[str, str]] = []
    if raw != normalized:
        warnings.append({"code": "TEXT_NORMALIZED", "message": "Raw claim differed from normalized claim."})
    if len(normalized) >= VERY_LONG_CLAIM_CHARS:
        warnings.append({"code": "VERY_LONG_CLAIM", "message": f"Claim is unusually long ({len(normalized)} chars)."})

    # 4) Emit output object (must match output_schema additionalProperties=false)
    out: Dict[str, Any] = {
        "request_id": request_id,
        "claim_id": claim_id,
        "user_claim": raw,
        "normalized_claim": normalized,
        "meta": {
            "received_at": _now_iso8601_utc(),
            "version": PIPELINE_VERSION,
        },
    }

    if warnings:
        out["meta"]["warnings"] = warnings

    return out


def ingest_from_user_text(user_text: Any, *, request_id: str, claim_id: str) -> Dict[str, Any]:
    """
    Convenience wrapper:
      user_text -> input_schema -> run step -> output_schema
    """
    step_input = adapt_user_text_to_input_schema(user_text)
    return run_1_ingest_claim(step_input, request_id=request_id, claim_id=claim_id)


# -------------------------------------------------------------------
# Endpoint-friendly wrapper (keeps api/routes/production.py minimal)
# -------------------------------------------------------------------
def _make_claim_id_from_normalized(normalized_claim: str) -> str:
    h = hashlib.sha256(normalized_claim.encode("utf-8")).hexdigest()
    return f"claim_{h}"


def process_user_claim(user_claim: Any) -> Dict[str, Any]:
    """
    Single entrypoint for /process:
    - validates user_claim
    - assigns request_id + stable claim_id (sha256(normalized_claim))
    - runs step and returns output_schema dict
    """
    if not isinstance(user_claim, str):
        raise IngestClaimError(code="INVALID_CLAIM_TEXT", message="user_claim must be a string.")
    if user_claim.strip() == "":
        raise IngestClaimError(code="INVALID_CLAIM_TEXT", message="Claim text is empty or whitespace-only.")

    request_id = str(uuid4())
    normalized_for_id = _normalize_claim(user_claim)
    claim_id = _make_claim_id_from_normalized(normalized_for_id)

    return ingest_from_user_text(user_claim, request_id=request_id, claim_id=claim_id)