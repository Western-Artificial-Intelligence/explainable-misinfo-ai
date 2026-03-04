from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

PIPELINE_VERSION = os.getenv("PIPELINE_VERSION", "dev")
CLAIM_NORMALIZATION_VERSION = os.getenv("CLAIM_NORMALIZATION_VERSION", "norm_v1")
VERY_LONG_CLAIM_CHARS = 5000

_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")


class IngestClaimError(Exception):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def _now_iso8601_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_claim(raw: str) -> str:
    normalized = unicodedata.normalize("NFKC", raw)
    normalized = _ZERO_WIDTH_RE.sub("", normalized)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.strip()
    normalized = " ".join(normalized.split())
    return normalized


def _make_claim_id_from_normalized(normalized_claim: str, *, norm_version: str) -> str:
    payload = f"{norm_version}\n{normalized_claim}".encode("utf-8")
    return f"claim_{hashlib.sha256(payload).hexdigest()}"


def run_1_ingest_claim(step_input: Dict[str, Any], *, request_id: str, claim_id: str) -> Dict[str, Any]:
    if not isinstance(step_input, dict) or "user_claim" not in step_input:
        raise IngestClaimError(code="INVALID_CLAIM_TEXT", message="Missing user_claim.")

    user_claim = step_input.get("user_claim")
    if not isinstance(user_claim, str):
        raise IngestClaimError(code="INVALID_CLAIM_TEXT", message="user_claim must be a string.")
    if user_claim.strip() == "":
        raise IngestClaimError(code="INVALID_CLAIM_TEXT", message="Claim text is empty or whitespace-only.")

    raw = user_claim
    normalized = _normalize_claim(raw)

    warnings: List[Dict[str, str]] = []
    if raw != normalized:
        warnings.append({"code": "TEXT_NORMALIZED", "message": "Raw claim differed from normalized claim."})
    if len(normalized) >= VERY_LONG_CLAIM_CHARS:
        warnings.append({"code": "VERY_LONG_CLAIM", "message": f"Claim is unusually long ({len(normalized)} chars)."})

    out: Dict[str, Any] = {
        "request_id": request_id,
        "claim_id": claim_id,
        "user_claim": raw,
        "normalized_claim": normalized,
        "meta": {
            "received_at": _now_iso8601_utc(),
            "version": PIPELINE_VERSION,
            "normalization_version": CLAIM_NORMALIZATION_VERSION,
        },
    }
    if warnings:
        out["meta"]["warnings"] = warnings
    return out


def process_user_claim(user_claim: Any) -> Dict[str, Any]:
    # Keep this wrapper thin — let the step do validation
    request_id = str(uuid4())
    if not isinstance(user_claim, str):
        raise IngestClaimError(code="INVALID_CLAIM_TEXT", message="user_claim must be a string.")

    normalized_for_id = _normalize_claim(user_claim)
    claim_id = _make_claim_id_from_normalized(
        normalized_for_id,
        norm_version=CLAIM_NORMALIZATION_VERSION,
    )
    return run_1_ingest_claim({"user_claim": user_claim}, request_id=request_id, claim_id=claim_id)