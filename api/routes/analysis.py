from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from ..utils.analysis_store import add_analysis_record, get_analysis_store_mode, list_analysis_records

router = APIRouter(prefix="/analysis", tags=["analysis"])
logger = logging.getLogger(__name__)


class AnalysisRecordIn(BaseModel):
    id: str | None = None
    session_id: str | None = None
    input_type: str | None = None
    input_text: str | None = None
    transcript: str | None = None
    page_url: str | None = None
    analysis_result: dict[str, Any] | None = None
    confidence: float | int | None = None
    reasoning: str | None = None
    verdict: str | None = None
    timestamp: int | None = None
    source_context: str | None = None


@router.post("")
def create_analysis_record(payload: AnalysisRecordIn):
    if hasattr(payload, "model_dump"):
        data = payload.model_dump(exclude_none=True)
    else:
        data = payload.dict(exclude_none=True)
    saved = add_analysis_record(data)
    logger.info(
        "🧠 Analysis saved (%s): session=%s input_type=%s",
        get_analysis_store_mode(),
        saved.get("session_id"),
        saved.get("input_type"),
    )
    return {"ok": True, "mode": get_analysis_store_mode(), "record": saved}


@router.get("")
def get_analysis_records(limit: int = 100):
    return {
        "mode": get_analysis_store_mode(),
        "records": list_analysis_records(limit=limit),
    }
