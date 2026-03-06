from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from ..utils.history_store import (
    MAX_HISTORY_ENTRIES,
    add_history_entry,
    clear_history_entries,
    delete_history_entry,
    get_store_mode,
    list_history_entries,
)

router = APIRouter(prefix="/history", tags=["history"])
logger = logging.getLogger(__name__)


class HistoryEntryIn(BaseModel):
    id: str | None = None
    text: str = Field(min_length=1, max_length=5000)
    source: str = Field(default="Pasted text", max_length=400)
    prediction: str
    confidence: float
    detail: str = Field(default="", max_length=5000)
    model: str = "roberta"
    modelLabel: str = "RoBERTa"
    timestamp: int | None = None


class HistoryEntryOut(BaseModel):
    id: str
    text: str
    source: str
    prediction: Literal["TRUE", "FALSE", "MIXED"]
    confidence: float
    detail: str
    model: str
    modelLabel: str
    timestamp: int


class HistoryListResponse(BaseModel):
    entries: list[HistoryEntryOut]
    mode: Literal["mongodb", "memory"]


@router.get("", response_model=HistoryListResponse)
def get_history(
    limit: int = Query(default=MAX_HISTORY_ENTRIES, ge=1, le=MAX_HISTORY_ENTRIES),
    include_central: bool = Query(default=False),
):
    entries = list_history_entries(limit=limit) if include_central else []
    return {"entries": entries, "mode": get_store_mode()}


@router.post("", response_model=HistoryEntryOut)
def create_history_entry(payload: HistoryEntryIn):
    if hasattr(payload, "model_dump"):
        data = payload.model_dump(exclude_none=True)
    else:
        data = payload.dict(exclude_none=True)
    saved = add_history_entry(data)
    preview = saved.get("text", "").replace("\n", " ").strip()[:120]
    logger.info("📝 New history entry saved locally: %s", preview)

    if get_store_mode() == "mongodb":
        logger.info("📝 New history entry saved in MongoDB: %s", saved.get("id"))

    return saved


@router.delete("/{entry_id}")
def remove_history_entry(entry_id: str):
    deleted = delete_history_entry(entry_id)
    return {"ok": deleted}


@router.delete("")
def remove_all_history_entries():
    deleted_count = clear_history_entries()
    return {"ok": True, "deletedCount": deleted_count}
