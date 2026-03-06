from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

MAX_ANALYSIS_ENTRIES = 5000

_memory_lock = Lock()
_memory_entries: list[dict[str, Any]] = []

_mongo_lock = Lock()
_mongo_collection = None
_mongo_ready = False


def _now_ms() -> int:
    from time import time

    return int(time() * 1000)


def _normalize_verdict(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if raw in {"TRUE", "FALSE", "MIXED"}:
        return raw
    if raw in {"REAL", "RELIABLE", "FACTUAL"}:
        return "TRUE"
    if raw in {"FAKE", "MISINFORMATION", "NOT_REAL"}:
        return "FALSE"
    return "MIXED"


def _normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence > 1.0:
        confidence = confidence / 100.0
    return round(max(0.0, min(1.0, confidence)), 4)


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    analysis_result = record.get("analysis_result")
    if not isinstance(analysis_result, dict):
        analysis_result = {"raw": analysis_result}

    return {
        "id": str(record.get("id") or uuid4()),
        "session_id": str(record.get("session_id") or "unknown-session")[:200],
        "input_type": str(record.get("input_type") or "unknown")[:80],
        "input_text": str(record.get("input_text") or "")[:50000],
        "transcript": str(record.get("transcript") or "")[:200000],
        "page_url": str(record.get("page_url") or "")[:2000],
        "analysis_result": analysis_result,
        "confidence": _normalize_confidence(record.get("confidence")),
        "reasoning": str(record.get("reasoning") or "")[:50000],
        "verdict": _normalize_verdict(record.get("verdict")),
        "timestamp": int(record.get("timestamp") or _now_ms()),
        "source_context": str(record.get("source_context") or "unknown")[:120],
    }


def _get_mongo_collection():
    global _mongo_collection, _mongo_ready

    if _mongo_ready:
        return _mongo_collection

    uri = (os.getenv("MONGO_URI") or os.getenv("MONGODB_URI") or "").strip()
    if not uri:
        _mongo_ready = True
        return None

    with _mongo_lock:
        if _mongo_ready:
            return _mongo_collection

        try:
            from pymongo import DESCENDING, MongoClient

            timeout_ms = int(os.getenv("MONGODB_CONNECT_TIMEOUT_MS", "5000"))
            client_kwargs: dict[str, Any] = {"serverSelectionTimeoutMS": timeout_ms}
            try:
                import certifi

                client_kwargs["tlsCAFile"] = certifi.where()
            except Exception:  # noqa: BLE001
                logger.warning(
                    "certifi is not available. If Atlas SSL fails, install with: pip install certifi"
                )

            client = MongoClient(uri, **client_kwargs)
            client.admin.command("ping")

            db_name = os.getenv("MONGODB_DB_NAME", "truthlens")
            collection_name = os.getenv("MONGODB_ANALYSIS_COLLECTION", "analysis_records")
            collection = client[db_name][collection_name]
            collection.create_index("id", unique=True)
            collection.create_index([("timestamp", DESCENDING)])
            collection.create_index([("session_id", DESCENDING)])
            collection.create_index([("source_context", DESCENDING)])
            collection.create_index([("input_type", DESCENDING)])

            _mongo_collection = collection
            logger.info("✅ MongoDB analysis store connected: %s.%s", db_name, collection_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "MongoDB analysis store unavailable; falling back to memory (%s)",
                exc,
            )
            _mongo_collection = None
        finally:
            _mongo_ready = True

    return _mongo_collection


def get_analysis_store_mode() -> str:
    return "mongodb" if _get_mongo_collection() is not None else "memory"


def add_analysis_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_record(record)
    collection = _get_mongo_collection()
    if collection is not None:
        collection.update_one({"id": normalized["id"]}, {"$set": normalized}, upsert=True)
        return normalized

    with _memory_lock:
        _memory_entries.append(normalized)
        _memory_entries.sort(key=lambda e: int(e.get("timestamp", 0)), reverse=True)
        del _memory_entries[MAX_ANALYSIS_ENTRIES:]
    return normalized


def list_analysis_records(limit: int = 100) -> list[dict[str, Any]]:
    capped = max(1, min(int(limit), MAX_ANALYSIS_ENTRIES))
    collection = _get_mongo_collection()
    if collection is not None:
        docs = collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(capped)
        return [_normalize_record(doc) for doc in docs]
    with _memory_lock:
        return [_normalize_record(entry) for entry in _memory_entries[:capped]]
