from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

MAX_HISTORY_ENTRIES = 100

_memory_lock = Lock()
_memory_entries: list[dict[str, Any]] = []

_mongo_lock = Lock()
_mongo_collection = None
_mongo_ready = False


def _now_ms() -> int:
    from time import time

    return int(time() * 1000)


def _normalize_prediction(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if raw in {"TRUE", "FALSE", "MIXED"}:
        return raw
    if raw in {"REAL", "FACTUAL", "RELIABLE"}:
        return "TRUE"
    if raw in {"FAKE", "MISINFORMATION", "NOT_REAL"}:
        return "FALSE"
    return "MIXED"


def _normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence <= 1:
        confidence *= 100
    return round(max(0.0, min(100.0, confidence)), 2)


def _normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(entry.get("id") or uuid4()),
        "text": str(entry.get("text") or "")[:5000],
        "source": str(entry.get("source") or "Pasted text")[:400],
        "prediction": _normalize_prediction(entry.get("prediction")),
        "confidence": _normalize_confidence(entry.get("confidence")),
        "detail": str(entry.get("detail") or "")[:5000],
        "model": str(entry.get("model") or "roberta"),
        "modelLabel": str(entry.get("modelLabel") or "RoBERTa"),
        "timestamp": int(entry.get("timestamp") or _now_ms()),
    }


def _get_mongo_collection():
    global _mongo_collection, _mongo_ready

    if _mongo_ready:
        return _mongo_collection

    uri = (os.getenv("MONGO_URI") or os.getenv("MONGODB_URI") or "").strip()
    if not uri:
        logger.warning("❌ MongoDB connection failed, using local history only")
        logger.info("MongoDB is optional: set MONGO_URI to enable central history logging.")
        _mongo_ready = True
        return None

    logger.info("🔌 Attempting MongoDB connection...")

    with _mongo_lock:
        if _mongo_ready:
            return _mongo_collection

        try:
            from pymongo import DESCENDING, MongoClient

            timeout_ms = int(os.getenv("MONGODB_CONNECT_TIMEOUT_MS", "5000"))
            client_kwargs: dict[str, Any] = {"serverSelectionTimeoutMS": timeout_ms}

            # Atlas TLS verification can fail on some local Python installs if
            # system certs are not configured. Use certifi bundle when available.
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
            collection_name = os.getenv("MONGODB_HISTORY_COLLECTION", "history")
            collection = client[db_name][collection_name]
            collection.create_index("id", unique=True)
            collection.create_index([("timestamp", DESCENDING)])

            _mongo_collection = collection
            logger.info("✅ MongoDB connected successfully")
            logger.info("MongoDB target: %s.%s", db_name, collection_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("❌ MongoDB connection failed, using local history only")
            logger.exception("MongoDB connection error: %s", exc)
            _mongo_collection = None
        finally:
            _mongo_ready = True

    return _mongo_collection


def get_store_mode() -> str:
    return "mongodb" if _get_mongo_collection() is not None else "memory"


def list_history_entries(limit: int = MAX_HISTORY_ENTRIES) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), MAX_HISTORY_ENTRIES))
    collection = _get_mongo_collection()

    if collection is not None:
        docs = (
            collection.find({}, {"_id": 0})
            .sort("timestamp", -1)
            .limit(limit)
        )
        return [_normalize_entry(doc) for doc in docs]

    with _memory_lock:
        sorted_entries = sorted(_memory_entries, key=lambda e: int(e.get("timestamp", 0)), reverse=True)
        return [_normalize_entry(entry) for entry in sorted_entries[:limit]]


def add_history_entry(entry: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_entry(entry)
    collection = _get_mongo_collection()

    if collection is not None:
        collection.update_one({"id": normalized["id"]}, {"$set": normalized}, upsert=True)
        return normalized

    with _memory_lock:
        existing_index = next((i for i, item in enumerate(_memory_entries) if item.get("id") == normalized["id"]), None)
        if existing_index is not None:
            _memory_entries.pop(existing_index)
        _memory_entries.append(normalized)
        _memory_entries.sort(key=lambda e: int(e.get("timestamp", 0)), reverse=True)
        del _memory_entries[MAX_HISTORY_ENTRIES:]
    return normalized


def delete_history_entry(entry_id: str) -> bool:
    safe_id = str(entry_id or "").strip()
    if not safe_id:
        return False

    collection = _get_mongo_collection()
    if collection is not None:
        return bool(collection.delete_one({"id": safe_id}).deleted_count)

    with _memory_lock:
        before = len(_memory_entries)
        _memory_entries[:] = [entry for entry in _memory_entries if entry.get("id") != safe_id]
        return len(_memory_entries) < before


def clear_history_entries() -> int:
    collection = _get_mongo_collection()
    if collection is not None:
        result = collection.delete_many({})
        return int(result.deleted_count)

    with _memory_lock:
        count = len(_memory_entries)
        _memory_entries.clear()
        return count
