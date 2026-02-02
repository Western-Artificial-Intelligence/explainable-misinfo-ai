from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _safe_json(v: Any) -> Any:
    if is_dataclass(v):
        return asdict(v)
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_safe_json(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _safe_json(val) for k, val in v.items()}
    return repr(v)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Pull "extra" fields from record.__dict__
        for k, v in record.__dict__.items():
            if k.startswith("_"):
                continue
            if k in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
            }:
                continue
            payload[k] = _safe_json(v)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def new_run_id() -> str:
    return uuid.uuid4().hex


def setup_logging(
    *,
    name: str = "LLM_explainable",
    level: str = "INFO",
    log_dir: Optional[str] = None,
    json_logs: bool = True,
) -> logging.Logger:
    """
    Idempotent logger setup (safe to call multiple times).
    Logs to stdout + optional file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if getattr(logger, "_configured", False):
        return logger
    logger._configured = True  # type: ignore[attr-defined]
    logger.propagate = False

    fmt = (
        JsonFormatter()
        if json_logs
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logger.level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
        fh.setLevel(logger.level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def log_event(
    logger: logging.Logger, event: str, *, run_id: Optional[str] = None, **fields: Any
) -> None:
    extra = {"event": event, **{k: _safe_json(v) for k, v in fields.items()}}
    if run_id:
        extra["run_id"] = run_id
    logger.info(event, extra=extra)
