from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv


class OllamaBlackboxError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details


def _get_timeout():
    """
    Read env vars at CALL-TIME (not import-time), so .env can be loaded later.
    Also self-load .env as a fallback since main.py loads it after route imports.
    """
    load_dotenv(override=False)

    raw = os.getenv("OLLAMA_TIMEOUT_S", "60").strip().lower()
    if raw in ("0", "none", "null", "inf", "infinite", "unlimited"):
        return None
    try:
        return float(raw)
    except Exception:
        return 60.0


def _get_base_url() -> str:
    load_dotenv(override=False)
    return os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


def _get_model() -> str:
    load_dotenv(override=False)
    return os.getenv("OLLAMA_MODEL", "qwen3:4b")


async def ollama_chat(
    *,
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    top_p: float = 0.9,
    seed: int = 0,
    num_ctx: int = 4096,
    num_predict: int = 64,  # ✅ cap output length (prevents rambling/timeouts)
) -> Dict[str, Any]:
    """
    Blackbox LLM call. Returns a stable JSON envelope.
    messages: [{"role":"system"|"user"|"assistant","content":"..."}]
    """
    base_url = _get_base_url()
    model = model or _get_model()
    timeout = _get_timeout()

    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "stop": ["\n\n", "\n- ", "\n1.", "\n2.", "\n3."],  # ✅ stop list-y outputs
        },
    }

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=payload)
    except httpx.TimeoutException as e:
        raise OllamaBlackboxError(
            "OLLAMA_TIMEOUT",
            "Ollama request timed out.",
            details={"error": str(e), "timeout": str(timeout)},
        )
    except httpx.RequestError as e:
        raise OllamaBlackboxError("OLLAMA_CONNECTION", "Could not reach Ollama server.", details={"error": str(e)})

    latency_ms = int(round((time.perf_counter() - t0) * 1000))

    if r.status_code != 200:
        raise OllamaBlackboxError(
            "OLLAMA_BAD_STATUS",
            f"Ollama returned HTTP {r.status_code}.",
            status_code=r.status_code,
            details={"body": r.text[:5000]},
        )

    data = r.json()
    content = (data.get("message") or {}).get("content", "")

    return {
        "ok": True,
        "model": model,
        "latency_ms": latency_ms,
        "content": content,
        "raw": data,
    }