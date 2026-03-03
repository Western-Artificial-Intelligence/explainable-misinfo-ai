# llm_blackbox.py
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx


class LLMBlackboxError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


def _load_dotenv_if_available() -> None:
    # Optional: supports local ".env" without forcing a dependency
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(override=False)
    except Exception:
        pass


def _get_timeout_s() -> Optional[float]:
    raw = os.getenv("OLLAMA_TIMEOUT_S", "60").strip().lower()
    if raw in ("0", "none", "null", "inf", "infinite", "unlimited"):
        return None
    try:
        return float(raw)
    except Exception:
        return 60.0


def _get_base_url() -> str:
    # Common env names people use for Ollama
    url = (
        os.getenv("OLLAMA_BASE_URL")
        or os.getenv("OLLAMA_HOST")
        or "http://localhost:11434"
    ).strip()
    return url.rstrip("/")


def _get_model() -> str:
    return (os.getenv("LLM_MODEL") or "gpt-oss:20b-cloud").strip()


def _auth_headers_for_base_url(base_url: str) -> Dict[str, str]:
    """
    If you set base_url to https://ollama.com, you typically need:
      export OLLAMA_API_KEY=...
    and send Authorization: Bearer <key>.
    """
    api_key = (os.getenv("OLLAMA_API_KEY") or "").strip()
    if not api_key:
        return {}
    # Works for both remote ollama.com and any custom reverse proxy that expects bearer auth.
    return {"Authorization": f"Bearer {api_key}"}


def _extract_text(resp_json: Dict[str, Any]) -> str:
    # /api/chat (non-stream) typically returns {"message": {"content": "..."}}
    msg = resp_json.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, str):
            return content

    # Some setups might return {"response": "..."} (e.g., /api/generate)
    content2 = resp_json.get("response")
    if isinstance(content2, str):
        return content2

    raise LLMBlackboxError(
        code="bad_response_shape",
        message="Unexpected Ollama response shape (missing assistant content).",
        details={"keys": list(resp_json.keys())[:50]},
    )


@dataclass(frozen=True)
class LLMBlackbox:
    """
    Minimal blackbox wrapper around Ollama /api/chat.

    Defaults:
      - model: gpt-oss:20b-cloud
      - base_url: http://localhost:11434

    Notes:
      - If you point base_url to https://ollama.com (Cloud API access), you usually want
        model="gpt-oss:20b" (not the *-cloud tag) and must set OLLAMA_API_KEY.
    """

    model: str = _get_model()
    base_url: str = _get_base_url()
    timeout_s: Optional[float] = _get_timeout_s()
    max_retries: int = 3

    def generate(
        self,
        system_context: str,
        user_context: str,
        *,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> str:
        """
        Returns a single assistant string.
        """
        _load_dotenv_if_available()

        if system_context is None or user_context is None:
            raise ValueError("system_context and user_context must be provided (non-None).")

        url = f"{self.base_url}/api/chat"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "llm_blackbox/1.0",
            **_auth_headers_for_base_url(self.base_url),
        }

        # Some models/providers might ignore 'system' role; but it's still the standard structure.
        messages = [
            {"role": "system", "content": str(system_context)},
            {"role": "user", "content": str(user_context)},
        ]

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        options: Dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = float(temperature)
        if num_predict is not None:
            options["num_predict"] = int(num_predict)
        if options:
            payload["options"] = options

        last_err: Optional[Exception] = None

        timeout = None if self.timeout_s is None else httpx.Timeout(self.timeout_s)
        with httpx.Client(timeout=timeout) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    r = client.post(url, headers=headers, content=json.dumps(payload))
                    if r.status_code >= 400:
                        # Retry on common transient codes
                        if r.status_code in (408, 409, 429) or 500 <= r.status_code <= 599:
                            raise LLMBlackboxError(
                                code="http_error_retryable",
                                message=f"Ollama HTTP {r.status_code}",
                                status_code=r.status_code,
                                details={"body": _safe_text(r)},
                            )
                        raise LLMBlackboxError(
                            code="http_error",
                            message=f"Ollama HTTP {r.status_code}",
                            status_code=r.status_code,
                            details={"body": _safe_text(r)},
                        )

                    data = r.json()
                    return _extract_text(data).strip()

                except (httpx.TimeoutException, httpx.TransportError, LLMBlackboxError) as e:
                    last_err = e
                    if attempt >= self.max_retries:
                        break
                    # exponential backoff + jitter
                    sleep_s = min(8.0, (2.0 ** attempt) * 0.5) + random.uniform(0.0, 0.25)
                    time.sleep(sleep_s)

        # If we got here, we failed all retries
        if isinstance(last_err, LLMBlackboxError):
            raise last_err
        raise LLMBlackboxError(
            code="request_failed",
            message="Request failed after retries.",
            details={"error": repr(last_err)},
        )

    async def generate_async(
        self,
        system_context: str,
        user_context: str,
        *,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> str:
        """
        Async version. Returns a single assistant string.
        """
        _load_dotenv_if_available()

        if system_context is None or user_context is None:
            raise ValueError("system_context and user_context must be provided (non-None).")

        url = f"{self.base_url}/api/chat"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "llm_blackbox/1.0",
            **_auth_headers_for_base_url(self.base_url),
        }

        messages = [
            {"role": "system", "content": str(system_context)},
            {"role": "user", "content": str(user_context)},
        ]

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        options: Dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = float(temperature)
        if num_predict is not None:
            options["num_predict"] = int(num_predict)
        if options:
            payload["options"] = options

        last_err: Optional[Exception] = None

        timeout = None if self.timeout_s is None else httpx.Timeout(self.timeout_s)
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    r = await client.post(url, headers=headers, content=json.dumps(payload))
                    if r.status_code >= 400:
                        if r.status_code in (408, 409, 429) or 500 <= r.status_code <= 599:
                            raise LLMBlackboxError(
                                code="http_error_retryable",
                                message=f"Ollama HTTP {r.status_code}",
                                status_code=r.status_code,
                                details={"body": _safe_text(r)},
                            )
                        raise LLMBlackboxError(
                            code="http_error",
                            message=f"Ollama HTTP {r.status_code}",
                            status_code=r.status_code,
                            details={"body": _safe_text(r)},
                        )

                    data = r.json()
                    return _extract_text(data).strip()

                except (httpx.TimeoutException, httpx.TransportError, LLMBlackboxError) as e:
                    last_err = e
                    if attempt >= self.max_retries:
                        break
                    sleep_s = min(8.0, (2.0 ** attempt) * 0.5) + random.uniform(0.0, 0.25)
                    await _asleep(sleep_s)

        if isinstance(last_err, LLMBlackboxError):
            raise last_err
        raise LLMBlackboxError(
            code="request_failed",
            message="Request failed after retries.",
            details={"error": repr(last_err)},
        )


def _safe_text(resp: httpx.Response) -> str:
    try:
        return resp.text[:4000]
    except Exception:
        return "<unreadable>"


async def _asleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)