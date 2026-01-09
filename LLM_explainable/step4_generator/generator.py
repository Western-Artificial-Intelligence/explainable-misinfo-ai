from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import json
import urllib.request
import urllib.error
from .json_utils import extract_json

from .mode import choose_mode
from .prompts import build_prompt


@dataclass
class GeneratorOutput:
    mode: str
    content: Dict[str, Any]  # parsed JSON from model


class GeneratorLLM:
    """
    Generator LLM used ONLY for explanation/correction grounded in evidence.

    Default provider: Ollama local server (http://localhost:11434).
    """

    def __init__(self, model_name: str, *, ollama_url: str = "http://localhost:11434/api/generate", timeout_s: int = 45):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.timeout_s = timeout_s

    def call_llm(self, prompt: str) -> str:
        """
        Calls Ollama and returns a JSON string (no extra text).
        Requires Ollama running locally.

        We set:
        - format="json" to force valid JSON
        - temperature=0 for determinism
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
            },
        }

        req = urllib.request.Request(
            self.ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            raise RuntimeError(f"Ollama HTTPError {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                "Failed to reach Ollama. Is it running? "
                "Start it and ensure the URL is correct (default http://localhost:11434/api/generate)."
            ) from e

        obj = json.loads(data)

        # Ollama returns JSON like: {"response":"...","done":true,...}
        raw = obj.get("response")
        if not isinstance(raw, str) or not raw.strip():
            raise RuntimeError(f"Ollama returned empty response: keys={list(obj.keys())}")

        return raw.strip()

    def generate(
        self,
        claim: str,
        snippets: List[Dict[str, Any]],
        *,
        verifier_label_id: Optional[int] = None,
        contradiction: bool = False,
    ) -> GeneratorOutput:
        mode = choose_mode(verifier_label_id, contradiction)
        prompt = build_prompt(mode, claim, snippets)
        raw = self.call_llm(prompt)

        # Strict JSON parse (fail fast)
        content = extract_json(raw)
        return GeneratorOutput(mode=mode, content=content)
