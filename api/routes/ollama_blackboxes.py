import os
import requests

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..utils.cache import add_to_cache, get_from_cache

router = APIRouter()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")


class OllamaInput(BaseModel):
    system_instruction: str
    input_text: str


class OllamaOutput(BaseModel):
    result: str


def ollama_logic(system_instruction: str, input_text: str) -> str:
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": f"{system_instruction}\n\n{input_text}",
            "stream": False,
        },
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()

    # generate response returns {"response": "...", ...}
    content = data.get("response")
    if not content:
        raise HTTPException(status_code=502, detail=f"Unexpected Ollama response: {data}")

    return content


@router.post("/llm_response", response_model=OllamaOutput)
def llm_response(payload: OllamaInput):
    # Cache key: include system + input so different system prompts don't collide
    cache_key = f"SYS:{payload.system_instruction}\nUSER:{payload.input_text}"

    cached = get_from_cache(cache_key)
    if cached:
        return cached

    try:
        result_text = ollama_logic(payload.system_instruction, payload.input_text)
        result = {"result": result_text}
        add_to_cache(cache_key, result)
        return result

    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Could not connect to Ollama (is it running on :11434?)")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Ollama request timed out")
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Ollama HTTP error: {str(e)}")