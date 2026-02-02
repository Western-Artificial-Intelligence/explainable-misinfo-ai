from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v is not None and v != "" else default


def _env_int(name: str, default: int) -> int:
    v = _env(name)
    return int(v) if v is not None else default


def _env_float(name: str, default: float) -> float:
    v = _env(name)
    return float(v) if v is not None else default


def _env_bool(name: str, default: bool) -> bool:
    v = _env(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class GooglePSEConfig:
    api_key: str
    cx: str
    num_results: int = 5
    safe: str = "active"  # "active" | "off" (PSE supports)
    language_restrict: Optional[str] = None  # e.g., "lang_en"


@dataclass(frozen=True)
class RAGConfig:
    cache_dir: str = ".cache/rag"
    use_cache: bool = True
    top_k_urls: int = 5
    top_k_snippets: int = 8
    chunk_size: int = 900
    chunk_overlap: int = 120
    timeout_s: int = 15
    user_agent: str = "verifier_pipeline/1.0 (+https://example.invalid)"
    allow_domains: Optional[list[str]] = None
    deny_domains: Optional[list[str]] = None


@dataclass(frozen=True)
class GuardrailConfig:
    min_snippets: int = 2
    min_unique_sources: int = 2
    contradiction_threshold: float = 0.55
    support_threshold: float = 0.55


@dataclass(frozen=True)
class GeneratorConfig:
    provider: str = "ollama"  # "ollama" | "llamacpp" | "vllm" | "hf"
    model: str = "llama3.1:8b"
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_s: int = 45

    # Provider endpoints (only used depending on provider)
    ollama_url: str = "http://localhost:11434/api/generate"
    llamacpp_url: str = "http://localhost:8080/completion"
    vllm_url: str = "http://localhost:8000/v1/chat/completions"


@dataclass(frozen=True)
class VerifierConfig:
    model_path: str  # local path or HF id depending on your step1 implementation


@dataclass(frozen=True)
class AppConfig:
    verifier: VerifierConfig
    google_pse: Optional[GooglePSEConfig] = None
    rag: RAGConfig = field(default_factory=RAGConfig)
    guardrail: GuardrailConfig = field(default_factory=GuardrailConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)

    log_dir: str = ".logs"
    log_level: str = "INFO"
    json_logs: bool = True


def load_config_from_env() -> AppConfig:
    """
    Environment-driven config with safe defaults.
    Set these env vars as needed:

    VERIFIER_MODEL_PATH
    GOOGLE_PSE_API_KEY
    GOOGLE_PSE_CX

    RAG_CACHE_DIR, RAG_USE_CACHE, RAG_TOP_K_URLS, RAG_TOP_K_SNIPPETS, RAG_TIMEOUT_S
    GEN_PROVIDER, GEN_MODEL, GEN_TEMPERATURE, GEN_MAX_TOKENS, GEN_TIMEOUT_S
    GEN_OLLAMA_URL, GEN_LLAMACPP_URL, GEN_VLLM_URL

    LOG_DIR, LOG_LEVEL, LOG_JSON
    """
    verifier_model_path = _env("VERIFIER_MODEL_PATH", "").strip()
    if not verifier_model_path:
        raise ValueError("Missing VERIFIER_MODEL_PATH in environment.")

    api_key = _env("GOOGLE_PSE_API_KEY")
    cx = _env("GOOGLE_PSE_CX")

    google_pse = None
    if api_key and cx:
        google_pse = GooglePSEConfig(
            api_key=api_key,
            cx=cx,
            num_results=_env_int("GOOGLE_PSE_NUM_RESULTS", 5),
            safe=_env("GOOGLE_PSE_SAFE", "active") or "active",
            language_restrict=_env("GOOGLE_PSE_LR"),
        )

    rag = RAGConfig(
        cache_dir=_env("RAG_CACHE_DIR", ".cache/rag") or ".cache/rag",
        use_cache=_env_bool("RAG_USE_CACHE", True),
        top_k_urls=_env_int("RAG_TOP_K_URLS", 5),
        top_k_snippets=_env_int("RAG_TOP_K_SNIPPETS", 8),
        chunk_size=_env_int("RAG_CHUNK_SIZE", 900),
        chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", 120),
        timeout_s=_env_int("RAG_TIMEOUT_S", 15),
        user_agent=_env(
            "RAG_USER_AGENT", "verifier_pipeline/1.0 (+https://example.invalid)"
        )
        or "verifier_pipeline/1.0 (+https://example.invalid)",
    )

    generator = GeneratorConfig(
        provider=_env("GEN_PROVIDER", "ollama") or "ollama",
        model=_env("GEN_MODEL", "llama3.1:8b") or "llama3.1:8b",
        temperature=_env_float("GEN_TEMPERATURE", 0.0),
        max_tokens=_env_int("GEN_MAX_TOKENS", 512),
        timeout_s=_env_int("GEN_TIMEOUT_S", 45),
        ollama_url=_env("GEN_OLLAMA_URL", "http://localhost:11434/api/generate")
        or "http://localhost:11434/api/generate",
        llamacpp_url=_env("GEN_LLAMACPP_URL", "http://localhost:8080/completion")
        or "http://localhost:8080/completion",
        vllm_url=_env("GEN_VLLM_URL", "http://localhost:8000/v1/chat/completions")
        or "http://localhost:8000/v1/chat/completions",
    )

    cfg = AppConfig(
        verifier=VerifierConfig(model_path=verifier_model_path),
        google_pse=google_pse,
        rag=rag,
        guardrail=GuardrailConfig(
            min_snippets=_env_int("GR_MIN_SNIPPETS", 2),
            min_unique_sources=_env_int("GR_MIN_UNIQUE_SOURCES", 2),
            contradiction_threshold=_env_float("GR_CONTRADICTION_TH", 0.55),
            support_threshold=_env_float("GR_SUPPORT_TH", 0.55),
        ),
        generator=generator,
        log_dir=_env("LOG_DIR", ".logs") or ".logs",
        log_level=_env("LOG_LEVEL", "INFO") or "INFO",
        json_logs=_env_bool("LOG_JSON", True),
    )

    # Ensure dirs exist (non-fatal)
    Path(cfg.rag.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

    return cfg
