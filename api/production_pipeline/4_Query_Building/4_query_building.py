# 4_query_building.py
from __future__ import annotations

import re
import time
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib

# ---- load Ollama blackbox by path (works even with file-loaded steps) ----
_OLLAMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "middlewares"
    / "ollama_blackbox.py"
)

_spec = importlib.util.spec_from_file_location("ollama_blackbox", _OLLAMA_PATH)
_ollama = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_ollama)

ollama_chat = _ollama.ollama_chat
OllamaBlackboxError = _ollama.OllamaBlackboxError

# ----------------------------
# Constants
# ----------------------------
SYS_PARAPHRASE = (
    "Paraphrase the claim WITHOUT changing meaning. "
    "Must be DIFFERENT wording than the original (do not copy it). "
    "ONE short sentence (max 12 words). "
    "Preserve names, numbers, dates, and negations exactly. "
    "Output ONLY the paraphrase text (no quotes, no numbering)."
)

N_PARAPHRASES = 3
REQUIRE_NUMBERS_MATCH = True
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
_LEADING_JUNK_RE = re.compile(r"^\s*(?:[-*•]\s*|\d+[\)\.]\s*)")

# for heuristic fallback (only used when LLM yields none)
_IS_DEAD_RE = re.compile(r"^(?P<subj>.+?)\s+is\s+dead\s*$", re.IGNORECASE)


class QueryBuildingError(Exception):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def _copy_meta(meta_in: Dict[str, Any]) -> Dict[str, Any]:
    meta_out: Dict[str, Any] = {"received_at": meta_in["received_at"], "version": meta_in["version"]}
    for k in ("warnings", "latency_ms", "stage_latencies_ms", "total_latency_ms", "stage_errors"):
        if k in meta_in:
            meta_out[k] = meta_in[k]
    return meta_out


def _ensure_warnings(meta: Dict[str, Any]) -> List[Dict[str, str]]:
    w = meta.get("warnings")
    if isinstance(w, list):
        return w
    meta["warnings"] = []
    return meta["warnings"]


def _collapse_ws(s: str) -> str:
    return " ".join(s.strip().split())


def _clean_llm(s: str) -> str:
    s = s.strip()
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return ""
    s = lines[0]
    s = _LEADING_JUNK_RE.sub("", s)
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return _collapse_ws(s)


def _nums(text: str) -> set[str]:
    return {m.replace(",", "") for m in _NUM_RE.findall(text)}


async def _paraphrase_once(claim: str, seed: int, *, extra_system: str = "") -> str:
    system = SYS_PARAPHRASE if not extra_system else (SYS_PARAPHRASE + " " + extra_system)
    out = await ollama_chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": claim},
        ],
        # slightly higher temp helps it stop copying
        temperature=0.55,
        top_p=0.9,
        seed=seed,
        num_predict=64,
    )
    return str(out.get("content") or "")


async def _paraphrase_with_retry(claim: str, claim_id: str, i: int) -> str:
    """
    If the model copies the original, retry with a stronger constraint.
    Deterministic seeds: stable across runs.
    """
    seed_i = int(hashlib.sha256(f"{claim_id}:{i}:try0".encode("utf-8")).hexdigest()[:8], 16)
    p0 = _clean_llm(await _paraphrase_once(claim, seed_i))
    if p0 and p0 != claim:
        return p0

    seed_1 = int(hashlib.sha256(f"{claim_id}:{i}:try1".encode("utf-8")).hexdigest()[:8], 16)
    p1 = _clean_llm(await _paraphrase_once(
        claim,
        seed_1,
        extra_system="Your last output was too similar. Use different synonyms/rewording."
    ))
    if p1 and p1 != claim:
        return p1

    seed_2 = int(hashlib.sha256(f"{claim_id}:{i}:try2".encode("utf-8")).hexdigest()[:8], 16)
    p2 = _clean_llm(await _paraphrase_once(
        claim,
        seed_2,
        extra_system="You MUST change wording. If stuck, use synonyms like 'has died'/'died'/'is deceased'."
    ))
    return p2  # may still equal claim; caller filters


def _heuristic_fallback_variants(claim: str) -> List[str]:
    """
    Only kicks in when LLM produced 0 kept paraphrases.
    Keep these conservative + meaning-preserving for search queries.
    """
    c = _collapse_ws(claim)

    m = _IS_DEAD_RE.match(c)
    if m:
        subj = m.group("subj").strip()
        return [
            f"{subj} has died",
            f"{subj} died",
            f"{subj} is deceased",
        ]

    # generic fallback (very safe): add keyword-style query form
    return [
        f"{c} news",
        f"fact check {c}",
    ]


async def process_step3_output(step3_out: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step3 -> Step4
    Adds: query_plan
    """
    try:
        request_id = step3_out["request_id"]
        claim_id = step3_out["claim_id"]
        user_claim = step3_out["user_claim"]
        normalized_claim = step3_out["normalized_claim"]
        roberta = step3_out["roberta"]
        routing = step3_out["routing"]
        meta_in = step3_out["meta"]

        enabled = bool(routing["rag"]["use_query_expansion"])
    except Exception as e:
        raise QueryBuildingError("INVALID_INPUT", "Step4 expected Step3 output shape.", {"error": str(e)})

    t0 = time.perf_counter()
    meta_out = _copy_meta(meta_in)
    warnings = _ensure_warnings(meta_out)

    queries: List[Dict[str, str]] = [{"q": normalized_claim, "variant": "original"}]
    kept = 0
    mode = "original_only"

    if enabled:
        raw: List[str] = []
        try:
            for i in range(N_PARAPHRASES):
                raw.append(await _paraphrase_with_retry(normalized_claim, claim_id, i))
        except OllamaBlackboxError as e:
            warnings.append({"code": "PARAPHRASE_LLM_FAILED", "message": f"LLM paraphrase failed: {e.code}"})
            raw = []

        nums0 = _nums(normalized_claim)
        kept_paras: List[str] = []

        for p in raw:
            p = _collapse_ws(p)
            if not p:
                continue
            if p == normalized_claim:
                continue
            if p in kept_paras:
                continue
            if REQUIRE_NUMBERS_MATCH and _nums(p) != nums0:
                continue
            kept_paras.append(p)
            if len(kept_paras) == N_PARAPHRASES:
                break

        # heuristic fallback if LLM gave nothing useful
        if len(kept_paras) == 0:
            fb = _heuristic_fallback_variants(normalized_claim)
            for p in fb:
                p = _collapse_ws(p)
                if not p or p == normalized_claim or p in kept_paras:
                    continue
                if REQUIRE_NUMBERS_MATCH and _nums(p) != nums0:
                    continue
                kept_paras.append(p)
                if len(kept_paras) == N_PARAPHRASES:
                    break

            if len(kept_paras) > 0:
                warnings.append({"code": "PARAPHRASE_HEURISTIC_FALLBACK", "message": "Used heuristic fallback paraphrases (LLM copied original)."})
            else:
                warnings.append({"code": "PARAPHRASE_NONE_KEPT", "message": "No paraphrases passed filters."})

        for p in kept_paras:
            queries.append({"q": p, "variant": "paraphrase"})

        kept = len(kept_paras)
        mode = "original_plus_paraphrases" if kept > 0 else "original_only"

    step4_ms = int(round((time.perf_counter() - t0) * 1000))
    meta_out["latency_ms"] = max(0, step4_ms)

    query_plan = {
        "mode": mode,
        "primary_query": normalized_claim,
        "queries": queries,  # 1..4
        "paraphrase_meta": {
            "enabled": enabled,
            "n_requested": (N_PARAPHRASES if enabled else 0),
            "kept": kept,
            "filter": {"require_numbers_match": REQUIRE_NUMBERS_MATCH},
        },
    }

    return {
        "request_id": request_id,
        "claim_id": claim_id,
        "user_claim": user_claim,
        "normalized_claim": normalized_claim,
        "roberta": roberta,
        "routing": routing,
        "query_plan": query_plan,
        "meta": meta_out,
    }