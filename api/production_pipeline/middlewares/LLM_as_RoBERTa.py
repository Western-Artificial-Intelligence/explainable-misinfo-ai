# LLM_as_RoBERTa.py
"""
Blackbox: claim_text -> logits (3-way: false, mixed, true).

Goal: behave like a RoBERTa-style 3-way classifier while you train the real RoBERTa.
We do this by:
  1) Loading personality prompts from middlewares/personalities/*.json
  2) Running one vote per personality (cycling if n_votes > number of personalities)
  3) Converting vote distribution into logits (calibrated uncertainty)

Personalities are split into positive, negative, and neutral drives so that
mixed claims naturally produce a spread vote distribution.

Env:
  LLM_AS_ROBERTA_N_VOTES: number of LLM calls (default 6; one per personality)
  LLM_AS_ROBERTA_VOTE_TEMPERATURE: temperature per vote (default 0.0)
  LLM_AS_ROBERTA_DEBUG_ALWAYS_FALSE: if true, always return "false" (no LLM call)
  ROBERTA_LLM_DEBUG_RESPONSE: if true, print each vote + final distribution
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


LABELS_3WAY = ["false", "mixed", "true"]


def _load_dotenv() -> None:
    """Load repo env files so local overrides work when server runs from any cwd."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    path = Path(__file__).resolve()
    for _ in range(6):
        path = path.parent
        env_file = path / ".env"
        env_local_file = path / ".env.local"
        loaded = False
        if env_file.is_file():
            load_dotenv(env_file, override=False)
            loaded = True
        if env_local_file.is_file():
            load_dotenv(env_local_file, override=True)
            loaded = True
        if loaded:
            return
    load_dotenv(override=True)


def _debug_always_false_from_env_file() -> bool:
    """Read repo env files and return True if LLM_AS_ROBERTA_DEBUG_ALWAYS_FALSE is true."""
    path = Path(__file__).resolve()
    for _ in range(6):
        path = path.parent
        for env_file in [path / ".env", path / ".env.local"]:
            if not env_file.is_file():
                continue
            try:
                text = env_file.read_text(encoding="utf-8", errors="replace")
                for line in text.splitlines():
                    line = line.strip()
                    if line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    if key.strip() == "LLM_AS_ROBERTA_DEBUG_ALWAYS_FALSE":
                        return val.strip().lower() in ("1", "true", "yes")
            except Exception:
                pass
    return False


def _n_votes() -> int:
    raw = (os.getenv("LLM_AS_ROBERTA_N_VOTES") or "6").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 6


def _vote_temperature() -> float:
    raw = os.getenv("LLM_AS_ROBERTA_VOTE_TEMPERATURE")
    if raw is None or str(raw).strip() == "":
        return 0.0
    try:
        return max(0.0, min(2.0, float(str(raw).strip())))
    except ValueError:
        return 0.0


def _parse_label(raw: str) -> str:
    """Extract 'false', 'mixed', or 'true' from LLM response; default 'mixed' if unclear."""
    if not raw or not isinstance(raw, str):
        return "mixed"
    txt = raw.strip().lower()
    if txt in LABELS_3WAY:
        return txt
    words = txt.replace(",", " ").replace(".", " ").replace("\n", " ").split()
    for w in words:
        if w in LABELS_3WAY:
            return w
    for lab in LABELS_3WAY:
        if lab in txt:
            return lab
    if any(x in txt for x in ("myth", "debunked", "hoax")):
        return "false"
    if any(x in txt for x in ("established fact", "scientific fact", "well-known fact")):
        return "true"
    return "mixed"


def _counts_to_logits(
    counts: Dict[str, int],
    *,
    alpha: float = 0.35,
    single_vote_alpha: Optional[float] = None,
) -> List[float]:
    """
    Convert vote counts to logits so softmax(logits) ~= smoothed vote proportions.

    alpha is Dirichlet smoothing. When n==1, single_vote_alpha overrides alpha if provided
    (0.01 for crisp one-word response, 0.25 when parsed from reasoning).
    """
    n = sum(counts.values())
    if n == 1 and single_vote_alpha is not None:
        alpha = single_vote_alpha
    elif n == 1:
        alpha = 0.01
    denom = n + alpha * len(LABELS_3WAY)
    probs = [
        (counts.get(lab, 0) + alpha) / denom if denom > 0 else 1.0 / 3.0
        for lab in LABELS_3WAY
    ]
    return [math.log(max(1e-12, p)) for p in probs]


def _get_llm():
    """Return the underlying LLM blackbox instance."""
    try:
        from api.production_pipeline.middlewares.llm_blackbox import LLMBlackbox
        return LLMBlackbox()
    except Exception:
        import importlib.util
        here = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location("llm_blackbox", here / "llm_blackbox.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load llm_blackbox for LLM_as_RoBERTa.")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "LLMBlackbox")()


_PERSONALITY_FILES = [
    "negative_1.json",
    "negative_2.json",
    "neutral_1.json",
    "neutral_2.json",
    "positive_1.json",
    "positive_2.json",
]


def _load_personalities() -> List[dict]:
    """Load personality JSON files from middlewares/personalities/."""
    personalities_dir = Path(__file__).resolve().parent / "personalities"
    personalities = []
    for filename in _PERSONALITY_FILES:
        with (personalities_dir / filename).open(encoding="utf-8") as fh:
            personalities.append(json.load(fh))
    return personalities


_PERSONALITIES: List[dict] = _load_personalities()


def _classify_once(
    claim_text: str,
    llm,
    *,
    temperature: float,
    personality: dict,
) -> Tuple[str, str]:
    """Returns (parsed_label, raw_response) for one personality vote."""
    response = llm.generate(
        system_context=personality["system_prompt"],
        user_context=f"Claim to classify (reply with exactly one word: false, mixed, or true):\n{claim_text}",
        temperature=temperature,
        num_predict=256,
    )
    label = _parse_label(response)
    if (os.getenv("ROBERTA_LLM_DEBUG_RESPONSE") or "").strip().lower() in ("1", "true", "yes"):
        print(f"[LLM_as_RoBERTa] [{personality['name']}] raw: {repr(response)[:120]} -> {label}", flush=True)
    return (label, response or "")


def classify(
    claim_text: str,
    *,
    n_votes: Optional[int] = None,
    temperature: Optional[float] = None,
) -> List[float]:
    """
    Input:  claim_text (str)
    Output: logits (list[float] length 3, order: false, mixed, true)

    n_votes: override LLM_AS_ROBERTA_N_VOTES (default 6, one per personality).
    temperature: override LLM_AS_ROBERTA_VOTE_TEMPERATURE.
    """
    _load_dotenv()

    debug_always_false = (
        (os.getenv("LLM_AS_ROBERTA_DEBUG_ALWAYS_FALSE") or "").strip().lower() in ("1", "true", "yes")
        or _debug_always_false_from_env_file()
    )
    if debug_always_false:
        print("[LLM_as_RoBERTa] DEBUG_ALWAYS_FALSE: returning false (no LLM call)", flush=True)
        return _counts_to_logits({"false": 1, "mixed": 0, "true": 0}, alpha=0.01)

    claim_text = (claim_text or "").strip() or "(empty)"
    llm = _get_llm()
    n = n_votes if n_votes is not None else _n_votes()
    n = max(1, n)
    temp = temperature if temperature is not None else _vote_temperature()

    counts: Dict[str, int] = {lab: 0 for lab in LABELS_3WAY}
    debug = (os.getenv("ROBERTA_LLM_DEBUG_RESPONSE") or "").strip().lower() in ("1", "true", "yes")
    last_raw: str = ""

    for i in range(n):
        personality = _PERSONALITIES[i % len(_PERSONALITIES)]
        lab, raw = _classify_once(claim_text, llm, temperature=temp, personality=personality)
        last_raw = raw
        counts[lab] += 1
        if debug:
            print(f"[LLM_as_RoBERTa] vote {i+1}/{n} [{personality['name']}] -> {lab}", flush=True)

    if debug and n > 1:
        print(f"[LLM_as_RoBERTa] votes: {counts}", flush=True)

    single_alpha: Optional[float] = None
    if n == 1 and last_raw:
        single_alpha = 0.01 if last_raw.strip().lower() in ("false", "mixed", "true") else 0.25

    return _counts_to_logits(counts, single_vote_alpha=single_alpha)
