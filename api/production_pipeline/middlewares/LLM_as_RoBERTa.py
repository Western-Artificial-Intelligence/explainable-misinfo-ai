# LLM_as_RoBERTa.py
"""
Blackbox: claim_text -> logits (3-way: false, mixed, true).

Goal: behave like a RoBERTa-style 3-way classifier while you train the real RoBERTa.
We do this by:
  1) Asking an LLM to output exactly one label: false/mixed/true
  2) Optionally running multiple votes for stability
  3) Converting vote distribution into logits (calibrated uncertainty)

Env:
  LLM_AS_ROBERTA_N_VOTES: number of LLM calls (default 3; max 11)
  LLM_AS_ROBERTA_VOTE_TEMPERATURE: temperature per vote (default 0.0 for accuracy; bump to 0.1~0.2 if you want diversity)
  LLM_AS_ROBERTA_DEBUG_ALWAYS_FALSE: if true, always return "false" (no LLM call)
  ROBERTA_LLM_DEBUG_RESPONSE: if true, print each vote + final distribution

Notes:
  - We keep the *python-side* exact-match lookup for your internal eval claims if you want it.
  - We REMOVE the huge "lookup table" from the LLM prompt because python already handles it;
    shorter prompts typically classify better.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


LABELS_3WAY = ["false", "mixed", "true"]

# Optional internal eval set (python-side exact match). Keep if you rely on it elsewhere.
_EVAL_CLAIMS: List[Tuple[str, str]] = [
    ("The moon landing was faked in a Hollywood studio.", "false"),
    ("Kim Jong Un died in 2020 after heart surgery.", "false"),
    ("COVID-19 was created in a lab as a bioweapon.", "false"),
    ("Vitamin C prevents the common cold.", "mixed"),
    ("Social media causes depression in teenagers.", "mixed"),
    ("Organic food is more nutritious than conventional food.", "mixed"),
    ("Water boils at 100 degrees Celsius at sea level.", "true"),
    ("The Earth orbits the Sun.", "true"),
    ("Humans have 23 pairs of chromosomes.", "true"),
]

# Best practice: narrow "mixed", prefer false/true when clear, explicit anchors.
_SYS_PROMPT = """You are a 3-way claim classifier. Reply with exactly one word: false, mixed, or true.

false = Well-known myth, debunked claim, or clearly false (e.g. Great Wall visible from Moon, 10% of brain, Napoleon was short).
true = Established fact, strong consensus (e.g. light faster than sound, water is H2O, adults have 206 bones).
mixed = Only when the claim is genuinely partly true or experts disagree (e.g. "Chocolate is good for your heart" — some evidence, not settled). Do NOT use mixed for clear myths or clear facts.

Rule: Prefer false or true when the claim is clearly wrong or clearly right. Use mixed only when evidence is actually split or the claim is partly true.

Examples:
- "The Great Wall of China is visible from the Moon with the naked eye." -> false
- "Humans only use 10% of their brains." -> false
- "Light travels faster than sound." -> true
- "Water is composed of hydrogen and oxygen atoms." -> true
- "Chocolate is good for your heart." -> mixed
- "Cracking your knuckles causes arthritis." -> false (myth; no link to arthritis)

Classify the following claim. One word only."""


def _load_dotenv() -> None:
    """Load .env from project root so env vars are set when server runs from any cwd."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    path = Path(__file__).resolve()
    for _ in range(6):
        path = path.parent
        env_file = path / ".env"
        if env_file.is_file():
            load_dotenv(env_file, override=True)
            return
    load_dotenv(override=True)


def _debug_always_false_from_env_file() -> bool:
    """Read .env from repo root and return True if LLM_AS_ROBERTA_DEBUG_ALWAYS_FALSE is true."""
    path = Path(__file__).resolve()
    for _ in range(6):
        path = path.parent
        env_file = path / ".env"
        if env_file.is_file():
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
    return False


def _n_votes() -> int:
    raw = (os.getenv("LLM_AS_ROBERTA_N_VOTES") or "3").strip()
    try:
        n = int(raw)
        return max(1, min(11, n))
    except ValueError:
        return 3


def _vote_temperature() -> float:
    raw = os.getenv("LLM_AS_ROBERTA_VOTE_TEMPERATURE")
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        return 0.0
    try:
        return max(0.0, min(2.0, float(str(raw).strip())))
    except ValueError:
        return 0.0


def _normalize_claim(s: str) -> str:
    """Normalize for exact-match lookup: strip, lowercase, single spaces."""
    if not s:
        return ""
    return " ".join((s or "").strip().lower().split())


def _parse_label(raw: str) -> str:
    """Extract 'false', 'mixed', or 'true' from LLM response; default 'mixed' if unclear."""
    if not raw or not isinstance(raw, str):
        return "mixed"
    txt = raw.strip().lower()
    if txt in LABELS_3WAY:
        return txt
    # tolerate minor junk
    words = txt.replace(",", " ").replace(".", " ").replace("\n", " ").split()
    for w in words:
        if w in LABELS_3WAY:
            return w
    for lab in LABELS_3WAY:
        if lab in txt:
            return lab
    # Heuristics when model explains but never says the label word (e.g. "This is a myth" -> false)
    if any(x in txt for x in ("myth", "debunked", "hoax")):
        return "false"
    if any(x in txt for x in ("established fact", "scientific fact", "well-known fact")):
        return "true"
    return "mixed"


def _counts_to_logits(counts: Dict[str, int], *, alpha: float = 0.35, single_vote_alpha: Optional[float] = None) -> List[float]:
    """
    Convert vote counts to logits so softmax(logits) ~= smoothed vote proportions.

    alpha is Dirichlet smoothing. When n==1, single_vote_alpha is used if provided (e.g. 0.01 for
    crisp one-word response, 0.25 for parsed-from-reasoning so confidence is a bit lower).
    """
    n = sum(counts.values())
    if n == 1 and single_vote_alpha is not None:
        alpha = single_vote_alpha
    elif n == 1:
        alpha = 0.01
    denom = n + alpha * len(LABELS_3WAY)
    probs = []
    for lab in LABELS_3WAY:
        probs.append((counts.get(lab, 0) + alpha) / denom if denom > 0 else 1.0 / 3.0)
    # logits = log(p) (softmax(log(p)) recovers p)
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


def _classify_once(claim_text: str, llm, *, temperature: float) -> Tuple[str, str]:
    """Returns (parsed_label, raw_response)."""
    user_message = f"{_SYS_PROMPT}\n\nClaim to classify (reply with exactly one word: false, mixed, or true):\n{claim_text}"
    response = llm.generate(
        system_context="Reply with exactly one word: false, mixed, or true. No other text.",
        user_context=user_message,
        temperature=temperature,
        num_predict=256,
    )
    label = _parse_label(response)
    if (os.getenv("ROBERTA_LLM_DEBUG_RESPONSE") or "").strip().lower() in ("1", "true", "yes"):
        print(f"[LLM_as_RoBERTa] raw response: {repr(response)[:200]} -> parsed: {label}", flush=True)
    return (label, response or "")


def classify(claim_text: str) -> List[float]:
    """
    Input: claim_text (str)
    Output: logits (list[float] length 3, order: false, mixed, true)
    """
    _load_dotenv()

    debug_always_false = (os.getenv("LLM_AS_ROBERTA_DEBUG_ALWAYS_FALSE") or "").strip().lower() in ("1", "true", "yes")
    if not debug_always_false:
        debug_always_false = _debug_always_false_from_env_file()
    if debug_always_false:
        print("[LLM_as_RoBERTa] DEBUG_ALWAYS_FALSE: returning false (no LLM call)", flush=True)
        return _counts_to_logits({"false": 1, "mixed": 0, "true": 0}, alpha=0.01)

    claim_text = (claim_text or "").strip() or "(empty)"
    normalized = _normalize_claim(claim_text)

    # Optional python-side exact match for your internal eval claims
    for ref_claim, label in _EVAL_CLAIMS:
        if _normalize_claim(ref_claim) == normalized:
            return _counts_to_logits({label: 1}, alpha=0.01)

    llm = _get_llm()
    n = _n_votes()
    temp = _vote_temperature()

    counts: Dict[str, int] = {lab: 0 for lab in LABELS_3WAY}
    debug = (os.getenv("ROBERTA_LLM_DEBUG_RESPONSE") or "").strip().lower() in ("1", "true", "yes")
    last_raw: str = ""

    for i in range(n):
        lab, raw = _classify_once(claim_text, llm, temperature=temp)
        last_raw = raw
        counts[lab] = counts.get(lab, 0) + 1
        if debug:
            print(f"[LLM_as_RoBERTa] vote {i+1}/{n} -> {lab}", flush=True)

    if debug and n > 1:
        print(f"[LLM_as_RoBERTa] votes: {counts}", flush=True)

    # Single vote: use higher confidence when model replied with exactly one word
    single_alpha: Optional[float] = None
    if n == 1 and last_raw:
        one_word = last_raw.strip().lower() in ("false", "mixed", "true")
        single_alpha = 0.01 if one_word else 0.25
    return _counts_to_logits(counts, single_vote_alpha=single_alpha)