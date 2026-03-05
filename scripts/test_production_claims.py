#!/usr/bin/env python3
"""
Test LLM_as_RoBERTa classifier performance only (not the full pipeline).

This script measures the classifier in api/production_pipeline/middlewares/LLM_as_RoBERTa.py.

  python scripts/test_production_claims.py
      Uses API (POST /api/process); only roberta label is checked. Needs server.
  python scripts/test_production_claims.py --direct
      Calls LLM_as_RoBERTa.classify(claim) once per claim in-process. No server.
      Use --direct to verify the classifier runs independently for each claim.

For API mode, ensure inference_source is "llm" (ROBERTA_USE_LLM=true).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Repo root for --direct import
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import httpx
except ImportError:
    print("Install httpx: pip install httpx")
    sys.exit(1)

BASE_URL = "http://127.0.0.1:8000"
PROCESS_URL = f"{BASE_URL}/api/process"

# Test-only set: 3 false, 3 mixed, 3 true (different from pipeline eval set so LLM runs)
CLAIMS: Dict[str, List[str]] = {
    "false": [
        "Bats are blind.",
        "Goldfish have a three-second memory.",
        "Humans have only five senses.",
    ],
    "mixed": [
        "Coffee stunts your growth.",
        "Eating late at night makes you gain weight.",
        "Stress causes gray hair.",
    ],
    "true": [
        "The speed of light is approximately 299,792 kilometers per second.",
        "DNA carries genetic information.",
        "The heart pumps blood through the body.",
    ],
}


def run_one(
    client: httpx.Client, claim: str, expected: str
) -> Tuple[str, str, float, str, bool, str]:
    """POST one claim; return (expected, predicted, confidence, status, override_applied, inference_source)."""
    try:
        r = client.post(PROCESS_URL, json={"user_claim": claim}, timeout=120.0)
        if r.status_code != 200:
            return (expected, "—", 0.0, f"HTTP {r.status_code}: {r.text[:200]}", False, "—")
        data = r.json()
        roberta = data.get("roberta") or {}
        label = roberta.get("label") or {}
        pred = label.get("class_name", "—")
        conf = float(label.get("confidence") or roberta.get("confidence") or 0.0)
        override = roberta.get("claim_override_applied") is True
        source = roberta.get("inference_source", "—")
        return (expected, pred, conf, "OK", override, source)
    except Exception as e:
        return (expected, "—", 0.0, str(e), False, "—")


LABELS_3WAY = ["false", "mixed", "true"]


def _softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps] if s else [1.0 / 3] * 3


def run_one_direct(claim: str, expected: str) -> Tuple[str, str, float, str]:
    """Call LLM_as_RoBERTa.classify(claim) directly; one independent call per claim. Returns (expected, predicted, confidence, status)."""
    try:
        from api.production_pipeline.middlewares.LLM_as_RoBERTa import classify
        logits = classify(claim)
        probs = _softmax(logits)
        idx = max(range(3), key=lambda i: probs[i])
        pred = LABELS_3WAY[idx]
        conf = float(probs[idx])
        return (expected, pred, conf, "OK")
    except Exception as e:
        return (expected, "—", 0.0, str(e))


def main() -> None:
    parser = argparse.ArgumentParser(description="Test LLM_as_RoBERTa classifier (9 claims).")
    parser.add_argument("--direct", action="store_true", help="Call classifier directly for each claim (no API); proves independent run per claim")
    args = parser.parse_args()

    if args.direct:
        print("Test: LLM_as_RoBERTa classifier only — DIRECT (one classify(claim) per claim, no API)")
        print("9 claims: 3 false, 3 mixed, 3 true")
        print("=" * 60)
        correct = 0
        total = 0
        for category, claims in CLAIMS.items():
            for claim in claims:
                result = run_one_direct(claim, category)
                expected, predicted, confidence, status = result
                total += 1
                if predicted == expected:
                    correct += 1
                match = "[OK]" if predicted == expected else "[MISS]"
                print(f"\n[{category.upper()}] {match}")
                print(f"  Claim: {claim[:70]}{'...' if len(claim) > 70 else ''}")
                print(f"  Expected: {expected}  ->  Got: {predicted}  (conf: {confidence:.3f})  {status}")
        print("\n" + "=" * 60)
        print(f"Summary: {correct}/{total} correct (LLM_as_RoBERTa, direct)")
        print("Done.")
        return

    print("Test: LLM_as_RoBERTa classifier only (middlewares/LLM_as_RoBERTa.py)")
    print("9 claims: 3 false, 3 mixed, 3 true (via API; only roberta label is checked)")
    print("=" * 60)
    try:
        with httpx.Client() as client:
            r = client.get(BASE_URL, timeout=5.0)
            if r.status_code != 200:
                print("Warning: server returned", r.status_code)
    except Exception as e:
        print("Server not reachable at", BASE_URL)
        print("Start it with: uvicorn api.main:app --port 8000")
        print("Or use --direct to call the classifier without the server.")
        print("Error:", e)
        sys.exit(1)
    print()

    correct = 0
    total = 0
    inference_source: str = "—"
    with httpx.Client() as client:
        for category, claims in CLAIMS.items():
            for claim in claims:
                result = run_one(client, claim, category)
                expected, predicted, confidence, status, override, source = result
                if inference_source == "—" and source != "—":
                    inference_source = source
                total += 1
                if predicted == expected:
                    correct += 1
                match = "[OK]" if predicted == expected else "[MISS]"
                print(f"\n[{category.upper()}] {match}")
                print(f"  Claim: {claim[:70]}{'...' if len(claim) > 70 else ''}")
                print(f"  Expected: {expected}  ->  Got: {predicted}  (conf: {confidence:.3f})  {status}")
                if override:
                    print(f"  (claim_override applied)")
    print("\n" + "=" * 60)
    print(f"Summary: {correct}/{total} correct (LLM_as_RoBERTa classifier)")
    if inference_source != "—":
        print(f"Inference source: {inference_source}")
        if inference_source != "llm":
            print("(Warning: not 'llm' — classifier was not used for these claims. Set ROBERTA_USE_LLM=true and restart server to test LLM_as_RoBERTa.)")
    print("Done.")


if __name__ == "__main__":
    main()
