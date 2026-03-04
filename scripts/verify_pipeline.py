#!/usr/bin/env python3
"""
Verify the full /api/process pipeline (Steps 1–8).

POSTs one or two claims and checks:
- 200 response
- Required top-level keys and roberta.label shape
- inference_source present when expected
- No missing/corrupt keys that would break downstream

Run from repo root:  python scripts/verify_pipeline.py
Requires API server:  uvicorn api.main:app --port 8000
"""
from __future__ import annotations

import sys
from typing import Any, Dict

try:
    import httpx
except ImportError:
    print("Install httpx: pip install httpx")
    sys.exit(1)

BASE_URL = "http://127.0.0.1:8000"
PROCESS_URL = f"{BASE_URL}/api/process"

REQUIRED_TOP_LEVEL = {"request_id", "claim_id", "user_claim", "normalized_claim", "roberta", "meta"}
REQUIRED_ROBERTA = {"label", "confidence"}
REQUIRED_LABEL = {"class_name", "class_id", "logits", "probs"}
VALID_CLASS_NAMES = {"false", "mixed", "true"}


def verify_response(data: Dict[str, Any], claim_desc: str) -> list[str]:
    """Check response shape and Step 2 (roberta) output. Returns list of error messages."""
    errors: list[str] = []
    missing_top = REQUIRED_TOP_LEVEL - set(data.keys())
    if missing_top:
        errors.append(f"[{claim_desc}] Missing top-level keys: {missing_top}")

    roberta = data.get("roberta")
    if not isinstance(roberta, dict):
        errors.append(f"[{claim_desc}] roberta is not a dict")
        return errors

    missing_roberta = REQUIRED_ROBERTA - set(roberta.keys())
    if missing_roberta:
        errors.append(f"[{claim_desc}] roberta missing keys: {missing_roberta}")

    label = roberta.get("label")
    if not isinstance(label, dict):
        errors.append(f"[{claim_desc}] roberta.label is not a dict")
    else:
        missing_label = REQUIRED_LABEL - set(label.keys())
        if missing_label:
            errors.append(f"[{claim_desc}] roberta.label missing keys: {missing_label}")
        class_name = label.get("class_name")
        if class_name not in VALID_CLASS_NAMES:
            errors.append(f"[{claim_desc}] roberta.label.class_name invalid: {class_name!r}")

    if "inference_source" not in roberta:
        errors.append(f"[{claim_desc}] roberta.inference_source missing (expected for pipeline transparency)")

    meta = data.get("meta")
    if not isinstance(meta, dict):
        errors.append(f"[{claim_desc}] meta is not a dict")
    elif "version" not in meta:
        errors.append(f"[{claim_desc}] meta.version missing")

    return errors


def main() -> None:
    print("Pipeline verification: POST /api/process and check full response shape")
    print("=" * 60)

    # 1) Server reachable
    try:
        with httpx.Client() as client:
            r = client.get(BASE_URL, timeout=5.0)
            if r.status_code != 200:
                print("Warning: GET / returned", r.status_code)
    except Exception as e:
        print("Server not reachable at", BASE_URL)
        print("Start it with: uvicorn api.main:app --port 8000")
        print("Error:", e)
        sys.exit(1)

    # 2) First claim – full pipeline
    claim1 = "Bats are blind."
    print(f"\n1. POST claim: {claim1!r}")
    try:
        with httpx.Client(timeout=120.0) as client:
            r = client.post(PROCESS_URL, json={"user_claim": claim1})
    except Exception as e:
        print("Request failed:", e)
        sys.exit(1)

    if r.status_code != 200:
        print(f"FAIL: status {r.status_code}")
        print(r.text[:500])
        sys.exit(1)

    try:
        data1 = r.json()
    except Exception as e:
        print("FAIL: response is not JSON:", e)
        sys.exit(1)

    errors1 = verify_response(data1, "claim1")
    if errors1:
        for e in errors1:
            print("  ", e)
    else:
        roberta1 = data1.get("roberta") or {}
        label1 = roberta1.get("label") or {}
        src = roberta1.get("inference_source", "—")
        print(f"  OK: class_name={label1.get('class_name')}, confidence={roberta1.get('confidence')}, inference_source={src}")

    # 3) Second claim – ensure independent handling
    claim2 = "DNA carries genetic information."
    print(f"\n2. POST claim: {claim2!r}")
    try:
        with httpx.Client(timeout=120.0) as client:
            r2 = client.post(PROCESS_URL, json={"user_claim": claim2})
    except Exception as e:
        print("Request failed:", e)
        sys.exit(1)

    if r2.status_code != 200:
        print(f"FAIL: status {r2.status_code}")
        sys.exit(1)

    try:
        data2 = r2.json()
    except Exception as e:
        print("FAIL: response is not JSON:", e)
        sys.exit(1)

    errors2 = verify_response(data2, "claim2")
    if errors2:
        for e in errors2:
            print("  ", e)
    else:
        roberta2 = data2.get("roberta") or {}
        label2 = roberta2.get("label") or {}
        src2 = roberta2.get("inference_source", "—")
        print(f"  OK: class_name={label2.get('class_name')}, confidence={roberta2.get('confidence')}, inference_source={src2}")

    # 4) Cross-check: different claims should yield different request_id and ideally different class_name
    rid1 = data1.get("request_id")
    rid2 = data2.get("request_id")
    if rid1 == rid2:
        print("\n  Warning: same request_id for two requests (unexpected)")
    else:
        print("\n  request_id differs per request (OK)")

    class1 = (data1.get("roberta") or {}).get("label") or {}
    class2 = (data2.get("roberta") or {}).get("label") or {}
    if class1.get("class_name") == class2.get("class_name") and claim1 != claim2:
        print("  (Same class_name for two different claims is valid if model classifies both the same.)")

    all_errors = errors1 + errors2
    print("\n" + "=" * 60)
    if all_errors:
        print("VERIFY FAILED:", len(all_errors), "issue(s)")
        sys.exit(1)
    print("Pipeline verification passed.")
    print("Done.")


if __name__ == "__main__":
    main()
