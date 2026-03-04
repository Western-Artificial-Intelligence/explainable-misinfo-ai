# 3_routing_policy.py
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

POLICY_NAME = "routing_policy"
POLICY_VERSION = "v1"

HI_CONF = 0.80
LO_CONF = 0.55

STANDARD = {"M": 50,  "N": 12, "K": 6,  "MIN_REL": 0.25, "QEXP": True,  "WEB": True, "LAMBDA": 0.70}
EXPANDED = {"M": 100, "N": 20, "K": 10, "MIN_REL": 0.20, "QEXP": True,  "WEB": True, "LAMBDA": 0.65}
CONSERV  = {"M": 80,  "N": 16, "K": 8,  "MIN_REL": 0.30, "QEXP": False, "WEB": True, "LAMBDA": 0.75}


class RoutingPolicyError(Exception):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def _copy_meta(meta_in: Dict[str, Any]) -> Dict[str, Any]:
    """Copy only allowed meta fields (schema has additionalProperties=false)."""
    meta_out: Dict[str, Any] = {
        "received_at": meta_in["received_at"],
        "version": meta_in["version"],
    }
    for k in ("warnings", "latency_ms", "stage_latencies_ms", "total_latency_ms", "stage_errors"):
        if k in meta_in:
            meta_out[k] = meta_in[k]
    return meta_out


def _ensure_warnings_list(meta: Dict[str, Any]) -> List[Dict[str, str]]:
    w = meta.get("warnings")
    if isinstance(w, list):
        return w
    meta["warnings"] = []
    return meta["warnings"]


def _ensure_stage_latencies(meta: Dict[str, Any]) -> Dict[str, int]:
    v = meta.get("stage_latencies_ms")
    if isinstance(v, dict):
        return v
    meta["stage_latencies_ms"] = {}
    return meta["stage_latencies_ms"]


def _normalize_and_clamp_knobs(
    knobs: Dict[str, Any],
    *,
    warnings: List[Dict[str, str]],
) -> Tuple[int, int, int, float, bool, bool, float]:
    """Return (M, N, K, min_rel, qexp, web, mmr_lambda) after clamps + warnings."""
    orig_M, orig_N, orig_K = knobs["M"], knobs["N"], knobs["K"]
    orig_min_rel = float(knobs["MIN_REL"])
    orig_lambda = float(knobs["LAMBDA"])

    M = max(1, int(orig_M))
    N = max(1, int(orig_N))
    K = max(1, int(orig_K))

    if N > M:
        N = M
    if K > N:
        K = N

    if (M, N, K) != (int(orig_M), int(orig_N), int(orig_K)):
        warnings.append({"code": "ROUTING_KNOBS_CLAMPED", "message": "Clamped K/N/M to satisfy K<=N<=M."})

    min_rel = min(1.0, max(0.0, orig_min_rel))
    if min_rel != orig_min_rel:
        warnings.append({"code": "ROUTING_MIN_REL_CLAMPED", "message": "Clamped min_relevance to [0,1]."})

    mmr_lambda = min(1.0, max(0.0, orig_lambda))
    if mmr_lambda != orig_lambda:
        warnings.append({"code": "ROUTING_LAMBDA_CLAMPED", "message": "Clamped mmr_lambda to [0,1]."})

    qexp = bool(knobs["QEXP"])
    web = bool(knobs["WEB"])

    return M, N, K, min_rel, qexp, web, mmr_lambda


def run_3_routing_policy(step2_out: Dict[str, Any]) -> Dict[str, Any]:
    # minimal contract validation
    try:
        request_id = step2_out["request_id"]
        claim_id = step2_out["claim_id"]
        user_claim = step2_out["user_claim"]
        normalized_claim = step2_out["normalized_claim"]
        roberta = step2_out["roberta"]
        meta_in = step2_out["meta"]

        conf = float(roberta["confidence"])
        label = str(roberta["label"]["class_name"])
        is_truncated = bool(roberta["tokenization"]["truncated"])
    except Exception as e:
        raise RoutingPolicyError(
            code="INVALID_INPUT",
            message="Step3 expected Step2 output shape.",
            details={"error": str(e)},
        )

    # 1) Derive routing tier
    if conf < LO_CONF:
        tier = "expanded"
        knobs = dict(EXPANDED)
    elif conf >= HI_CONF and (is_truncated is False):
        tier = "standard"
        knobs = dict(STANDARD)
    else:
        tier = "conservative"
        knobs = dict(CONSERV)

    # 2) Copy meta + warnings
    meta_out = _copy_meta(meta_in)
    warnings = _ensure_warnings_list(meta_out)

    # 3) Optional label-based tweaks (+ trace)
    if label == "false" and knobs.get("QEXP") is False:
        knobs["QEXP"] = True
        warnings.append({"code": "ROUTING_TWEAK_FALSE_QEXP", "message": "Enabled query expansion for 'false' prediction."})

    if label == "true" and conf >= HI_CONF:
        before = int(knobs["M"])
        knobs["M"] = max(30, before - 20)
        if int(knobs["M"]) != before:
            warnings.append({"code": "ROUTING_TWEAK_TRUE_REDUCE_M", "message": "Reduced candidate pool for high-confidence 'true'."})

    # 4) Clamp/invariants
    M, N, K, min_rel, qexp, web, mmr_lambda = _normalize_and_clamp_knobs(knobs, warnings=warnings)

    routing = {
        "policy_name": POLICY_NAME,
        "policy_version": POLICY_VERSION,
        "tier": tier,
        "rag": {
            "candidate_pool_size_m": M,
            "mmr_top_n": N,
            "mmr_lambda": mmr_lambda,
            "final_top_k": K,
            "min_relevance": min_rel,
            "use_query_expansion": qexp,
            "allow_web_search": web,
        },
    }

    out: Dict[str, Any] = {
        "request_id": request_id,
        "claim_id": claim_id,
        "user_claim": user_claim,
        "normalized_claim": normalized_claim,
        "roberta": roberta,
        "routing": routing,
        "meta": meta_out,
    }
    return out


def process_step2_output(step2_out: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    out = run_3_routing_policy(step2_out)
    dt_ms = int(round((time.perf_counter() - t0) * 1000))

    # Add stage latency if schema allows it (your _copy_meta allows stage_latencies_ms)
    stage_lat = _ensure_stage_latencies(out["meta"])
    stage_lat["step3_routing_policy"] = max(0, dt_ms)

    return out