# 3_routing_policy.py
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

# ----------------------------
# CONSTANTS (tunable) — per pseudo
# ----------------------------
POLICY_NAME = "routing_policy"
POLICY_VERSION = "v1"

HI_CONF = 0.80
LO_CONF = 0.55

# Add mmr_lambda (required by output schema)
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


def run_3_routing_policy(step2_out: Dict[str, Any]) -> Dict[str, Any]:
    """
    INPUT: Step2 output
    OUTPUT: Step2 passthrough + routing block
    """

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

    # ----------------------------
    # 1) Derive routing tier (per pseudo)
    # ----------------------------
    if conf < LO_CONF:
        tier = "expanded"
        knobs = dict(EXPANDED)
    elif conf >= HI_CONF and (is_truncated is False):
        tier = "standard"
        knobs = dict(STANDARD)
    else:
        tier = "conservative"
        knobs = dict(CONSERV)

    # Optional label-based tweaks
    if label == "false":
        knobs["QEXP"] = True
    if label == "true" and conf >= HI_CONF:
        knobs["M"] = max(30, int(knobs["M"]) - 20)

    # ----------------------------
    # 2) Safety clamps / invariants (per pseudo)
    # ----------------------------
    meta_out = _copy_meta(meta_in)
    warnings = _ensure_warnings_list(meta_out)

    orig = (knobs["M"], knobs["N"], knobs["K"], knobs["MIN_REL"])

    # clamp integer knobs
    M = max(1, int(knobs["M"]))
    N = max(1, int(knobs["N"]))
    K = max(1, int(knobs["K"]))

    # enforce K <= N <= M
    if N > M:
        N = M
    if K > N:
        K = N

    if (M, N, K) != (orig[0], orig[1], orig[2]):
        warnings.append({"code": "ROUTING_KNOBS_CLAMPED", "message": "Clamped K/N/M to satisfy K<=N<=M."})

    # clamp MIN_REL to [0,1]
    min_rel = float(knobs["MIN_REL"])
    min_rel_clamped = min(1.0, max(0.0, min_rel))
    if min_rel_clamped != min_rel:
        warnings.append({"code": "ROUTING_MIN_REL_CLAMPED", "message": "Clamped min_relevance to [0,1]."})
    min_rel = min_rel_clamped

    # mmr_lambda is required by schema; ensure in [0,1]
    mmr_lambda = float(knobs["LAMBDA"])
    mmr_lambda = min(1.0, max(0.0, mmr_lambda))

    # ----------------------------
    # 3) Build routing block
    # ----------------------------
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
            "use_query_expansion": bool(knobs["QEXP"]),
            "allow_web_search": bool(knobs["WEB"]),
        },
    }

    # ----------------------------
    # 4) Emit output (passthrough + routing)
    # ----------------------------
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
    """Public entrypoint for production.py."""
    _ = time.perf_counter()  # kept if you later want to add latency
    return run_3_routing_policy(step2_out)