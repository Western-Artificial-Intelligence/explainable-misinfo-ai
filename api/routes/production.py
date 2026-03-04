from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ..production_pipeline.middlewares.llm_blackbox import LLMBlackbox

router = APIRouter()


# ----------------------------
# Dynamic step loader
# ----------------------------

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    # Register to support module-level caches + avoid duplicate module instances.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_API_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE = _API_ROOT / "production_pipeline"


# ----------------------------
# Load Step 1
# ----------------------------
_step1 = _load_module(
    "step1_ingest_claim",
    _PIPELINE / "1_Ingest_claim" / "1_ingest_claim.py",
)
process_user_claim = _step1.process_user_claim
IngestClaimError = getattr(_step1, "IngestClaimError", Exception)


# ----------------------------
# Load Step 2
# ----------------------------
_step2 = _load_module(
    "step2_roberta_inference",
    _PIPELINE / "2_RoBERTa_inference" / "2_RoBERTa_inference.py",
)
process_step1_output = _step2.process_step1_output
RobertaInferenceError = getattr(_step2, "RobertaInferenceError", Exception)


# ----------------------------
# Load Step 3
# ----------------------------
_step3 = _load_module(
    "step3_routing_policy",
    _PIPELINE / "3_Routing_policy" / "3_routing_policy.py",
)
process_step2_output = _step3.process_step2_output
RoutingPolicyError = getattr(_step3, "RoutingPolicyError", Exception)


# ----------------------------
# Load Step 4
# ----------------------------
_step4 = _load_module(
    "step4_query_building",
    _PIPELINE / "4_Query_Building" / "4_query_building.py",
)
process_step3_output = _step4.process_step3_output
QueryBuildingError = getattr(_step4, "QueryBuildingError", Exception)


# ----------------------------
# Load Step 5
# ----------------------------
_step5 = _load_module(
    "step5_rag_retrieval",
    _PIPELINE / "5_RAG_retrieval" / "5_rag_retrieval.py",
)
process_step4_output = _step5.process_step4_output
RAGRetrievalError = getattr(_step5, "RAGRetrievalError", Exception)


# ----------------------------
# Load Step 6
# ----------------------------
_step6 = _load_module(
    "step6_mmr_selection",
    _PIPELINE / "6_MMR_selection" / "6_mmr_selection.py",
)
process_step5_output = _step6.process_step5_output
MMRSelectionError = getattr(_step6, "MMRSelectionError", Exception)


# ----------------------------
# Load Step 7
# ----------------------------
_step7 = _load_module(
    "step7_light_relevance_gate",
    _PIPELINE / "7_Light_relevance_gate" / "7_light_relevance_gate.py",
)
process_step6_output = _step7.process_step6_output
LightRelevanceGateError = getattr(_step7, "LightRelevanceGateError", Exception)


# ----------------------------
# Load Step 8
# ----------------------------
_step8 = _load_module(
    "step8_explainability_shap",
    _PIPELINE / "8_Explainability_SHAP" / "8_explainability_shap.py",
)
process_step7_output = _step8.process_step7_output
ExplainabilitySHAPError = getattr(_step8, "ExplainabilitySHAPError", Exception)


@router.post("/process")
async def process(request: Request):
    body = await request.json()
    user_claim = body.get("user_claim")

    if not isinstance(user_claim, str) or not user_claim.strip():
        raise HTTPException(
            status_code=400,
            detail={"code": "BAD_REQUEST", "message": "user_claim must be a non-empty string"},
        )

    try:
        # Steps 1–3: sync, cheap → run in threadpool for event-loop hygiene.
        step1_out = await run_in_threadpool(process_user_claim, user_claim)
        step2_out = await run_in_threadpool(process_step1_output, step1_out)
        step3_out = await run_in_threadpool(process_step2_output, step2_out)

        # Step 4: async (LLM query expansion)
        step4_out = await process_step3_output(step3_out)

        # Steps 5–8: async (HTTP + CPU-bound but already handled with async APIs)
        step5_out = await process_step4_output(step4_out)
        step6_out = await process_step5_output(step5_out)
        step7_out = await process_step6_output(step6_out)
        step8_out = await process_step7_output(step7_out)

        return step8_out

    except IngestClaimError as e:
        raise HTTPException(status_code=400, detail={"code": getattr(e, "code", "INGEST_ERROR"), "message": getattr(e, "message", str(e))})

    except RobertaInferenceError as e:
        raise HTTPException(status_code=400, detail={"code": getattr(e, "code", "ROBERTA_ERROR"), "message": getattr(e, "message", str(e))})

    except RoutingPolicyError as e:
        raise HTTPException(status_code=400, detail={"code": getattr(e, "code", "ROUTING_ERROR"), "message": getattr(e, "message", str(e))})

    except QueryBuildingError as e:
        raise HTTPException(status_code=400, detail={"code": getattr(e, "code", "QUERY_BUILDING_ERROR"), "message": getattr(e, "message", str(e)), "details": getattr(e, "details", None)})

    except RAGRetrievalError as e:
        raise HTTPException(status_code=400, detail={"code": getattr(e, "code", "RAG_RETRIEVAL_ERROR"), "message": getattr(e, "message", str(e)), "details": getattr(e, "details", None)})

    except MMRSelectionError as e:
        raise HTTPException(status_code=400, detail={"code": getattr(e, "code", "MMR_ERROR"), "message": getattr(e, "message", str(e)), "details": getattr(e, "details", None)})

    except LightRelevanceGateError as e:
        raise HTTPException(status_code=400, detail={"code": getattr(e, "code", "RELEVANCE_GATE_ERROR"), "message": getattr(e, "message", str(e)), "details": getattr(e, "details", None)})

    except ExplainabilitySHAPError as e:
        raise HTTPException(status_code=400, detail={"code": getattr(e, "code", "SHAP_ERROR"), "message": getattr(e, "message", str(e)), "details": getattr(e, "details", None)})


class LLMTestRequest(BaseModel):
    user_context: str


@router.post("/llm_test")
async def llm_test(request: LLMTestRequest):
    system_context = "You are a helpful assistant. Answer in 1 or 2 sentences."
    user_context = request.user_context

    if not isinstance(user_context, str) or not user_context.strip():
        raise HTTPException(status_code=400, detail="user_context must be provided")

    try:
        llm = LLMBlackbox()
        result = await llm.generate_async(system_context=system_context, user_context=user_context)
        return JSONResponse(content={"result": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
