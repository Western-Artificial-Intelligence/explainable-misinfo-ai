from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from ..production_pipeline.middlewares.ollama_blackbox import OllamaBlackboxError, ollama_chat

router = APIRouter()

# ----------------------------
# Load Step 1
# ----------------------------
_STEP1_PATH = (
    Path(__file__).resolve().parents[1]
    / "production_pipeline"
    / "1_Ingest_claim"
    / "1_ingest_claim.py"
)

_spec1 = importlib.util.spec_from_file_location("step1_ingest_claim", _STEP1_PATH)
_step1 = importlib.util.module_from_spec(_spec1)
assert _spec1 and _spec1.loader
_spec1.loader.exec_module(_step1)

process_user_claim = _step1.process_user_claim
IngestClaimError = _step1.IngestClaimError

# ----------------------------
# Load Step 2
# ----------------------------
_STEP2_PATH = (
    Path(__file__).resolve().parents[1]
    / "production_pipeline"
    / "2_RoBERTa_inference"
    / "2_RoBERTa_inference.py"
)

_spec2 = importlib.util.spec_from_file_location("step2_roberta_inference", _STEP2_PATH)
_step2 = importlib.util.module_from_spec(_spec2)
assert _spec2 and _spec2.loader
_spec2.loader.exec_module(_step2)

process_step1_output = _step2.process_step1_output
RobertaInferenceError = getattr(_step2, "RobertaInferenceError", Exception)

# Load Step 3
_STEP3_PATH = (
    Path(__file__).resolve().parents[1]
    / "production_pipeline"
    / "3_Routing_policy"
    / "3_routing_policy.py"
)
_spec3 = importlib.util.spec_from_file_location("step3_routing_policy", _STEP3_PATH)
_step3 = importlib.util.module_from_spec(_spec3)
assert _spec3 and _spec3.loader
_spec3.loader.exec_module(_step3)

process_step2_output = _step3.process_step2_output
RoutingPolicyError = getattr(_step3, "RoutingPolicyError", Exception)

# ----------------------------
# Load Step 4
# ----------------------------
_STEP4_PATH = (
    Path(__file__).resolve().parents[1]
    / "production_pipeline"
    / "4_Query_Building"
    / "4_query_building.py"
)

_spec4 = importlib.util.spec_from_file_location("step4_query_building", _STEP4_PATH)
_step4 = importlib.util.module_from_spec(_spec4)
assert _spec4 and _spec4.loader
_spec4.loader.exec_module(_step4)

process_step3_output = _step4.process_step3_output
QueryBuildingError = getattr(_step4, "QueryBuildingError", Exception)

# ----------------------------
# Load Step 5
# ----------------------------
_STEP5_PATH = (
    Path(__file__).resolve().parents[1]
    / "production_pipeline"
    / "5_RAG_retrieval"
    / "5_rag_retrieval.py"
)

_spec5 = importlib.util.spec_from_file_location("step5_rag_retrieval", _STEP5_PATH)
assert _spec5 and _spec5.loader

_step5 = importlib.util.module_from_spec(_spec5)
sys.modules[_spec5.name] = _step5   # ✅ add this

_spec5.loader.exec_module(_step5)

process_step4_output = _step5.process_step4_output
RAGRetrievalError = getattr(_step5, "RAGRetrievalError", Exception)

# ----------------------------
# Load Step 6
# ----------------------------
_STEP6_PATH = (
    Path(__file__).resolve().parents[1]
    / "production_pipeline"
    / "6_MMR_selection"
    / "6_mmr_selection.py"
)

_spec6 = importlib.util.spec_from_file_location("step6_mmr_selection", _STEP6_PATH)
assert _spec6 and _spec6.loader
_step6 = importlib.util.module_from_spec(_spec6)
sys.modules[_spec6.name] = _step6
_spec6.loader.exec_module(_step6)

process_step5_output = _step6.process_step5_output
MMRSelectionError = getattr(_step6, "MMRSelectionError", Exception)

# ----------------------------
# Load Step 7
# ----------------------------
_STEP7_PATH = (
    Path(__file__).resolve().parents[1]
    / "production_pipeline"
    / "7_Light_relevance_gate"
    / "7_light_relevance_gate.py"
)

_spec7 = importlib.util.spec_from_file_location("step7_light_relevance_gate", _STEP7_PATH)
assert _spec7 and _spec7.loader
_step7 = importlib.util.module_from_spec(_spec7)
sys.modules[_spec7.name] = _step7
_spec7.loader.exec_module(_step7)

process_step6_output = _step7.process_step6_output
LightRelevanceGateError = getattr(_step7, "LightRelevanceGateError", Exception)

# ----------------------------
# Load Step 8
# ----------------------------
_STEP8_PATH = (
    Path(__file__).resolve().parents[1]
    / "production_pipeline"
    / "8_Explainability_SHAP"
    / "8_explainability_shap.py"
)

_spec8 = importlib.util.spec_from_file_location("step8_explainability_shap", _STEP8_PATH)
assert _spec8 and _spec8.loader
_step8 = importlib.util.module_from_spec(_spec8)
sys.modules[_spec8.name] = _step8
_spec8.loader.exec_module(_step8)

process_step7_output = _step8.process_step7_output
ExplainabilitySHAPError = getattr(_step8, "ExplainabilitySHAPError", Exception)



@router.post("/process")
async def process(request: Request):
    body = await request.json()
    try:
        step1_out = process_user_claim(body.get("user_claim"))
        step2_out = process_step1_output(step1_out)
        step3_out = process_step2_output(step2_out)
        step4_out = await process_step3_output(step3_out)
        step5_out = await process_step4_output(step4_out)
        step6_out = await process_step5_output(step5_out)
        step7_out = await process_step6_output(step6_out)
        step8_out = await process_step7_output(step7_out)
        return step8_out

    except IngestClaimError as e:
        raise HTTPException(status_code=400, detail={"code": e.code, "message": e.message})

    except RobertaInferenceError as e:
        code = getattr(e, "code", "ROBERTA_ERROR")
        msg = getattr(e, "message", str(e))
        raise HTTPException(status_code=400, detail={"code": code, "message": msg})
    
    except RoutingPolicyError as e:
        raise HTTPException(status_code=400, detail={"code": e.code, "message": e.message})
    
    except QueryBuildingError as e:
        raise HTTPException(status_code=400, detail={"code": e.code, "message": e.message})
    
    except RAGRetrievalError as e:
        raise HTTPException(status_code=400, detail={"code": e.code, "message": e.message, "details": getattr(e, "details", None)})
    
    except MMRSelectionError as e:
        raise HTTPException(
            status_code=400,
            detail={"code": e.code, "message": e.message, "details": getattr(e, "details", None)},
        )
    
    
@router.get("/llm_test")
async def llm_test():
    try:
        out = await ollama_chat(
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "hi, what can you do?"},
            ]
        )
        return out
    except OllamaBlackboxError as e:
        raise HTTPException(status_code=500, detail={"code": e.code, "message": e.message, "details": e.details})