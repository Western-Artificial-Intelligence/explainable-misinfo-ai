from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..utils.cache import add_to_cache, get_from_cache

router = APIRouter()


class InputText(BaseModel):
    text: str


# POST endpoint at /classify — runs TruthLens pipeline
@router.post("/classify")
def classify_text(payload: InputText):
    """
    Run the TruthLens pipeline (ingest → RoBERTa → output).
    Returns: { request_id, claim_id, public: { claim, verdict, confidence, summary, ... }, meta }
    """
    cached = get_from_cache(payload.text)
    if cached:
        return cached

    try:
        from pipeline.orchestrator import run_pipeline
        result = run_pipeline(payload.text)
        add_to_cache(payload.text, result)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
