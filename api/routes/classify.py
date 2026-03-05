import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services.predictor import predict_text
from ..utils.cache import add_to_cache, get_from_cache

# Create router instance for this module
router = APIRouter()
logger = logging.getLogger(__name__)


# Define input and output data models
class InputText(BaseModel):
    text: str


class Output(BaseModel):
    label: str
    confidence: float
    explanation: str


class PredictOutput(BaseModel):
    prediction: str
    confidence: float
    explanation: str


class SourceItem(BaseModel):
    title: str
    url: str
    snippet: str = ""


class AnalyzeOutput(BaseModel):
    prediction: str
    confidence: float
    explanation: str
    sources: list[SourceItem] = []


def classify_logic(text: str) -> dict:
    """
    Core classification logic.
    Falls back to random outputs when no model is configured.
    """
    return predict_text(text)


def _to_predict_output(result: dict) -> dict:
    label = str(result.get("label", "")).lower().strip()
    real_labels = {"real", "true", "factual", "reliable"}
    prediction = "REAL" if label in real_labels else "FAKE"
    return {
        "prediction": prediction,
        "confidence": float(result.get("confidence", 0)),
        "explanation": str(result.get("explanation", "")),
    }


# POST endpoint at /classify
@router.post("/classify", response_model=Output)
def classify_text(payload: InputText):
    """
    Handle a classification request:
    1. Check cache first.
    2. If not cached, run model logic.
    3. Save result in cache.
    4. Return result.
    """
    # Step 1: Check cache
    cached_result = get_from_cache(payload.text)
    if cached_result:
        logger.info("INPUT: %r", payload.text)
        logger.info("OUTPUT (cache): %s", cached_result)
        return cached_result

    # Step 2: Run dummy classification logic
    result = classify_logic(payload.text)

    # Step 3: Save to cache
    add_to_cache(payload.text, result)

    # Step 4: Return the result
    logger.info("INPUT: %r", payload.text)
    logger.info("OUTPUT: %s", result)
    return result


@router.post("/predict", response_model=PredictOutput)
def predict_text_endpoint(payload: InputText):
    result = classify_logic(payload.text)
    output = _to_predict_output(result)
    logger.info("INPUT: %r", payload.text)
    logger.info("PREDICT_OUTPUT: %s", output)
    return output


@router.post("/analyze", response_model=AnalyzeOutput)
async def analyze_with_reasoning(payload: InputText):
    """
    Analyze text using Ollama + web search only. No RoBERTa/heuristic fallback.
    Used by Chrome extension "Analyze Selected Text".
    """
    from ..services.ollama_misinfo_analyzer import analyze_with_ollama
    from ..production_pipeline.middlewares.ollama_blackbox import OllamaBlackboxError

    try:
        result = await analyze_with_ollama(payload.text, use_web_search=True)
        logger.info("ANALYZE INPUT: %r", payload.text)
        logger.info("ANALYZE OUTPUT: %s", result)
        return result
    except OllamaBlackboxError as e:
        logger.warning("Ollama error in /analyze: %s", e.message)
        raise HTTPException(
            status_code=503,
            detail=f"Ollama unavailable: {e.message}. Is Ollama running? Try: ollama run llama3.2",
        )
    except ValueError as e:
        logger.warning("Analyze error: %s", e)
        raise HTTPException(status_code=422, detail=str(e))
