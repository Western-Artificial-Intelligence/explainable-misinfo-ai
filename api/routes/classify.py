import logging

from fastapi import APIRouter
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
