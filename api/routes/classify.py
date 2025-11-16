import random

from fastapi import APIRouter
from pydantic import BaseModel

from ..utils.cache import add_to_cache, get_from_cache

# Create router instance for this module
router = APIRouter()


# Define input and output data models
class InputText(BaseModel):
    text: str


class Output(BaseModel):
    label: str
    confidence: float
    explanation: str


def classify_logic(text: str) -> dict:
    """
    Core classification logic (currently dummy).
    Later, replace with real ML model inference.
    """
    label = random.choice(["factual", "mixed", "false"])
    confidence = round(random.uniform(0.6, 0.98), 2)
    explanation = f"This is a placeholder explanation for '{label}' classification."
    return {"label": label, "confidence": confidence, "explanation": explanation}


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
        return cached_result

    # Step 2: Run dummy classification logic
    result = classify_logic(payload.text)

    # Step 3: Save to cache
    add_to_cache(payload.text, result)

    # Step 4: Return the result
    return result
