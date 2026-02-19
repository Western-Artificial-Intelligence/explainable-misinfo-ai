"""Pipeline orchestrator. Runs stages in sequence."""

import uuid

from pipeline.config import PIPELINE_VERSION
from pipeline.stages import stage_01_ingest, stage_02_roberta, stage_10_output

# MVP: stages 1, 2, 10 only
STAGES = [
    ("ingest", stage_01_ingest.run),
    ("roberta", stage_02_roberta.run),
    ("output", stage_10_output.run),
]


def run_pipeline(user_claim: str) -> dict:
    """
    Run the full pipeline and return the final output.
    """
    request_id = str(uuid.uuid4())
    claim_id = str(uuid.uuid4())
    state = {
        "request_id": request_id,
        "claim_id": claim_id,
        "user_claim": user_claim,
        "PIPELINE_VERSION": PIPELINE_VERSION,
    }
    for name, stage_fn in STAGES:
        state = stage_fn(state)
        if name == "output":
            return state
    return state
