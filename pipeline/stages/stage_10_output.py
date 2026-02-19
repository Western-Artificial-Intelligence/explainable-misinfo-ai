"""Stage 10: Final output packaging."""

from datetime import datetime, timezone


def run(state: dict) -> dict:
    roberta = state["roberta"]
    public = {
        "claim": state["user_claim"],
        "verdict": roberta["label"]["class_name"],
        "confidence": roberta["confidence"],
        "summary": f"{roberta['label']['class_name']} (confidence {roberta['confidence']:.2f})",
        "bullets": [],
        "citations": [],
        "explainability": {
            "top_positive_tokens": [],
            "top_negative_tokens": [],
        },
    }
    return {
        "request_id": state["request_id"],
        "claim_id": state["claim_id"],
        "public": public,
        "meta": {
            "version": state["meta"].get("version", "0.1.0"),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    }
