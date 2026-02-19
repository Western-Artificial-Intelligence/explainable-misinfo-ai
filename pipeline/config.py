"""Pipeline configuration."""
import os

PIPELINE_VERSION = "0.1.0"
ROBERTA_CHECKPOINT = os.getenv(
    "ROBERTA_CHECKPOINT",
    "baseline_outputs/baseline/checkpoints/best_model.pt",
)
ROBERTA_BACKBONE = "roberta-base"
T_CONTEXT = 256
LABELS_3WAY = ["false", "mixed", "true"]
