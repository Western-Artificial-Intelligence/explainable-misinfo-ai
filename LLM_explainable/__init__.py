"""
LLM_explainable: verification-first misinformation pipeline.
Keep imports light to avoid optional-dependency import failures.
"""

from .config import AppConfig, load_config_from_env
from .schemas import FinalResult

__all__ = ["run_claim", "AppConfig", "load_config_from_env", "FinalResult"]


def run_claim(*args, **kwargs):
    # Lazy import so importing the package doesn't require heavy deps.
    from .pipeline import run_claim as _run_claim
    return _run_claim(*args, **kwargs)
