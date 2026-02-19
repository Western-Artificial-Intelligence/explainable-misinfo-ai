"""Shared types for the pipeline. Each stage receives a dict and returns a dict."""

from typing import Any

# Pipeline state passed between stages (grows as it moves through)
PipelineState = dict[str, Any]
