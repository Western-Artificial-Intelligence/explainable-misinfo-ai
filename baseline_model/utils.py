# ===== baseline_model/utils.py =====
"""
Utility functions used by training and evaluation scripts.
"""

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, Any]:
    """
    Load YAML config from path and return as dict.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_checkpoint(state: dict, path: str):
    """
    Save a PyTorch checkpoint to disk and ensure parent dirs exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"[train] Saved checkpoint to {path}")
