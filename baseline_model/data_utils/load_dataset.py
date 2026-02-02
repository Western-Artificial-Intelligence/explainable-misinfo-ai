# ===== baseline_model/data_utils/load_dataset.py =====
"""
Dataset loading utilities for baseline misinformation classification.

Assumptions:
- The merged dataset is stored as a HuggingFace dataset at `data/merged_dataset`.
- Columns:
    - input_text: str
    - label_3way: int (0,1,2) or -100
    - label_bin: int (0,1) or -100
    - source_id: int
    - split: str in {"train", "val", "test"}
- If dataset directory does not exist and `create_synthetic_if_missing=True`,
  a small synthetic dataset will be created via synthetic_data.py and saved to that path.
"""

import os
from pathlib import Path

from datasets import DatasetDict, load_from_disk

from .synthetic_data import create_and_save_synthetic_dataset


def load_merged_dataset(
    path: str, create_synthetic_if_missing: bool = True
) -> DatasetDict:
    """
    Load a merged HuggingFace dataset from disk.

    If not present and `create_synthetic_if_missing=True`,
    create a small synthetic dataset and save it to the path.

    Args:
        path (str): Path to the dataset directory.
        create_synthetic_if_missing (bool): Whether to create synthetic dataset if missing.

    Returns:
        DatasetDict: HuggingFace dataset with splits 'train', 'val', 'test'.

    Raises:
        FileNotFoundError: If dataset not found and synthetic creation disabled.
        ValueError: If loaded dataset is not a DatasetDict.
    """
    p = Path(path)
    if not p.exists():
        if create_synthetic_if_missing:
            print(
                f"[load_dataset] Dataset not found at {path}. Creating a small synthetic dataset."
            )
            os.makedirs(path, exist_ok=True)
            create_and_save_synthetic_dataset(path)
        else:
            raise FileNotFoundError(f"Merged dataset not found at {path}")

    print(f"[load_dataset] Loading dataset from {path}")
    ds = load_from_disk(path)

    if isinstance(ds, DatasetDict):
        return ds
    else:
        # If user saved a single dataset, ensure it has splits
        raise ValueError(
            "Expected a DatasetDict with splits (train/val/test). Please provide a split column."
        )
