import os
from pathlib import Path

from datasets import load_from_disk, DatasetDict

from .synthetic_data import create_and_save_synthetic_dataset


def load_merged_dataset(path: str, create_synthetic_if_missing: bool = True):
    """
    Load a merged HuggingFace dataset from disk.

    Assumptions:
    - The merged dataset is stored at `path` using Dataset.save_to_disk(...).
    - The dataset contains the columns:
        - input_text: str
        - label_3way: int (0,1,2) or -100
        - label_bin: int (0,1) or -100
        - source_id: int
        - split: str in {"train", "val", "test"}

    Behavior:
    - If the dataset directory does not exist and `create_synthetic_if_missing`
      is True, a small synthetic dataset will be created and saved to that path.
    - If the directory does not exist and `create_synthetic_if_missing`
      is False, a FileNotFoundError is raised.

    Returns:
        DatasetDict with keys {"train", "val", "test"}.
    """

    p = Path(path)

    if not p.exists():
        if create_synthetic_if_missing:
            print(
                f"[load_dataset] Dataset not found at {path}. "
                "Creating a small synthetic dataset."
            )
            os.makedirs(path, exist_ok=True)
            create_and_save_synthetic_dataset(path)
        else:
            raise FileNotFoundError(f"Merged dataset not found at {path}")

    print(f"[load_dataset] Loading dataset from {path}")

    ds = load_from_disk(path)

    # Ensure we have splits
    if isinstance(ds, DatasetDict):
        return ds
    else:
        raise ValueError(
            "Expected a DatasetDict with splits (train/val/test). "
            "Please save a DatasetDict or regenerate the dataset."
        )