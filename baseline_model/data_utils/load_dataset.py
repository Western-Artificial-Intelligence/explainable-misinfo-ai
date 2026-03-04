import os
from pathlib import Path

from datasets import load_from_disk, Dataset, DatasetDict

from .synthetic_data import create_and_save_synthetic_dataset


def _build_input_text(example):
    """Combine claim_text and article_text into a single input_text field with special tokens.

    Only includes <ARTICLE> when content_status is 'full_article', per the model architecture spec.
    """
    claim = (example.get("claim_text") or "").strip()
    article = (example.get("article_text") or "").strip()
    content_status = example.get("content_status", "")
    if article and content_status in ("full_article", "partial_article"):
        return {"input_text": f"<CLAIM> {claim} </CLAIM> <ARTICLE> {article} </ARTICLE>"}
    else:
        return {"input_text": f"<CLAIM> {claim} </CLAIM>"}


def load_merged_dataset(path: str, create_synthetic_if_missing: bool = True):
    """
    Load a merged HuggingFace dataset from disk.

    Supports two formats:
    - HuggingFace DatasetDict saved with save_to_disk()
    - Directory of parquet files named unified_{split}.parquet

    The dataset must contain the columns:
        - label_3way: int (0,1,2) or -100
        - label_bin: int (0,1) or -100
        - source_id: int

    For parquet files, input_text is built from claim_text + article_text
    using <CLAIM>/<ARTICLE> special tokens.

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

    # Check if this is a directory of parquet files
    parquet_files = {
        "train": p / "unified_train.parquet",
        "val": p / "unified_val.parquet",
        "test": p / "unified_test.parquet",
    }

    if all(f.exists() for f in parquet_files.values()):
        print("[load_dataset] Found parquet files, loading from parquet")
        splits = {}
        for split_name, parquet_path in parquet_files.items():
            ds = Dataset.from_parquet(str(parquet_path))
            ds = ds.map(_build_input_text, desc=f"Building input_text ({split_name})")
            splits[split_name] = ds
        return DatasetDict(splits)

    # Fall back to HuggingFace load_from_disk
    ds = load_from_disk(path)

    if isinstance(ds, DatasetDict):
        return ds
    else:
        raise ValueError(
            "Expected a DatasetDict with splits (train/val/test) "
            "or parquet files named unified_{train,val,test}.parquet."
        )