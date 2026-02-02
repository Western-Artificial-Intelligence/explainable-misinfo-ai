# ===== baseline_model/data_utils/synthetic_data.py =====
"""
Creates a small synthetic merged dataset and saves it to disk.

This ensures the repo is runnable end-to-end without needing external dataset files.
"""

import os
from typing import Dict, List

from datasets import Dataset, DatasetDict


def create_example_rows() -> List[Dict]:
    """
    Return a small list of example rows following the dataset schema.

    Examples cover:
    - LIAR-style 3-way labels
    - Binary-labeled rows
    - Multiple sources (source_id)
    - train/val/test splits
    """
    rows: List[Dict] = []

    # LIAR examples (3-way)
    rows += [
        {
            "input_text": "<CLAIM> Vaccines cause autism </CLAIM> <ARTICLE> Short note: This claim is false. </ARTICLE>",
            "label_3way": 0,  # false
            "label_bin": -100,
            "source_id": 0,
            "split": "train",
        },
        {
            "input_text": "<CLAIM> New study shows coffee reduces risk of cancer </CLAIM> <ARTICLE> Summary: Study has mixed results. </ARTICLE>",
            "label_3way": 1,  # mixed
            "label_bin": -100,
            "source_id": 0,
            "split": "val",
        },
        {
            "input_text": "<CLAIM> Historic artifact discovered in 2024 confirms lost city </CLAIM> <ARTICLE> Press release indicates legitimate find. </ARTICLE>",
            "label_3way": 2,  # true
            "label_bin": -100,
            "source_id": 0,
            "split": "test",
        },
    ]

    # Binary labeled examples from other sources
    rows += [
        {
            "input_text": "<CLAIM> Drinking bleach cures disease </CLAIM> <ARTICLE> Blog post: extremely dangerous and false. </ARTICLE>",
            "label_3way": -100,
            "label_bin": 0,  # false
            "source_id": 1,
            "split": "train",
        },
        {
            "input_text": "<CLAIM> Breakthrough in energy tech can power city on one battery </CLAIM> <ARTICLE> News: claims are overhyped. </ARTICLE>",
            "label_3way": -100,
            "label_bin": 1,  # true-ish
            "source_id": 1,
            "split": "val",
        },
        {
            "input_text": "<CLAIM> Celebrity endorses miracle product </CLAIM> <ARTICLE> Influencer post with no evidence. </ARTICLE>",
            "label_3way": -100,
            "label_bin": 0,
            "source_id": 2,
            "split": "test",
        },
    ]

    # Additional synthetic rows for minimal training batches
    for i in range(8):
        rows.append(
            {
                "input_text": f"<CLAIM> Synthetic claim {i} </CLAIM> <ARTICLE> Example article content {i}. </ARTICLE>",
                "label_3way": 0 if i % 3 == 0 else 2,
                "label_bin": -100,
                "source_id": i % 3,
                "split": "train" if i < 6 else "val",
            }
        )

    return rows


def create_and_save_synthetic_dataset(path: str):
    """
    Create a small synthetic dataset and save it to disk.

    Args:
        path (str): Directory to save the dataset.

    The function creates a DatasetDict with splits 'train', 'val', 'test'
    and saves it using HuggingFace's `.save_to_disk(path)`.
    """
    rows = create_example_rows()

    # Split by 'split' field
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]

    ds_train = Dataset.from_list(train_rows)
    ds_val = Dataset.from_list(val_rows)
    ds_test = Dataset.from_list(test_rows)

    ds = DatasetDict({"train": ds_train, "val": ds_val, "test": ds_test})

    os.makedirs(path, exist_ok=True)
    ds.save_to_disk(path)
    print(f"[synthetic_data] Saved synthetic dataset to {path}")
