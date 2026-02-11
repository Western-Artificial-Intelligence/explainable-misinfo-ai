import os
from typing import Dict, List

from datasets import Dataset, DatasetDict


def create_example_rows() -> List[Dict]:
    """
    Return a small list of example rows following the expected schema.

    The synthetic examples cover:
    - LIAR-style 3-way labels
    - Binary-labeled rows
    - Multiple dataset sources (source_id)
    - train/val/test splits
    """

    rows: List[Dict] = []

    # LIAR-style examples (3-way)
    rows += [
        {
            "input_text": (
                "<CLAIM> Vaccines cause autism </CLAIM> "
                "<ARTICLE> Short note: This claim is false. </ARTICLE>"
            ),
            "label_3way": 0,  # false
            "label_bin": -100,
            "source_id": 0,
            "split": "train",
        },
        {
            "input_text": (
                "<CLAIM> New study shows coffee reduces risk of cancer </CLAIM> "
                "<ARTICLE> Summary: Study has mixed results. </ARTICLE>"
            ),
            "label_3way": 1,  # mixed
            "label_bin": -100,
            "source_id": 0,
            "split": "val",
        },
        {
            "input_text": (
                "<CLAIM> Historic artifact discovered in 2024 confirms lost city </CLAIM> "
                "<ARTICLE> Press release indicates legitimate find. </ARTICLE>"
            ),
            "label_3way": 2,  # true
            "label_bin": -100,
            "source_id": 0,
            "split": "test",
        },
    ]

    # Binary-labeled examples from other sources
    rows += [
        {
            "input_text": (
                "<CLAIM> Drinking bleach cures disease </CLAIM> "
                "<ARTICLE> Blog post: extremely dangerous and false. </ARTICLE>"
            ),
            "label_3way": -100,
            "label_bin": 0,  # false
            "source_id": 1,
            "split": "train",
        },
        {
            "input_text": (
                "<CLAIM> Breakthrough in energy tech can power city on one battery </CLAIM> "
                "<ARTICLE> News: claims are overhyped. </ARTICLE>"
            ),
            "label_3way": -100,
            "label_bin": 1,  # true-ish (coarse)
            "source_id": 1,
            "split": "val",
        },
        {
            "input_text": (
                "<CLAIM> Celebrity endorses miracle product </CLAIM> "
                "<ARTICLE> Influencer post with no evidence. </ARTICLE>"
            ),
            "label_3way": -100,
            "label_bin": 0,
            "source_id": 2,
            "split": "test",
        },
    ]

    # Additional rows to allow minimal training batches
    for i in range(8):
        rows.append(
            {
                "input_text": (
                    f"<CLAIM> Synthetic claim {i} </CLAIM> "
                    f"<ARTICLE> Example article content {i}. </ARTICLE>"
                ),
                "label_3way": 0 if i % 3 == 0 else 2,
                "label_bin": -100,
                "source_id": i % 3,
                "split": "train" if i < 6 else "val",
            }
        )

    return rows


def create_and_save_synthetic_dataset(path: str):
    """
    Create a small synthetic DatasetDict with train/val/test splits
    and save it to disk at the provided path.

    Args:
        path: Directory where the dataset will be saved.
    """

    rows = create_example_rows()

    # Split rows by the "split" field
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]

    ds_train = Dataset.from_list(train_rows)
    ds_val = Dataset.from_list(val_rows)
    ds_test = Dataset.from_list(test_rows)

    ds = DatasetDict(
        {
            "train": ds_train,
            "val": ds_val,
            "test": ds_test,
        }
    )

    os.makedirs(path, exist_ok=True)
    ds.save_to_disk(path)

    print(f"[synthetic_data] Saved synthetic dataset to {path}")