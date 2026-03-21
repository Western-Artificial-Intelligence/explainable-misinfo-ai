"""
Prepare combined training data for 3-way RoBERTa classifier.

Sources:
  - LIAR : unified_liar.parquet (local, already in repo)
  - FEVER: v1.0 from HuggingFace (claim → false / mixed / true)

Label mapping (FEVER):
  SUPPORTS        → true
  REFUTES         → false
  NOT ENOUGH INFO → mixed

Output: combined_data.parquet  [claim_text, label_3way, source]
        Balanced: up to MAX_PER_CLASS per label

Usage:
  python api/production_pipeline/RoBERTa_model/prepare_data.py
"""

import pandas as pd
from pathlib import Path
from datasets import load_dataset

HERE      = Path(__file__).resolve().parent
LIAR_PATH = HERE / "unified_liar.parquet"
OUT_PATH  = HERE / "combined_data.parquet"

MAX_PER_CLASS = 20_000
SEED          = 42

FEVER_LABEL_MAP = {
    "SUPPORTS":        "true",
    "REFUTES":         "false",
    "NOT ENOUGH INFO": "mixed",
}


def load_liar():
    df = pd.read_parquet(LIAR_PATH)[["claim_text", "label_3way"]].dropna()
    df["source"] = "liar"
    print(f"[data] LIAR       : {len(df):>6}  | {dict(df['label_3way'].value_counts())}")
    return df


def load_fever():
    print("[data] Downloading FEVER v1.0 from HuggingFace...")
    ds = load_dataset("fever", "v1.0")

    rows = []
    for split in ["train", "paper_dev"]:
        if split not in ds:
            continue
        for ex in ds[split]:
            label = FEVER_LABEL_MAP.get(ex.get("label", ""))
            if label is None:
                continue
            claim = (ex.get("claim") or "").strip()
            if claim:
                rows.append({"claim_text": claim, "label_3way": label, "source": "fever"})

    df = pd.DataFrame(rows).drop_duplicates(subset=["claim_text"]).reset_index(drop=True)
    print(f"[data] FEVER      : {len(df):>6}  | {dict(df['label_3way'].value_counts())}")
    return df


def main():
    liar  = load_liar()
    fever = load_fever()

    combined = (
        pd.concat([liar, fever], ignore_index=True)
        .drop_duplicates(subset=["claim_text"])
        .reset_index(drop=True)
    )
    print(f"\n[data] Combined   : {len(combined):>6}  | {dict(combined['label_3way'].value_counts())}")

    # Balance: sample up to MAX_PER_CLASS per label, then shuffle
    balanced = (
        combined.groupby("label_3way", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), MAX_PER_CLASS), random_state=SEED))
        .reset_index(drop=True)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )
    print(f"[data] Balanced   : {len(balanced):>6}  | {dict(balanced['label_3way'].value_counts())}")

    balanced.to_parquet(OUT_PATH, index=False)
    print(f"\n[data] Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
