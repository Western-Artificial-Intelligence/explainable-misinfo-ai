import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent                 # .../data/unified_schema/hover
PROJECT_ROOT = THIS_DIR.parents[2]                         # .../WAI

# add project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# add unified_schema dir directly (so we can import cleanup_schema.py reliably)
UNIFIED_SCHEMA_DIR = PROJECT_ROOT / "data" / "unified_schema"
if str(UNIFIED_SCHEMA_DIR) not in sys.path:
    sys.path.insert(0, str(UNIFIED_SCHEMA_DIR))

DATA_DIR = PROJECT_ROOT / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

import os
import json
import pandas as pd
from data.unified_schema.hover.deprecate_article_scraping import threaded_hydrate
from data.unified_schema.cleanup_schema import cleanup
import argparse

# =====================================================================
# Load HoVer JSON
# =====================================================================

def load_hover_json(path, test_mode=False, max_claims=None):
    print("\n================================================================================")
    print("LOADING HOVER DATA ")
    print("================================================================================\n")
    print(f"Loading: {path}")

    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Rename to internal format
    df = df.rename(columns={
        "claim": "claim_text"
    })

    if max_claims:
        df = df.head(max_claims)

    print(f"Loaded {len(df)} rows.\n")
    return df


# =====================================================================
# Claim Schema — includes LABEL
# =====================================================================

def claim_schema(df, text_col="claim_text"):
    print("\n================================================================================")
    print("APPLYING CLAIM SCHEMA")
    print("================================================================================\n")

    df["needs_hydration"] = df[text_col].isnull() | (df[text_col] == "")
    total_rows = len(df)
    to_hydrate = df["needs_hydration"].sum()
    existing = total_rows - to_hydrate

    print(f"Total rows: {total_rows}")
    print(f"Rows needing hydration: {to_hydrate}")
    print(f"Rows with existing claim text: {existing}\n")

    # HoVer claims do not need hydration — but this is kept for consistency
    if to_hydrate > 0:
        print(f"Hydrating {to_hydrate} claims...")
        sub = df[df['needs_hydration']].copy()
        res_df = threaded_hydrate(sub, title_col = text_col, url_col = None, hydrate_fn = None, dataset='hover')
        if text_col in res_df.columns:
            df.loc[sub.index,text_col] = res_df[text_col].values
        elif 'claim_text' in res_df.columns:
            df.loc[sub.index,text_col] = res_df['claim_text'].values
        print("Hydration complete.\n")

    df["content_status"] = "full_claim"
    df["dataset"] = "hover"

    # Ensure label column exists (HoVer has "SUPPORTED"/"NOT_SUPPORTED")
    if "label" not in df.columns:
        df["label"] = None  # safety guard

    df['label'] = df['label'].map({'SUPPORTED':'true','NOT_SUPPORTED':'false'})

    # Final schema
    df_unified = df[[
        "uid",
        "claim_text",
        "label",
        "content_status",
        "dataset"
    ]]

    df_unified['label_confidence'] = 'gold'

    return df_unified


# =====================================================================
# MAIN
# =====================================================================

def main(test_mode=False, max_claims=None):
    raw_dir = PROJECT_ROOT / 'data' / 'raw' / 'hover' / 'data' / 'hover'

    split_files = [
        ('train', raw_dir / 'hover_train_release_v1.1.json'),
        ('dev', raw_dir / 'hover_dev_release_v1.1.json'),
        ('test', raw_dir / 'hover_test_release_v1.1.json'),
    ]

    frames =[]
    for split_name, path in split_files:
        if not path.exists():
            print(f'Warning: Missing HoVer file for {split_name}: {path}')
            continue
        frames.append(load_hover_json(str(path), test_mode=test_mode, max_claims=max_claims))

    if not frames:
        raise FileNotFoundError(f'No HoVer split files found under: {raw_dir}')

    df = pd.concat(frames, ignore_index=True)

    df_unified = claim_schema(df, text_col="claim_text")

    print("\n" + "="*80)
    print("CLEANING UNIFIED OUTPUT")
    print("="*80)

    final = cleanup(
        df_unified,
        id_col="uid",
        claim_col="claim_text",
        article_col=None,
        label_col="label",
        dataset="hover"
    )


    print("\n================================================================================")
    print("FINAL UNIFIED SCHEMA SAMPLE")
    print("================================================================================\n")
    print(final.head())

    # Your desired parquet output path
    output_dir = PROJECT_ROOT / 'data' / 'processed' / 'hover'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / ('unified_hover_test.parquet' if test_mode else 'unified_hover.parquet')

    final.to_parquet(output_file, index=False)

    print(f"\nSaved unified parquet file to: {output_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HoVer dataset into unified schema.")
    parser.add_argument("--test", action="store_true",
                        help="Run in test mode (smaller subset and different output filename).")
    parser.add_argument("--max-claims", type=int, default=None, help="Maximum number of claims to load per split.")
    args = parser.parse_args()

    main(test_mode=args.test, max_claims=args.max_claims)
