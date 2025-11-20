"""
HoVer Unified Schema Processing Script

This script converts the HoVer dataset to the unified schema.

Usage:
    python hover_unified.py [--test] [--max-claims N]

    --test: Process only first 100 claims for testing
    --max-claims N: Process only first N claims

Example:
    python hover_unified.py --test
    python hover_unified.py --max-claims 1000
    python hover_unified.py  # Process all claims
"""

import os
import json
import argparse
import pandas as pd
from article_scraping import threaded_hydrate
from tqdm import tqdm

def load_hover_json(path, test_mode=False, max_claims=None):
    """
    Load a HoVer JSON file into a DataFrame
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Rename columns to match unified schema
    df = df.rename(columns={'claim': 'claim_text'})
    df['split'] = 'train'  # Default, can be adjusted if loading dev/test

    if test_mode:
        df = df.head(100)
    if max_claims:
        df = df.head(max_claims)

    return df

def claim_schema(df, text_col='claim_text'):
    """
    Convert HoVer claims to unified schema
    """
    print("\n================================================================================")
    print("APPLYING CLAIM SCHEMA")
    print("================================================================================\n")

    df['needs_hydration'] = df[text_col].isnull() | (df[text_col] == '')
    total_rows = len(df)
    to_hydrate = df['needs_hydration'].sum()
    existing = total_rows - to_hydrate

    print(f"Total rows: {total_rows}")
    print(f"Rows needing hydration: {to_hydrate}")
    print(f"Rows with existing claim text: {existing}\n")

    if to_hydrate > 0:
        print(f"Hydrating {to_hydrate} claims...")
        hydrate_df = threaded_hydrate(df[df['needs_hydration']], text_col=text_col)
        # Merge hydrated claims back
        df.loc[df['needs_hydration'], text_col] = hydrate_df['claim_text'].values
        print("Hydration complete.\n")

    df['content_status'] = 'full_claim'
    df['dataset'] = 'hover'
    return df[['uid', 'claim_text', 'split', 'content_status', 'dataset']]

def main(test_mode=False, max_claims=None):
    print("="*80)
    print("LOADING HOVER DATA")
    print("="*80)

    raw_path = os.path.join('..', '..', 'raw', 'hover', 'data', 'hover', 'hover_train_release_v1.1.json')
    df = load_hover_json(raw_path, test_mode=test_mode, max_claims=max_claims)

    print("\nTotal claims loaded:", len(df))
    print("\n" + "="*80)
    print("SAMPLE DATA")
    print("="*80)
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    print("\nShape:", df.shape)

    df_unified = claim_schema(df)

    # Save unified dataset
    output_dir = os.path.join('..', '..', 'processed', 'hover')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'unified_hover_test.parquet' if test_mode else 'unified_hover.parquet')
    df_unified.to_parquet(output_file, index=False)
    print("\n" + "="*80)
    print("SAVING OUTPUT")
    print("="*80)
    print(f"Saved to: {output_file}")
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"Processed {len(df_unified)} claims")
    print("Output:", output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Process only first 100 claims')
    parser.add_argument('--max-claims', type=int, help='Process only first N claims')
    args = parser.parse_args()
    main(test_mode=args.test, max_claims=args.max_claims)
