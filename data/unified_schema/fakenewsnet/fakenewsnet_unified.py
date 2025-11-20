"""
FakeNewsNet Unified Schema Processing Script

This script converts the FakeNewsNet dataset to the unified schema.

Usage:
    python fakenewsnet_unified.py [--test] [--max-articles N]
    
    --test: Process only first 100 articles for testing
    --max-articles N: Process only first N articles

Example:
    python fakenewsnet_unified.py --test
    python fakenewsnet_unified.py --max-articles 1000
    python fakenewsnet_unified.py  # Process all articles
"""

import numpy as np
import pandas as pd
import json
import os
import argparse
from pathlib import Path
from urllib.parse import urlparse
from article_scraping import threaded_hydrate, hydrate_claim


# ============================================================================
# SECTION 1: Load FakeNewsNet Data
# ============================================================================

def load_fakenewsnet_articles(base_path="../../raw/fakenewsnet/dataset", max_articles=None):
    """
    Load FakeNewsNet from CSV files (your setup).

    Expected structure (YOUR ACTUAL STRUCTURE):
    - fakenewsnet/
      - dataset/
         - gossipcop_fake.csv
         - gossipcop_real.csv
         - politifact_fake.csv
         - politifact_real.csv

    Returns:
        DataFrame with: id, title, text, url, label, source_dataset
    """

    csv_files = {
        "politifact_fake":   ("politifact", "fake"),
        "politifact_real":   ("politifact", "real"),
        "gossipcop_fake":    ("gossipcop", "fake"),
        "gossipcop_real":    ("gossipcop", "real"),
    }

    rows = []
    total = 0

    for filename, (source, label) in csv_files.items():
        csv_path = Path(base_path) / f"{filename}.csv"

        if not csv_path.exists():
            print(f"Warning: Missing file: {csv_path}")
            continue

        print(f"Loading {csv_path} ...")
        df = pd.read_csv(csv_path)

        # Normalize columns in case some CSVs use slightly different names
        df = df.rename(columns={
            "news_id": "id",
            "article_id": "id",
            "content": "text",
            "body": "text",
            "claim": "title"
        })

        # Add metadata
        df["source_dataset"] = source
        df["label"] = "true" if label == "real" else "false"

        rows.append(df)

        total += len(df)
        if max_articles and total >= max_articles:
            break

    if not rows:
        print("ERROR: No CSV files loaded.")
        return pd.DataFrame()

    # Merge everything
    df_all = pd.concat(rows, ignore_index=True)

    # Trim to max_articles if needed
    if max_articles:
        df_all = df_all.head(max_articles)

    print("\n===== LOADED CSV VERSION =====")
    print(f"Total rows loaded: {len(df_all)}")
    print(df_all["label"].value_counts())
    print(df_all["source_dataset"].value_counts())
    print(df_all.columns)

    return df_all



# ============================================================================
# SECTION 2: FakeNewsNet Schema Function
# ============================================================================

def fakenewsnet_schema(
    df: pd.DataFrame,
    split: str = "train",
    dataset: str = "fakenewsnet",
    title_col: str = "title",
    url_col: str = "url",
    text_col: str = "text",
    max_workers: int = 12,
    batch_size: int = 100,
    save_batches_dir: str = None,
    throttle_seconds: float = 1.0,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Convert FakeNewsNet DataFrame to unified schema.
    
    FakeNewsNet already has article text in many cases, so we:
    1. Use existing text where available
    2. Only hydrate URLs when text is missing or too short
    3. Preserve all original metadata
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input FakeNewsNet dataframe
    split : str
        Dataset split (train/val/test)
    dataset : str
        Dataset name
    title_col : str
        Column name for article title
    url_col : str
        Column name for article URL
    text_col : str
        Column name for existing article text
    max_workers : int
        Number of parallel workers for hydration
    batch_size : int
        Batch size for saving intermediate results
    save_batches_dir : str
        Directory to save batches (for resumability)
    throttle_seconds : float
        Delay between requests to same domain
    show_progress : bool
        Show progress bar
    
    Returns:
    --------
    pd.DataFrame with unified schema
    """
    print("\n" + "="*80)
    print("APPLYING UNIFIED SCHEMA")
    print("="*80)
    
    # Make a copy to avoid modifying original
    temp = df.copy()
    
    # Add metadata columns
    temp["dataset"] = dataset
    temp["split"] = split
    temp["label_confidence"] = "gold"  # FakeNewsNet has verified labels
    
    # Rename columns to match unified schema
    if title_col in temp.columns:
        temp.rename(columns={title_col: "claim_text"}, inplace=True)
    
    if url_col in temp.columns and "news_url" not in temp.columns:
        temp.rename(columns={url_col: "news_url"}, inplace=True)
    
    # Handle existing article text
    if text_col in temp.columns:
        temp.rename(columns={text_col: "article_text_original"}, inplace=True)
        
        # Determine which rows need hydration
        temp["needs_hydration"] = temp["article_text_original"].isna() | \
                                   (temp["article_text_original"].str.len() < 200)
    else:
        temp["article_text_original"] = None
        temp["needs_hydration"] = True
    
    print(f"\nTotal rows: {len(temp)}")
    print(f"Rows needing hydration: {temp['needs_hydration'].sum()}")
    print(f"Rows with existing text: {(~temp['needs_hydration']).sum()}")
    
    # Only hydrate rows that need it
    rows_to_hydrate = temp[temp["needs_hydration"]].copy()
    
    if len(rows_to_hydrate) > 0:
        print(f"\nHydrating {len(rows_to_hydrate)} articles...")
        
        # Run hydration
        res_df = threaded_hydrate(
            rows_to_hydrate,
            title_col="claim_text",
            url_col="news_url",
            hydrate_fn=hydrate_claim,
            dataset=dataset,
            max_workers=max_workers,
            batch_size=batch_size,
            save_batches_dir=save_batches_dir,
            throttle_seconds=throttle_seconds,
            show_progress=show_progress,
        )
        
        # Expected columns from hydration
        hydration_cols = [
            "article_text",
            "content_status",
            "archive_url",
            "is_archived",
            "source_domain",
            "is_hydrated",
            "fetch_status",
            "lang",
            "content_char_len",
            "claim_norm_hash",
            "ingested_at",
            "fetch_attempts",
            "last_fetch_at",
        ]
        
        # Keep only available columns
        available_cols = [c for c in hydration_cols if c in res_df.columns]
        
        # Join hydration results back to main dataframe
        for col in available_cols:
            if col not in temp.columns:
                temp[col] = None
        
        # Update hydrated rows
        for col in available_cols:
            temp.loc[rows_to_hydrate.index, col] = res_df[col]
    
    else:
        print("\nNo hydration needed - all articles have sufficient text")
        
        # Initialize hydration columns for consistency
        temp["article_text"] = None
        temp["content_status"] = "title_only"
        temp["archive_url"] = None
        temp["is_archived"] = False
        temp["source_domain"] = None
        temp["is_hydrated"] = False
        temp["fetch_status"] = "not_needed"
        temp["lang"] = None
        temp["content_char_len"] = 0
        temp["claim_norm_hash"] = None
        temp["ingested_at"] = pd.Timestamp.now(tz='UTC').isoformat()
        temp["fetch_attempts"] = 0
        temp["last_fetch_at"] = None
    
    # Merge original text with hydrated text
    # Priority: hydrated text > original text > None
    temp["article_text_final"] = temp["article_text"].fillna(temp["article_text_original"])
    
    # Update content status based on final text
    def determine_content_status(row):
        if pd.isna(row["article_text_final"]) or len(str(row["article_text_final"])) < 50:
            return "title_only"
        elif len(str(row["article_text_final"])) < 500:
            return "partial"
        else:
            return "full_article"
    
    temp["content_status"] = temp.apply(determine_content_status, axis=1)
    
    # Update content char length
    temp["content_char_len"] = temp["article_text_final"].fillna("").str.len()
    
    # Set is_hydrated flag
    temp["is_hydrated"] = ~temp["article_text_final"].isna()
    
    # Extract source domain from news_url
    if "source_domain" not in temp.columns or temp["source_domain"].isna().all():
        temp["source_domain"] = temp["news_url"].apply(
            lambda x: urlparse(x).netloc.lower() if pd.notna(x) else ""
        )
    
    # Clean up temporary columns
    temp.drop(columns=["needs_hydration", "article_text_original", "article_text"], 
              inplace=True, errors="ignore")
    
    # Rename final article text column
    temp.rename(columns={"article_text_final": "article_text"}, inplace=True)
    
    # Fill missing values with sensible defaults
    if "fetch_status" in temp.columns:
        temp["fetch_status"] = temp["fetch_status"].fillna("existing_text")
    if "is_archived" in temp.columns:
        temp["is_archived"] = temp["is_archived"].fillna(False).astype(bool)
    if "is_hydrated" in temp.columns:
        temp["is_hydrated"] = temp["is_hydrated"].fillna(True).astype(bool)
    if "fetch_attempts" in temp.columns:
        temp["fetch_attempts"] = temp["fetch_attempts"].fillna(0).astype(int)
    
    print(f"\n=== Final Statistics ===")
    print(f"Total rows: {len(temp)}")
    print(f"\nContent status distribution:")
    print(temp["content_status"].value_counts(normalize=True))
    print(f"\nHydration status:")
    print(f"Hydrated: {temp['is_hydrated'].sum()} ({temp['is_hydrated'].mean():.1%})")
    print(f"Not hydrated: {(~temp['is_hydrated']).sum()}")
    
    if "fetch_status" in temp.columns:
        print(f"\nFetch status distribution:")
        print(temp["fetch_status"].value_counts())
    
    return temp


# ============================================================================
# SECTION 3: Main Execution
# ============================================================================

def main(test_mode=False, max_articles=None):
    """
    Main execution function.
    
    Parameters:
    -----------
    test_mode : bool
        If True, process only 100 articles
    max_articles : int, optional
        Maximum number of articles to process
    """
    
    # Load FakeNewsNet data
    print("="*80)
    print("LOADING FAKENEWSNET DATA")
    print("="*80)
    
    if test_mode:
        max_articles = 100
        print("TEST MODE: Processing only 100 articles")
    
    df = load_fakenewsnet_articles(
        "../../raw/fakenewsnet/dataset",
        max_articles=max_articles
    )
    
    # Display sample
    print("\n" + "="*80)
    print("SAMPLE DATA")
    print("="*80)
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    print(f"\nShape: {df.shape}")
    
    # Apply unified schema
    df_unified = fakenewsnet_schema(
        df,
        split="train",
        dataset="fakenewsnet",
        title_col="title",
        url_col="url",
        text_col="text",
        max_workers=12,
        batch_size=100,
        save_batches_dir="../../processed/fakenewsnet/batches",
        throttle_seconds=1.0,
        show_progress=True
    )
    
    # Save output
    print("\n" + "="*80)
    print("SAVING OUTPUT")
    print("="*80)
    
    output_dir = Path("../../processed/fakenewsnet")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output filename
    if test_mode:
        output_path = output_dir / "unified_fakenewsnet_test.parquet"
        sample_path = output_dir / "unified_fakenewsnet_test.csv"
    else:
        output_path = output_dir / "unified_fakenewsnet.parquet"
        sample_path = output_dir / "unified_fakenewsnet_sample.csv"
    
    df_unified.to_parquet(output_path)
    print(f"Saved to: {output_path}")
    
    # Also save a sample CSV for inspection
    df_unified.head(100).to_csv(sample_path, index=False)
    print(f"Sample saved to: {sample_path}")
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"Processed {len(df_unified)} articles")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process FakeNewsNet dataset to unified schema'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: process only 100 articles'
    )
    parser.add_argument(
        '--max-articles',
        type=int,
        help='Maximum number of articles to process'
    )
    
    args = parser.parse_args()
    
    main(test_mode=args.test, max_articles=args.max_articles)