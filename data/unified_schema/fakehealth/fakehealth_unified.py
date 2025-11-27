"""
FakeHealth Unified Schema Processing Script

This script converts the FakeHealth dataset to the unified schema.

FakeHealth has:
- Individual article JSON files with full text
- Expert reviews with 10-point quality criteria
- No binary labels (must derive from expert ratings)

Usage:
    python fakehealth_unified.py [--test] [--max-articles N]
    
    --test: Process only first 100 articles for testing
    --max-articles N: Process only first N articles

Example:
    python fakehealth_unified.py --test
    python fakehealth_unified.py --max-articles 500
    python fakehealth_unified.py  # Process all articles
"""

import numpy as np
import pandas as pd
import json
import os
import argparse
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime, timezone
from article_scraping import threaded_hydrate, hydrate_claim


# ============================================================================
# SECTION 1: Load FakeHealth Data
# ============================================================================

def load_fakehealth_content(base_path="../../raw/fakehealth", max_articles=None):
    """
    Load FakeHealth article content from individual JSON files.
    
    Expected structure:
    - fakehealth/
      - dataset/
        - content/
          - HealthRelease/ (606 files)
          - HealthStory/ (1,700 files)
        - reviews/
          - HealthRelease.json
          - HealthStory.json
    
    Parameters:
    -----------
    base_path : str
        Path to FakeHealth data directory
    max_articles : int, optional
        Maximum number of articles to load (for testing)
    
    Returns:
    --------
    pd.DataFrame with article content
    """
    articles = []
    total_loaded = 0
    
    for content_type in ['HealthRelease', 'HealthStory']:
        content_dir = Path(base_path) / 'dataset' / 'content' / content_type
        
        if not content_dir.exists():
            print(f"Warning: Directory not found: {content_dir}")
            continue
        
        json_files = sorted(content_dir.glob("*.json"))
        print(f"Found {len(json_files)} articles in {content_type}")
        
        for json_file in json_files:
            if max_articles and total_loaded >= max_articles:
                break
                
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract article ID from filename
                # e.g., "news_reviews_00000.json" -> "news_reviews_00000"
                article_id = json_file.stem
                
                # Add metadata
                data['article_id'] = article_id
                data['content_type'] = content_type
                data['file_path'] = str(json_file)
                
                articles.append(data)
                total_loaded += 1
                
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        if max_articles and total_loaded >= max_articles:
            break
    
    df = pd.DataFrame(articles)
    print(f"\nTotal articles loaded: {len(df)}")
    print(f"Content type distribution:\n{df['content_type'].value_counts()}")
    
    return df


def load_fakehealth_reviews(base_path="../../raw/fakehealth"):
    """
    Load FakeHealth expert reviews.
    
    Returns:
    --------
    pd.DataFrame with expert review data
    """
    reviews = []
    
    for content_type in ['HealthRelease', 'HealthStory']:
        review_file = Path(base_path) / 'dataset' / 'reviews' / f'{content_type}.json'
        
        if not review_file.exists():
            print(f"Warning: Review file not found: {review_file}")
            continue
        
        try:
            with open(review_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # data is an array of review objects
            for review in data:
                review['content_type'] = content_type
                reviews.append(review)
            
            print(f"Loaded {len(data)} reviews from {content_type}")
            
        except Exception as e:
            print(f"Error loading {review_file}: {e}")
            continue
    
    df = pd.DataFrame(reviews)
    print(f"\nTotal reviews loaded: {len(df)}")
    
    return df


def derive_labels_from_reviews(reviews_df):
    """
    Derive binary labels from expert review ratings.
    
    FakeHealth uses expert ratings (typically 1-5 scale):
    - Low ratings (1-2) -> poor quality -> "false" or "mixed"
    - High ratings (4-5) -> good quality -> "true"
    - Middle ratings (3) -> ambiguous -> "mixed"
    
    Also considers the 10 evaluation criteria:
    - Count "Not Satisfactory" answers
    - High count -> poor quality -> "false"
    
    Parameters:
    -----------
    reviews_df : pd.DataFrame
        DataFrame with expert reviews
    
    Returns:
    --------
    pd.DataFrame with added label columns
    """
    df = reviews_df.copy()
    
    # Count "Not Satisfactory" answers in criteria
    def count_unsatisfactory(criteria_list):
        if not isinstance(criteria_list, list):
            return None
        count = 0
        for criterion in criteria_list:
            if isinstance(criterion, dict):
                answer = criterion.get('answer', '')
                if 'Not Satisfactory' in str(answer):
                    count += 1
        return count
    
    df['unsatisfactory_count'] = df['criteria'].apply(count_unsatisfactory)
    
    # Derive label based on rating and unsatisfactory count
    def derive_label(row):
        rating = row.get('rating')
        unsatisfactory = row.get('unsatisfactory_count')
        
        # If we have both rating and criteria
        if pd.notna(rating) and pd.notna(unsatisfactory):
            # Low rating OR many unsatisfactory -> false
            if rating <= 2 or unsatisfactory >= 7:
                return 'false'
            # High rating AND few unsatisfactory -> true
            elif rating >= 4 and unsatisfactory <= 3:
                return 'true'
            # Middle ground -> mixed
            else:
                return 'mixed'
        
        # If only rating available
        elif pd.notna(rating):
            if rating <= 2:
                return 'false'
            elif rating >= 4:
                return 'true'
            else:
                return 'mixed'
        
        # If only criteria available
        elif pd.notna(unsatisfactory):
            if unsatisfactory >= 7:
                return 'false'
            elif unsatisfactory <= 3:
                return 'true'
            else:
                return 'mixed'
        
        # No data -> unknown
        return None
    
    df['label'] = df.apply(derive_label, axis=1)
    
    # Label confidence
    # "gold" if we have expert reviews, "weak" if derived
    df['label_confidence'] = 'gold'  # All FakeHealth labels are from experts
    
    print("\n=== Label Distribution ===")
    print(df['label'].value_counts(dropna=False))
    print(f"\nLabeled articles: {df['label'].notna().sum()}")
    print(f"Unlabeled articles: {df['label'].isna().sum()}")
    
    return df


# ============================================================================
# SECTION 2: FakeHealth Schema Function
# ============================================================================

def fakehealth_schema(
    content_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    split: str = "train",
    dataset: str = "fakehealth",
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
    Convert FakeHealth DataFrames to unified schema.
    
    FakeHealth already has article text, but we may need to:
    1. Use existing text where available (most cases)
    2. Only hydrate URLs when text is missing or too short
    3. Merge with expert reviews to get labels
    4. Preserve expert evaluation metadata
    
    Parameters:
    -----------
    content_df : pd.DataFrame
        DataFrame with article content
    reviews_df : pd.DataFrame
        DataFrame with expert reviews (includes labels)
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
    
    # Merge content with reviews on article_id
    # reviews_df has 'news_id', content_df has 'article_id'
    df = content_df.merge(
        reviews_df,
        left_on='article_id',
        right_on='news_id',
        how='left',
        suffixes=('', '_review')
    )
    
    print(f"\nMerged {len(df)} articles with reviews")
    print(f"Articles with reviews: {df['news_id'].notna().sum()}")
    print(f"Articles without reviews: {df['news_id'].isna().sum()}")
    
    # Make a copy
    temp = df.copy()
    
    # Add metadata columns
    temp["dataset"] = dataset
    temp["split"] = split
    
    # Use label_confidence from reviews (already set to 'gold')
    if 'label_confidence' not in temp.columns:
        temp['label_confidence'] = 'gold'
    
    # Rename columns to match unified schema
    if title_col in temp.columns:
        temp.rename(columns={title_col: "claim_text"}, inplace=True)
    
    if url_col in temp.columns and "news_url" not in temp.columns:
        temp.rename(columns={url_col: "news_url"}, inplace=True)
    
    # Handle existing article text
    if text_col in temp.columns:
        temp.rename(columns={text_col: "article_text_original"}, inplace=True)
        
        # FakeHealth has good text coverage, but check for short/missing text
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
        text = row["article_text_final"]
        if pd.isna(text) or len(str(text)) < 50:
            return "title_only"
        elif len(str(text)) < 500:
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
    
    # Keep expert review metadata as additional columns
    # These can be useful for analysis even though they're not in core unified schema
    expert_cols = ['rating', 'reviewers', 'unsatisfactory_count', 'criteria', 
                   'summary', 'category', 'tags', 'news_source']
    for col in expert_cols:
        if col in temp.columns:
            temp[f'expert_{col}'] = temp[col]
    
    print(f"\n=== Final Statistics ===")
    print(f"Total rows: {len(temp)}")
    print(f"\nLabel distribution:")
    print(temp["label"].value_counts(dropna=False, normalize=True))
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
    
    # Load FakeHealth content
    print("="*80)
    print("LOADING FAKEHEALTH CONTENT")
    print("="*80)
    
    if test_mode:
        max_articles = 100
        print("TEST MODE: Processing only 100 articles")
    
    content_df = load_fakehealth_content(
        "../../raw/fakehealth",
        max_articles=max_articles
    )
    
    # Load expert reviews
    print("\n" + "="*80)
    print("LOADING EXPERT REVIEWS")
    print("="*80)
    
    reviews_df = load_fakehealth_reviews("../../raw/fakehealth")
    
    # Derive labels from expert reviews
    print("\n" + "="*80)
    print("DERIVING LABELS FROM EXPERT REVIEWS")
    print("="*80)
    
    reviews_df = derive_labels_from_reviews(reviews_df)
    
    # Display sample
    print("\n" + "="*80)
    print("SAMPLE DATA")
    print("="*80)
    print("\nContent sample:")
    print(content_df.head())
    print("\nReviews sample:")
    print(reviews_df[['news_id', 'rating', 'label', 'unsatisfactory_count']].head())
    print(f"\nContent shape: {content_df.shape}")
    print(f"Reviews shape: {reviews_df.shape}")
    
    # Apply unified schema
    df_unified = fakehealth_schema(
        content_df,
        reviews_df,
        split="train",
        dataset="fakehealth",
        title_col="title",
        url_col="url",
        text_col="text",
        max_workers=12,
        batch_size=100,
        save_batches_dir="../../processed/fakehealth/batches",
        throttle_seconds=1.0,
        show_progress=True
    )
    
    # Save output
    print("\n" + "="*80)
    print("SAVING OUTPUT")
    print("="*80)
    
    output_dir = Path("../../processed/fakehealth")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output filename
    if test_mode:
        output_path = output_dir / "unified_fakehealth_test.parquet"
        sample_path = output_dir / "unified_fakehealth_test.csv"
    else:
        output_path = output_dir / "unified_fakehealth.parquet"
        sample_path = output_dir / "unified_fakehealth_sample.csv"
    
    df_unified.to_parquet(output_path)
    print(f"Saved to: {output_path}")
    
    # Also save a sample CSV for inspection
    df_unified.head(100).to_csv(sample_path, index=False)
    print(f"Sample saved to: {sample_path}")
    
    # Save label statistics
    label_stats_path = output_dir / "label_statistics.txt"
    with open(label_stats_path, 'w') as f:
        f.write("FakeHealth Label Statistics\n")
        f.write("="*50 + "\n\n")
        f.write("Label Distribution:\n")
        f.write(str(df_unified['label'].value_counts(dropna=False)))
        f.write("\n\nLabel by Content Type:\n")
        f.write(str(pd.crosstab(df_unified['content_type'], df_unified['label'], dropna=False)))
        f.write("\n\nExpert Rating Distribution:\n")
        if 'expert_rating' in df_unified.columns:
            f.write(str(df_unified['expert_rating'].value_counts(sort=True)))
    print(f"Label statistics saved to: {label_stats_path}")
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"Processed {len(df_unified)} articles")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process FakeHealth dataset to unified schema'
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