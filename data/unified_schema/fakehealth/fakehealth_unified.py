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
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent                 # .../data/unified_schema/fakehealth
PROJECT_ROOT = THIS_DIR.parents[2]                         # .../WAI

# add project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# add unified_schema dir directly (so we can import cleanup_schema.py reliably)
UNIFIED_SCHEMA_DIR = PROJECT_ROOT / "data" / "unified_schema"
if str(UNIFIED_SCHEMA_DIR) not in sys.path:
    sys.path.insert(0, str(UNIFIED_SCHEMA_DIR))

import numpy as np
import pandas as pd
import json
import argparse
from urllib.parse import urlparse
from article_scraping import threaded_hydrate, hydrate_claim
from data.unified_schema.cleanup_schema import cleanup


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

    if df.empty:
        print("No articles found. Expected content under:")
        print(Path(base_path) / "dataset" / "content")
        return pd.DataFrame(columns=["article_id", "content_type", "file_path", "title", "url", "text"])

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
    dataset: str = "fakehealth",
    title_col: str = "title",
    url_col: str = "url",
    text_col: str = "text",
    max_workers: int = 12,
    batch_size: int = None,
    save_batches_dir: str = None,
    throttle_seconds: float = 1.0,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Claim-schema style wrapper for FakeHealth:
      - preserve ALL original columns after merge
      - add only hydration columns (and do not overwrite originals)
      - hydrate only when original text is missing/too short
      - produce final unified columns: claim_text, news_url, article_text, content_status, etc.
    """

    print("\n" + "="*80)
    print("APPLYING UNIFIED SCHEMA (FakeHealth)")
    print("="*80)

    # 1) Merge content with reviews (preserve everything)
    df = content_df.merge(
        reviews_df,
        left_on="article_id",
        right_on="news_id",
        how="left",
        suffixes=("", "_review"),
    ).copy()

    print(f"\nMerged rows: {len(df)}")
    print(f"With reviews: {df['news_id'].notna().sum()}")
    print(f"Without reviews: {df['news_id'].isna().sum()}")

    # 2) Canonical metadata
    df["dataset"] = dataset
    if "label_confidence" not in df.columns:
        df["label_confidence"] = "gold"

    # 3) Canonical naming for title/url into unified names WITHOUT overwriting if already present
    #    (FakeHealth often uses title/url/text, but keep defensive.)
    if "claim_text" not in df.columns and title_col in df.columns:
        df = df.rename(columns={title_col: "claim_text"})
    if "news_url" not in df.columns and url_col in df.columns:
        df = df.rename(columns={url_col: "news_url"})

    # 4) Keep original article text in a dedicated column (never overwrite)
    if "article_text_original" not in df.columns:
        if text_col in df.columns:
            df = df.rename(columns={text_col: "article_text_original"})
        else:
            df["article_text_original"] = None

    # 5) Decide which rows need hydration
    #    Note: handle non-string text safely
    orig_len = df["article_text_original"].fillna("").astype(str).str.len()
    df["needs_hydration"] = df["article_text_original"].isna() | (orig_len < 200)

    n_need = int(df["needs_hydration"].sum())
    print(f"\nRows needing hydration: {n_need}")
    print(f"Rows with sufficient original text: {len(df) - n_need}")

    # 6) Hydrate only subset; then JOIN results back by index (this is the key fix)
    hydration_cols = [
        "article_text",        # hydrated text
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

    if n_need > 0:
        rows_to_hydrate = df.loc[df["needs_hydration"]].copy()

        # threaded_hydrate should return DF indexed like input
        res_df = threaded_hydrate(
            rows_to_hydrate,
            title_col="claim_text",
            url_col="news_url",
            hydrate_fn=hydrate_claim,     # keep your hydrate fn
            dataset=dataset,
            max_workers=max_workers,
            batch_size=batch_size,
            save_batches_dir=save_batches_dir,   # IMPORTANT: don’t ignore this
            throttle_seconds=throttle_seconds,
            show_progress=show_progress,
        )

        # Defensive: only keep known hydration cols that exist
        available = [c for c in hydration_cols if c in res_df.columns]

        # IMPORTANT: do not overwrite any existing columns in df
        # If you already have a column with same name, we will store hydrated values in a prefixed version
        cols_to_add = []
        rename_map = {}
        for c in available:
            if c in df.columns:
                # avoid overwrite: put into "hydrated_<col>"
                new_c = f"hydrated_{c}"
                rename_map[c] = new_c
                cols_to_add.append(new_c)
            else:
                cols_to_add.append(c)

        res_use = res_df[available].copy()
        if rename_map:
            res_use = res_use.rename(columns=rename_map)

        # Align indices and join
        res_use = res_use.reindex(df.index)  # ensures full index; non-hydrated rows will be NaN
        df = df.join(res_use, how="left")

        # For convenience, define "article_text_hydrated" consistently
        if "article_text" in res_use.columns:
            df["article_text_hydrated"] = df["article_text"]
        elif "hydrated_article_text" in res_use.columns:
            df["article_text_hydrated"] = df["hydrated_article_text"]
        else:
            df["article_text_hydrated"] = None
    else:
        df["article_text_hydrated"] = None

    # 7) Build FINAL article_text (hydrated wins, else original)
    df["article_text"] = df["article_text_hydrated"].fillna(df["article_text_original"])

    # 8) Compute content_status + lengths based on FINAL text (not based on partial cols)
    final_len = df["article_text"].fillna("").astype(str).str.len()
    df["content_char_len"] = final_len.astype(int)

    def _content_status_from_len(n: int) -> str:
        if n < 50:
            return "title_only"
        if n < 500:
            return "partial"
        return "full_article"

    df["content_status"] = df["content_char_len"].apply(_content_status_from_len)

    # 9) Fill hydration-ish defaults consistently
    df["is_hydrated"] = df["article_text"].notna() & (df["content_char_len"] > 0)

    # fetch_status: only fill missing values; choose based on whether we attempted/needed hydration
    if "fetch_status" not in df.columns:
        df["fetch_status"] = None

    missing_fetch = df["fetch_status"].isna()
    df.loc[missing_fetch & df["needs_hydration"], "fetch_status"] = "not_fetched"
    df.loc[missing_fetch & ~df["needs_hydration"], "fetch_status"] = "existing_text"

    if "is_archived" in df.columns:
        df["is_archived"] = df["is_archived"].fillna(False).astype(bool)
    if "fetch_attempts" in df.columns:
        df["fetch_attempts"] = df["fetch_attempts"].fillna(0).astype(int)

    # source_domain: infer from URL if missing
    if "source_domain" not in df.columns or df["source_domain"].isna().all():
        df["source_domain"] = df["news_url"].apply(
            lambda x: urlparse(x).netloc.lower() if pd.notna(x) else ""
        )
    else:
        df["source_domain"] = df["source_domain"].fillna(
            df["news_url"].apply(lambda x: urlparse(x).netloc.lower() if pd.notna(x) else "")
        ).fillna("")

    # 10) Keep expert metadata WITHOUT clobbering originals
    expert_cols = ["rating", "reviewers", "unsatisfactory_count", "criteria",
                   "summary", "category", "tags", "news_source"]
    for col in expert_cols:
        if col in df.columns and f"expert_{col}" not in df.columns:
            df[f"expert_{col}"] = df[col]

    # 11) Cleanup only helper cols
    df = df.drop(columns=["needs_hydration"], errors="ignore")

    print(f"\n=== Final Statistics ===")
    print(f"Total rows: {len(df)}")
    if "label" in df.columns:
        print("\nLabel distribution:")
        print(df["label"].value_counts(dropna=False, normalize=True))
    print("\nContent status distribution:")
    print(df["content_status"].value_counts(normalize=True))
    print(f"\nHas article_text: {(df['content_char_len'] > 0).mean():.1%}")

    return df

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

    raw_dir = PROJECT_ROOT / 'data' / 'raw' / 'fakehealth'

    content_df = load_fakehealth_content(
        str(raw_dir),
        max_articles=max_articles
    )

    if content_df.empty:
        raise FileNotFoundError(
            f"No FakeHealth articles found. \nChecked: {raw_dir / 'dataset' / 'content'}"
        )

    # Load expert reviews
    print("\n" + "="*80)
    print("LOADING EXPERT REVIEWS")
    print("="*80)

    reviews_df = load_fakehealth_reviews(str(raw_dir))

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
        dataset="fakehealth",
        title_col="title",
        url_col="url",
        text_col="text",
        max_workers=12,
        batch_size=None,
        save_batches_dir=None,
        throttle_seconds=1.0,
        show_progress=True
    )

    # Save output
    print("\n" + "="*80)
    print("SAVING OUTPUT")
    print("="*80)

    output_dir = PROJECT_ROOT / 'data'/'processed' / 'fakehealth'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output filename
    if test_mode:
        output_path = output_dir / "unified_fakehealth_test.parquet"
        sample_path = output_dir / "unified_fakehealth_test.csv"
    else:
        output_path = output_dir / "unified_fakehealth.parquet"
        sample_path = output_dir / "unified_fakehealth_sample.csv"

    # Pick the correct ID column that exists
    id_col = "id" if "id" in df_unified.columns else "article_id"

    final = cleanup(
        df_unified,
        id_col=id_col,
        claim_col="claim_text",
        article_col="article_text",
        label_col="label",
        dataset="fakehealth"
    )

    final.to_parquet(output_path, index=False)
    print(f"Saved to: {output_path}")

    # Also save a sample CSV for inspection
    final.head(100).to_csv(sample_path, index=False)
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
    print(f"Processed {len(final)} articles")
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