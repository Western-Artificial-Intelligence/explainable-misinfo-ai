from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm

# Import shared utilities
THIS_DIR = Path(__file__).resolve().parent
HELPERS_DIR = THIS_DIR.parent
if str(HELPERS_DIR) not in sys.path:
    sys.path.insert(0, str(HELPERS_DIR))

from article_scraper import scrap_from_web
from cleanup_text import normalize_text
from cleanup_schema import cleanup as cleanup_schema

logger = logging.getLogger("fakenewsnet")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


CSV_SPECS: Dict[str, Tuple[str, str]] = {
    "politifact_fake.csv": ("politifact", "false"),
    "politifact_real.csv": ("politifact", "true"),
    "gossipcop_fake.csv": ("gossipcop", "false"),
    "gossipcop_real.csv": ("gossipcop", "true"),
}

# for verification
UNIFIED_COLS = [
    "dataset",
    "id",
    "claim_text",
    "article_text",
    "content_status",
    "label_raw",
    "label",
    "label_confidence",
    "label_mode",
    "label_3way",
    "label_bin",
    "source_id",
    "claim_norm_hash",
    "lang",
    "content_char_len",
]


def _is_abs_path(p: Path) -> bool:
    try:
        return p.is_absolute()
    except Exception:
        return os.path.isabs(str(p))


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8", encoding_errors="replace")
    except TypeError:
        return pd.read_csv(path, dtype=str, encoding="utf-8")


def _looks_like_fakenewsnet_dir(d: Path) -> bool:
    if not d.exists() or not d.is_dir():
        return False
    return all((d / fn).exists() for fn in CSV_SPECS.keys())


def _resolve_dataset_dir() -> Path:
    """Resolve dataset directory robustly without adding extra CLI flags."""
    candidates: list[Path] = []

    try:
        project_root = THIS_DIR.parents[2]
        candidates.append(project_root / "data" / "raw" / "fakenewsnet" / "dataset")
    except Exception:
        pass

    cwd = Path.cwd()
    candidates.append(cwd / "data" / "raw" / "fakenewsnet" / "dataset")
    candidates.append(cwd / "fakenewsnet" / "dataset")
    candidates.append(cwd / "dataset")

    for c in candidates:
        if _looks_like_fakenewsnet_dir(c):
            return c

    expected = ", ".join(sorted(CSV_SPECS.keys()))
    raise FileNotFoundError(
        "Could not locate FakeNewsNet CSVs. Expected a directory containing: "
        f"{expected}. Tried: " + "; ".join(str(x) for x in candidates)
    )


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort standardization across FakeNewsNet CSV variants."""
    rename: Dict[str, str] = {}

    if "news_id" in df.columns and "id" not in df.columns:
        rename["news_id"] = "id"
    if "article_id" in df.columns and "id" not in df.columns:
        rename["article_id"] = "id"

    if "url" in df.columns and "news_url" not in df.columns:
        rename["url"] = "news_url"

    if "claim" in df.columns and "title" not in df.columns:
        rename["claim"] = "title"

    if "content" in df.columns and "text" not in df.columns:
        rename["content"] = "text"
    if "body" in df.columns and "text" not in df.columns:
        rename["body"] = "text"

    if rename:
        df = df.rename(columns=rename)

    for col in ["id", "news_url", "title"]:
        if col not in df.columns:
            df[col] = None

    df["id"] = df["id"].fillna("").astype(str)
    df["news_url"] = df["news_url"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)

    return df

def _load_fakenewsnet(base_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for filename, (source, label) in CSV_SPECS.items():
        csv_path = base_dir / filename
        logger.info("Loading %s", csv_path)
        df = _safe_read_csv(csv_path)
        df = _standardize_columns(df)

        df["source_dataset"] = source
        df["label"] = label

        if "tweet_ids" in df.columns:
            df = df.drop(columns=["tweet_ids"], errors="ignore")

        rows.append(df[["id", "news_url", "title", "source_dataset", "label"]])

    out = pd.concat(rows, ignore_index=True)

    out["id"] = out["id"].astype(str)
    out = out.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)
    return out


def _hydrate_one(url: object) -> Tuple[Optional[str], str]:
    """Return (normalized_article_text_or_None, content_status)."""
    if url is None or (isinstance(url, float) and pd.isna(url)):
        return None, "no_url"

    u = str(url).strip()
    if not u:
        return None, "no_url"

    try:
        txt, meta = scrap_from_web(
            u,
            timeout=(5.0, 12.0),
            max_attempts=2,
            use_wayback=True,
            respect_robots=True,
            return_meta=True,
        )
    except Exception:
        return None, "worker_exception"

    status = str(meta.get("fetch_status") or "unknown")

    if txt and str(txt).strip():
        cleaned = normalize_text(str(txt), strip_boilerplate=True)
        if cleaned and cleaned.strip():
            return cleaned, status

    return None, status

def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has the unified output schema and column order."""
    for c in UNIFIED_COLS:
        if c not in df.columns:
            df[c] = pd.Series([None] * len(df))
    return df[UNIFIED_COLS]

def run(*, test: bool, threads: int, save_dir: Path) -> None:
    base_dir = _resolve_dataset_dir()
    df = _load_fakenewsnet(base_dir)

    if test:
        logger.info("TEST MODE: Processing only first 100 articles")
        df = df.head(100).copy()

    n = len(df)
    if n == 0:
        logger.warning("No rows loaded. Writing empty parquet files.")
        empty = _ensure_schema(pd.DataFrame())
        for name in ["fakenewsnet_true", "fakenewsnet_false", "fakenewsnet_mixed"]:
            empty.to_parquet(save_dir / f"{name}.parquet", index=False)
        return

    urls = df["news_url"].tolist()

    article_texts: list[Optional[str]] = [None] * n
    statuses: list[str] = [""] * n
    status_counts: Dict[str, int] = {}

    if threads <= 1:
        for i, url in enumerate(tqdm(urls, total=n, desc="Scraping x1")):
            txt, st = _hydrate_one(url)
            article_texts[i] = txt
            statuses[i] = st
            status_counts[st] = status_counts.get(st, 0) + 1
    else:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            fut_to_idx = {ex.submit(_hydrate_one, url): i for i, url in enumerate(urls)}
            for fut in tqdm(as_completed(fut_to_idx), total=n, desc=f"Scraping x{threads}"):
                i = fut_to_idx[fut]
                try:
                    txt, st = fut.result()
                except Exception:
                    txt, st = None, "worker_exception"
                article_texts[i] = txt
                statuses[i] = st
                status_counts[st] = status_counts.get(st, 0) + 1

    df["article_text"] = article_texts
    df["content_status"] = statuses
    df["claim_text"] = df["title"].apply(lambda x: normalize_text(x) if pd.notna(x) else None)
    df["label_confidence"] = 1.0

    cleaned = cleanup_schema(
        df[["id", "claim_text", "article_text", "content_status", "label", "label_confidence"]],
        id_col="id",
        claim_col="claim_text",
        article_col="article_text",
        label_col="label",
        dataset="fakenewsnet",
    )

    true_df = cleaned[cleaned["label"] == "true"].reset_index(drop=True)
    false_df = cleaned[cleaned["label"] == "false"].reset_index(drop=True)
    mixed_df = cleaned[cleaned["label"] == "mixed"].reset_index(drop=True)

    true_df = _ensure_schema(true_df)
    false_df = _ensure_schema(false_df)
    mixed_df = _ensure_schema(mixed_df)

    out_true = save_dir / "fakenewsnet_true.parquet"
    out_false = save_dir / "fakenewsnet_false.parquet"
    out_mixed = save_dir / "fakenewsnet_mixed.parquet"

    true_df.to_parquet(out_true, index=False)
    false_df.to_parquet(out_false, index=False)
    mixed_df.to_parquet(out_mixed, index=False)

    logger.info("Saved: %s (%d rows)", out_true, len(true_df))
    logger.info("Saved: %s (%d rows)", out_false, len(false_df))
    logger.info("Saved: %s (%d rows)", out_mixed, len(mixed_df))

    if status_counts:
        top = sorted(status_counts.items(), key=lambda x: (-x[1], x[0]))
        logger.info("fetch_status summary (top 12): %s", top[:12])

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FakeNewsNet -> unified schema (true/false/mixed parquet splits)"
    )
    p.add_argument("--test", action="store_true", help="Process only first 100 articles.")
    p.add_argument(
        "--threads",
        type=int,
        default=1,
        choices=range(1, 15),
        help="Number of threads (1-14).",
    )
    p.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Absolute path to output directory.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)

    if not _is_abs_path(save_dir):
        raise ValueError(f"--save-dir must be an absolute path: {save_dir}")

    save_dir.mkdir(parents=True, exist_ok=True)
    run(test=bool(args.test), threads=int(args.threads), save_dir=save_dir)


if __name__ == "__main__":
    main()
