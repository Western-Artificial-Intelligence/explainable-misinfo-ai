from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from tqdm.auto import tqdm
import pandas as pd

sys.path.append("../")
from article_scraper import scrap_from_web
from cleanup_text import normalize_text
from cleanup_schema import cleanup as cleanup_schema

logger = logging.getLogger("fakehealth")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Helpers
def _is_abs_path(p: Path) -> bool:
    try:
        return p.is_absolute()
    except Exception:
        return os.path.isabs(str(p))

def _find_fakehealth_root() -> Path:
    """
    Best-effort discovery for FakeHealth raw folder.
    Expected structure:
      data/raw/fakehealth/dataset/content/HealthRelease/*.json
      data/raw/fakehealth/dataset/reviews/HealthRelease.json
    """
    here = Path(__file__).resolve()
    candidates = []

    candidates.append(Path.cwd() / "data" / "raw" / "fakehealth")
    candidates.append(Path.cwd() / "raw" / "fakehealth")

    for base in [here.parent, *here.parents]:
        candidates.append(base / "data" / "raw" / "fakehealth")
        candidates.append(base / "raw" / "fakehealth")

    def _looks_like_fakehealth(p: Path) -> bool:
        content_ok = (p / "dataset" / "content" / "HealthRelease").exists() or (p / "dataset" / "content" / "HealthStory").exists()
        reviews_ok = (p / "dataset" / "reviews" / "HealthRelease.json").exists() or (p / "dataset" / "reviews" / "HealthStory.json").exists()
        return p.exists() and p.is_dir() and content_ok and reviews_ok

    for c in candidates:
        if _looks_like_fakehealth(c):
            return c

    raise FileNotFoundError(
        "Could not locate FakeHealth raw directory. Expected something like:\n"
        "  data/raw/fakehealth/dataset/content/HealthRelease/news_reviews_00000.json\n"
        "  data/raw/fakehealth/dataset/reviews/HealthRelease.json\n"
        "Make sure FakeHealth is extracted into data/raw/fakehealth."
    )

def _content_status_from_text(txt: Optional[str]) -> str:
    if not txt or not str(txt).strip():
        return "title_only"
    L = len(str(txt))
    return "full_article" if L >= 200 else "partial"

def _count_unsatisfactory(criteria_list: Any) -> Optional[int]:
    if not isinstance(criteria_list, list):
        return None
    c = 0
    for item in criteria_list:
        if isinstance(item, dict):
            ans = str(item.get("answer", "") or "")
            if "Not Satisfactory" in ans:
                c += 1
    return c

def _derive_label(rating: Any, unsat: Any) -> Optional[str]:
    has_rating = pd.notna(rating)
    has_unsat = pd.notna(unsat)

    if has_rating:
        try:
            rating = float(rating)
        except Exception:
            has_rating = False

    if has_unsat:
        try:
            unsat = int(unsat)
        except Exception:
            has_unsat = False

    if has_rating and has_unsat:
        if rating <= 2 or unsat >= 7:
            return "false"
        if rating >= 4 and unsat <= 3:
            return "true"
        return "mixed"

    if has_rating:
        if rating <= 2:
            return "false"
        if rating >= 4:
            return "true"
        return "mixed"

    if has_unsat:
        if unsat >= 7:
            return "false"
        if unsat <= 3:
            return "true"
        return "mixed"

    return None

# Hydration
_domain_lock = threading.Lock()
_domain_next_time: Dict[str, float] = {}
_url_cache_lock = threading.Lock()
_url_cache: Dict[str, Tuple[Optional[str], Optional[str]]] = {}

def _domain_throttle(url: str, min_interval_secs: float) -> None:
    if min_interval_secs <= 0:
        return
    try:
        dom = urlparse(url).netloc.lower() or "<nohost>"
    except Exception:
        dom = "<badurl>"

    now = time.time()
    with _domain_lock:
        nxt = _domain_next_time.get(dom, 0.0)
        wait = max(0.0, nxt - now)
        _domain_next_time[dom] = max(nxt, now) + min_interval_secs

    if wait > 0:
        time.sleep(wait)

def _hydrate_one(url: str, throttle_seconds: float) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (normalized_article_text_or_none, fetch_status_or_none)
    """
    if not url or not str(url).strip():
        return None, "no_url"

    u = str(url).strip()

    with _url_cache_lock:
        if u in _url_cache:
            return _url_cache[u]

    _domain_throttle(u, throttle_seconds)

    text, meta = scrap_from_web(
        u,
        timeout=(5.0, 12.0),
        max_attempts=2,
        use_wayback=False,
        respect_robots=True,
        return_meta=False,
    )

    fetch_status = None
    try:
        fetch_status = (meta or {}).get("fetch_status", None)
    except Exception:
        fetch_status = None

    norm_text = normalize_text(text, strip_boilerplate=True) if text else None

    with _url_cache_lock:
        _url_cache[u] = (norm_text, fetch_status)

    return norm_text, fetch_status

def _hydrate_dataframe(df: pd.DataFrame, threads: int, throttle_seconds: float = 1.0) -> pd.DataFrame:
    """
    Hydrate rows where article_text is missing/too short and url exists.
    Updates df (copy) and returns it.
    """
    out = df.copy()

    art = out.get("article_text", pd.Series([None] * len(out)))
    url = out.get("news_url", pd.Series([None] * len(out)))

    art_len = art.fillna("").astype(str).str.len()
    mask = art.isna() | (art.astype(str).str.strip() == "") | (art_len < 200)
    mask &= url.notna() & (url.astype(str).str.strip() != "")

    idxs = out.index[mask].tolist()
    if not idxs:
        return out

    logger.info("Hydrating %d rows (threads=%d)...", len(idxs), threads)

    status_counts: Dict[str, int] = {}

    def worker(i: int) -> Tuple[int, Optional[str]]:
        t, st = _hydrate_one(str(out.at[i, "news_url"]), throttle_seconds=throttle_seconds)
        if st:
            status_counts[st] = status_counts.get(st, 0) + 1
        return i, t

    if threads <= 1:
        it = tqdm(idxs, total=len(idxs), desc="Hydrating", unit="url")
        for i in it:
            _, t = worker(i)
            if t:
                out.at[i, "article_text"] = t
    else:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            futs = [ex.submit(worker, i) for i in idxs]
            it = tqdm(as_completed(futs), total=len(futs), desc="Hydrating", unit="url")
            for fut in it:
                i, t = fut.result()
                if t:
                    out.at[i, "article_text"] = t

    out["content_status"] = out["article_text"].apply(_content_status_from_text).astype(str)

    if status_counts:
        top = sorted(status_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("Hydration fetch_status (top): %s", top)

    return out

# FakeHealth loading + merge
def _load_content(root: Path) -> pd.DataFrame:
    rows = []
    base = root / "dataset" / "content"

    for content_type in ["HealthRelease", "HealthStory"]:
        d = base / content_type
        if not d.exists():
            logger.warning("Missing content dir: %s", d)
            continue
        files = sorted(d.glob("*.json"))
        if not files:
            continue

        for fp in tqdm(files, desc=f"Loading {content_type}", unit="file"):
            try:
                data = json.loads(fp.read_text(encoding="utf-8", errors="replace"))
                if not isinstance(data, dict):
                    continue
                rows.append(
                    {
                        "article_id": fp.stem,
                        "content_type": content_type,
                        "title": data.get("title", None),
                        "news_url": data.get("url", None),
                        "text": data.get("text", None),
                    }
                )
            except Exception as e:
                logger.warning("Failed loading %s: %s", fp, e)

    if not rows:
        return pd.DataFrame(columns=["article_id", "content_type", "title", "news_url", "text"])
    return pd.DataFrame(rows)


def _load_reviews(root: Path) -> pd.DataFrame:
    rows = []
    base = root / "dataset" / "reviews"

    for content_type in ["HealthRelease", "HealthStory"]:
        fp = base / f"{content_type}.json"
        if not fp.exists():
            logger.warning("Missing review file: %s", fp)
            continue

        try:
            data = json.loads(fp.read_text(encoding="utf-8", errors="replace"))
            if not isinstance(data, list):
                continue
            for r in data:
                if not isinstance(r, dict):
                    continue
                crit = r.get("criteria", None)
                unsat = _count_unsatisfactory(crit)
                rating = r.get("rating", None)
                label = _derive_label(rating, unsat)

                rows.append(
                    {
                        "news_id": r.get("news_id", None),
                        "rating": rating,
                        "unsatisfactory_count": unsat,
                        "label": label,
                        "label_confidence": "gold",
                        "content_type_review": content_type,
                    }
                )
        except Exception as e:
            logger.warning("Failed loading %s: %s", fp, e)

    if not rows:
        return pd.DataFrame(columns=["news_id", "rating", "unsatisfactory_count", "label", "label_confidence"])
    return pd.DataFrame(rows)


def build_base_dataframe(root: Path) -> pd.DataFrame:
    """
    Build base rows:
      id=article_id, claim_text=title, article_text=text, news_url=url, label from reviews.
    Drops unlabeled rows (no expert review => cannot go into true/false/mixed outputs).
    """
    content = _load_content(root)
    if content.empty:
        raise RuntimeError("No FakeHealth content JSONs found under: " + str(root / "dataset" / "content"))

    reviews = _load_reviews(root)

    df = content.merge(
        reviews,
        left_on="article_id",
        right_on="news_id",
        how="left",
        suffixes=("", "_review"),
    ).copy()

    before = len(df)
    df = df[df["label"].notna()].copy()
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d unlabeled rows (no expert review).", dropped)

    df["claim_text"] = df["title"].apply(lambda x: normalize_text(x) if pd.notna(x) else None)
    df["article_text"] = df["text"].apply(
        lambda x: normalize_text(x, strip_boilerplate=False) if pd.notna(x) else None
    )

    df["content_status"] = df["article_text"].apply(_content_status_from_text).astype(str)

    out = pd.DataFrame(
        {
            "article_id": df["article_id"],
            "claim_text": df["claim_text"],
            "article_text": df["article_text"],
            "news_url": df["news_url"],
            "label": df["label"].astype(str),
            "label_confidence": df.get("label_confidence", "gold"),
        }
    )

    out = out[out["claim_text"].notna() & (out["claim_text"].astype(str).str.strip() != "")].reset_index(drop=True)
    out["content_status"] = out["article_text"].apply(_content_status_from_text).astype(str)

    return out


def finalize_and_write(df: pd.DataFrame, save_dir: Path) -> None:
    """
    Run cleanup_schema.cleanup() to enforce unified columns, then write 3 parquet files.
    """
    cleaned = cleanup_schema(
        df,
        id_col="article_id",
        claim_col="claim_text",
        article_col="article_text",
        label_col="label",
        dataset="fakehealth",
    )

    save_dir.mkdir(parents=True, exist_ok=True)

    for lbl in ("true", "false", "mixed"):
        part = cleaned[cleaned["label"] == lbl].reset_index(drop=True)
        out_path = save_dir / f"fakehealth_{lbl}.parquet"
        part.to_parquet(out_path, index=False)
        logger.info("Wrote %s (%d rows)", out_path, len(part))

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build unified FakeHealth parquet files (true/false/mixed).")
    p.add_argument("--test", action="store_true", help="If set, run only on first 100 labeled rows.")
    p.add_argument("--threads", type=int, default=1, help="Number of threads for hydration (1-14).")
    p.add_argument("--save-dir", type=str, required=True, help="Absolute path to directory for outputs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    threads = int(args.threads)
    if threads < 1 or threads > 14:
        raise ValueError("--threads must be in [1, 14].")

    save_dir = Path(args.save_dir).expanduser()
    if not _is_abs_path(save_dir):
        raise ValueError("--save-dir must be an absolute path.")

    root = _find_fakehealth_root()
    logger.info("FakeHealth root: %s", root)

    base = build_base_dataframe(root)

    if args.test:
        base = base.head(100).reset_index(drop=True)
        logger.info("TEST MODE: using first 100 labeled rows")

    base = _hydrate_dataframe(base, threads=threads, throttle_seconds=1.0)

    base["content_status"] = base["article_text"].apply(_content_status_from_text).astype(str)

    finalize_and_write(base, save_dir=save_dir)


if __name__ == "__main__":
    main()
