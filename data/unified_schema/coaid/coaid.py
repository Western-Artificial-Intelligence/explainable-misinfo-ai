from __future__ import annotations

import argparse
import logging
import os
import re
import threading
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse
from tqdm.auto import tqdm

sys.path.append("../")
from article_scraper import scrap_from_web
from cleanup_text import normalize_text
from cleanup_schema import cleanup as cleanup_schema

import pandas as pd

logger = logging.getLogger("coaid")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# Helpers
_SNAPSHOT_RE = re.compile(r"^\d{2}-\d{2}-\d{4}$")

def _is_abs_path(p: Path) -> bool:
    try:
        return p.is_absolute()
    except Exception:
        return os.path.isabs(str(p))

def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8", encoding_errors="replace")
    except TypeError:
        return pd.read_csv(path, low_memory=False, encoding="utf-8")

def _find_coaid_root() -> Path:
    """
    Best-effort discovery for CoAID raw folder.
    Expected structure:
      data/raw/coaid/<SNAPSHOT>/*.csv
    Notebook used ../../raw/coaid/<SNAPSHOT>/... historically.
    """
    here = Path(__file__).resolve()
    candidates = []

    candidates.append(Path.cwd() / "data" / "raw" / "coaid")
    candidates.append(Path.cwd() / "raw" / "coaid")

    for base in [here.parent, *here.parents]:
        candidates.append(base / "data" / "raw" / "coaid")
        candidates.append(base / "raw" / "coaid")

    for c in candidates:
        if c.exists() and c.is_dir():
            # must contain at least one snapshot dir
            snaps = [d for d in c.iterdir() if d.is_dir() and _SNAPSHOT_RE.match(d.name)]
            if snaps:
                return c

    raise FileNotFoundError(
        "Could not locate CoAID raw directory. Expected something like:\n"
        "  data/raw/coaid/05-01-2020/ClaimFakeCOVID-19.csv\n"
        "Make sure the CoAID repo snapshot is extracted into data/raw/coaid."
    )

def _infer_label_from_filename(name: str) -> Optional[str]:
    n = name.lower()
    if "real" in n:
        return "true"
    if "fake" in n:
        return "false"
    return None

def _content_status_from_text(txt: Optional[str]) -> str:
    if not txt or not str(txt).strip():
        return "title_only"
    L = len(str(txt))
    return "full_article" if L >= 200 else "partial"

def _pick_best_url_row(row: pd.Series) -> Optional[str]:
    for col in ("news_url", "news_url2", "news_url3", "news_url4", "news_url5", "archive"):
        v = row.get(col, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def _id_token(x: object) -> str:
    try:
        if pd.isna(x):
            return "na"
    except Exception:
        pass

    s = str(x)
    try:
        f = float(s)
        if f.is_integer():
            s = str(int(f))
    except Exception:
        pass

    s = re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_")
    return (s[:60] if s else "na")

def _claim_hash_for_conflicts(claim_text: Optional[str]) -> Optional[str]:
    """
    We need a stable grouping key to detect conflicts across snapshots.
    Use normalize_text() (shared utility) then hash it.
    """
    norm = normalize_text(claim_text)
    if not norm:
        return None
    import hashlib
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()

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
    Hydrate rows missing article_text but with a URL.
    Writes back into df (copy) and returns it.
    """
    out = df.copy()

    art = out.get("article_text", pd.Series([None] * len(out)))
    url = out.get("news_url", pd.Series([None] * len(out)))

    mask = art.isna() | (art.astype(str).str.strip() == "")
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
        it = idxs
        if tqdm is not None:
            it = tqdm(it, total=len(idxs), desc="Hydrating", unit="url")
        for i in it:
            _, t = worker(i)
            if t:
                out.at[i, "article_text"] = t

    else:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            futs = [ex.submit(worker, i) for i in idxs]
            it = as_completed(futs)
            if tqdm is not None:
                it = tqdm(it, total=len(futs), desc="Hydrating", unit="url")
            for fut in it:
                i, t = fut.result()
                if t:
                    out.at[i, "article_text"] = t

    out["content_status"] = out["article_text"].apply(_content_status_from_text).astype(str)

    if status_counts:
        top = sorted(status_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        logger.info("Hydration fetch_status (top): %s", top)

    return out

# main
def build_base_dataframe(root: Path) -> pd.DataFrame:
    """
    Load CoAID snapshots and build a unified row table from:
      - Claim{Real,Fake}COVID-19.csv
      - News{Real,Fake}COVID-19.csv
    Skips tweets/replies mapping files by construction.
    """
    rows = []

    snapshots = sorted([d for d in root.iterdir() if d.is_dir() and _SNAPSHOT_RE.match(d.name)], key=lambda p: p.name)

    wanted = [
        "ClaimFakeCOVID-19.csv",
        "ClaimRealCOVID-19.csv",
        "NewsFakeCOVID-19.csv",
        "NewsRealCOVID-19.csv",
    ]

    for snap in snapshots:
        for fname in wanted:
            fpath = snap / fname
            if not fpath.exists():
                continue

            label = _infer_label_from_filename(fname)
            kind = "claim" if fname.lower().startswith("claim") else "news"
            if label not in {"true", "false"}:
                continue

            df = _safe_read_csv(fpath)

            raw_id = df["Unnamed: 0"] if "Unnamed: 0" in df.columns else df.index
            ids = [f"coaid_{snap.name}_{kind}_{label}_{_id_token(x)}" for x in raw_id]

            if kind == "claim":
                claim_text = df.get("title", pd.Series([None] * len(df)))
                news_url = df.get("news_url", pd.Series([None] * len(df)))
                
                claim_text = claim_text.apply(lambda x: normalize_text(x) if pd.notna(x) else None)
                article_text = pd.Series([None] * len(df))
            else:
                claim_text = df.get("title", df.get("newstitle", pd.Series([None] * len(df))))
                claim_text = claim_text.apply(lambda x: normalize_text(x) if pd.notna(x) else None)

                content = df.get("content", pd.Series([None] * len(df)))
                article_text = content.apply(lambda x: normalize_text(x, strip_boilerplate=False) if pd.notna(x) else None)

                if any(c in df.columns for c in ("news_url", "news_url2", "news_url3", "news_url4", "news_url5", "archive")):
                    news_url = df.apply(_pick_best_url_row, axis=1)
                else:
                    news_url = pd.Series([None] * len(df))

            temp = pd.DataFrame(
                {
                    "id": ids,
                    "claim_text": claim_text,
                    "article_text": article_text,
                    "news_url": news_url,
                    "label": label,
                    "label_confidence": "gold",
                }
            )

            temp["content_status"] = temp["article_text"].apply(_content_status_from_text).astype(str)
            rows.append(temp)

            logger.info("Loaded %s/%s: %d rows", snap.name, fname, len(temp))

    if not rows:
        raise RuntimeError("No CoAID claim/news CSVs found under: " + str(root))

    out = pd.concat(rows, ignore_index=True)
    out = out[out["claim_text"].notna() & (out["claim_text"].astype(str).str.strip() != "")].reset_index(drop=True)
    return out

def apply_mixed_labeling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark label='mixed' for any claim hash that appears with both true and false.
    """
    out = df.copy()
    out["_claim_hash"] = out["claim_text"].apply(_claim_hash_for_conflicts)

    valid = out["_claim_hash"].notna()
    grp = out.loc[valid].groupby("_claim_hash")["label"].agg(lambda s: set([x for x in s if x in {"true", "false"}]))

    mixed_hashes = set([h for h, labs in grp.items() if len(labs) >= 2])

    if mixed_hashes:
        out.loc[out["_claim_hash"].isin(mixed_hashes), "label"] = "mixed"

    out.drop(columns=["_claim_hash"], inplace=True, errors="ignore")
    return out

def finalize_and_write(df: pd.DataFrame, save_dir: Path) -> None:
    """
    Run cleanup_schema.cleanup() to enforce unified columns, then patch label_* fields
    to support true/false/mixed outputs, and write 3 parquet files.
    """
    cleaned = cleanup_schema(
        df,
        id_col="id",
        claim_col="claim_text",
        article_col="article_text",
        label_col="label",
        dataset="coaid",
    )

    label_3way_map = {"false": 0, "mixed": 1, "true": 2}
    label_bin_map = {"false": 0, "true": 1}

    cleaned["label_3way"] = cleaned["label"].map(label_3way_map).fillna(-100).astype(int)
    cleaned["label_bin"] = cleaned["label"].map(label_bin_map).fillna(-100).astype(int)

    save_dir.mkdir(parents=True, exist_ok=True)

    for lbl in ("true", "false", "mixed"):
        part = cleaned[cleaned["label"] == lbl].reset_index(drop=True)
        out_path = save_dir / f"coaid_{lbl}.parquet"
        part.to_parquet(out_path, index=False)
        logger.info("Wrote %s (%d rows)", out_path, len(part))

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build unified CoAID parquet files (true/false/mixed).")
    p.add_argument("--test", action="store_true", help="If set, run only on first 100 loaded rows.")
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

    root = _find_coaid_root()
    logger.info("CoAID root: %s", root)

    base = build_base_dataframe(root)

    if args.test:
        base = base.head(100).reset_index(drop=True)
        logger.info("TEST MODE: using first 100 rows")

    # label conflicts -> mixed
    base = apply_mixed_labeling(base)

    base = _hydrate_dataframe(base, threads=threads, throttle_seconds=1.0)

    base["content_status"] = base["article_text"].apply(_content_status_from_text).astype(str)

    # write
    finalize_and_write(base, save_dir=save_dir)

if __name__ == "__main__":
    main()
