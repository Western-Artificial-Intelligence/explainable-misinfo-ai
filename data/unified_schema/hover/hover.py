from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm

sys.path.append("../")
from cleanup_text import normalize_text
from cleanup_schema import cleanup as cleanup_schema

logger = logging.getLogger("hover")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

_SPLITS: Dict[str, str] = {
    "train": "hover_train_release_v1.1.json",
    "dev": "hover_dev_release_v1.1.json",
    "test": "hover_test_release_v1.1.json",
}

_LABEL_MAP = {
    "SUPPORTED": "true",
    "NOT_SUPPORTED": "false",
}

# Helpers
def _is_abs_path(p: Path) -> bool:
    try:
        return p.is_absolute()
    except Exception:
        return os.path.isabs(str(p))

def _find_hover_root() -> Path:
    """Best-effort discovery for HoVer raw folder.

    Expected structure:
      data/raw/hover/data/hover/hover_train_release_v1.1.json
      data/raw/hover/data/hover/hover_dev_release_v1.1.json
      data/raw/hover/data/hover/hover_test_release_v1.1.json
    """
    here = Path(__file__).resolve()
    candidates: List[Path] = []

    candidates.append(Path.cwd() / "data" / "raw" / "hover" / "data" / "hover")
    candidates.append(Path.cwd() / "raw" / "hover" / "data" / "hover")

    for base in [here.parent, *here.parents]:
        candidates.append(base / "data" / "raw" / "hover" / "data" / "hover")
        candidates.append(base / "raw" / "hover" / "data" / "hover")

    def _looks_like_hover(p: Path) -> bool:
        if not (p.exists() and p.is_dir()):
            return False
        return all((p / fn).exists() for fn in _SPLITS.values())

    for c in candidates:
        if _looks_like_hover(c):
            return c.resolve()

    raise FileNotFoundError(
        "Could not locate HoVer raw directory. Expected something like:\n"
        "  data/raw/hover/data/hover/hover_train_release_v1.1.json\n"
        "  data/raw/hover/data/hover/hover_dev_release_v1.1.json\n"
        "  data/raw/hover/data/hover/hover_test_release_v1.1.json\n"
        "Make sure HoVer is extracted into data/raw/hover."
    )

def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data

def _safe_norm(x: object) -> Optional[str]:
    try:
        return normalize_text(x)
    except Exception:
        return None

def _load_split(split: str, path: Path) -> pd.DataFrame:
    rows = _load_json_list(path)
    df = pd.DataFrame(rows)

    if "uid" not in df.columns:
        raise KeyError(f"Missing required field 'uid' in {path}")

    if "claim" in df.columns and "claim_text" not in df.columns:
        df = df.rename(columns={"claim": "claim_text"})

    if "claim_text" not in df.columns:
        raise KeyError(f"Missing required field 'claim'/'claim_text' in {path}")

    if "label" in df.columns:
        df["label_src"] = df["label"]
        df["label_norm"] = df["label"].map(_LABEL_MAP)
    else:
        df["label_src"] = None
        df["label_norm"] = "mixed"

    df["label_norm"] = df["label_norm"].fillna("mixed")

    df["article_text"] = None
    df["content_status"] = "title_only"
    df["label_confidence"] = df["label_norm"].apply(lambda x: "gold" if x in ("true", "false") else None)

    out = pd.DataFrame(
        {
            "uid": df["uid"].astype(str),
            "claim_text": df["claim_text"],
            "label_norm": df["label_norm"],
            "label_src": df["label_src"],
            "article_text": df["article_text"],
            "content_status": df["content_status"],
            "label_confidence": df["label_confidence"],
            "split": split,
        }
    )

    return out

def _normalize_claims_in_parallel(texts: List[object], threads: int) -> List[Optional[str]]:
    n = len(texts)
    if n == 0:
        return []

    if threads <= 1:
        out: List[Optional[str]] = []
        for t in tqdm(texts, total=n, desc="Normalizing claim_text", unit="row"):
            out.append(_safe_norm(t))
        return out

    chunk_size = max(256, (n + threads - 1) // threads)

    def work(lo: int, hi: int) -> Tuple[int, List[Optional[str]]]:
        return lo, [_safe_norm(t) for t in texts[lo:hi]]

    results: List[Optional[str]] = [None] * n
    futs = []
    with ThreadPoolExecutor(max_workers=threads) as ex:
        for lo in range(0, n, chunk_size):
            hi = min(n, lo + chunk_size)
            futs.append(ex.submit(work, lo, hi))

        for fut in tqdm(as_completed(futs), total=len(futs), desc="Normalizing claim_text", unit="chunk"):
            lo, chunk_out = fut.result()
            results[lo : lo + len(chunk_out)] = chunk_out

    return results

# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build unified HoVer parquet files (true/false/mixed).")
    p.add_argument("--test", action="store_true", help="If set, run only on first 100 rows total.")
    p.add_argument("--threads", type=int, default=1, help="Number of threads (1-14).")
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
    save_dir.mkdir(parents=True, exist_ok=True)

    hover_root = _find_hover_root()
    logger.info("HoVer root: %s", hover_root)

    paths = {s: hover_root / fn for s, fn in _SPLITS.items()}

    loaded: Dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=min(threads, 3)) as ex:
        futs = {ex.submit(_load_split, s, paths[s]): s for s in ["train", "dev", "test"]}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Loading HoVer splits", unit="split"):
            s = futs[fut]
            loaded[s] = fut.result()

    df = pd.concat([loaded["train"], loaded["dev"], loaded["test"]], ignore_index=True)

    if args.test:
        df = df.head(100).reset_index(drop=True)
        logger.info("TEST MODE: using first 100 rows total")

    df["claim_text"] = _normalize_claims_in_parallel(df["claim_text"].tolist(), threads)
    df = df[df["claim_text"].notna() & (df["claim_text"].astype(str).str.strip() != "")].reset_index(drop=True)

    cleaned = cleanup_schema(
        df,
        id_col="uid",
        claim_col="claim_text",
        article_col="article_text",
        label_col="label_norm",
        dataset="hover",
    )

    raw_map = df[["uid", "label_src"]].drop_duplicates(subset=["uid"])
    cleaned = cleaned.merge(raw_map, left_on="id", right_on="uid", how="left")
    cleaned["label_raw"] = cleaned["label_src"]
    cleaned = cleaned.drop(columns=[c for c in ["uid", "label_src"] if c in cleaned.columns], errors="ignore")

    cleaned["label"] = cleaned["label"].fillna("mixed")
    label_3way_map = {"false": 0, "mixed": 1, "true": 2}
    label_bin_map = {"false": 0, "true": 1}
    cleaned["label_3way"] = cleaned["label"].map(label_3way_map).fillna(-100).astype(int)
    cleaned["label_bin"] = cleaned["label"].map(label_bin_map).fillna(-100).astype(int)
    cleaned["lang"] = "en"

    out_cols = [
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
    cleaned = cleaned[out_cols]

    # Split + write
    true_df = cleaned[cleaned["label"] == "true"].reset_index(drop=True)
    false_df = cleaned[cleaned["label"] == "false"].reset_index(drop=True)
    mixed_df = cleaned[cleaned["label"] == "mixed"].reset_index(drop=True)

    out_true = save_dir / "hover_true.parquet"
    out_false = save_dir / "hover_false.parquet"
    out_mixed = save_dir / "hover_mixed.parquet"

    true_df.to_parquet(out_true, index=False)
    false_df.to_parquet(out_false, index=False)
    mixed_df.to_parquet(out_mixed, index=False)

    logger.info("Label distribution: %s", cleaned["label"].value_counts(dropna=False).to_dict())
    logger.info("Wrote %s (%d rows)", out_true, len(true_df))
    logger.info("Wrote %s (%d rows)", out_false, len(false_df))
    logger.info("Wrote %s (%d rows)", out_mixed, len(mixed_df))

if __name__ == "__main__":
    main()
