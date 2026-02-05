from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm

sys.path.append("../")
from cleanup_text import normalize_text
from cleanup_schema import cleanup as cleanup_schema

logger = logging.getLogger("liar")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Helpers
def _is_abs_path(p: Path) -> bool:
    try:
        return p.is_absolute()
    except Exception:
        return os.path.isabs(str(p))


def _find_liar_root() -> Path:
    """Best-effort discovery for LIAR raw folder.

    Expected structure:
      data/raw/liar/train.tsv
      data/raw/liar/valid.tsv
      data/raw/liar/test.tsv
    """
    here = Path(__file__).resolve()
    candidates: List[Path] = []

    candidates.append(Path.cwd() / "data" / "raw" / "liar")
    candidates.append(Path.cwd() / "raw" / "liar")

    for base in [here.parent, *here.parents]:
        candidates.append(base / "data" / "raw" / "liar")
        candidates.append(base / "raw" / "liar")

    for c in candidates:
        if not (c.exists() and c.is_dir()):
            continue
        if (c / "train.tsv").exists() and (c / "valid.tsv").exists() and (c / "test.tsv").exists():
            return c.resolve()

    raise FileNotFoundError(
        "Could not locate LIAR raw directory. Expected something like:\n"
        "  data/raw/liar/train.tsv\n"
        "  data/raw/liar/valid.tsv\n"
        "  data/raw/liar/test.tsv\n"
        "Make sure liar_dataset.zip is extracted into data/raw/liar."
    )


def _safe_read_tsv(path: Path) -> pd.DataFrame:
    cols = [
        "claim_id",
        "label_raw",
        "statement",
        "subject",
        "speaker",
        "job_title",
        "state_info",
        "party_affiliation",
        "barely_true_counts",
        "false_counts",
        "half_true_counts",
        "mostly_true_counts",
        "pants_fire_counts",
        "context",
    ]

    try:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=cols,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
            encoding_errors="replace",
        )
    except TypeError:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=cols,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
        )

    for c in [
        "barely_true_counts",
        "false_counts",
        "half_true_counts",
        "mostly_true_counts",
        "pants_fire_counts",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")

    return df


def _map_label_3way(label_raw: object) -> str:
    """LIAR 6-way -> 3-way mapping."""
    x = str(label_raw).strip().lower() if label_raw is not None else ""
    if x == "true":
        return "true"
    if x in {"mostly-true", "half-true", "barely-true"}:
        return "mixed"
    if x in {"false", "pants-fire"}:
        return "false"
    return "mixed"


def _normalize_claims(texts: List[object], threads: int) -> List[Optional[str]]:
    """Normalize claim texts using shared normalize_text(), optionally threaded."""
    n = len(texts)
    if n == 0:
        return []

    if threads <= 1:
        out: List[Optional[str]] = []
        for x in tqdm(texts, total=n, desc="Normalizing claims", unit="row"):
            try:
                out.append(normalize_text(x))
            except Exception:
                out.append(None)
        return out

    results: List[Optional[str]] = [None] * n

    def worker(i: int, x: object) -> Tuple[int, Optional[str]]:
        try:
            return i, normalize_text(x)
        except Exception:
            return i, None

    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = [ex.submit(worker, i, x) for i, x in enumerate(texts)]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Normalizing claims", unit="row"):
            i, v = fut.result()
            results[i] = v

    return results

# Main pipeline
def build_base_dataframe(root: Path, *, threads: int, test: bool) -> pd.DataFrame:
    """Load LIAR TSVs and build a minimal dataframe ready for cleanup_schema."""
    splits = [("train", "train.tsv"), ("valid", "valid.tsv"), ("test", "test.tsv")]

    frames: List[pd.DataFrame] = []
    for split, fname in splits:
        fpath = root / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing required file: {fpath}")
        df = _safe_read_tsv(fpath)
        df["split"] = split
        frames.append(df)
        logger.info("Loaded %s (%d rows)", fname, len(df))

    raw = pd.concat(frames, ignore_index=True)

    if test:
        raw = raw.head(100).reset_index(drop=True)
        logger.info("TEST MODE: using first 100 rows")

    total_counts = (
        raw["barely_true_counts"]
        + raw["false_counts"]
        + raw["half_true_counts"]
        + raw["mostly_true_counts"]
        + raw["pants_fire_counts"]
    )

    label_conf = total_counts.apply(lambda x: "weak" if int(x) < 10 else "gold")
    norm_claims = _normalize_claims(raw["statement"].tolist(), threads=threads)

    out = pd.DataFrame(
        {
            "id": raw["claim_id"].astype(str),
            "claim_text": norm_claims,
            "article_text": None,
            "content_status": "title_only",
            "label": raw["label_raw"].apply(_map_label_3way),
            "label_confidence": label_conf.astype(str),
            "label_raw_6": raw["label_raw"].apply(lambda x: str(x).strip().lower() if x is not None else None),
        }
    )

    out = out[out["claim_text"].notna() & (out["claim_text"].astype(str).str.strip() != "")].reset_index(drop=True)

    return out


def finalize_and_write(df: pd.DataFrame, save_dir: Path) -> None:
    """Run cleanup_schema, restore 6-way label_raw, and write 3 parquet files."""

    orig_label_raw = df[["id", "label_raw_6"]].rename(columns={"label_raw_6": "label_raw"}).drop_duplicates("id")

    cleaned = cleanup_schema(
        df.drop(columns=["label_raw_6"], errors="ignore"),
        id_col="id",
        claim_col="claim_text",
        article_col="article_text",
        label_col="label",
        dataset="liar",
    )

    cleaned = cleaned.drop(columns=["label_raw"], errors="ignore").merge(orig_label_raw, on="id", how="left")
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

    save_dir.mkdir(parents=True, exist_ok=True)

    dist = cleaned["label"].value_counts(dropna=False).to_dict()
    logger.info("Label distribution: %s", dist)

    for lbl in ("true", "false", "mixed"):
        part = cleaned[cleaned["label"] == lbl].reset_index(drop=True)
        out_path = save_dir / f"liar_{lbl}.parquet"
        part.to_parquet(out_path, index=False)
        logger.info("Wrote %s (%d rows)", out_path, len(part))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build unified LIAR parquet files (true/false/mixed).")
    p.add_argument("--test", action="store_true", help="If set, run only on first 100 rows total.")
    p.add_argument("--threads", type=int, default=1, help="Number of threads for normalization (1-14).")
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

    root = _find_liar_root()
    logger.info("LIAR root: %s", root)

    base = build_base_dataframe(root, threads=threads, test=bool(args.test))
    finalize_and_write(base, save_dir=save_dir)


if __name__ == "__main__":
    main()
