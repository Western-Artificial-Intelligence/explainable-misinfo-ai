from __future__ import annotations

import argparse
import csv
import hashlib
import re
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset, DatasetDict

# false=0, mixed=1, true=2, nei=3
LABEL_FALSE = 0
LABEL_MIXED = 1
LABEL_TRUE = 2
LABEL_NEI = 3


def clean_text(s: object) -> str:
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def claim_hash(s: str) -> str:
    s = clean_text(s).casefold()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def map_liar_label_to_4way(raw: object) -> int:
    """
    LIAR has 6 labels:
      pants-fire, false, barely-true, half-true, mostly-true, true

    Best-practice (transparent collapse to 4-way):
      - FALSE: pants-fire, false, barely-true
      - MIXED: half-true
      - TRUE: mostly-true, true
      - NEI: (not present) -> none
    """
    if raw is None:
        return -100
    s = str(raw).strip().casefold()
    if not s:
        return -100

    if s in {"pants-fire", "pantsfire", "false", "barely-true", "barely true"}:
        return LABEL_FALSE
    if s in {"half-true", "half true"}:
        return LABEL_MIXED
    if s in {"mostly-true", "mostly true", "true"}:
        return LABEL_TRUE

    return -100


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        f.write(r.read())


def _extract(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted"
    if marker.exists():
        return
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    marker.write_text("ok", encoding="utf-8")


def _find_file(root: Path, filename: str) -> Path:
    for p in root.rglob(filename):
        return p
    raise FileNotFoundError(f"Could not find {filename} under {root}")


def _read_liar_tsv(path: Path) -> Dict[str, List[object]]:
    """
    Mirrors the original HF dataset script parsing: TSV, QUOTE_NONE.
    Columns (14):
      0 id
      1 label
      2 statement
      3 subject
      4 speaker
      5 job_title
      6 state_info
      7 party_affiliation
      8 barely_true_counts
      9 false_counts
      10 half_true_counts
      11 mostly_true_counts
      12 pants_on_fire_counts
      13 context
    """
    cols = {
        "orig_id": [],
        "label": [],
        "statement": [],
        "subject": [],
        "speaker": [],
        "job_title": [],
        "state_info": [],
        "party_affiliation": [],
        "barely_true_counts": [],
        "false_counts": [],
        "half_true_counts": [],
        "mostly_true_counts": [],
        "pants_on_fire_counts": [],
        "context": [],
    }

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            if len(row) < 14:
                row = row + [""] * (14 - len(row))
            cols["orig_id"].append(row[0])
            cols["label"].append(row[1])
            cols["statement"].append(row[2])
            cols["subject"].append(row[3])
            cols["speaker"].append(row[4])
            cols["job_title"].append(row[5])
            cols["state_info"].append(row[6])
            cols["party_affiliation"].append(row[7])
            cols["barely_true_counts"].append(row[8])
            cols["false_counts"].append(row[9])
            cols["half_true_counts"].append(row[10])
            cols["mostly_true_counts"].append(row[11])
            cols["pants_on_fire_counts"].append(row[12])

            cols["context"].append(row[13])

    return cols


@dataclass(frozen=True)
class LiarAdapter:
    """
    LIAR adapter (no HF dataset scripts).
    Downloads the official UCSB zip used by the HF script and parses TSVs. :contentReference[oaicite:1]{index=1}
    """

    name: str = "liar"
    source_id: int = 5

    url: str = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"

    def load(self, *, cache_dir: Optional[str] = None) -> DatasetDict:
        cache_root = Path(cache_dir or ".cache") / self.name
        tag = hashlib.sha1(self.url.encode("utf-8")).hexdigest()[:12]
        work = cache_root / tag
        zip_path = work / "liar_dataset.zip"
        extract_dir = work / "src"

        _download(self.url, zip_path)
        _extract(zip_path, extract_dir)

        train_path = _find_file(extract_dir, "train.tsv")
        valid_path = _find_file(extract_dir, "valid.tsv")
        test_path = _find_file(extract_dir, "test.tsv")

        train = Dataset.from_dict(_read_liar_tsv(train_path))
        valid = Dataset.from_dict(_read_liar_tsv(valid_path))
        test = Dataset.from_dict(_read_liar_tsv(test_path))

        return DatasetDict(train=train, validation=valid, test=test)

    def normalize(self, dd: DatasetDict, *, min_len: int = 10) -> DatasetDict:
        out = DatasetDict()

        for split, ds in dd.items():

            def _map_batch(batch, indices):
                statements = [clean_text(x) for x in batch["statement"]]
                raw_labels = batch["label"]
                orig_ids = batch["orig_id"]

                label_ids = [map_liar_label_to_4way(x) for x in raw_labels]
                ids = [
                    f"{self.name}_{split}_{clean_text(oid) or i}"
                    for oid, i in zip(orig_ids, indices)
                ]
                hashes = [claim_hash(s) for s in statements]

                return {
                    "source_id": [self.source_id] * len(indices),
                    "id": ids,
                    "claim_text": statements,
                    "label_id": label_ids,
                    "label_raw": [str(x) for x in raw_labels],
                    "claim_hash": hashes,
                }

            norm = ds.map(
                _map_batch,
                batched=True,
                with_indices=True,
                remove_columns=ds.column_names,
            )
            norm = norm.filter(
                lambda x: x["claim_text"]
                and len(x["claim_text"]) >= min_len
                and x["label_id"] != -100
            )

            out[split] = norm

        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--max_rows", type=int, default=5)
    ap.add_argument("--min_len", type=int, default=10)
    args = ap.parse_args()

    ad = LiarAdapter()
    dd = ad.load(cache_dir=args.cache_dir)

    print("Loaded splits:", {k: len(v) for k, v in dd.items()})
    print("Raw columns (train):", dd["train"].column_names)

    norm = ad.normalize(dd, min_len=args.min_len)

    print("Normalized splits:", {k: len(v) for k, v in norm.items()})
    print("Normalized columns:", norm["train"].column_names)

    sample_n = min(5000, len(norm["train"]))
    print("Label counts (train sample):", Counter(norm["train"]["label_id"][:sample_n]))

    for i in range(min(args.max_rows, len(norm["train"]))):
        print(f"Row {i}:", norm["train"][i])


if __name__ == "__main__":
    main()
