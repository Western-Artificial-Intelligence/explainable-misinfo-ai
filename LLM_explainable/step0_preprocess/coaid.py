from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Iterable
import hashlib
import re
import math
from collections import Counter
from datasets import DatasetDict, load_dataset

# false=0, mixed=1, true=2, nei=3
LABEL_FALSE = 0
LABEL_MIXED = 1
LABEL_TRUE = 2
LABEL_NEI = 3

def clean_text(s: object) -> str:
    if s is None:
        return ""
    if isinstance(s, float) and math.isnan(s):
        return ""

    s = str(s).strip()
    if s.lower() in {"nan", "none", "null", "n/a", ""}:
        return ""

    s = re.sub(r"\s+", " ", s).strip()
    return s


def claim_hash(s: str) -> str:
    s = clean_text(s).casefold()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def pick_first_col(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = set(cols)
    for c in candidates:
        if c in cols:
            return c
    return None

def map_label_to_4way(raw: object) -> int:
    if raw is None:
        return -100

    if isinstance(raw, (int, float)) and str(raw).strip().lower() != "nan":
        v = int(raw)
        if v == 1:
            return LABEL_TRUE
        if v == 0:
            return LABEL_FALSE
        return -100

    s = str(raw).strip().casefold()
    if s == "":
        return -100

    try:
        v = int(s)
        if v == 1:
            return LABEL_TRUE
        if v == 0:
            return LABEL_FALSE
    except:
        pass

    if any(k in s for k in ["nei", "not enough", "unknown", "unverified", "insufficient"]):
        return LABEL_NEI

    if any(k in s for k in ["fake", "false", "refut", "misinfo", "misinformation", "rumor", "hoax"]):
        return LABEL_FALSE

    if any(k in s for k in ["real", "true", "support", "verified", "correct"]):
        return LABEL_TRUE

    return -100

@dataclass(frozen=True)
class CoAIDAdapter:
    name: str = "coaid"
    source_id: int = 3

    revision: str = "e5a2329f72d7785c714982b67c7fb63475fe1fcc"

    def load(self) -> DatasetDict:
        base = f"hf://datasets/ComplexDataLab/Misinfo_Datasets@{self.revision}/coaid"
        data_files = {
            "train": f"{base}/coaid_train.parquet",
            "validation": f"{base}/coaid_validation.parquet",
            "test": f"{base}/coaid_test.parquet",
        }
        return load_dataset("parquet", data_files=data_files)

    def normalize(self, dd: DatasetDict) -> DatasetDict:
        out = DatasetDict()

        for split, ds in dd.items():
            cols = ds.column_names

            claim_candidates = [c for c in ["claim_en", "claim", "initial_claim", "tweet_text", "article_headline"] if c in cols]
            if not claim_candidates:
                raise RuntimeError(f"[CoAIDAdapter] No claim candidates found. cols={cols}")

            def claim_coverage(col: str, n: int = 5000) -> float:
                sample = ds[col][: min(n, len(ds))]
                cleaned = [clean_text(x) for x in sample]
                good = sum(1 for t in cleaned if len(t) >= 10)
                return good / max(1, len(cleaned))

            claim_cov = {c: claim_coverage(c) for c in claim_candidates}
            claim_col = max(claim_cov, key=claim_cov.get)

            fallback_col = "claim" if ("claim" in cols and claim_col != "claim") else None

            if claim_cov[claim_col] == 0.0:
                peek = ds[claim_col][:20]
                raise RuntimeError(f"[CoAIDAdapter] claim_col='{claim_col}' has 0% usable text. peek={peek}")

            label_candidates = [c for c in ["label", "veracity", "binary_label", "tweet_label", "contain_misinfo"] if c in cols]
            if not label_candidates:
                raise RuntimeError(f"[CoAIDAdapter] No label candidates found. cols={cols}")

            def label_coverage(col: str, n: int = 5000) -> float:
                sample = ds[col][: min(n, len(ds))]
                mapped = [map_label_to_4way(x) for x in sample]
                good = sum(1 for m in mapped if m != -100)
                return good / max(1, len(mapped))

            coverages = {c: label_coverage(c) for c in label_candidates}
            label_col = max(coverages, key=coverages.get)

            if coverages[label_col] == 0.0:
                peek = ds[label_col][:20]
                raise RuntimeError(f"[CoAIDAdapter] label_col='{label_col}' has 0% mappable labels. peek={peek}")

            print(
                f"[CoAIDAdapter] split={split} claim_col={claim_col} (cov={claim_cov[claim_col]:.3f}) "
                f"fallback={fallback_col} label_col={label_col} (cov={coverages[label_col]:.3f})"
            )

            def _map_batch(batch, indices):
                primary_vals = [clean_text(x) for x in batch[claim_col]]
                if fallback_col:
                    fallback_vals = [clean_text(x) for x in batch[fallback_col]]
                    claims = [p if p else f for p, f in zip(primary_vals, fallback_vals)]
                else:
                    claims = primary_vals

                raw_labels = batch[label_col]

                ids = [f"{self.name}_{split}_{i}" for i in indices]
                label_ids = [map_label_to_4way(x) for x in raw_labels]
                hashes = [claim_hash(c) for c in claims]

                return {
                    "id": ids,
                    "claim_text": claims,
                    "label_id": label_ids,
                    "label_raw": [str(x) for x in raw_labels],
                    "source_id": [self.source_id] * len(claims),
                    "claim_hash": hashes,
                }

            mapped = ds.map(_map_batch, batched=True, with_indices=True, remove_columns=cols)

            mapped = mapped.filter(lambda x: x["claim_text"] and len(x["claim_text"]) >= 10 and x["label_id"] != -100)

            out[split] = mapped

        return out

if __name__ == "__main__":
    from collections import Counter

    ad = CoAIDAdapter()
    dd = ad.load()
    print("Loaded splits:", {k: len(v) for k, v in dd.items()})
    print("Raw columns (train):", dd["train"].column_names)

    norm = ad.normalize(dd)
    print("Normalized columns:", norm["train"].column_names)
    print("Train size:", len(norm["train"]))

    if len(norm["train"]) > 0:
        print("Label counts (train sample):", Counter(norm["train"]["label_id"][:5000]))
        print("Sample row:", norm["train"][0])
    else:
        print("Train is empty after filtering. Check claim/label mapping.")