from __future__ import annotations

import argparse
import hashlib
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional

from datasets import DatasetDict, load_dataset

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


def pick_first_col(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = set(cols)
    for c in candidates:
        if c in cols:
            return c
    return None


def map_label_to_4way(raw: object, *, one_is_fake: bool) -> int:
    """
    FakeNewsNet label mapping:
    - If label is numeric (0/1): default assumes 1=fake, 0=real (can override).
    - If label is string: map fake/false -> FALSE, real/true -> TRUE.
    Returns -100 if unknown.
    """
    if raw is None:
        return -100

    if isinstance(raw, (int, float)) and str(raw).strip().lower() != "nan":
        v = int(raw)
        if v not in (0, 1):
            return -100
        if one_is_fake:
            return LABEL_FALSE if v == 1 else LABEL_TRUE
        else:
            return LABEL_TRUE if v == 1 else LABEL_FALSE

    s = str(raw).strip().casefold()
    if s == "":
        return -100

    if s in {"0", "1"}:
        v = int(s)
        if one_is_fake:
            return LABEL_FALSE if v == 1 else LABEL_TRUE
        else:
            return LABEL_TRUE if v == 1 else LABEL_FALSE

    if any(
        k in s
        for k in [
            "fake",
            "false",
            "refut",
            "misinfo",
            "misinformation",
            "rumor",
            "hoax",
        ]
    ):
        return LABEL_FALSE
    if any(k in s for k in ["real", "true", "support", "verified", "correct"]):
        return LABEL_TRUE

    return -100


@dataclass(frozen=True)
class FakeNewsNetAdapter:
    """
    Loader for Hugging Face dataset: rickstello/FakeNewsNet.
    The repo contains a single CSV: FakeNewsNet.csv.
    We pin to a specific commit for reproducibility.
    """

    name: str = "fakenewsnet"
    source_id: int = 2

    revision: str = "e1bffec6dc9f2db705efaf542875e32246421d5f"

    def load(self, *, cache_dir: Optional[str] = None) -> DatasetDict:
        csv_path = (
            f"hf://datasets/rickstello/FakeNewsNet@{self.revision}/FakeNewsNet.csv"
        )
        data_files = {"train": csv_path}
        return load_dataset("csv", data_files=data_files, cache_dir=cache_dir)

    def normalize(
        self,
        dd: DatasetDict,
        *,
        one_is_fake: bool = True,
        min_len: int = 10,
    ) -> DatasetDict:
        out = DatasetDict()

        ds = dd["train"]
        cols = ds.column_names

        title_col = pick_first_col(
            cols, ["title", "headline", "news_title", "article_title"]
        )
        text_col = pick_first_col(cols, ["text", "content", "article", "body"])
        url_col = pick_first_col(cols, ["news_url", "url", "link", "source_url"])

        if title_col is None and text_col is None:
            raise RuntimeError(
                f"[FakeNewsNetAdapter] No usable text columns found. cols={cols}"
            )

        label_col = pick_first_col(
            cols, ["real", "label", "veracity", "class", "target"]
        )

        if label_col == "real":
            one_is_fake = False

        if label_col is None:
            raise RuntimeError(
                f"[FakeNewsNetAdapter] No label column found. cols={cols}"
            )

        sample_n = min(5000, len(ds))
        sample = ds[label_col][:sample_n]
        mapped = [map_label_to_4way(x, one_is_fake=one_is_fake) for x in sample]
        cov = sum(1 for m in mapped if m != -100) / max(1, len(mapped))

        print(
            f"[FakeNewsNetAdapter] claim_title={title_col} text={text_col} url={url_col} "
            f"label_col={label_col} label_cov={cov:.3f} one_is_fake={one_is_fake}"
        )

        def _map_batch(batch, indices):
            titles = (
                [clean_text(x) for x in batch[title_col]]
                if title_col
                else [""] * len(indices)
            )
            texts = (
                [clean_text(x) for x in batch[text_col]]
                if text_col
                else [""] * len(indices)
            )

            claims = []
            for t, x in zip(titles, texts):
                if t:
                    claims.append(t)
                else:
                    claims.append(x)

            raw_labels = batch[label_col]
            label_ids = [
                map_label_to_4way(x, one_is_fake=one_is_fake) for x in raw_labels
            ]

            ids = [f"{self.name}_train_{i}" for i in indices]
            hashes = [claim_hash(c) for c in claims]

            urls = (
                [clean_text(u) for u in batch[url_col]]
                if url_col
                else [""] * len(indices)
            )

            return {
                "source_id": [self.source_id] * len(indices),
                "id": ids,
                "claim_text": claims,
                "label_id": label_ids,
                "label_raw": ["real" if int(x) == 1 else "fake" for x in raw_labels],
                "url": urls,
                "claim_hash": hashes,
            }

        norm = ds.map(_map_batch, batched=True, with_indices=True, remove_columns=cols)

        norm = norm.filter(
            lambda x: x["claim_text"]
            and len(x["claim_text"]) >= min_len
            and x["label_id"] != -100
        )

        out["train"] = norm
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--max_rows", type=int, default=5)
    ap.add_argument(
        "--one_is_fake",
        type=int,
        default=int(os.environ.get("FAKENEWSNET_ONE_IS_FAKE", "1")),
        help="1 => numeric label 1=fake,0=real (default). 0 => flip mapping.",
    )
    args = ap.parse_args()

    ad = FakeNewsNetAdapter()
    dd = ad.load(cache_dir=args.cache_dir)

    print("Loaded splits:", {k: len(v) for k, v in dd.items()})
    print("Raw columns (train):", dd["train"].column_names)

    norm = ad.normalize(dd, one_is_fake=bool(args.one_is_fake))
    print("Normalized columns:", norm["train"].column_names)
    print("Train size:", len(norm["train"]))

    sample_n = min(5000, len(norm["train"]))
    if sample_n > 0:
        print(
            "Label counts (train sample):",
            Counter(norm["train"]["label_id"][:sample_n]),
        )
        for i in range(min(args.max_rows, len(norm["train"]))):
            print(f"Row {i}:", norm["train"][i])
    else:
        print("Train is empty after filtering. Check claim/label mapping.")


if __name__ == "__main__":
    main()
