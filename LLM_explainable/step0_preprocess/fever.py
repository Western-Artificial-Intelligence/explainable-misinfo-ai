from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from datasets import ClassLabel, DatasetDict, load_dataset
from huggingface_hub import HfApi

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


def map_fever_label_to_4way(label_str: str) -> int:
    """
    FEVER labels:
      SUPPORTS         -> TRUE
      REFUTES          -> FALSE
      NOT ENOUGH INFO  -> NEI
    """
    s = (label_str or "").strip().casefold()
    if not s:
        return -100

    if "support" in s:
        return LABEL_TRUE
    if "refut" in s:
        return LABEL_FALSE
    if "not enough info" in s or s == "nei":
        return LABEL_NEI

    return -100


@dataclass(frozen=True)
class FeverAdapter:
    """
    FEVER adapter using *parquet* files directly (no loading scripts).
    Repo: fever/fever
    """

    name: str = "fever"
    source_id: int = 6
    repo_id: str = "fever/fever"

    revision: str = "787a300a2863f6e91d6142678ed8e73fe3a69249"

    def _parquet_data_files(self) -> Dict[str, List[str]]:
        api = HfApi()
        files = api.list_repo_files(
            repo_id=self.repo_id, repo_type="dataset", revision=self.revision
        )
        parquet_paths = [p for p in files if p.endswith(".parquet")]

        if not parquet_paths:
            raise RuntimeError(
                f"[FeverAdapter] No parquet files found at {self.repo_id}@{self.revision}"
            )

        split_to_paths = defaultdict(list)
        for p in parquet_paths:
            pl = p.lower()

            if "/train/" in pl or pl.startswith("train/") or "train" in pl.split("/"):
                split_to_paths["train"].append(p)
            elif (
                "/validation/" in pl
                or "/dev/" in pl
                or "labelled_dev" in pl
                or "valid" in pl
            ):
                split_to_paths["validation"].append(p)
            elif "/test/" in pl or "labelled_test" in pl:
                split_to_paths["test"].append(p)
            else:
                split_to_paths["train"].append(p)

        for k in list(split_to_paths.keys()):
            split_to_paths[k] = sorted(split_to_paths[k])

        return {
            split: [
                f"hf://datasets/{self.repo_id}@{self.revision}/{path}" for path in paths
            ]
            for split, paths in split_to_paths.items()
        }

    def load(self, *, cache_dir: Optional[str] = None) -> DatasetDict:
        """
        Best practice: choose labelled splits explicitly, but resolve the actual paths
        from the repo tree so we don't hardcode a folder layout.
        """
        api = HfApi()
        files = api.list_repo_files(
            repo_id=self.repo_id, repo_type="dataset", revision=self.revision
        )

        def pick_file(filename: str) -> str:
            if filename in files:
                return filename
            matches = [
                f for f in files if f.endswith("/" + filename) or f.endswith(filename)
            ]
            if not matches:
                raise FileNotFoundError(
                    f"[FeverAdapter] Could not find '{filename}' in {self.repo_id}@{self.revision}. "
                    f"Available parquet examples: {[f for f in files if f.endswith('.parquet')][:10]}"
                )
            matches = sorted(matches, key=lambda x: (len(x), x))
            return matches[0]

        train_path = pick_file("fever-train.parquet")
        val_path = pick_file("fever-labelled_dev.parquet")
        test_path = pick_file("fever-paper_test.parquet")

        base = f"hf://datasets/{self.repo_id}@{self.revision}"
        data_files = {
            "train": f"{base}/{train_path}",
            "validation": f"{base}/{val_path}",
            "test": f"{base}/{test_path}",
        }
        return load_dataset("parquet", data_files=data_files, cache_dir=cache_dir)

    def normalize(
        self,
        dd: DatasetDict,
        *,
        min_len: int = 10,
        keep_evidence: bool = True,
    ) -> DatasetDict:
        out = DatasetDict()

        for split, ds in dd.items():
            cols = ds.column_names

            claim_col = pick_first_col(cols, ["claim", "Claim", "statement", "text"])
            label_col = pick_first_col(
                cols, ["label", "Label", "gold_label", "verdict"]
            )
            id_col = pick_first_col(cols, ["id", "orig_id", "claim_id"])

            evidence_col = pick_first_col(cols, ["evidence", "Evidence"])

            if claim_col is None or label_col is None:
                raise RuntimeError(
                    f"[FeverAdapter] Missing required columns in split='{split}'. "
                    f"need claim+label. cols={cols}"
                )

            label_feat = ds.features.get(label_col)
            if isinstance(label_feat, ClassLabel):

                def to_label_str(x):
                    try:
                        return label_feat.int2str(int(x))
                    except Exception:
                        return str(x)

            else:

                def to_label_str(x):
                    return str(x)

            sample_n = min(5000, len(ds))
            mapped = [
                map_fever_label_to_4way(to_label_str(x))
                for x in ds[label_col][:sample_n]
            ]
            cov = sum(1 for m in mapped if m != -100) / max(1, len(mapped))

            print(
                f"[FeverAdapter] split={split} claim_col={claim_col} label_col={label_col} "
                f"label_cov={cov:.3f} id_col={id_col} evidence_col={evidence_col}"
            )

            def _map_batch(batch, indices):
                claims = [clean_text(x) for x in batch[claim_col]]

                raw_label_strs = [to_label_str(x) for x in batch[label_col]]
                label_ids = [map_fever_label_to_4way(s) for s in raw_label_strs]

                claim_ids = batch[id_col] if id_col else [""] * len(indices)

                ids = [f"{self.name}_{split}_{i}" for i in indices]

                hashes = [claim_hash(c) for c in claims]

                if keep_evidence and evidence_col:
                    ev = []
                    for x in batch[evidence_col]:
                        try:
                            ev.append(json.dumps(x, ensure_ascii=False))
                        except Exception:
                            ev.append(clean_text(x))
                else:
                    ev = [""] * len(indices)

                return {
                    "source_id": [self.source_id] * len(indices),
                    "id": ids,
                    "claim_id": [str(x) for x in claim_ids],
                    "claim_text": claims,
                    "label_id": label_ids,
                    "label_raw": raw_label_strs,
                    "claim_hash": hashes,
                    "evidence": ev,
                }

            norm = ds.map(
                _map_batch, batched=True, with_indices=True, remove_columns=cols
            )
            norm = norm.filter(
                lambda x: x["claim_text"]
                and len(x["claim_text"]) >= min_len
                and x["label_id"] != -100
            )

            claim_ids = norm["claim_id"]
            first_idx = {}
            for i, cid in enumerate(claim_ids):
                if cid not in first_idx:
                    first_idx[cid] = i
            norm = norm.select(sorted(first_idx.values()))

            out[split] = norm

        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--max_rows", type=int, default=5)
    ap.add_argument("--min_len", type=int, default=10)
    ap.add_argument("--no_evidence", action="store_true")
    args = ap.parse_args()

    ad = FeverAdapter()
    dd = ad.load(cache_dir=args.cache_dir)

    print("Loaded splits:", {k: len(v) for k, v in dd.items()})
    first_split = next(iter(dd.keys()))
    print(f"Raw columns ({first_split}):", dd[first_split].column_names)

    norm = ad.normalize(dd, min_len=args.min_len, keep_evidence=not args.no_evidence)

    print("Normalized splits:", {k: len(v) for k, v in norm.items()})
    print("Normalized columns:", norm[next(iter(norm.keys()))].column_names)

    show_split = "train" if "train" in norm else next(iter(norm.keys()))
    sample_n = min(5000, len(norm[show_split]))
    if sample_n > 0:
        print(
            f"Label counts ({show_split} sample):",
            Counter(norm[show_split]["label_id"][:sample_n]),
        )
        for i in range(min(args.max_rows, len(norm[show_split]))):
            print(f"Row {i}:", norm[show_split][i])
    else:
        print(f"{show_split} is empty after filtering. Check claim/label mapping.")


if __name__ == "__main__":
    main()
