from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable, Dict, List
import argparse
import hashlib
import re
from collections import Counter, defaultdict

from datasets import DatasetDict, load_dataset
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

def map_hover_label_to_4way(raw: object) -> int:
    """
    HoVer label mapping (per dataset card):
      0 => Supports  => TRUE
      1 => Refutes   => FALSE
    Accepts ints, numeric strings, and 'supports/refutes' strings.
    Returns -100 if unknown.
    """
    if raw is None:
        return -100

    if isinstance(raw, (int, float)) and str(raw).strip().lower() != "nan":
        v = int(raw)
        if v == 0:
            return LABEL_TRUE
        if v == 1:
            return LABEL_FALSE
        return -100

    s = str(raw).strip().casefold()
    if s == "":
        return -100

    if s in {"0", "1"}:
        v = int(s)
        return LABEL_TRUE if v == 0 else LABEL_FALSE

    if "support" in s:
        return LABEL_TRUE
    if "refut" in s:
        return LABEL_FALSE

    return -100

@dataclass(frozen=True)
class HoverAdapter:
    """
    HoVer adapter (Dzeniks/hover).
    Best-practice: load Parquet (data-only) + pin revision for reproducibility.
    """
    name: str = "hover"
    source_id: int = 4
    repo_id: str = "Dzeniks/hover"

    revision: str = "f84e030f1249cd9b9b4a5261233a5c8f17fd08e9"

    def _parquet_data_files(self) -> Dict[str, List[str]]:
        api = HfApi()
        files = api.list_repo_files(repo_id=self.repo_id, repo_type="dataset", revision=self.revision)
        parquet_paths = [p for p in files if p.endswith(".parquet")]

        if not parquet_paths:
            raise RuntimeError(f"[HoverAdapter] No parquet files found at {self.repo_id}@{self.revision}")

        split_to_paths = defaultdict(list)
        for p in parquet_paths:
            pl = p.lower()
            if "/train/" in pl:
                split_to_paths["train"].append(p)
            elif "/validation/" in pl or "/valid/" in pl or "/dev/" in pl:
                split_to_paths["validation"].append(p)
            elif "/test/" in pl:
                split_to_paths["test"].append(p)
            else:
                split_to_paths["train"].append(p)

        for k in list(split_to_paths.keys()):
            split_to_paths[k] = sorted(split_to_paths[k])

        data_files = {
            split: [f"hf://datasets/{self.repo_id}@{self.revision}/{path}" for path in paths]
            for split, paths in split_to_paths.items()
        }
        return data_files

    def load(self, *, cache_dir: Optional[str] = None) -> DatasetDict:
        data_files = self._parquet_data_files()
        dd = load_dataset("parquet", data_files=data_files, cache_dir=cache_dir)
        return dd

    def normalize(
        self,
        dd: DatasetDict,
        *,
        min_len: int = 10,
        make_validation_if_missing: bool = True,
        val_frac: float = 0.10,
        seed: int = 42,
    ) -> DatasetDict:
        out = DatasetDict()

        for split, ds in dd.items():
            cols = ds.column_names

            claim_col = pick_first_col(cols, ["claim", "Claim", "sentence", "text"])
            label_col = pick_first_col(cols, ["label", "Label", "verdict", "gold_label"])
            expl_col = pick_first_col(cols, ["explanation", "Explanation", "justification"])
            evid_col = pick_first_col(cols, ["evidence", "Evidence", "context"])

            if claim_col is None or label_col is None:
                raise RuntimeError(
                    f"[HoverAdapter] Missing required columns in split='{split}'. "
                    f"need claim+label. cols={cols}"
                )

            sample_n = min(5000, len(ds))
            cov = sum(
                1 for x in ds[label_col][:sample_n]
                if map_hover_label_to_4way(x) != -100
            ) / max(1, sample_n)

            print(
                f"[HoverAdapter] split={split} claim_col={claim_col} label_col={label_col} "
                f"label_cov={cov:.3f} expl_col={expl_col} evid_col={evid_col}"
            )

            def _map_batch(batch, indices):
                claims = [clean_text(x) for x in batch[claim_col]]
                raw_labels = batch[label_col]
                label_ids = [map_hover_label_to_4way(x) for x in raw_labels]
                ids = [f"{self.name}_{split}_{i}" for i in indices]
                hashes = [claim_hash(c) for c in claims]

                explanations = [clean_text(x) for x in batch[expl_col]] if expl_col else [""] * len(indices)
                evidences = [clean_text(x) for x in batch[evid_col]] if evid_col else [""] * len(indices)

                return {
                    "source_id": [self.source_id] * len(indices),
                    "id": ids,
                    "claim_text": claims,
                    "label_id": label_ids,
                    "label_raw": [str(x) for x in raw_labels],
                    "claim_hash": hashes,
                    "explanation": explanations,
                    "evidence": evidences,
                }

            norm = ds.map(_map_batch, batched=True, with_indices=True, remove_columns=cols)
            norm = norm.filter(lambda x: x["claim_text"] and len(x["claim_text"]) >= min_len and x["label_id"] != -100)

            out[split] = norm

        if make_validation_if_missing and "train" in out and "validation" not in out:
            tmp = out["train"].map(lambda x: {"label_str": str(x["label_raw"])})

            try:
                tmp = tmp.class_encode_column("label_str")
                split_dd = tmp.train_test_split(
                    test_size=val_frac,
                    seed=seed,
                    stratify_by_column="label_str",
                )
                out["train"] = split_dd["train"].remove_columns(["label_str"])
                out["validation"] = split_dd["test"].remove_columns(["label_str"])
                print(f"[HoverAdapter] created validation split from train (stratified): val_frac={val_frac}")
            except Exception as e:
                split_dd = out["train"].train_test_split(test_size=val_frac, seed=seed)
                out["train"] = split_dd["train"]
                out["validation"] = split_dd["test"]
                print(f"[HoverAdapter] created validation split from train (non-stratified fallback): val_frac={val_frac} err={e}")

        return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--max_rows", type=int, default=5)
    ap.add_argument("--min_len", type=int, default=10)
    ap.add_argument("--no_make_validation", action="store_true")
    args = ap.parse_args()

    ad = HoverAdapter()
    dd = ad.load(cache_dir=args.cache_dir)
    print("Loaded splits:", {k: len(v) for k, v in dd.items()})
    first_split = next(iter(dd.keys()))
    print(f"Raw columns ({first_split}):", dd[first_split].column_names)

    norm = ad.normalize(
        dd,
        min_len=args.min_len,
        make_validation_if_missing=not args.no_make_validation,
    )

    print("Normalized splits:", {k: len(v) for k, v in norm.items()})
    print("Normalized columns:", norm["train"].column_names)

    sample_n = min(5000, len(norm["train"]))
    if sample_n > 0:
        print("Label counts (train sample):", Counter(norm["train"]["label_id"][:sample_n]))
        for i in range(min(args.max_rows, len(norm["train"]))):
            print(f"Row {i}:", norm["train"][i])
    else:
        print("Train is empty after filtering. Check claim/label mapping.")

if __name__ == "__main__":
    main()
