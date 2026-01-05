from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import hashlib
import json
import re
import zipfile
import urllib.request

from datasets import Dataset, DatasetDict

# false=0, mixed=1, true=2, nei=3
LABEL_FALSE = 0
LABEL_MIXED = 1
LABEL_TRUE = 2
LABEL_NEI = 3

def clean_text(x: object) -> str:
    if x is None:
        return ""
    s = str(x)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def claim_hash(text: str) -> str:
    s = clean_text(text).casefold()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def first_present(d: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    return None

def to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(str(x).strip())
    except Exception:
        return None

@dataclass(frozen=True)
class FakeHealthAdapter:
    """
    Loader for FakeHealth (GitHub repo).
    We primarily use dataset/reviews/HealthStory.json and dataset/reviews/HealthRelease.json.
    Fake/Real is derived from rating score: rating < 3 => fake, else real (per FakeHealth paper).
    """
    name: str = "fakehealth"
    source_id: int = 1

    revision: str = "ec9379de8f8f13af8c436dd6dd9bfaddacd2df30"

    repo_zip_url: str = "https://github.com/EnyanDai/FakeHealth/archive/{rev}.zip"

    def _cache_root(self, cache_dir: Path) -> Path:
        return cache_dir / "fakehealth" / self.revision

    def fetch(self, cache_dir: str = ".cache") -> Path:
        cache_dir_p = Path(cache_dir).expanduser().resolve()
        root = self._cache_root(cache_dir_p)
        root.mkdir(parents=True, exist_ok=True)

        zip_path = root / "repo.zip"
        extract_dir = root / "src"

        if not extract_dir.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)

        if not zip_path.exists():
            url = self.repo_zip_url.format(rev=self.revision)
            print(f"[FakeHealth] downloading: {url}")
            urllib.request.urlretrieve(url, zip_path)

        marker = extract_dir / ".extracted"
        if not marker.exists():
            print(f"[FakeHealth] extracting -> {extract_dir}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            marker.write_text("ok", encoding="utf-8")

        candidates = [p for p in extract_dir.iterdir() if p.is_dir() and p.name.lower().startswith("fakehealth-")]
        if not candidates:
            raise RuntimeError(f"[FakeHealth] Could not find extracted repo folder in {extract_dir}")
        return candidates[0]

    def _load_reviews_json(self, repo_root: Path, subset: str) -> List[Dict[str, Any]]:
        p = repo_root / "dataset" / "reviews" / f"{subset}.json"
        if not p.exists():
            raise RuntimeError(
                f"[FakeHealth] Missing {p}. The repo README says reviews live in dataset/reviews. "
                f"If your checkout is incomplete, re-fetch or download the dataset as instructed by the repo."
            )
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        if not isinstance(obj, list):
            raise RuntimeError(f"[FakeHealth] Expected a list in {p}, got: {type(obj)}")
        return obj

    def load(self, cache_dir: str = ".cache") -> DatasetDict:
        repo_root = self.fetch(cache_dir=cache_dir)

        hs = self._load_reviews_json(repo_root, "HealthStory")
        hr = self._load_reviews_json(repo_root, "HealthRelease")

        rows: List[Dict[str, Any]] = []
        rows.extend(self._normalize_reviews(hs, subset="HealthStory"))
        rows.extend(self._normalize_reviews(hr, subset="HealthRelease"))

        ds = Dataset.from_list(rows)
        return DatasetDict({"train": ds})

    def _normalize_reviews(self, items: List[Dict[str, Any]], subset: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        for idx, r in enumerate(items):
            if not isinstance(r, dict):
                continue

            rid = first_present(r, ["news_id", "newsId", "id", "ID"])
            title = first_present(r, ["title", "Title", "news_title", "headline", "Headline"])
            summary = first_present(r, ["summary", "Summary", "review_summary", "description", "Description"])
            url = first_present(r, ["url", "URL", "news_url", "link", "Link"])

            rating_raw = first_present(r, ["rating", "Rating", "score", "Score", "overall_rating"])
            rating = to_float(rating_raw)

            claim_text = clean_text(title) or clean_text(summary)

            if not claim_text or rating is None:
                continue

            # rating < 3 => fake, else real
            label_id = LABEL_FALSE if rating < 3.0 else LABEL_TRUE
            label_raw = "fake" if label_id == LABEL_FALSE else "real"

            stable_id = str(rid) if rid not in (None, "", "nan") else f"{subset}_{idx}"
            example_id = f"{self.name}_{subset}_{stable_id}"

            out.append(
                {
                    "source_id": self.source_id,
                    "id": example_id,
                    "subset": subset,
                    "claim_text": claim_text,
                    "label_id": label_id,
                    "label_raw": label_raw,
                    "rating": rating,
                    "url": clean_text(url),
                    "claim_hash": claim_hash(claim_text),
                }
            )

        return out

if __name__ == "__main__":
    from collections import Counter
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, default=".cache")
    ap.add_argument("--max_rows", type=int, default=5)
    args = ap.parse_args()

    ad = FakeHealthAdapter()
    dd = ad.load(cache_dir=args.cache_dir)
    ds = dd["train"]

    print("Loaded split sizes:", {k: len(v) for k, v in dd.items()})
    print("Columns:", ds.column_names)
    print("Label counts (sample):", Counter(ds["label_id"][: min(len(ds), 5000)]))

    n = min(args.max_rows, len(ds))
    for i in range(n):
        print(f"Row {i}:", ds[i])
