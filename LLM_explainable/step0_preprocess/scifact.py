from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable, Dict, Any, Iterator, List
import argparse
import hashlib
import json
import os
import re
import tarfile
import urllib.request
from pathlib import Path
from collections import Counter

from datasets import Dataset, DatasetDict

# false=0, mixed=1, true=2, nei=3
LABEL_FALSE = 0
LABEL_MIXED = 1
LABEL_TRUE = 2
LABEL_NEI = 3

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
SCIFACT_SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"

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

def map_scifact_label_to_4way(raw: object) -> int:
    """
    SciFact evidence labels commonly:
      SUPPORT / SUPPORTS         -> TRUE
      CONTRADICT / CONTRADICTS   -> FALSE
      NOT_ENOUGH_INFO            -> NEI
    Returns -100 if unknown/empty.
    """
    if raw is None:
        return -100
    s = str(raw).strip().casefold()
    if not s:
        return -100

    if "support" in s:
        return LABEL_TRUE
    if "contrad" in s or "refut" in s:
        return LABEL_FALSE
    if "not_enough_info" in s or "not enough info" in s or s == "nei":
        return LABEL_NEI

    return -100

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dst)

def _is_within_directory(directory: Path, target: Path) -> bool:
    directory = directory.resolve()
    target = target.resolve()
    return str(target).startswith(str(directory) + os.sep)

def _safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for member in tar.getmembers():
        member_path = path / member.name
        if not _is_within_directory(path, member_path):
            raise RuntimeError(f"Unsafe tar member path: {member.name}")

    try:
        tar.extractall(path, filter="data")
    except TypeError:
        tar.extractall(path)

def _prepare_scifact_data_dir(cache_dir: Optional[str]) -> Path:
    """
    Returns extracted data directory containing:
      data/claims_train.jsonl, data/claims_dev.jsonl, data/claims_test.jsonl, data/corpus.jsonl
    """
    root = Path(cache_dir) if cache_dir else (Path.home() / ".cache" / "scifact")
    root.mkdir(parents=True, exist_ok=True)

    archive = root / "scifact_data.tar.gz"
    extracted = root / "scifact_extracted"
    data_dir = extracted / "data"

    if not archive.exists():
        _download(SCIFACT_URL, archive)

    got = _sha256_file(archive)
    if got != SCIFACT_SHA256:
        raise RuntimeError(
            f"[SciFactAdapter] SHA256 mismatch for {archive}.\n"
            f"Expected: {SCIFACT_SHA256}\n"
            f"Got:      {got}\n"
            f"If the upstream 'latest' changed, update SCIFACT_SHA256 intentionally."
        )

    marker = data_dir / "claims_train.jsonl"
    if not marker.exists():
        if extracted.exists():
            for p in extracted.rglob("*"):
                try:
                    if p.is_file():
                        p.unlink()
                except Exception:
                    pass
        with tarfile.open(archive, "r:gz") as tf:
            _safe_extract(tf, extracted)

    return data_dir

def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def _iter_claim_rows(claims_path: Path, *, is_test: bool) -> Iterator[Dict[str, Any]]:
    """
    Expands each claim into (claim, evidence_doc_id, evidence_label, evidence_sentences).
    Matches the HF script behavior:
      - train/dev: expand evidence dict into multiple rows; if no evidence => 1 row with NOT_ENOUGH_INFO
      - test: 1 row per claim with empty evidence fields
    """
    for ex in _read_jsonl(claims_path):
        cid = ex.get("id", "")
        claim = ex.get("claim", "")
        cited = ex.get("cited_doc_ids", []) or []

        if is_test:
            yield {
                "id": int(cid) if str(cid).isdigit() else cid,
                "claim": claim,
                "evidence_doc_id": "",
                "evidence_label": "",
                "evidence_sentences": [],
                "cited_doc_ids": cited,
            }
            continue

        evidence = ex.get("evidence", {}) or {}
        if not evidence:
            yield {
                "id": int(cid) if str(cid).isdigit() else cid,
                "claim": claim,
                "evidence_doc_id": "",
                "evidence_label": "NOT_ENOUGH_INFO",
                "evidence_sentences": [],
                "cited_doc_ids": cited,
            }
            continue

        for doc_id, ev_list in evidence.items():
            if not isinstance(ev_list, list):
                continue
            for ev in ev_list:
                if not isinstance(ev, dict):
                    continue
                label = ev.get("label", "")
                sents = ev.get("sentences", []) or []
                yield {
                    "id": int(cid) if str(cid).isdigit() else cid,
                    "claim": claim,
                    "evidence_doc_id": str(doc_id),
                    "evidence_label": str(label),
                    "evidence_sentences": sents,
                    "cited_doc_ids": cited,
                }

@dataclass(frozen=True)
class SciFactAdapter:
    """
    Adapter for SciFact without HF dataset scripts (datasets now forbids them).
    We download/extract SciFact tarball and build DatasetDict manually.
    """
    name: str = "scifact"
    source_id: int = 7

    def load(self, *, cache_dir: Optional[str] = None) -> DatasetDict:
        data_dir = _prepare_scifact_data_dir(cache_dir)

        train_path = data_dir / "claims_train.jsonl"
        val_path = data_dir / "claims_dev.jsonl"
        test_path = data_dir / "claims_test.jsonl"

        train = Dataset.from_generator(lambda: _iter_claim_rows(train_path, is_test=False))
        validation = Dataset.from_generator(lambda: _iter_claim_rows(val_path, is_test=False))
        test = Dataset.from_generator(lambda: _iter_claim_rows(test_path, is_test=True))

        return DatasetDict({"train": train, "validation": validation, "test": test})

    def _load_corpus_map(self, *, cache_dir: Optional[str]) -> Dict[int, Dict[str, Any]]:
        data_dir = _prepare_scifact_data_dir(cache_dir)
        corpus_path = data_dir / "corpus.jsonl"

        m: Dict[int, Dict[str, Any]] = {}
        for ex in _read_jsonl(corpus_path):
            try:
                doc_id = int(ex.get("doc_id"))
            except Exception:
                continue
            m[doc_id] = {
                "title": ex.get("title", "") or "",
                "abstract": ex.get("abstract", []) or [],
                "structured": bool(ex.get("structured", False)),
            }
        return m

    def normalize(
        self,
        dd: DatasetDict,
        *,
        cache_dir: Optional[str] = None,
        min_len: int = 10,
        join_corpus: bool = True,
        max_evidence_chars: int = 1500,
    ) -> DatasetDict:
        out = DatasetDict()

        corpus_map: Optional[Dict[int, Dict[str, Any]]] = None
        if join_corpus:
            corpus_map = self._load_corpus_map(cache_dir=cache_dir)

        for split, ds in dd.items():
            cols = ds.column_names

            claim_col = pick_first_col(cols, ["claim"])
            label_col = pick_first_col(cols, ["evidence_label", "label"])
            claim_id_col = pick_first_col(cols, ["id"])
            doc_id_col = pick_first_col(cols, ["evidence_doc_id"])
            sent_ids_col = pick_first_col(cols, ["evidence_sentences"])
            cited_col = pick_first_col(cols, ["cited_doc_ids"])

            if claim_col is None or label_col is None or claim_id_col is None:
                raise RuntimeError(f"[SciFactAdapter] Unexpected columns in split='{split}': {cols}")

            sample_n = min(5000, len(ds))
            mapped = [map_scifact_label_to_4way(x) for x in ds[label_col][:sample_n]]
            cov = sum(1 for m in mapped if m != -100) / max(1, len(mapped))
            print(f"[SciFactAdapter] split={split} label_cov={cov:.3f} join_corpus={join_corpus}")

            def _map_batch(batch, indices):
                claims = [clean_text(x) for x in batch[claim_col]]
                raw_labels = batch[label_col]
                label_ids = [map_scifact_label_to_4way(x) for x in raw_labels]

                claim_ids = batch[claim_id_col]
                doc_ids_raw = batch[doc_id_col] if doc_id_col else [""] * len(indices)
                sent_ids_raw = batch[sent_ids_col] if sent_ids_col else [[]] * len(indices)
                cited_raw = batch[cited_col] if cited_col else [[]] * len(indices)

                ids = [f"{self.name}_{split}_{i}" for i in indices]
                hashes = [claim_hash(c) for c in claims]

                evidence_title = [""] * len(indices)
                evidence_text = [""] * len(indices)

                if join_corpus and corpus_map is not None and doc_id_col:
                    for j, (doc_raw, sent_ids) in enumerate(zip(doc_ids_raw, sent_ids_raw)):
                        try:
                            doc_id = int(doc_raw) if str(doc_raw).strip() else None
                        except Exception:
                            doc_id = None
                        if doc_id is None:
                            continue
                        doc = corpus_map.get(doc_id)
                        if not doc:
                            continue

                        title = clean_text(doc.get("title", ""))
                        abstract = doc.get("abstract", []) or []

                        selected: List[str] = []
                        if isinstance(sent_ids, (list, tuple)) and sent_ids:
                            for k in sent_ids:
                                try:
                                    kk = int(k)
                                except Exception:
                                    continue
                                if 0 <= kk < len(abstract):
                                    selected.append(clean_text(abstract[kk]))
                        if not selected:
                            selected = [clean_text(x) for x in abstract]

                        txt = " ".join([t for t in selected if t]).strip()
                        if max_evidence_chars and len(txt) > max_evidence_chars:
                            txt = txt[:max_evidence_chars].rstrip() + "…"

                        evidence_title[j] = title
                        evidence_text[j] = txt

                return {
                    "source_id": [self.source_id] * len(indices),
                    "id": ids,
                    "claim_id": [str(x) for x in claim_ids],
                    "claim_text": claims,
                    "label_id": label_ids,
                    "label_raw": [str(x) for x in raw_labels],
                    "claim_hash": hashes,
                    "evidence_doc_id": [clean_text(x) for x in doc_ids_raw],
                    "evidence_sentences": [json.dumps(x, ensure_ascii=False) for x in sent_ids_raw],
                    "cited_doc_ids": [json.dumps(x, ensure_ascii=False) for x in cited_raw],
                    "evidence_title": evidence_title,
                    "evidence": evidence_text,
                }

            norm = ds.map(_map_batch, batched=True, with_indices=True, remove_columns=cols)

            def _keep(x):
                if not x["claim_text"] or len(x["claim_text"]) < min_len:
                    return False
                if split != "test" and x["label_id"] == -100:
                    return False
                return True

            norm = norm.filter(_keep)
            out[split] = norm

        return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--max_rows", type=int, default=5)
    ap.add_argument("--min_len", type=int, default=10)
    ap.add_argument("--no_join_corpus", action="store_true")
    ap.add_argument("--max_evidence_chars", type=int, default=1500)
    args = ap.parse_args()

    ad = SciFactAdapter()
    dd = ad.load(cache_dir=args.cache_dir)
    print("Loaded splits:", {k: len(v) for k, v in dd.items()})
    first_split = next(iter(dd.keys()))
    print(f"Raw columns ({first_split}):", dd[first_split].column_names)

    norm = ad.normalize(
        dd,
        cache_dir=args.cache_dir,
        min_len=args.min_len,
        join_corpus=not args.no_join_corpus,
        max_evidence_chars=args.max_evidence_chars,
    )

    print("Normalized splits:", {k: len(v) for k, v in norm.items()})
    show_split = "train" if "train" in norm else next(iter(norm.keys()))
    print("Normalized columns:", norm[show_split].column_names)

    sample_n = min(5000, len(norm[show_split]))
    if sample_n > 0:
        print(f"Label counts ({show_split} sample):", Counter(norm[show_split]["label_id"][:sample_n]))
        for i in range(min(args.max_rows, len(norm[show_split]))):
            print(f"Row {i}:", norm[show_split][i])
    else:
        print(f"{show_split} is empty after filtering. Check label mapping.")

if __name__ == "__main__":
    main()
