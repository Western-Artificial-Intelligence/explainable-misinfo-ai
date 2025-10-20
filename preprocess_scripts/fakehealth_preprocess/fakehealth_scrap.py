from __future__ import annotations

import argparse, json, os
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from tqdm import tqdm

DATASETS = ("HealthStory", "HealthRelease")

def map_rating_to_label(rating: Any) -> Optional[str]:
    try:
        r = int(rating)
    except Exception:
        return None
    return "real" if r >= 3 else "fake"

def load_reviews(reviews_root: str, dataset: str) -> List[Dict[str, Any]]:
    path = os.path.join(reviews_root, f"{dataset}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing reviews file")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [dict(v, **({"news_id": k} if "news_id" not in v else {}))
                for k, v in data.items() if isinstance(v, dict)]
    raise ValueError(f"You shouldn't hit this")

def load_content_record(content_root: str, dataset: str, news_id: str) -> Dict[str, Any]:
    fpath = os.path.join(content_root, dataset, f"{news_id}.json")
    if os.path.isfile(fpath):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def get_news_id(rec: Dict[str, Any]) -> Optional[str]:
    for k in ("news_id", "newsId", "id"):
        if k in rec and str(rec[k]).strip():
            return str(rec[k]).strip()
    return None

def safe_get_str(d: Dict[str, Any], *names: str) -> Optional[str]:
    for n in names:
        if n in d and isinstance(d[n], str) and d[n].strip():
            return d[n]
    return None

def build_text(content_rec: Dict[str, Any]) -> str:
    main  = safe_get_str(content_rec, "text", "article", "content", "body", "articleBody")
    title = safe_get_str(content_rec, "title", "Title", "headline", "meta_title")
    desc  = safe_get_str(content_rec, "meta_description", "description")
    kw    = content_rec.get("keywords") or content_rec.get("Keywords") or content_rec.get("Key words")
    if isinstance(kw, list):
        kw = " ".join(str(x) for x in kw if isinstance(x, (str, int, float)))
    elif not isinstance(kw, str):
        kw = None
        
    parts = [p for p in (main, title, desc, kw) if isinstance(p, str) and p.strip()]
    return " ".join(parts).strip()


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def write_split_csv(df: pd.DataFrame, out_dir: str) -> None:
    cols = ["label", "text"]
    ensure_dir(out_dir)
    df[df["label"] == "real"][cols].to_csv(os.path.join(out_dir, "real_1.csv"), index=False)
    df[df["label"] == "fake"][cols].to_csv(os.path.join(out_dir, "fake_1.csv"), index=False)

def write_summary(out_dir: str, df: pd.DataFrame) -> None:
    lines = [f"Folder: {out_dir}"]
    if not df.empty and "label" in df.columns:
        pct = df["label"].value_counts(normalize=True).mul(100).round(2)
        lines.append("label distribution (%):")
        lines.append(pct.apply(lambda v: f"{v:.2f}%").to_string())
    else:
        lines.append("label distribution (%): N/A")
    avg_len = (df["text"].astype(str).str.split().apply(len).mean() if not df.empty else 0.0)
    lines.append(f"Avg text length: {0.0 if pd.isna(avg_len) else round(avg_len, 2)}")
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def process_dataset(fakehealth_root: str, dataset: str) -> pd.DataFrame:
    reviews_root = os.path.join(fakehealth_root, "reviews")
    content_root = os.path.join(fakehealth_root, "content")
    reviews = load_reviews(reviews_root, dataset)

    rows: List[Tuple[str, str]] = []

    for rec in tqdm(reviews, total=len(reviews), desc=f"{dataset} reviews", unit="item", leave=False):
        nid = get_news_id(rec)
        if not nid:
            continue
        label = map_rating_to_label(rec.get("rating"))
        if label is None:
            continue

        content_rec = load_content_record(content_root, dataset, nid)
        text = build_text(content_rec)
        if label and text:
            rows.append((label, text))


    return pd.DataFrame(rows, columns=["label", "text"])

def main():
    ap = argparse.ArgumentParser(description="Export FakeHealth to real_1.csv / fake_1.csv per dataset (with summary).")
    ap.add_argument("--fakehealth-dir", required=True, help="Path to root that contains content/, reviews/, engagements/")
    ap.add_argument("--out-dir", required=True, help="Output root, e.g., data/fakehealth")
    args = ap.parse_args()

    for ds in tqdm(DATASETS, desc="Datasets", unit="dataset"):
        df = process_dataset(args.fakehealth_dir, ds)
        out_ds_dir = os.path.join(args.out_dir, ds)
        write_split_csv(df, out_ds_dir)
        write_summary(out_ds_dir, df)
        
        if not df.empty:
            pct = df["label"].value_counts(normalize=True).mul(100).round(2)
            print(f"\n[REPORT] {ds}")
            print(pct.apply(lambda v: f"{v:.2f}%"))
            avg_len = df["text"].str.split().apply(len).mean()
            print("Avg text length:", round(avg_len if pd.notna(avg_len) else 0.0, 2))
        else:
            print(f"\n[REPORT] {ds} -> no rows produced")

if __name__ == "__main__":
    main()
