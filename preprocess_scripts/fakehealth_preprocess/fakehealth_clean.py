from __future__ import annotations

import argparse, os, re
from typing import List, Optional
import pandas as pd
from tqdm import tqdm

TARGET_FILES = {"real_1.csv", "fake_1.csv"}

def trim_after_hash3(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = re.search(r"#{3,}", text)
    return text[:m.start()] if m else text

SLANG_DICT = {
    "fr": "for real",
    "idk": "i don't know",
    "smh": "shaking my head",
    "lol": "laughing out loud",
}

URL_RE = re.compile(r"http\S+")
AT_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s<>]")
WS_RE = re.compile(r"\s+")
EMPTY_ANGLE_RUNS_RE = re.compile(r"(?:\s*<>\s*)+")

def expand_slang(text: str) -> str:
    for word, full in SLANG_DICT.items():
        text = re.sub(rf"\b{re.escape(word)}\b", full, text)
    return text

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    text = trim_after_hash3(text)
    text = text.lower()
    text = URL_RE.sub("<URL>", text)
    text = AT_RE.sub("<USER>", text)
    text = HASHTAG_RE.sub(r"\1", text)
    text = expand_slang(text)
    text = NON_ALNUM_RE.sub("", text)
    text = EMPTY_ANGLE_RUNS_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text

PREFERRED_TEXT_COLS = ["text", "content", "body"]
def pick_text_col(cols: List[str]) -> Optional[str]:
    for c in PREFERRED_TEXT_COLS:
        if c in cols:
            return c
    return cols[1] if len(cols) >= 2 else (cols[0] if cols else None)

def drop_missing_first_two(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] < 2:
        return df.iloc[0:0]
    c0, c1 = df.columns[0], df.columns[1]
    mask = df[c0].astype(str).str.strip().ne("") & df[c1].astype(str).str.strip().ne("")
    return df.loc[mask].copy()

def process_file(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8", engine="python", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(path, dtype=str, encoding="latin-1", engine="python", on_bad_lines="skip")

    if df.empty:
        df.to_csv(path, index=False)
        return df

    df = drop_missing_first_two(df)

    tcol = "text" if "text" in df.columns else pick_text_col(df.columns.tolist())
    if not tcol:
        df.to_csv(path, index=False)
        return df

    df.loc[:, tcol] = df[tcol].map(clean_text)

    df = df[df[tcol].str.strip().astype(bool)]

    df.to_csv(path, index=False)

    rep = df.copy()
    if "text" not in rep.columns and tcol != "text":
        rep["text"] = rep[tcol]
    if "label" not in rep.columns and df.shape[1] >= 1:
        rep["label"] = df.columns[0]
    return rep

def gather_dirs(root: str):
    out = {}
    for dirpath, _, files in os.walk(root):
        hits = [os.path.join(dirpath, f) for f in files if f in TARGET_FILES]
        if hits:
            out[dirpath] = sorted(hits)
    return dict(sorted(out.items()))

def write_summary(dirpath: str, df: pd.DataFrame) -> None:
    lines = [f"Folder: {dirpath}"]
    if "label" in df.columns and not df.empty:
        pct = df["label"].value_counts(normalize=True).mul(100).round(2)
        lines.append("label distribution (%):")
        lines.append(pct.apply(lambda v: f"{v:.2f}%").to_string())
    else:
        lines.append("label distribution (%): N/A")
    if "text" in df.columns and not df.empty:
        avg_len = df["text"].astype(str).str.split().apply(len).mean()
        lines.append(f"Avg text length: {0.0 if pd.isna(avg_len) else round(avg_len, 2)}")
    else:
        lines.append("Avg text length: N/A")
    with open(os.path.join(dirpath, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Clean FakeHealth CSVs in place (trim at ###, clean text, slang).")
    ap.add_argument("--input-dir", required=True, help="Root folder (e.g., ./data/fakehealth)")
    args = ap.parse_args()

    dir_map = gather_dirs(args.input_dir)
    if not dir_map:
        print("No target files (real_1.csv / fake_1.csv) found.")
        return

    for dirpath, files in tqdm(dir_map.items(), desc="Folders", unit="folder"):
        reps = []
        for f in tqdm(files, desc=os.path.basename(dirpath), unit="file", leave=False):
            reps.append(process_file(f))
        if not reps:
            continue
        report_df = pd.concat(reps, ignore_index=True)
        print(f"\n[REPORT] {dirpath}")
        write_summary(dirpath, report_df)

if __name__ == "__main__":
    main()
