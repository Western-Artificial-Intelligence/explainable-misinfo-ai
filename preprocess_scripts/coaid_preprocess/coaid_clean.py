from __future__ import annotations

import argparse, os, re
from typing import List, Optional
import pandas as pd
from tqdm import tqdm

_EMOJI_RE = re.compile(
    "[" 
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "\U0001F018-\U0001F270"
    "\U0001F650-\U0001F67F"
    "]",
    flags=re.UNICODE,
)
_VARIATION_SELECTOR_RE = re.compile("[\uFE0F\uFE0E\u20E3\U0001F3FB-\U0001F3FF]")

def normalize_emoji(s: str) -> str:
    s = _EMOJI_RE.sub(" <EMOJI> ", s)
    s = _VARIATION_SELECTOR_RE.sub("", s)
    return s

SLANG_DICT = {
    "fr": "for real",
    "idk": "i do not know",
    "smh": "shaking my head",
    "lol": "laughing out loud",
}
_SLANG_PATTERNS = [(re.compile(rf"\b{re.escape(k)}\b"), v) for k, v in SLANG_DICT.items()]

def expand_slang(s: str) -> str:
    for pat, repl in _SLANG_PATTERNS:
        s = pat.sub(repl, s)
    return s

# removing bullshit
URL_RE = re.compile(r"http\S+")
AT_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s<>]")
WS_RE = re.compile(r"\s+")
EMPTY_ANGLE_RUNS_RE = re.compile(r"(?:\s*<>\s*)+")
UNKNOWN_ANGLE_TOKEN_RE = re.compile(r"<(?!url|user|emoji)\w+>")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = normalize_emoji(text)
    text = URL_RE.sub("<URL>", text)
    text = AT_RE.sub("<USER>", text)
    text = HASHTAG_RE.sub(r"\1", text)
    text = expand_slang(text)
    text = NON_ALNUM_RE.sub("", text)
    text = UNKNOWN_ANGLE_TOKEN_RE.sub(" ", text)
    text = EMPTY_ANGLE_RUNS_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text

PREFERRED_TEXT_COLS = ["text", "content", "tweet", "body", "full_text"]

def pick_text_col(cols: List[str]) -> Optional[str]:
    for c in PREFERRED_TEXT_COLS:
        if c in cols:
            return c
    return cols[1] if len(cols) >= 2 else (cols[0] if cols else None)

def row_has_first_two(df: pd.DataFrame, i: int) -> bool:
    if df.shape[1] < 2:
        return False
    c0, c1 = df.columns[0], df.columns[1]
    v0, v1 = df.at[i, c0], df.at[i, c1]
    s0 = "" if pd.isna(v0) else str(v0).strip()
    s1 = "" if pd.isna(v1) else str(v1).strip()
    return bool(s0) and bool(s1)

def infer_label_from_filename(path: str) -> Optional[str]:
    base = os.path.basename(path).lower()
    if base.startswith("fake"):
        return "False" if False else "Fake"
    if base.startswith("real"):
        return "Real"
    return None

def process_file(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8", engine="python", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(path, dtype=str, encoding="latin-1", engine="python", on_bad_lines="skip")

    if df.empty:
        df.to_csv(path, index=False)
        return df

    keep = [row_has_first_two(df, i) for i in range(len(df))]
    df = df.loc[keep].copy()

    tcol = pick_text_col(df.columns.tolist())
    if tcol:
        df.loc[:, tcol] = df[tcol].map(clean_text)
        df = df[df[tcol].str.strip().astype(bool)]

    df.to_csv(path, index=False)

    stats_df = df.copy()
    if "text" not in stats_df.columns and tcol and tcol != "text":
        stats_df["text"] = stats_df[tcol]
    if "label" not in stats_df.columns:
        lab = infer_label_from_filename(path)
        if lab is not None:
            stats_df["label"] = lab
    return stats_df

def gather_target_dirs(root: str):
    targets = {"fake_1.csv", "real_1.csv"}
    dirs = {}
    for dirpath, _, filenames in os.walk(root):
        files = [os.path.join(dirpath, f) for f in filenames if f in targets]
        if files:
            dirs[dirpath] = sorted(files)
    return dict(sorted(dirs.items()))

def write_summary(dirpath: str, report_df: pd.DataFrame, txt_col: Optional[str]) -> None:
    lines = []
    lines.append(f"Folder: {dirpath}")
    if "label" in report_df.columns:
        lines.append("label distribution (%):")
        pct = report_df["label"].value_counts(normalize=True).mul(100).round(2)
        lines.append(pct.apply(lambda v: f"{v:.2f}%").to_string())
    else:
        lines.append("label distribution (%): N/A (no label column)")

    if txt_col:
        avg_len = (
            report_df[txt_col].astype(str).str.split().apply(len).mean()
        )
        lines.append(f"Avg text length: {0.0 if pd.isna(avg_len) else round(avg_len, 2)}")
    else:
        lines.append("Avg text length: N/A")
    out_path = os.path.join(dirpath, "summary.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Clean CoAID CSVs (in place) and write per-folder summary.txt")
    ap.add_argument("--input-dir", required=True)
    args = ap.parse_args()

    dir_map = gather_target_dirs(args.input_dir)
    if not dir_map:
        print("No target files found (fake_1.csv / real_1.csv).")
        return

    for dirpath in tqdm(dir_map.keys(), desc="Folders", unit="folder"):
        dfs = []
        for fpath in tqdm(dir_map[dirpath], desc=os.path.basename(dirpath), leave=False, unit="file"):
            dfs.append(process_file(fpath))
        if not dfs:
            continue

        report_df = pd.concat(dfs, ignore_index=True)

        print(f"\n[REPORT] {dirpath}")
        if "label" in report_df.columns:
            pct = report_df["label"].value_counts(normalize=True).mul(100).round(2)
            print(pct.apply(lambda v: f"{v:.2f}%"))
        else:
            print("No 'label' column present (and could not infer).")

        txt_col = "text" if "text" in report_df.columns else pick_text_col(report_df.columns.tolist())
        if txt_col:
            avg_len = report_df[txt_col].astype(str).str.split().apply(len).mean()
            print("Avg text length:", round(avg_len if pd.notna(avg_len) else 0.0, 2))
        else:
            print("No suitable text column to compute average length.")

        write_summary(dirpath, report_df, txt_col)

if __name__ == "__main__":
    main()
