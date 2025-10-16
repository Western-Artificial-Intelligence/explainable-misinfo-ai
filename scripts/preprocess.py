from __future__ import annotations
import argparse
from pathlib import Path
import re
import pandas as pd
from typing import List, Tuple

UNIFIED_MAP = {
    0: {"true", "mostly-true", "real"},
    1: {"half-true", "partly-true", "barely-true"},
    2: {"false", "pants-fire", "fake"}
}

def map_labels(label):
    label = label.lower()
    if label in ["true", "mostly-true", "real"]:
        return 0
    elif label in ["half-true", "partly-true", "barely-true"]:
        return 1
    elif label in ["false", "pants-fire", "fake"]:
        return 2
    print(label)
    raise ValueError("Label is unknown")

slang_dict = {
    "fr": "for real",
    "idk": "i don't know", # AHHHHH... punctuation... do we keep that...? idk...
    "smh": "shaking my head",
    "lol": "laughing out loud"
}

def slang_modifier(sentence: str) -> str:
    for word, full in slang_dict.items():
        sentence = re.sub(rf"\b{word}\b", full, sentence)
    return sentence

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "<url>", text)       # replace URLs
    text = re.sub(r"@\w+", "<user>", text)         # replace @mentions
    text = re.sub(r"#(\w+)", r"\1", text)          # remove hashtags
    text = re.sub(r"[^a-z0-9\s<>]", "", text)      # remove non-alphanumeric chars
    text = slang_modifier(text)
    text = re.sub(r"\s+", " ", text).strip()       # collapse whitespace
    return text

def process_one(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "label_raw" in df.columns:
        col = df["label_raw"].map(map_labels)
    elif "label_name" in df.columns:
        col = df["label_name"].map(map_labels)
    else:
        raise ValueError("You're not suppose to hit this")

    df["text"] = df["text"].astype(str).map(clean_text)
    
    out = pd.DataFrame({
        "text": df["text"],
        "label": col
    })
    
    out = out.dropna().drop_duplicates(subset=["text"]).reset_index(drop=True)
    return out

def file_out_name(path) -> str:
    s = str(path)
    start = max(s.rfind('/'), s.rfind('\\')) + 1
    dot = s.rfind('.', start)
    if dot == -1:
        dot = len(s)
    return s[start:dot]

def expand_inputs(files: list[str]) -> list[Path]:
    path_list: list[Path] = []
    for f in files:
        path = Path(f)
        if not path.exists():
            raise FileNotFoundError(f"--input not found: {path}")
        if path.suffix.lower() != ".csv":
            raise ValueError(f"--input must be .csv files: {path}")
        path_list.append(path)
    return path_list

def outputs_for_dir(out_dir: Path, in_list: list[Path]) -> list[Path]:
    if out_dir.exists() and out_dir.is_file():
        raise ValueError(f"--output must be a directory, but is a file: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return [out_dir / f"{file_out_name(inp)}_clean.csv" for inp in in_list]

def label_distribution(df: pd.DataFrame) -> pd.Series:
    order = [0, 1, 2]
    report = df["label"].value_counts(normalize=True).reindex(order, fill_value=0)
    report_named = report.rename({0: "true", 1: "mixed", 2: "false"})
    return (report_named * 100).round(2)

def build_report(pairs: list[tuple[Path, Path]], outputs: list[pd.DataFrame]) -> str:
    lines: list[str] = ["# Preprocess Report\n"]
    aggs: list[pd.DataFrame] = []
    for (inp, outp), df in zip(pairs, outputs):
        lines.append(f"## {inp.name}")
        lines.append(f"- **Input**: `{inp}`")
        lines.append(f"- **Output**: `{outp}`")
        lines.append(f"- **Rows (output)**: {len(df):,}")
        dist = label_distribution(df)
        lines.append("- **Label distribution (%):**")
        for lbl, pct in dist.items():
            lines.append(f"  - {lbl}: {pct}%")
        avg_len = round(df["text"].str.split().map(len).mean(), 2) if len(df) else 0.0
        lines.append(f"- **Avg text length (tokens)**: {avg_len}\n")
        aggs.append(df)

    if aggs:
        combo = pd.concat(aggs, ignore_index=True)
        lines.append("## Combined Summary")
        lines.append(f"- **Total rows (output)**: {len(combo):,}")
        dist = label_distribution(combo)
        lines.append("- **Label distribution (%):**")
        for lbl, pct in dist.items():
            lines.append(f"  - {lbl}: {pct}%")
        avg_len = round(combo["text"].str.split().map(len).mean(), 2) if len(combo) else 0.0
        lines.append(f"- **Avg text length (tokens)**: {avg_len}\n")
    return "\n".join(lines)

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Preprocess CSVs: clean text + unify labels.")
    ap.add_argument("--input", "-i", nargs="+", required=True,
                    help="One or more CSV files.")
    ap.add_argument("--output", "-o", required=True,
                    help="Output directory to write *_clean.csv files and summary.txt.")
    args = ap.parse_args(argv)

    inputs = expand_inputs(args.input)
    out_dir = Path(args.output)
    outputs = outputs_for_dir(out_dir, inputs)

    processed: list[pd.DataFrame] = []
    pairs: list[tuple[Path, Path]] = []

    for inp, outp in zip(inputs, outputs):
        clean = process_one(inp)
        clean.to_csv(outp, index=False)
        print(f"DONE: saved {len(clean):,} rows -> {outp}")
        processed.append(clean)
        pairs.append((inp, outp))

    summary_path = out_dir / "summary.txt"
    summary_path.write_text(build_report(pairs, processed), encoding="utf-8")
    print(f"DONE: wrote summary -> {summary_path}")
    return 0

if __name__ == "__main__":
    main()