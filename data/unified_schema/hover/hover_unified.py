import os
import json
import pandas as pd
from article_scraping import threaded_hydrate


# =====================================================================
# Load HoVer JSON
# =====================================================================

def load_hover_json(path, test_mode=False, max_claims=None):
    print("\n================================================================================")
    print("LOADING HOVER DATA (train split only)")
    print("================================================================================\n")
    print(f"Loading: {path}")

    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Rename to internal format
    df = df.rename(columns={
        "claim": "claim_text"
    })

    df["split"] = "train"  # Only train file loaded here

    if max_claims:
        df = df.head(max_claims)

    print(f"Loaded {len(df)} rows.\n")
    return df


# =====================================================================
# Claim Schema — includes LABEL
# =====================================================================

def claim_schema(df, text_col="claim_text"):
    print("\n================================================================================")
    print("APPLYING CLAIM SCHEMA")
    print("================================================================================\n")

    df["needs_hydration"] = df[text_col].isnull() | (df[text_col] == "")
    total_rows = len(df)
    to_hydrate = df["needs_hydration"].sum()
    existing = total_rows - to_hydrate

    print(f"Total rows: {total_rows}")
    print(f"Rows needing hydration: {to_hydrate}")
    print(f"Rows with existing claim text: {existing}\n")

    # HoVer claims do not need hydration — but this is kept for consistency
    if to_hydrate > 0:
        print(f"Hydrating {to_hydrate} claims...")
        hydrated = threaded_hydrate(df[df["needs_hydration"]], text_col=text_col)
        hydrated_texts = [h["claim_text"] for h in hydrated]
        df.loc[df["needs_hydration"], text_col] = hydrated_texts
        print("Hydration complete.\n")

    df["content_status"] = "full_claim"
    df["dataset"] = "hover"

    # Ensure label column exists (HoVer has "SUPPORTED"/"NOT_SUPPORTED")
    if "label" not in df.columns:
        df["label"] = None  # safety guard

    # Final schema
    df_unified = df[[
        "uid",
        "claim_text",
        "label",
        "split",
        "content_status",
        "dataset"
    ]]

    return df_unified


# =====================================================================
# MAIN
# =====================================================================

def main(test_mode=False, max_claims=None):
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Raw input file: hover_train_release_v1.1.json
    raw_path = os.path.join(
        base_path,
        "..",
        "..",
        "raw",
        "hover",
        "data",
        "hover",
        "hover_train_release_v1.1.json",
    )

    df = load_hover_json(raw_path, test_mode=test_mode, max_claims=max_claims)

    df_unified = claim_schema(df, text_col="claim_text")

    print("\n================================================================================")
    print("FINAL UNIFIED SCHEMA SAMPLE")
    print("================================================================================\n")
    print(df_unified.head())

    # Your desired parquet output path
    output_dir = os.path.join(base_path, "..", "..", "processed", "hover")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        "unified_hover_test.parquet" if test_mode else "unified_hover.parquet"
    )

    df_unified.to_parquet(output_file, index=False)

    print(f"\nSaved unified parquet file to: {output_file}\n")


if __name__ == "__main__":
    main()
