import argparse, csv, os, sys, time
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

import jitter_1

def load_cache(path: str) -> dict:
    cache = {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8", newline="") as f:
            r = csv.reader(f); next(r, None)
            for row in r:
                if len(row) >= 2 and row[0]:
                    cache[row[0]] = row[1]
    return cache

def save_cache(path: str, cache: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tweet_id", "text"])
        for k, v in cache.items():
            w.writerow([k, v])
    os.replace(tmp, path)

def is_fake(fname: str) -> bool:
    n = fname.lower()
    return "fake" in n and n.endswith("_tweets.csv")

def is_real(fname: str) -> bool:
    n = fname.lower()
    return "real" in n and n.endswith("_tweets.csv")

def read_tweet_ids(csv_path: str) -> list[str]:
    ids = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if "tweet_id" not in r.fieldnames:
            raise RuntimeError(f"'tweet_id' column not found in {csv_path}")
        for row in r:
            tid = (row.get("tweet_id") or "").strip()
            if tid:
                ids.append(tid)
    return ids

def process_date_folder(date_dir: str, out_dir: str, cache: dict, sleep_s: float, max_workers: int, flush_every: int):
    import threading
    tweet_csvs = [p for p in glob(os.path.join(date_dir, "*_tweets.csv")) if os.path.isfile(p)]
    if not tweet_csvs:
        return

    file_entries, unique_ids, label_by_id = [], set(), {}
    for csv_path in sorted(tweet_csvs):
        fname = os.path.basename(csv_path)
        target = "fake" if is_fake(fname) else ("real" if is_real(fname) else None)
        if target is None:
            continue
        try:
            ids = read_tweet_ids(csv_path)
        except Exception as e:
            print(f"[warn] Skipping {csv_path}: {e}", file=sys.stderr)
            continue
        file_entries.append((target, ids))
        for tid in ids:
            if tid not in label_by_id:
                label_by_id[tid] = "False" if target == "fake" else "Factual"
        unique_ids.update(ids)
    if not file_entries:
        return

    os.makedirs(out_dir, exist_ok=True)
    fake_out = os.path.join(out_dir, "fake_1.csv")
    real_out = os.path.join(out_dir, "real_1.csv")
    fake_f = open(fake_out, "w", encoding="utf-8", newline="")
    real_f = open(real_out, "w", encoding="utf-8", newline="")
    fake_w, real_w = csv.writer(fake_f), csv.writer(real_f)
    fake_w.writerow(["Factual/False", "content string"])
    real_w.writerow(["Factual/False", "content string"])
    fake_count_since_flush = 0
    real_count_since_flush = 0

    bar = tqdm(total=len(unique_ids), desc=os.path.basename(date_dir), unit="tw", dynamic_ncols=True) if tqdm else None

    lock = threading.Lock()
    ids_to_fetch = []
    for tid in unique_ids:
        text = cache.get(tid, "")
        if text:
            with lock:
                if label_by_id[tid] == "False":
                    fake_w.writerow(["False", text]); fake_count_since_flush += 1
                    if fake_count_since_flush >= flush_every: fake_f.flush(); fake_count_since_flush = 0
                else:
                    real_w.writerow(["Factual", text]); real_count_since_flush += 1
                    if real_count_since_flush >= flush_every: real_f.flush(); real_count_since_flush = 0
            if bar: bar.update(1)
        else:
            ids_to_fetch.append(tid)

    def worker(tid: str):
        try:
            txt = jitter_1.fetch_text(tid)
        except SystemExit:
            txt = ""
        except Exception:
            txt = ""
        if sleep_s > 0:
            time.sleep(sleep_s)
        return tid, (txt or "")

    if ids_to_fetch:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(worker, tid): tid for tid in ids_to_fetch}
            for fut in as_completed(futures):
                tid, text = fut.result()
                cache[tid] = text
                with lock:
                    if label_by_id[tid] == "False":
                        fake_w.writerow(["False", text]); fake_count_since_flush += 1
                        if fake_count_since_flush >= flush_every: fake_f.flush(); fake_count_since_flush = 0
                    else:
                        real_w.writerow(["Factual", text]); real_count_since_flush += 1
                        if real_count_since_flush >= flush_every: real_f.flush(); real_count_since_flush = 0
                if bar: bar.update(1)

    if bar: bar.close()
    fake_f.flush(); real_f.flush()
    fake_f.close(); real_f.close()
    print(f"[ok] {os.path.basename(date_dir)} -> fake_1.csv & real_1.csv")

def main():
    ap = argparse.ArgumentParser(description="COAID cleaner (fast, concurrent, streaming)")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--sleep", type=float, default=0.02, help="Throttle per network call (seconds)")
    ap.add_argument("--cache", default="tweet_text_cache.csv")
    ap.add_argument("--max-workers", type=int, default=16, help="Parallel fetchers")
    ap.add_argument("--flush-every", type=int, default=50, help="Flush to CSV every N rows")
    args = ap.parse_args()

    in_root = os.path.abspath(args.input_dir)
    out_root = os.path.abspath(args.output_dir)
    os.makedirs(out_root, exist_ok=True)

    cache_path = os.path.join(out_root, args.cache)
    cache = load_cache(cache_path)

    dates = [d for d in sorted(os.listdir(in_root)) if os.path.isdir(os.path.join(in_root, d))]
    if not dates:
        print("[error] No subfolders in input-dir.", file=sys.stderr); sys.exit(2)

    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

    for d in dates:
        date_in = os.path.join(in_root, d)
        date_out = os.path.join(out_root, d)
        process_date_folder(date_in, date_out, cache, args.sleep, args.max_workers, args.flush_every)
        save_cache(cache_path, cache)

if __name__ == "__main__":
    main()
