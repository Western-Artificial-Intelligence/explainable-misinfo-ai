"""
Article Scraping Utilities for HoVer

Provides threaded hydration for claim text. For HoVer, we mainly hydrate
short claims using Wikipedia links in supporting_facts if needed.
"""

import pandas as pd
from tqdm import tqdm
import threading
from queue import Queue
import time

def fetch_claim_text(row):
    """
    Dummy hydration function for HoVer claims.
    Returns the claim text itself, but can be extended to fetch supporting Wikipedia content.
    """
    # For HoVer, the main content is already in 'claim_text'.
    # Optionally, could fetch paragraphs from supporting_facts.
    return row['claim_text']

def worker(q, res_list, text_col):
    while True:
        row = q.get()
        if row is None:
            break
        hydrated_text = fetch_claim_text(row)
        res_list.append({
            'uid': row['uid'],
            'claim_text': hydrated_text
        })
        q.task_done()

def threaded_hydrate(df, text_col='claim_text', num_threads=4):
    """
    Hydrate claims in a DataFrame using threads.

    Args:
        df: pandas DataFrame containing claims
        text_col: column containing the claim text
        num_threads: number of threads to use

    Returns:
        pandas DataFrame with hydrated claim_text
    """
    q = Queue()
    res_list = []

    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker, args=(q, res_list, text_col))
        t.start()
        threads.append(t)

    for _, row in df.iterrows():
        q.put(row)

    # Stop workers
    for _ in threads:
        q.put(None)

    q.join()
    for t in threads:
        t.join()

    return pd.DataFrame(res_list)
