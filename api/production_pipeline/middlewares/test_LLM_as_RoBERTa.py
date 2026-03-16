"""
Performance test for LLM_as_RoBERTa.classify().

10 claims: 3 false, 3 mixed, 4 true
Records completion time and result per claim, prints a summary table.
"""

import math
import os
import sys
import time
from pathlib import Path

# Allow running directly from this folder
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

os.environ["ROBERTA_LLM_DEBUG_RESPONSE"] = "true"

from api.production_pipeline.middlewares.LLM_as_RoBERTa import classify

N_VOTES = int(os.environ.get("LLM_AS_ROBERTA_N_VOTES", 6))

LABELS = ["false", "mixed", "true"]

CLAIMS = [
    # --- FALSE (3) ---
    ("Lightning never strikes the same place twice.",                          "false"),
    ("Humans only use 10% of their brains.",                                   "false"),
    ("Shaving makes hair grow back thicker and darker.",                       "false"),
    # --- MIXED (3) ---
    ("Violent video games cause violent behavior.",                            "mixed"),
    ("Red wine is good for your heart when consumed in moderation.",           "mixed"),
    ("Multivitamins improve overall health.",                                  "mixed"),
    # --- TRUE (4) ---
    ("The speed of light in a vacuum is approximately 299,792 km/s.",         "true"),
    ("DNA has a double helix structure.",                                      "true"),
    ("Antibiotics are ineffective against viral infections.",                  "true"),
    ("The human body contains approximately 37 trillion cells.",               "true"),
]


def softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def predicted_label(logits):
    return LABELS[logits.index(max(logits))]


def run():
    print("=" * 80)
    print(f"{'#':<3}  {'Expected':<8}  {'Predicted':<9}  {'Correct':<7}  {'Time(s)':<8}  Claim")
    print("=" * 80)

    results = []
    for i, (claim, expected) in enumerate(CLAIMS, 1):
        t0 = time.perf_counter()
        logits = classify(claim, n_votes=N_VOTES, temperature=0.0)
        elapsed = time.perf_counter() - t0

        probs = softmax(logits)
        pred = predicted_label(logits)
        correct = pred == expected

        results.append({
            "claim": claim,
            "expected": expected,
            "predicted": pred,
            "correct": correct,
            "elapsed": elapsed,
            "probs": probs,
        })

        tick = "YES" if correct else "NO "
        print(f"{i:<3}  {expected:<8}  {pred:<9}  {tick:<7}  {elapsed:<8.2f}  {claim[:60]}")

    print("=" * 80)

    # Summary
    n_correct = sum(r["correct"] for r in results)
    total_time = sum(r["elapsed"] for r in results)
    avg_time = total_time / len(results)

    print(f"\nSummary")
    print(f"  Accuracy : {n_correct}/{len(results)} ({100*n_correct/len(results):.0f}%)")
    print(f"  Total    : {total_time:.2f}s")
    print(f"  Avg/call : {avg_time:.2f}s")

    # Per-label accuracy
    for label in LABELS:
        group = [r for r in results if r["expected"] == label]
        if not group:
            continue
        g_correct = sum(r["correct"] for r in group)
        print(f"  {label:<8}: {g_correct}/{len(group)} correct")

    # Detailed probability breakdown
    print("\nDetailed probabilities  [false   mixed   true  ]")
    print("-" * 80)
    for i, r in enumerate(results, 1):
        probs_str = "  ".join(f"{p:.3f}" for p in r["probs"])
        tick = "YES" if r["correct"] else "NO "
        print(f"{i:<3}  [{probs_str}]  {tick}  {r['claim'][:55]}")


if __name__ == "__main__":
    run()
