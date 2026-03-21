"""
Performance test for the RoBERTa inference step.

Switch backend via env var or the USE_LLM flag at the top of this file:

  USE_LLM=true  → LLM_as_RoBERTa.classify()  (3-way: false/mixed/true)
  USE_LLM=false → trained RoBERTa best.ckpt, single eval pass:
                    calibration_head.pt present → learned [p_false,p_true] -> 3-way
                    calibration_head.pt absent  → fallback diff threshold (|diff|>0.3)

15 claims: 5 false, 5 mixed, 5 true
Records completion time and result per claim, prints a summary table.
"""

import math
import os
import sys
import time
from pathlib import Path

# Allow running directly from this folder
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# ── SWITCH ────────────────────────────────────────────────────────────────────
# Set ROBERTA_USE_LLM=true/false in env, or change the default below.
USE_LLM = os.environ.get("ROBERTA_USE_LLM", "false").strip().lower() in ("1", "true", "yes")
# ─────────────────────────────────────────────────────────────────────────────

os.environ["ROBERTA_USE_LLM"] = "true" if USE_LLM else "false"

LABELS = ["false", "mixed", "true"]

CLAIMS = [
    # --- FALSE (5) ---
    ("You lose most body heat through your head.",                   "false"),
    ("Bats are completely blind.",                                   "false"),
    ("Bulls are enraged by the color red.",                         "false"),
    ("Carrots improve your eyesight.",                              "false"),
    ("Swimming right after eating causes dangerous cramps.",        "false"),
    # --- MIXED (5) ---
    ("Sugar causes hyperactivity in children.",                     "mixed"),
    ("Sitting too close to the TV damages your eyes.",              "mixed"),
    ("Gluten-free diets are healthier for everyone.",               "mixed"),
    ("Eight glasses of water a day is necessary for good health.",  "mixed"),
    ("Coffee is bad for your health.",                              "mixed"),
    # --- TRUE (5) ---
    ("The Earth is approximately 4.5 billion years old.",           "true"),
    ("Humans share approximately 98% of their DNA with chimpanzees.", "true"),
    ("Sound cannot travel through a vacuum.",                       "true"),
    ("The human brain uses approximately 20% of the body's energy.", "true"),
    ("Mount Everest is the highest mountain above sea level.",      "true"),
]


def softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


# ── Backend: LLM_as_RoBERTa ──────────────────────────────────────────────────

def _run_llm(claim: str):
    """Returns (predicted_label, probs3, elapsed, extra_info)."""
    os.environ["ROBERTA_LLM_DEBUG_RESPONSE"] = "true"
    from api.production_pipeline.middlewares.LLM_as_RoBERTa import classify
    n_votes = int(os.environ.get("LLM_AS_ROBERTA_N_VOTES", 6))
    t0 = time.perf_counter()
    logits = classify(claim, n_votes=n_votes, temperature=0.0)
    elapsed = time.perf_counter() - t0
    probs = softmax(logits)
    pred = LABELS[logits.index(max(logits))]
    return pred, probs, elapsed, {}


# ── Backend: real trained RoBERTa ─────────────────────────────────────────────

_step2 = None

def _get_step2():
    global _step2
    if _step2 is not None:
        return _step2
    import importlib.util
    here = Path(__file__).resolve()
    step2_path = here.parents[1] / "2_RoBERTa_inference" / "2_RoBERTa_inference.py"
    spec = importlib.util.spec_from_file_location("api.production_pipeline.step2", step2_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["api.production_pipeline.step2"] = mod
    spec.loader.exec_module(mod)
    _step2 = mod
    return mod


def _run_roberta(claim: str):
    """Returns (predicted_label, probs3, elapsed, extra_info)."""
    mod = _get_step2()
    text = f"<CLAIM> {claim} </CLAIM>"
    t0 = time.perf_counter()
    _logits3, probs3 = mod._infer_via_real_roberta(text)
    elapsed = time.perf_counter() - t0
    vote = dict(mod._last_roberta_vote)
    pred = vote["voted_label"]
    return pred, probs3, elapsed, vote


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    if USE_LLM:
        backend_name = "LLM_as_RoBERTa"
        mode_note = None
    else:
        mod = _get_step2()
        cal_path = mod._CALIBRATION_HEAD_PATH
        if cal_path.is_file():
            backend_name = "RoBERTa best.ckpt + calibration_head.pt"
            mode_note = "  mode : learned [p_false, p_true] -> 3-way (calibration head)"
        else:
            backend_name = "RoBERTa best.ckpt (fallback: diff threshold |diff|>0.3)"
            mode_note = "  mode : fixed diff threshold — train calibration head for better results"

    print("=" * 80)
    print(f"Backend : {backend_name}")
    if mode_note:
        print(mode_note)
    print("=" * 80)
    print(f"{'#':<3}  {'Expected':<8}  {'Predicted':<9}  {'Correct':<7}  {'Time(s)':<8}  Claim")
    print("=" * 80)

    results = []
    for i, (claim, expected) in enumerate(CLAIMS, 1):
        if USE_LLM:
            pred, probs, elapsed, extra = _run_llm(claim)
        else:
            pred, probs, elapsed, extra = _run_roberta(claim)

        correct = pred == expected
        results.append({
            "claim": claim,
            "expected": expected,
            "predicted": pred,
            "correct": correct,
            "elapsed": elapsed,
            "probs": probs,
            "extra": extra,
        })

        tick = "YES" if correct else "NO "
        print(f"{i:<3}  {expected:<8}  {pred:<9}  {tick:<7}  {elapsed:<8.2f}  {claim[:55]}")

    print("=" * 80)

    n_correct = sum(r["correct"] for r in results)
    total_time = sum(r["elapsed"] for r in results)
    avg_time = total_time / len(results)

    print(f"\nSummary")
    print(f"  Accuracy : {n_correct}/{len(results)} ({100*n_correct/len(results):.0f}%)")
    print(f"  Total    : {total_time:.2f}s")
    print(f"  Avg/call : {avg_time:.2f}s")

    for label in LABELS:
        group = [r for r in results if r["expected"] == label]
        if not group:
            continue
        g_correct = sum(r["correct"] for r in group)
        print(f"  {label:<8}: {g_correct}/{len(group)} correct")

    print("\nDetailed  [false   mixed   true  ]")
    print("-" * 80)
    for i, r in enumerate(results, 1):
        probs_str = "  ".join(f"{p:.3f}" for p in r["probs"])
        tick = "YES" if r["correct"] else "NO "
        extra_str = ""
        if r["extra"]:
            v = r["extra"]
            diff   = v.get("diff", 0)
            lean   = v.get("lean")
            detail = v.get("inference_detail", "")
            tag    = f"[{detail}]" if detail else ""
            lean_str = f"  lean->{lean['toward']}({lean['p']:.3f})" if lean else ""
            extra_str = f"  diff={diff:+.3f}{lean_str}  {tag}"
        print(f"{i:<3}  [{probs_str}]  {tick}{extra_str}  {r['claim'][:40]}")


if __name__ == "__main__":
    run()
