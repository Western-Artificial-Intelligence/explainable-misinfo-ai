"""
Train RoBERTa-base as 3-way classifier (false / mixed / true) from scratch.
Optimized for NVIDIA L4 (24 GB VRAM).

Pipeline:
  combined_data.parquet
      → tokenize → roberta-base + LoRA
      → Linear(768 → 3)
      → cross-entropy (class-weighted)
      → save best_3way.ckpt  (by val macro-F1)

Usage:
  python api/production_pipeline/RoBERTa_model/train_3way.py
"""

import random
import sys
import collections
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT))
for _p in [HERE.parents[2], HERE.parents[1]]:
    if (_p / "baseline_model").is_dir():
        sys.path.insert(0, str(_p))
        break

DATA_PATH = HERE / "combined_data.parquet"
OUT_PATH  = HERE / "best_3way.ckpt"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
BACKBONE     = "roberta-base"
EPOCHS       = 10
LR           = 2e-5
BATCH_SIZE   = 64       # L4 24 GB — bump to 128 if VRAM allows
MAX_LEN      = 256
SEED         = 42
VAL_FRAC     = 0.10
WARMUP_RATIO = 0.06
GRAD_CLIP    = 1.0
EARLY_STOP   = 3        # epochs without improvement

# LoRA
LORA_R        = 8
LORA_ALPHA    = 16
LORA_DROPOUT  = 0.1
LORA_TARGETS  = ["query", "value"]

LABEL2ID = {"false": 0, "mixed": 1, "true": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_split(claims, labels, val_frac, seed):
    rng = random.Random(seed)
    by_class = collections.defaultdict(list)
    for c, l in zip(claims, labels):
        by_class[l].append((c, l))

    train_c, train_l, val_c, val_l = [], [], [], []
    for lid in sorted(by_class):
        items = by_class[lid][:]
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_frac))
        for c, l in items[:n_val]:
            val_c.append(c); val_l.append(l)
        for c, l in items[n_val:]:
            train_c.append(c); train_l.append(l)
    return train_c, train_l, val_c, val_l


def macro_f1(preds, labels, n_classes=3):
    f1s = []
    for c in range(n_classes):
        tp = sum(p == c and l == c for p, l in zip(preds, labels))
        fp = sum(p == c and l != c for p, l in zip(preds, labels))
        fn = sum(p != c and l == c for p, l in zip(preds, labels))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return sum(f1s) / len(f1s)


# ── Dataset ───────────────────────────────────────────────────────────────────

class ClaimDataset(Dataset):
    def __init__(self, claims, labels, tokenizer, max_len):
        self.claims    = claims
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, idx):
        text = f"<CLAIM> {self.claims[idx]} </CLAIM>"
        enc  = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(tokenizer):
    from baseline_model.models.lora_utils import try_apply_peft_lora

    model = AutoModelForSequenceClassification.from_pretrained(
        BACKBONE,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    # Add special tokens <CLAIM> </CLAIM>
    model.resize_token_embeddings(len(tokenizer))

    model, peft_applied = try_apply_peft_lora(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
    )
    if not peft_applied:
        print("[train] LoRA not applied — training full model")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"[train] Trainable: {n_trainable:,} / {n_total:,} params")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, weights):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lbls  = batch["label"].to(device)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                out    = model(input_ids=ids, attention_mask=mask)
                logits = out.logits
            total_loss += F.cross_entropy(logits, lbls, weight=weights).item()
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(lbls.cpu().tolist())

    avg_loss = total_loss / max(1, len(loader))
    acc      = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    mf1      = macro_f1(all_preds, all_labels)
    per_class = {}
    for lid, lname in ID2LABEL.items():
        indices = [i for i, l in enumerate(all_labels) if l == lid]
        correct = sum(all_preds[i] == lid for i in indices)
        per_class[lname] = (correct, len(indices))
    model.train()
    return avg_loss, acc, mf1, per_class


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[train] Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"[train] Loading {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH).dropna(subset=["claim_text", "label_3way"])
    df = df[df["label_3way"].isin(LABEL2ID)].reset_index(drop=True)
    print(f"[train] {len(df)} examples | {dict(df['label_3way'].value_counts())}")

    claims = df["claim_text"].tolist()
    labels = [LABEL2ID[l] for l in df["label_3way"]]

    train_c, train_l, val_c, val_l = stratified_split(claims, labels, VAL_FRAC, SEED)
    print(f"[train] Split → train={len(train_c)} | val={len(val_c)}")

    counts  = collections.Counter(train_l)
    n_train = len(train_l)
    weights = torch.tensor(
        [n_train / (3 * counts[i]) for i in range(3)], dtype=torch.float32
    ).to(device)
    print(f"[train] Class weights → false={weights[0]:.3f}  mixed={weights[1]:.3f}  true={weights[2]:.3f}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    from baseline_model.data_utils.tokenize import build_tokenizer
    tokenizer = build_tokenizer(BACKBONE)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(tokenizer)
    model.to(device)

    # ── Loaders ───────────────────────────────────────────────────────────────
    train_ds = ClaimDataset(train_c, train_l, tokenizer, MAX_LEN)
    val_ds   = ClaimDataset(val_c,   val_l,   tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer    = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_f1   = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")
        for batch in pbar:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lbls  = batch["label"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                logits = model(input_ids=ids, attention_mask=mask).logits
                loss   = F.cross_entropy(logits, lbls, weight=weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), GRAD_CLIP
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            run_loss += loss.item()
            preds     = logits.argmax(dim=1)
            correct  += (preds == lbls).sum().item()
            total    += lbls.size(0)
            pbar.set_postfix(loss=f"{run_loss/(pbar.n+1):.4f}", acc=f"{correct/total:.2%}")

        val_loss, val_acc, val_f1, per_class = evaluate(model, val_loader, device, weights)
        pc_str = "  ".join(f"{k}={c}/{t}" for k, (c, t) in per_class.items())
        print(f"  [val] loss={val_loss:.4f}  acc={val_acc:.2%}  macro-F1={val_f1:.4f}  {pc_str}")

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            print(f"  [*] New best macro-F1 → {val_f1:.4f}")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP:
                print(f"\n[train] Early stopping at epoch {epoch} (no improvement for {EARLY_STOP} epochs)")
                break

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n[train] Best val macro-F1: {best_f1:.4f}")
    torch.save({
        "model_state":  best_state,
        "config": {
            "backbone":  BACKBONE,
            "num_labels": 3,
            "label2id":  LABEL2ID,
            "lora": {
                "enabled":        True,
                "r":              LORA_R,
                "alpha":          LORA_ALPHA,
                "dropout":        LORA_DROPOUT,
                "target_modules": LORA_TARGETS,
            },
        },
        "best_val_f1": best_f1,
    }, str(OUT_PATH))
    print(f"[train] Saved → {OUT_PATH}")


if __name__ == "__main__":
    train()
