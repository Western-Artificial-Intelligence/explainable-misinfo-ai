"""
Fine-tune best.ckpt (binary false/true) → 3-way (false/mixed/true).

Strategy:
  1. Load best.ckpt weights into the binary model
  2. Expand the final Linear of label_head: [256→2] → [256→3]
       false row and true row kept; mixed row initialised as their mean
  3. Freeze everything except label_head + LoRA adapter weights
  4. Fine-tune on unified_liar.parquet (label_3way) with class-weighted loss
  5. Save best checkpoint by val accuracy → best_3way.ckpt

Usage:
  python api/production_pipeline/RoBERTa_model/finetune_3way.py
"""

import random
import sys
import collections
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT))
for _p in [HERE.parents[2], HERE.parents[1]]:
    if (_p / "baseline_model").is_dir():
        sys.path.insert(0, str(_p))
        break

CKPT_PATH = HERE / "best.ckpt"
DATA_PATH = HERE / "unified_liar.parquet"
OUT_PATH  = HERE / "best_3way.ckpt"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
EPOCHS     = 10
LR         = 2e-5
BATCH_SIZE = 64
MAX_LEN    = 256
SEED       = 42
VAL_FRAC   = 0.10

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
    """Manual stratified train/val split (no sklearn dependency)."""
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


# ── Model loading + head surgery ──────────────────────────────────────────────

def load_model_binary():
    """Load best.ckpt exactly as it was trained (2-class label head)."""
    from baseline_model.models.domain_adversarial import DomainAdversarialClassifier
    from baseline_model.models.lora_utils import try_apply_peft_lora
    from baseline_model.data_utils.tokenize import build_tokenizer

    state       = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    cfg         = state.get("config", {})
    model_cfg   = cfg.get("model", {})
    lora_cfg    = cfg.get("lora", {})

    backbone    = model_cfg.get("backbone", "roberta-base")
    lambda_adv  = float(model_cfg.get("lambda_adv", 0.1))
    num_sources = state["model_state"]["domain_head.3.weight"].shape[0]

    tokenizer = build_tokenizer(backbone)

    model = DomainAdversarialClassifier(
        backbone_name=backbone,
        num_sources=num_sources,
        lambda_adv=lambda_adv,
    )
    model.encoder.resize_token_embeddings(len(tokenizer))

    if lora_cfg.get("enabled", False):
        model, _ = try_apply_peft_lora(
            model,
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("alpha", 16),
            lora_dropout=lora_cfg.get("dropout", 0.1),
            target_modules=lora_cfg.get("target_modules", ["query", "value"]),
        )

    model.load_state_dict(state["model_state"])
    return model, tokenizer, cfg


def expand_label_head(model):
    """Replace label_head[-1]: Linear(256,2) → Linear(256,3).

    Row layout:  0=false  1=mixed  2=true
    mixed row initialised as mean(false_row, true_row) so it starts
    between the two poles rather than random.
    """
    old = model.label_head[4]          # nn.Linear(256, 2)
    assert isinstance(old, nn.Linear) and old.out_features == 2, \
        f"Expected Linear(256,2) at label_head[4], got {old}"

    new = nn.Linear(old.in_features, 3)
    with torch.no_grad():
        new.weight[0] = old.weight[0]                               # false
        new.weight[2] = old.weight[1]                               # true
        new.weight[1] = (old.weight[0] + old.weight[1]) / 2        # mixed
        new.bias[0]   = old.bias[0]
        new.bias[2]   = old.bias[1]
        new.bias[1]   = (old.bias[0] + old.bias[1]) / 2
    model.label_head[4] = new
    print("[finetune] label_head expanded: Linear(256,2) → Linear(256,3)")


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, weights):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lbls  = batch["label"].to(device)
            logits = model(input_ids=ids, attention_mask=mask)["logits_label"]
            total_loss += F.cross_entropy(logits, lbls, weight=weights).item()
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(lbls.cpu().tolist())

    avg_loss = total_loss / max(1, len(loader))
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    per_class = {}
    for lid, lname in ID2LABEL.items():
        indices = [i for i, l in enumerate(all_labels) if l == lid]
        correct = sum(all_preds[i] == lid for i in indices)
        per_class[lname] = (correct, len(indices))
    model.train()
    return avg_loss, acc, per_class


# ── Main ──────────────────────────────────────────────────────────────────────

def finetune():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[finetune] Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"[finetune] Loading {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH).dropna(subset=["claim_text", "label_3way"])
    df = df[df["label_3way"].isin(LABEL2ID)].reset_index(drop=True)

    counts_str = " | ".join(f"{k}={v}" for k, v in df["label_3way"].value_counts().items())
    print(f"[finetune] {len(df)} examples — {counts_str}")

    claims = df["claim_text"].tolist()
    labels = [LABEL2ID[l] for l in df["label_3way"]]

    train_c, train_l, val_c, val_l = stratified_split(claims, labels, VAL_FRAC, SEED)
    print(f"[finetune] Split → train={len(train_c)} | val={len(val_c)}")

    # Class weights (inverse frequency, normalised to mean=1)
    counts = collections.Counter(train_l)
    n_train = len(train_l)
    raw_w   = [n_train / (3 * counts[i]) for i in range(3)]
    weights = torch.tensor(raw_w, dtype=torch.float32).to(device)
    print(f"[finetune] Class weights → false={weights[0]:.3f}  mixed={weights[1]:.3f}  true={weights[2]:.3f}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"[finetune] Loading {CKPT_PATH}")
    model, tokenizer, orig_cfg = load_model_binary()
    expand_label_head(model)
    model.to(device)

    # Freeze everything except label_head and LoRA adapter weights
    for name, param in model.named_parameters():
        trainable = "label_head" in name or "lora_" in name
        param.requires_grad_(trainable)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"[finetune] Trainable: {n_trainable:,} / {n_total:,} params")

    # ── Loaders ───────────────────────────────────────────────────────────────
    train_ds = ClaimDataset(train_c, train_l, tokenizer, MAX_LEN)
    val_ds   = ClaimDataset(val_c,   val_l,   tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scaler    = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_state   = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")
        for batch in pbar:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lbls  = batch["label"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(input_ids=ids, attention_mask=mask)["logits_label"]
                loss   = F.cross_entropy(logits, lbls, weight=weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            run_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total   += lbls.size(0)
            pbar.set_postfix(loss=f"{run_loss/(pbar.n+1):.4f}", acc=f"{correct/total:.2%}")

        val_loss, val_acc, per_class = evaluate(model, val_loader, device, weights)
        pc_str = "  ".join(f"{k}={c}/{t}" for k, (c, t) in per_class.items())
        print(f"  [val] loss={val_loss:.4f}  acc={val_acc:.2%}  {pc_str}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  [*] New best → {val_acc:.2%}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n[finetune] Best val acc: {best_val_acc:.2%}")
    save_cfg = dict(orig_cfg)
    save_cfg["model"] = dict(orig_cfg.get("model", {}))
    save_cfg["model"]["num_classes"] = 3

    torch.save({
        "model_state":  best_state,
        "config":       save_cfg,
        "best_val_acc": best_val_acc,
        "label2id":     LABEL2ID,
    }, str(OUT_PATH))
    print(f"[finetune] Saved → {OUT_PATH}")


if __name__ == "__main__":
    finetune()
