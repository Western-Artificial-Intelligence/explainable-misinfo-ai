# ===== baseline_model/data_utils/tokenize.py =====
from transformers import AutoTokenizer
from typing import Any, Dict, List

SPECIAL_TOKENS: List[str] = ["<CLAIM>", "</CLAIM>", "<ARTICLE>", "</ARTICLE>"]

def build_tokenizer(model_name: str, special_tokens: List[str] = SPECIAL_TOKENS):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if added:
        print(f"[tokenize] Added {added} special tokens to tokenizer.")
    return tokenizer

def tokenize_batch(batch: Dict[str, Any], tokenizer, max_len: int = 256):
    """
    Tokenize a batch for datasets.map(batched=True).
    IMPORTANT: return plain Python lists (not torch tensors).
    """
    texts = batch.get("input_text", [])
    out = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_attention_mask=True,
        return_token_type_ids=False  # roberta doesn't use token_type_ids
    )
    # tokenizer returns lists when return_tensors is not set — good for datasets.map
    return {
        "input_ids": out["input_ids"],
        "attention_mask": out["attention_mask"]
    }