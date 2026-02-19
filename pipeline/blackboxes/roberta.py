"""RoBERTa inference blackbox. Claim-only mode for production."""

import os
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_project_root))

_model = None
_tokenizer = None


def _load_model(checkpoint: str, backbone: str = "roberta-base", device: str = "cpu"):
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        return None, None

    try:
        import torch
        from baseline_model.data_utils.tokenize import build_tokenizer
        from baseline_model.models.domain_adversarial import DomainAdversarialClassifier
    except ImportError:
        return None, None

    _tokenizer = build_tokenizer(backbone)
    state = torch.load(checkpoint, map_location=device)
    _model = DomainAdversarialClassifier(
        backbone_name=backbone, num_sources=3, lambda_adv=0.5
    )
    _model.load_state_dict(state["model_state"])
    _model.to(device)
    _model.eval()
    return _model, _tokenizer


def roberta_infer(
    claim: str,
    article: str | None = None,
    checkpoint: str | None = None,
    backbone: str = "roberta-base",
    max_len: int = 256,
    device: str = "cpu",
) -> dict:
    """
    Returns: {h, label_logits, label_probs}
    For claim-only: pass article=None or empty string.
    """
    checkpoint = checkpoint or os.getenv(
        "ROBERTA_CHECKPOINT",
        "baseline_outputs/baseline/checkpoints/best_model.pt",
    )
    model, tokenizer = _load_model(checkpoint, backbone, device)

    if model is None or tokenizer is None:
        # Fallback: return dummy probs when no model available
        return {
            "h": None,
            "label_logits": [0.0, 0.0, 0.0],
            "label_probs": [0.33, 0.34, 0.33],
        }

    import torch

    text = f"<CLAIM> {claim} </CLAIM>"
    if article and article.strip():
        text += f" <ARTICLE> {article} </ARTICLE>"
    else:
        text += " <ARTICLE> </ARTICLE>"

    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = out["logits_label"].cpu().numpy()[0].tolist()
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy().tolist()
    h = out.get("pooled")
    if h is not None:
        h = h.cpu().numpy().tolist()

    return {
        "h": h,
        "label_logits": logits,
        "label_probs": probs,
    }
