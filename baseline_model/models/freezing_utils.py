"""
Utilities for freezing/unfreezing model parameters and constructing
per-layer parameter groups with layer-wise learning rate decay (LLRD).

Used by the two-phase training script (train_twophase.py).
"""

import re
from collections import defaultdict


def freeze_encoder_all(model):
    """Freeze all encoder parameters (base weights + LoRA adapters) for Phase 1."""
    for name, param in model.encoder.named_parameters():
        param.requires_grad = False


def unfreeze_lora_only(model):
    """Unfreeze only LoRA adapter parameters in the encoder for Phase 2 (RoBERTa)."""
    for name, param in model.encoder.named_parameters():
        if "lora_" in name:
            param.requires_grad = True


def unfreeze_all_encoder_layers(model):
    """Unfreeze all encoder parameters for Phase 2 (DistilBERT / no-LoRA)."""
    for name, param in model.encoder.named_parameters():
        param.requires_grad = True


def get_layer_index(name):
    """
    Extract the transformer layer index from a parameter name.

    Handles patterns like:
      - encoder.layer.3.attention...
      - roberta.encoder.layer.5...
      - base_model.model.encoder.layer.7...  (PEFT-wrapped)
      - transformer.layer.2...  (DistilBERT)

    Returns the integer layer index, or None if no layer index found
    (e.g. embeddings, pooler, head parameters).
    """
    match = re.search(r"\.layer\.(\d+)\.", name)
    if match:
        return int(match.group(1))
    return None


def detect_num_layers(model):
    """
    Detect the number of transformer layers from encoder parameter names.

    Returns the count of unique layer indices found.
    """
    layer_indices = set()
    for name, _ in model.encoder.named_parameters():
        idx = get_layer_index(name)
        if idx is not None:
            layer_indices.add(idx)
    return max(layer_indices) + 1 if layer_indices else 0


def build_phase1_param_groups(model, lr, weight_decay):
    """
    Build optimizer parameter groups for Phase 1 (head-only training).

    Only includes parameters with requires_grad=True (should be heads only
    after calling freeze_encoder_all).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    return [{"params": params, "lr": lr, "weight_decay": weight_decay}]


def build_phase2_param_groups_lora(model, head_lr, lora_top_lr, decay, weight_decay, num_layers):
    """
    Build optimizer parameter groups for Phase 2 with LoRA + LLRD.

    - Head parameters get `head_lr`
    - LoRA parameters get layer-wise decayed LR: top layer gets `lora_top_lr`,
      each lower layer is multiplied by `decay`
    - Non-LoRA encoder parameters stay frozen (not included)

    Args:
        model: The full model (with encoder, label_head, domain_head)
        head_lr: Learning rate for classification heads
        lora_top_lr: Learning rate for the topmost LoRA layer
        decay: Multiplicative decay per layer (e.g. 0.9)
        weight_decay: Weight decay for all groups
        num_layers: Number of transformer layers
    """
    # Collect head parameters
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "label_head" in name or "domain_head" in name or "grl" in name:
            head_params.append(param)

    # Collect LoRA parameters by layer
    lora_by_layer = defaultdict(list)
    lora_no_layer = []  # LoRA params without a layer index (e.g. in embeddings)

    for name, param in model.encoder.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            layer_idx = get_layer_index(name)
            if layer_idx is not None:
                lora_by_layer[layer_idx].append(param)
            else:
                lora_no_layer.append(param)

    groups = []

    # Head group
    if head_params:
        groups.append({
            "params": head_params,
            "lr": head_lr,
            "weight_decay": weight_decay,
        })

    # LoRA groups with LLRD (top layer = num_layers - 1)
    for layer_idx in sorted(lora_by_layer.keys()):
        distance_from_top = (num_layers - 1) - layer_idx
        lr = lora_top_lr * (decay ** distance_from_top)
        groups.append({
            "params": lora_by_layer[layer_idx],
            "lr": lr,
            "weight_decay": weight_decay,
        })

    # LoRA params without layer index get the lowest LR
    if lora_no_layer:
        lr = lora_top_lr * (decay ** (num_layers - 1))
        groups.append({
            "params": lora_no_layer,
            "lr": lr,
            "weight_decay": weight_decay,
        })

    return groups


def build_phase2_param_groups_distil(model, head_lr, encoder_top_lr, decay, weight_decay, num_layers):
    """
    Build optimizer parameter groups for Phase 2 with full encoder LLRD (no LoRA).

    - Head parameters get `head_lr`
    - Encoder parameters get layer-wise decayed LR based on their layer index
    - Embedding parameters get the lowest LR

    Args:
        model: The full model
        head_lr: Learning rate for classification heads
        encoder_top_lr: Learning rate for the topmost encoder layer
        decay: Multiplicative decay per layer
        weight_decay: Weight decay for all groups
        num_layers: Number of transformer layers
    """
    # Collect head parameters
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "label_head" in name or "domain_head" in name or "grl" in name:
            head_params.append(param)

    # Collect encoder parameters by layer
    encoder_by_layer = defaultdict(list)
    encoder_no_layer = []  # embeddings, pooler, etc.

    for name, param in model.encoder.named_parameters():
        if not param.requires_grad:
            continue
        layer_idx = get_layer_index(name)
        if layer_idx is not None:
            encoder_by_layer[layer_idx].append(param)
        else:
            encoder_no_layer.append(param)

    groups = []

    # Head group
    if head_params:
        groups.append({
            "params": head_params,
            "lr": head_lr,
            "weight_decay": weight_decay,
        })

    # Encoder groups with LLRD
    for layer_idx in sorted(encoder_by_layer.keys()):
        distance_from_top = (num_layers - 1) - layer_idx
        lr = encoder_top_lr * (decay ** distance_from_top)
        groups.append({
            "params": encoder_by_layer[layer_idx],
            "lr": lr,
            "weight_decay": weight_decay,
        })

    # Embedding / non-layer params get lowest LR
    if encoder_no_layer:
        lr = encoder_top_lr * (decay ** (num_layers - 1))
        groups.append({
            "params": encoder_no_layer,
            "lr": lr,
            "weight_decay": weight_decay,
        })

    return groups


def count_parameters(model):
    """
    Count total, trainable, and frozen parameters.

    Returns:
        dict with keys: total, trainable, frozen
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}
