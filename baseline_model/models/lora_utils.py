"""
LoRA utility.

Prefers using PEFT if available. If not installed, this module gracefully
falls back to a no-op (i.e., no LoRA applied) and prints a warning.
"""


def try_apply_peft_lora(model, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=None):
    """
    Try to apply PEFT LoRA to model.encoder. 

    Args:
        model: PyTorch model with an 'encoder' attribute (e.g., transformer backbone)
        r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: dropout probability
        target_modules: list of module names to apply LoRA (defaults to ["query","value"])

    Returns:
        model: potentially wrapped with LoRA
        peft_applied: bool indicating if PEFT was applied
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except Exception:
        print("[lora_utils] PEFT not available, continuing without LoRA. Install 'peft' to enable LoRA.")
        return model, False

    # Build LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules or ["query", "value"]
    )

    # Apply PEFT LoRA to model.encoder
    model.encoder = get_peft_model(model.encoder, lora_config)
    print("[lora_utils] Applied PEFT LoRA to encoder.")

    return model, True