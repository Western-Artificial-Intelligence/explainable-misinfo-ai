"""
baseline_model/models/domain_adversarial.py

DomainAdversarialClassifier implementation:

- Encoder: RoBERTa (AutoModel)
- Label head: maps CLS to logits over 3 classes
- Domain head: gradient reversal + MLP mapping to num_sources logits
- compute_loss implements partial-label loss (3-way + binary coarse) and domain loss
"""

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_adv: float):
        ctx.lambda_adv = float(lambda_adv)
        # Forward is identity
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward multiplies gradient by -lambda_adv
        return -ctx.lambda_adv * grad_output, None


class GradientReversal(nn.Module):
    """
    Gradient Reversal Layer wrapper that uses the custom autograd Function above.
    The module stores lambda_adv so it can be read by the model (for logging / weighting).
    """

    def __init__(self, lambda_adv: float = 1.0):
        super().__init__()
        self.lambda_adv = float(lambda_adv)

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_adv)


class DomainAdversarialClassifier(nn.Module):
    """
    Domain-adversarial classifier with:
     - encoder (transformer backbone)
     - label head (3-way classification)
     - domain head (adversarial)
    """

    def __init__(self, backbone_name: str = "roberta-base", num_sources: int = 3, lambda_adv: float = 0.5):
        super().__init__()

        # Encoder
        self.config = AutoConfig.from_pretrained(backbone_name)
        self.encoder = AutoModel.from_pretrained(backbone_name, config=self.config)
        hidden = self.config.hidden_size  # usually 768 for roberta-base

        # Label head (3-way)
        self.label_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3),
        )

        # Domain head (adversarial)
        self.grl = GradientReversal(lambda_adv=lambda_adv)
        self.domain_head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_sources),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_3way: Optional[torch.Tensor] = None,
        label_bin: Optional[torch.Tensor] = None,
        source_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass.

        Returns a dict with:
            - logits_label: [B, 3]
            - logits_domain: [B, num_sources] or None
            - loss: scalar tensor if labels provided
        """
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # CLS embedding (last_hidden_state[:, 0, :])
        h = enc_out.last_hidden_state[:, 0, :]  # shape [B, hidden]

        logits_label = self.label_head(h)  # [B, 3]
        logits_domain = self.domain_head(self.grl(h)) if source_id is not None else None

        outputs: Dict[str, Any] = {"logits_label": logits_label, "logits_domain": logits_domain}

        if (label_3way is not None) or (label_bin is not None):
            loss = self.compute_loss(logits_label, logits_domain, label_3way, label_bin, source_id)
            outputs["loss"] = loss

        return outputs

    def compute_loss(
        self,
        logits_label: torch.Tensor,
        logits_domain: Optional[torch.Tensor],
        label_3way: Optional[torch.Tensor],
        label_bin: Optional[torch.Tensor],
        source_id: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute combined loss:
         - Cross-entropy for 3-way labels where label_3way != -100
         - Coarse binary loss for binary-labeled rows where label_bin != -100
         - Domain adversarial loss (cross-entropy) weighted by lambda_adv
        """
        device = logits_label.device
        probs = F.softmax(logits_label, dim=-1)  # [B, 3]

        # 3-way loss
        loss_3 = torch.tensor(0.0, device=device)
        if label_3way is not None:
            mask_3 = (label_3way != -100)
            if mask_3.any():
                loss_3 = F.cross_entropy(logits_label[mask_3], label_3way[mask_3])

        # binary coarse loss
        loss_b = torch.tensor(0.0, device=device)
        if label_bin is not None:
            mask_b = (label_bin != -100)
            if mask_b.any():
                lb = label_bin[mask_b]  # [Nb]
                pb = probs[mask_b]      # [Nb, 3]

                p_false = pb[:, 0]
                p_mixed = pb[:, 1]
                p_true = pb[:, 2]

                loss_bin = torch.zeros_like(lb, dtype=torch.float32, device=device)
                false_mask = (lb == 0)
                true_mask = (lb == 1)

                # Negative log-likelihood style losses for coarse mapping:
                if false_mask.any():
                    loss_bin[false_mask] = -torch.log(p_false[false_mask] + 1e-8)
                if true_mask.any():
                    # treat 'true-ish' binary label as (true OR mixed)
                    loss_bin[true_mask] = -torch.log((p_true[true_mask] + p_mixed[true_mask]) + 1e-8)

                loss_b = loss_bin.mean()

        # domain adversarial loss
        loss_dom = torch.tensor(0.0, device=device)
        if (logits_domain is not None) and (source_id is not None):
            loss_dom = F.cross_entropy(logits_domain, source_id)

        lambda_adv = getattr(self.grl, "lambda_adv", 0.0)
        loss = loss_3 + loss_b + lambda_adv * loss_dom
        return loss