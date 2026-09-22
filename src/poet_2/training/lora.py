"""LoRA (Low-Rank Adaptation) for PoET-2 continued training.

Wraps attention Q/V projections with low-rank adapters so the pretrained base
weights stay frozen. This prevents catastrophic forgetting of the IFQ pathway
while allowing adaptation to new domains.

Because PoET-2's encoder, MLM decoder, and CLM decoder share weights via
``tie_module_weights``, LoRA adapters on the encoder's attention modules affect
all three code paths through the shared base weight tensors (frozen), while the
LoRA delta is only added through the encoder's forward pass.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with a low-rank adapter.

    output = base_linear(x) + (x @ A^T @ B^T) * (alpha / rank)

    The base linear's parameters are frozen; only A and B train.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        in_features = base.in_features
        out_features = base.out_features

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scale
        return base_out + lora_out

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: Optional[float] = None,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    module_prefix: str = "encoder",
) -> int:
    """Apply LoRA adapters to target Linear modules and freeze all base weights.

    Only modules whose path starts with ``module_prefix`` are wrapped. For PoET-2
    this should be ``"encoder"`` because the encoder, mlm_decoder, and clm_decoder
    share weights via ``tie_module_weights`` — wrapping all three would create
    conflicting LoRA adapters on the same shared base tensor.

    Args:
        model: The PoET-2 model (not wrapped in DDP yet).
        rank: LoRA rank (number of low-rank dimensions).
        alpha: LoRA scaling factor. Defaults to 2 * rank.
        target_modules: Names of Linear submodules to wrap (matched by suffix).
        module_prefix: Only wrap modules whose name starts with this prefix.

    Returns:
        Number of LoRA adapter parameters added.
    """
    if alpha is None:
        alpha = float(2 * rank)

    for p in model.parameters():
        p.requires_grad_(False)

    n_lora_params = 0

    for name, module in model.named_modules():
        if not name.startswith(module_prefix):
            continue
        for attr_name in target_modules:
            child = getattr(module, attr_name, None)
            if child is None or not isinstance(child, nn.Linear):
                continue
            if isinstance(child, LoRALinear):
                continue
            lora = LoRALinear(child, rank=rank, alpha=alpha)
            lora = lora.to(device=child.weight.device, dtype=child.weight.dtype)
            setattr(module, attr_name, lora)
            n_lora_params += lora.lora_A.numel() + lora.lora_B.numel()

    return n_lora_params


def lora_parameters(model: nn.Module):
    """Yield only the LoRA adapter parameters (A and B matrices)."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_A
            yield module.lora_B


def merge_lora(model: nn.Module) -> None:
    """Merge LoRA weights into base weights and remove adapters.

    After merging, the model behaves identically but without LoRA overhead.
    Call before saving the final checkpoint for inference.
    """
    for name, module in list(model.named_modules()):
        for attr_name in ("q_proj", "v_proj", "k_proj", "out_proj"):
            child = getattr(module, attr_name, None)
            if child is None or not isinstance(child, LoRALinear):
                continue
            with torch.no_grad():
                delta = (child.lora_B @ child.lora_A) * child.scale
                child.base.weight.add_(delta)
            setattr(module, attr_name, child.base)
