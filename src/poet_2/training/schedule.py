"""LR schedules: linear warmup -> cosine decay (default) or inverse-sqrt (spec §8.4)."""

from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR


def _warmup_cosine_fn(warmup_steps: int, total_steps: int, min_lr_frac: float):
    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if total_steps <= warmup_steps:
            return 1.0
        progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_frac + (1.0 - min_lr_frac) * cos

    return fn


def _warmup_inverse_sqrt_fn(warmup_steps: int):
    # peak at end of warmup, then ~1/sqrt(step) decay (matches the original recipe shape)
    w = max(1, warmup_steps)

    def fn(step: int) -> float:
        if step < w:
            return (step + 1) / w
        return (w / (step + 1)) ** 0.5

    return fn


def build_scheduler(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    kind: str = "cosine",
    min_lr_frac: float = 0.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """Return a LambdaLR scaling the optimizer's base LR by the chosen schedule.

    The multiplier is 0->1 over ``warmup_steps``, then decays per ``kind``. ``cosine``
    decays to ``min_lr_frac`` at ``total_steps``; ``inverse_sqrt`` decays as 1/sqrt(step).
    """
    if kind == "cosine":
        fn = _warmup_cosine_fn(warmup_steps, total_steps, min_lr_frac)
    elif kind == "inverse_sqrt":
        fn = _warmup_inverse_sqrt_fn(warmup_steps)
    else:
        raise ValueError(f"unknown schedule {kind!r}")
    return LambdaLR(optimizer, fn, last_epoch=last_epoch)
