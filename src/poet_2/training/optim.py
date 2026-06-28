"""Optimizer factory.

Default is AdamW (in-torch, zero extra deps) — a solid choice for a continual run and
what the plan uses for M1-M3 iteration. ``adafactor`` matches the original recipe (spec
§8.4) but PyTorch 2.6 ships no Adafactor, so it's sourced from ``transformers`` (opt-in;
a parity-critical optimizer is not hand-rolled here). With an external LR schedule it's
configured ``relative_step=False, warmup_init=False``.
"""

from __future__ import annotations

import torch


def build_optimizer(params, cfg) -> torch.optim.Optimizer:
    name = cfg.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(
            params, lr=cfg.peak_lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay
        )
    if name == "adafactor":
        try:
            from transformers.optimization import Adafactor
        except ImportError as e:  # pragma: no cover - depends on env
            raise ImportError(
                "optimizer='adafactor' needs the `transformers` package "
                "(`pixi add transformers`). Use optimizer='adamw' to avoid the dependency."
            ) from e
        return Adafactor(
            params,
            lr=cfg.peak_lr,
            scale_parameter=True,  # spec §8.4 underspecified; matches HF default
            relative_step=False,  # external LR schedule drives the rate
            warmup_init=False,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")
