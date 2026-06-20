"""The three continued-training cross-entropy losses, on the AA half only (spec §7).

``L = L_MLM_encoder + L_MLM_decoder + L_CLM_decoder``

* MLM terms (encoder + decoder): targets are the *clean* tokens; loss is scored only at
  positions that were sequence-masked. Any sequence whose realized sequence-track mask
  rate exceeds ``rate_cap`` (0.30) contributes no MLM loss (spec §7, notebook rule).
* CLM term: next-token CE on a *clean*, un-noised target — exactly the path in
  ``poet_2.models.poet_2_helpers.score_sequences_given_memory`` (logits[:, :-1] vs
  target[:, 1:], ``ignore_index`` = pad/mask). The CLM decoder predicts every real residue.

All losses use only the AA half of the 58-wide head (``logits[..., :AA_VOCAB]``, identical
to ``chunk(2, -1)[0]`` for the 58-wide case); the 3Di half never receives loss.

Distributed normalization: per-term we reduce ``(loss_sum, n_scored)`` and divide by the
**global** token count, then scale the grad-path term by ``world_size`` so that DDP's
gradient *averaging* (÷ world_size) yields the correct global ``(ΣΣ loss) / (ΣΣ tokens)``
rather than a per-rank mean of means. See :func:`total_loss`.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional

import torch
import torch.nn.functional as F

from poet_2.alphabet.sparse_uniref_cluster2 import Alphabet

#: Width of the AA half of the head (== model ``n_vocab``); full head is ``2 * AA_VOCAB``.
AA_VOCAB: int = 29
#: ``X`` == 24 — the mask token, which doubles as the pad / "missing" value.
MASK_TOKEN: int = int(Alphabet().mask_token)

AllReduce = Callable[[torch.Tensor], None]


def aa_logits(logits: torch.Tensor, n_aa: int = AA_VOCAB) -> torch.Tensor:
    """Slice the AA half of the head. For the 58-wide head this is ``chunk(2, -1)[0]``;
    for an already-29-wide tensor it is a no-op."""
    return logits[..., :n_aa]


def masked_mlm_loss_sum(
    logits: torch.Tensor,
    targets: torch.Tensor,
    was_masked: torch.Tensor,
    seq_mask_rate: torch.Tensor,
    *,
    rate_cap: float = 0.30,
    ignore_index: int = MASK_TOKEN,
    n_aa: int = AA_VOCAB,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed MLM cross-entropy over masked positions, with the >``rate_cap`` drop rule.

    Args:
        logits: ``(B, L, V)`` — V is 58 (full head) or 29 (AA half).
        targets: ``(B, L)`` long — the clean tokens.
        was_masked: ``(B, L)`` bool — positions that were sequence-masked.
        seq_mask_rate: ``(B,)`` float — per-sequence realized mask rate.

    Returns ``(loss_sum, n_scored)``: a scalar summed loss (with grad) and the scalar
    count of scored positions. ``F.cross_entropy(..., reduction="sum")`` returns 0.0 when
    every position is ignored, so an all-dropped batch gives ``(0, 0)`` (no NaN).
    """
    lg = aa_logits(logits, n_aa)  # (B, L, n_aa)
    targets = targets.long()
    keep = was_masked.bool().clone()
    over = seq_mask_rate > rate_cap  # (B,) — drop whole sequences over the cap
    keep[over] = False
    t = torch.where(keep, targets, torch.full_like(targets, ignore_index))
    loss_sum = F.cross_entropy(
        lg.transpose(1, 2), t, ignore_index=ignore_index, reduction="sum"
    )
    n_scored = (t != ignore_index).sum()
    return loss_sum.float(), n_scored


def clm_loss_sum(
    logits: torch.Tensor,
    clm_targets: torch.Tensor,
    *,
    ignore_index: int = MASK_TOKEN,
    n_aa: int = AA_VOCAB,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed next-token CE on a clean target (mirrors ``score_sequences_given_memory``).

    Returns ``(loss_sum, n_scored)``. Only pad positions (``ignore_index``) are skipped;
    every real residue (including the trailing ``*``) is predicted.
    """
    lg = aa_logits(logits, n_aa)[:, :-1]  # predict token t+1 from positions <= t
    tgt = clm_targets.long()[:, 1:]
    loss_sum = F.cross_entropy(
        lg.transpose(1, 2), tgt, ignore_index=ignore_index, reduction="sum"
    )
    n_scored = (tgt != ignore_index).sum()
    return loss_sum.float(), n_scored


def total_loss(
    xs_logits: torch.Tensor,
    mlm_logits: torch.Tensor,
    clm_logits: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    *,
    world_size: int = 1,
    all_reduce: Optional[AllReduce] = None,
    rate_cap: float = 0.30,
    ignore_index: int = MASK_TOKEN,
    n_aa: int = AA_VOCAB,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine the three terms into a single scalar with correct DDP normalization.

    ``batch`` must provide: ``xs_targets/xs_was_masked/xs_seq_mask_rate`` (encoder MLM),
    ``mlm_ys_targets/mlm_ys_was_masked/mlm_ys_seq_mask_rate`` (decoder MLM), and
    ``clm_ys_targets`` (decoder CLM).

    ``all_reduce`` (optional) does an in-place SUM all-reduce across ranks; pass
    ``lambda t: torch.distributed.all_reduce(t)`` under DDP. When ``None`` (single
    process / tests) the global counts equal the local counts.

    Returns ``(loss, logs)`` — ``loss`` is the scalar to call ``.backward()`` on; ``logs``
    holds detached per-term means and token counts.
    """
    enc_sum, enc_n = masked_mlm_loss_sum(
        xs_logits,
        batch["xs_targets"],
        batch["xs_was_masked"],
        batch["xs_seq_mask_rate"],
        rate_cap=rate_cap,
        ignore_index=ignore_index,
        n_aa=n_aa,
    )
    dec_sum, dec_n = masked_mlm_loss_sum(
        mlm_logits,
        batch["mlm_ys_targets"],
        batch["mlm_ys_was_masked"],
        batch["mlm_ys_seq_mask_rate"],
        rate_cap=rate_cap,
        ignore_index=ignore_index,
        n_aa=n_aa,
    )
    clm_sum, clm_n = clm_loss_sum(
        clm_logits, batch["clm_ys_targets"], ignore_index=ignore_index, n_aa=n_aa
    )

    names = ("L_mlm_enc", "L_mlm_dec", "L_clm_dec")
    local_sums = (enc_sum, dec_sum, clm_sum)
    # Detached counts and detached sums (sums for logging only); both reduced with SUM.
    global_ns = torch.stack([enc_n, dec_n, clm_n]).float().detach()
    log_sums = torch.stack([s.detach() for s in local_sums]).float()
    if all_reduce is not None:
        all_reduce(global_ns)
        all_reduce(log_sums)

    loss = xs_logits.new_zeros(())
    logs: dict[str, float] = {}
    for name, local_sum, gn, log_sum in zip(names, local_sums, global_ns, log_sums):
        if float(gn) > 0:
            # grad path: local_sum / global_n * world_size, so DDP's ÷world_size grad
            # averaging recovers (Σ_ranks local_sum) / global_n.
            loss = loss + local_sum / gn * world_size
            logs[name] = float(log_sum / gn)
        else:
            loss = loss + local_sum * 0.0
            logs[name] = 0.0
        logs[f"n_{name}"] = float(gn)
    logs["L_total"] = logs["L_mlm_enc"] + logs["L_mlm_dec"] + logs["L_clm_dec"]
    return loss, logs
