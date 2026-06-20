"""CPU unit tests for the §7 three-term loss (poet_2.training.losses)."""

import torch
import torch.nn.functional as F

from poet_2.training.losses import (
    AA_VOCAB,
    MASK_TOKEN,
    aa_logits,
    clm_loss_sum,
    masked_mlm_loss_sum,
    total_loss,
)

FULL = 2 * AA_VOCAB  # 58-wide head


def _ref_ce_sum(logits_aa, targets, keep):
    """Independent reference: -Σ log_softmax(logits)[target] over kept positions."""
    lp = F.log_softmax(logits_aa.float(), dim=-1)
    g = lp.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return -(g[keep]).sum()


def test_aa_logits_slices_first_half():
    x = torch.randn(2, 4, FULL)
    assert torch.equal(aa_logits(x), x[..., :AA_VOCAB])
    assert torch.equal(aa_logits(x), x.chunk(2, dim=-1)[0])  # matches inference slicing
    # no-op when already AA-width
    y = torch.randn(2, 4, AA_VOCAB)
    assert torch.equal(aa_logits(y), y)


def test_mlm_scores_only_masked_positions():
    B, L = 2, 5
    torch.manual_seed(0)
    logits = torch.randn(B, L, FULL)
    targets = torch.randint(0, 20, (B, L))
    was_masked = torch.zeros(B, L, dtype=torch.bool)
    was_masked[0, 1] = was_masked[0, 3] = True  # seq0: two masked; seq1: none
    rate = torch.tensor([0.4 / 5 * 2, 0.0])  # well under the 0.30 cap

    loss_sum, n = masked_mlm_loss_sum(logits, targets, was_masked, rate)
    assert int(n) == 2
    ref = _ref_ce_sum(aa_logits(logits), targets, was_masked)
    assert torch.allclose(loss_sum, ref, atol=1e-5)

    # corrupting an UNMASKED position's logits/target must not change the loss
    logits2 = logits.clone()
    logits2[1, :, :] += 100.0
    targets2 = targets.clone()
    targets2[1, :] = 7
    loss_sum2, n2 = masked_mlm_loss_sum(logits2, targets2, was_masked, rate)
    assert int(n2) == 2
    assert torch.allclose(loss_sum, loss_sum2, atol=1e-5)


def test_mlm_drops_sequences_over_rate_cap():
    B, L = 2, 6
    torch.manual_seed(1)
    logits = torch.randn(B, L, FULL)
    targets = torch.randint(0, 20, (B, L))
    was_masked = torch.zeros(B, L, dtype=torch.bool)
    was_masked[0, 1:4] = True
    was_masked[1, 1:3] = True
    rate = torch.tensor([0.5, 0.2])  # seq0 over the 0.30 cap -> dropped

    loss_sum, n = masked_mlm_loss_sum(logits, targets, was_masked, rate, rate_cap=0.30)
    keep = was_masked.clone()
    keep[0] = False  # only seq1 survives
    assert int(n) == int(keep.sum())
    ref = _ref_ce_sum(aa_logits(logits), targets, keep)
    assert torch.allclose(loss_sum, ref, atol=1e-5)


def test_mlm_all_dropped_is_zero_not_nan():
    B, L = 2, 4
    logits = torch.randn(B, L, FULL)
    targets = torch.randint(0, 20, (B, L))
    was_masked = torch.zeros(B, L, dtype=torch.bool)
    loss_sum, n = masked_mlm_loss_sum(
        logits, targets, was_masked, torch.zeros(B)
    )
    assert int(n) == 0
    assert float(loss_sum) == 0.0 and torch.isfinite(loss_sum)


def test_clm_matches_independent_reference_and_ignores_pad():
    B, L = 2, 7
    torch.manual_seed(2)
    logits = torch.randn(B, L, FULL)
    # clean target: real tokens incl. a trailing stop; pad (MASK_TOKEN) at the tail of seq1
    target = torch.randint(0, 20, (B, L))
    target[1, -2:] = MASK_TOKEN  # padding -> must be ignored
    loss_sum, n = clm_loss_sum(logits, target)

    lg = aa_logits(logits)[:, :-1]
    tgt = target[:, 1:]
    keep = tgt != MASK_TOKEN
    ref = _ref_ce_sum(lg, tgt, keep)
    assert torch.allclose(loss_sum, ref, atol=1e-5)
    assert int(n) == int(keep.sum())
    # equals the negative of the inference path's per-sequence logp summed
    logp = -F.cross_entropy(
        lg.transpose(1, 2), tgt, ignore_index=MASK_TOKEN, reduction="none"
    ).sum()
    assert torch.allclose(loss_sum, -logp, atol=1e-5)


def _toy_batch(B=2, L=5):
    torch.manual_seed(7)
    xs_logits = torch.randn(B, L, FULL, requires_grad=True)
    mlm_logits = torch.randn(B, L, FULL, requires_grad=True)
    clm_logits = torch.randn(B, L, FULL, requires_grad=True)
    xs_wm = torch.zeros(B, L, dtype=torch.bool)
    xs_wm[:, 1] = True
    mlm_wm = torch.zeros(B, L, dtype=torch.bool)
    mlm_wm[:, 2] = True
    batch = {
        "xs_targets": torch.randint(0, 20, (B, L)),
        "xs_was_masked": xs_wm,
        "xs_seq_mask_rate": torch.full((B,), 0.2),
        "mlm_ys_targets": torch.randint(0, 20, (B, L)),
        "mlm_ys_was_masked": mlm_wm,
        "mlm_ys_seq_mask_rate": torch.full((B,), 0.2),
        "clm_ys_targets": torch.randint(0, 20, (B, L)),
    }
    return xs_logits, mlm_logits, clm_logits, batch


def test_total_loss_sums_per_term_means_and_backprops():
    xs, mlm, clm, batch = _toy_batch()
    loss, logs = total_loss(xs, mlm, clm, batch)
    assert torch.isfinite(loss)
    assert logs["L_total"] == (logs["L_mlm_enc"] + logs["L_mlm_dec"] + logs["L_clm_dec"])
    loss.backward()
    assert xs.grad is not None and clm.grad is not None and mlm.grad is not None
    assert torch.isfinite(xs.grad).all()


def test_ddp_normalization_invariant():
    """Two identical ranks (counts+sums doubled, world_size=2) give the same loss/grad
    as one rank (world_size=1) — validating the global-denominator × world_size coupling."""
    xs, mlm, clm, batch = _toy_batch()

    loss1, _ = total_loss(xs, mlm, clm, batch, world_size=1, all_reduce=None)
    loss1.backward()
    g1 = xs.grad.clone()

    xs.grad = mlm.grad = clm.grad = None
    loss2, _ = total_loss(
        xs, mlm, clm, batch, world_size=2, all_reduce=lambda t: t.mul_(2)
    )
    loss2.backward()
    g2 = xs.grad.clone()

    assert torch.allclose(loss1, loss2, atol=1e-6)
    assert torch.allclose(g1, g2, atol=1e-6)
