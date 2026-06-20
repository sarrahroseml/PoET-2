"""CPU unit tests for the §8.2/§8.3 noise schedule (poet_2.training.noise)."""

import numpy as np
import pytest

from poet_2.training import noise
from poet_2.training.noise import (
    MASK_TOKEN,
    START_TOKEN,
    STOP_TOKEN,
    maybe_reverse,
    residue_mask,
    sample_mask_pattern,
    sample_seq_mask,
    sample_structure_mask,
)


def _toy_seq(n_res: int = 30) -> np.ndarray:
    """[$  res...  *] with residues drawn from AA ids 0..19."""
    rng = np.random.default_rng(0)
    body = rng.integers(0, 20, size=n_res).astype(np.int64)
    return np.concatenate(([START_TOKEN], body, [STOP_TOKEN]))


def test_token_ids_match_alphabet():
    from poet_2.alphabet.sparse_uniref_cluster2 import Alphabet

    a = Alphabet()
    assert (MASK_TOKEN, noise.GAP_TOKEN, START_TOKEN, STOP_TOKEN, noise.CLS_TOKEN) == (
        a.mask_token,
        a.gap_token,
        a.start_token,
        a.stop_token,
        a.cls_token,
    )


def test_residue_mask_excludes_specials():
    seq = _toy_seq(10)
    rm = residue_mask(seq)
    assert not rm[0] and not rm[-1]  # $ and *
    assert rm[1:-1].all()  # interior residues maskable
    assert rm.sum() == 10


def test_specials_never_masked_seq():
    seq = _toy_seq(40)
    for s in range(200):
        rng = np.random.default_rng(s)
        masked, was_masked, _ = sample_seq_mask(seq, rng, max_rate=0.30)
        assert not was_masked[0] and not was_masked[-1]
        assert masked[0] == START_TOKEN and masked[-1] == STOP_TOKEN
        # masked positions became X; everything else is unchanged
        assert np.all(masked[was_masked] == MASK_TOKEN)
        assert np.array_equal(masked[~was_masked], seq[~was_masked])


def test_mask_rate_is_realized_fraction_of_residues():
    seq = _toy_seq(50)
    for s in range(50):
        rng = np.random.default_rng(s)
        _, was_masked, rate = sample_seq_mask(seq, rng, max_rate=0.30)
        assert rate == pytest.approx(was_masked.sum() / 50)
        assert rate <= 1.0


@pytest.mark.parametrize("rate", [0.1, 0.3, 0.6])
def test_pattern_coverage_matches_rate_on_average(rate):
    """Averaged over many seeds (and all three modes), coverage ≈ the requested rate."""
    m = 200
    maskable = np.ones(m, dtype=bool)
    fracs = []
    for s in range(400):
        rng = np.random.default_rng(s)
        pat = sample_mask_pattern(maskable, rate, rng)
        assert pat.shape == (m,)
        fracs.append(pat.mean())
    assert np.mean(fracs) == pytest.approx(rate, abs=0.05)


def test_pattern_all_three_modes_reachable_and_bounded():
    maskable = np.ones(60, dtype=bool)
    seen_zero_one_span = False
    for s in range(300):
        rng = np.random.default_rng(s)
        pat = sample_mask_pattern(maskable, 0.25, rng)
        assert 0 <= pat.sum() <= 60
        # crude contiguity signal: at least sometimes we get runs > 1
        runs = np.diff(np.flatnonzero(np.r_[0, pat, 0]))
        if pat.any() and runs.max() > 1:
            seen_zero_one_span = True
    assert seen_zero_one_span


def test_zero_rate_masks_nothing():
    maskable = np.ones(20, dtype=bool)
    pat = sample_mask_pattern(maskable, 0.0, np.random.default_rng(1))
    assert pat.sum() == 0


def test_structure_mask_nan_propagation():
    seq = _toy_seq(30)
    plddt = np.full(seq.shape[0], 90.0, dtype=np.float32)
    atomx = np.zeros((seq.shape[0], 3, 3), dtype=np.float32)
    rng = np.random.default_rng(3)
    p2, a2, was = sample_structure_mask(seq, plddt, atomx, rng, max_rate=1.0)
    # masked positions are NaN in both tracks; unmasked unchanged; inputs not mutated
    assert np.isnan(p2[was]).all()
    assert np.isnan(a2[was]).reshape(was.sum(), -1).all()
    assert np.array_equal(p2[~was], plddt[~was])
    assert np.array_equal(a2[~was], atomx[~was])
    assert not np.isnan(plddt).any()  # original untouched
    # specials ($/*) are never selected
    assert not was[0] and not was[-1]


def test_maybe_reverse_p0_is_identity():
    seq = _toy_seq(15)
    plddt = np.arange(seq.shape[0], dtype=np.float32)
    atomx = np.random.default_rng(0).random((seq.shape[0], 3, 3)).astype(np.float32)
    s2, p2, a2, did = maybe_reverse(seq, plddt, atomx, np.random.default_rng(0), p=0.0)
    assert did is False
    assert np.array_equal(s2, seq)


def test_maybe_reverse_keeps_terminals_and_is_involution():
    seq = _toy_seq(15)
    plddt = np.arange(seq.shape[0], dtype=np.float32)
    atomx = (
        np.arange(seq.shape[0] * 9, dtype=np.float32).reshape(seq.shape[0], 3, 3)
    )
    s2, p2, a2, did = maybe_reverse(seq, plddt, atomx, np.random.default_rng(0), p=1.0)
    assert did is True
    # terminals preserved, interior reversed in lockstep
    assert s2[0] == START_TOKEN and s2[-1] == STOP_TOKEN
    assert np.array_equal(s2[1:-1], seq[1:-1][::-1])
    assert np.array_equal(p2[1:-1], plddt[1:-1][::-1])
    assert np.array_equal(a2[1:-1], atomx[1:-1][::-1])
    # reversing twice restores the original
    s3, p3, a3, _ = maybe_reverse(s2, p2, a2, np.random.default_rng(0), p=1.0)
    assert np.array_equal(s3, seq)
    assert np.array_equal(p3, plddt)
    assert np.array_equal(a3, atomx)
