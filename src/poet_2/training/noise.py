"""Sequence- and structure-track noising for continued training (spec §8.2, §8.3).

Pure NumPy — no torch / model imports — so it unit-tests on any machine. Operates on a
single tokenized sequence whose token ids already include the ``$``/``*`` start/stop
wrappers produced by the tokenizer, plus its per-residue structure tracks ``plddt``
``(L,)`` and ``atomx`` ``(L, 3, 3)`` (``np.nan`` = unknown).

This module never touches ``atomb``: the downstream collator recomputes the 36 local
backbone distances from the (possibly NaN-masked / reversed) ``atomx`` via the existing
``poet_2.models.poet_2_helpers.atomb_from_atomx``, so NaN coords propagate to NaN atomb
and the model routes them to its missing/low-confidence attention-bias bucket.

Mask-pattern mixture (spec §8.2), applied to whichever track is being noised:
  * 50% independent per-residue (Bernoulli at the sampled rate)
  * 25% contiguous spans, span length ``L ~ Poisson(3) + 1``, placed until the rate is met
  * 25% ``N`` contiguous spans, ``N ~ Poisson(2.5)`` half the time / ``Poisson(13)`` the
    other half; per-span length is derived so total coverage ≈ the sampled rate.

The third mode's per-span length is not pinned by the spec; deriving it from the target
rate is a documented choice that keeps all three modes at ≈ the sampled rate, matching
the planned ``sample_mask_pattern(length, rate, rng)`` contract.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from poet_2.alphabet.sparse_uniref_cluster2 import Alphabet

_ALPHABET = Alphabet()
MASK_TOKEN: int = int(_ALPHABET.mask_token)  # 24 — also pad / "missing"
GAP_TOKEN: int = int(_ALPHABET.gap_token)  # 25
START_TOKEN: int = int(_ALPHABET.start_token)  # 26
STOP_TOKEN: int = int(_ALPHABET.stop_token)  # 27
CLS_TOKEN: int = int(_ALPHABET.cls_token)  # 28

#: Tokens that are never masked and never count as maskable residues.
SPECIAL_TOKENS: tuple[int, ...] = (GAP_TOKEN, START_TOKEN, STOP_TOKEN, CLS_TOKEN)

BoolArray = npt.NDArray[np.bool_]


def residue_mask(
    seq_tokens: npt.NDArray[np.integer],
    special_tokens: tuple[int, ...] = SPECIAL_TOKENS,
) -> BoolArray:
    """Boolean over positions that are real (maskable) residues, i.e. not ``$``/``*``/``-``/``|``."""
    seq_tokens = np.asarray(seq_tokens)
    return ~np.isin(seq_tokens, np.asarray(special_tokens))


def sample_mask_pattern(
    maskable: BoolArray, rate: float, rng: np.random.Generator
) -> BoolArray:
    """Sample a boolean mask over ``maskable`` positions covering ≈ ``rate`` of them.

    ``maskable`` is a boolean array over the full sequence length marking residue
    positions (``$``/``*`` and other specials are ``False`` and are never selected).
    Returns a boolean array of the same length.
    """
    maskable = np.asarray(maskable, dtype=bool)
    length = maskable.shape[0]
    out = np.zeros(length, dtype=bool)
    idx = np.flatnonzero(maskable)
    m = int(idx.shape[0])
    if m == 0 or rate <= 0.0:
        return out
    target = min(m, int(round(rate * m)))
    if target <= 0:
        return out

    u = rng.random()
    if u < 0.5:
        # independent per-residue Bernoulli(rate)
        out[idx[rng.random(m) < rate]] = True
        return out

    # span-based modes operate in compressed residue coordinates [0, m); because the
    # specials sit at the sequence ends, the maskable region is contiguous and a span in
    # compressed coords maps back to a contiguous residue span.
    comp = np.zeros(m, dtype=bool)
    if u < 0.75:
        # contiguous spans, length ~ Poisson(3) + 1, until the target coverage is met
        guard = 0
        guard_max = 10 * m + 100
        while int(comp.sum()) < target and guard < guard_max:
            span_len = int(rng.poisson(3)) + 1
            start = int(rng.integers(0, m))
            comp[start : start + span_len] = True
            guard += 1
    else:
        # N spans; N ~ Poisson(2.5) half the time / Poisson(13) otherwise. Per-span length
        # is derived so total coverage ≈ target.
        lam = 2.5 if rng.random() < 0.5 else 13.0
        n_spans = max(1, int(rng.poisson(lam)))
        span_len = max(1, int(round(target / n_spans)))
        for _ in range(n_spans):
            if int(comp.sum()) >= target:
                break
            start = int(rng.integers(0, m))
            comp[start : start + span_len] = True
    out[idx[comp]] = True
    return out


def sample_seq_mask(
    seq_tokens: npt.NDArray[np.integer],
    rng: np.random.Generator,
    max_rate: float = 0.30,
    special_tokens: tuple[int, ...] = SPECIAL_TOKENS,
    mask_token: int = MASK_TOKEN,
) -> tuple[npt.NDArray[np.integer], BoolArray, float]:
    """Noise the sequence track (spec §8.2).

    Draws ``rate ~ U(0, max_rate)``, builds a pattern, and replaces masked residues with
    ``mask_token`` (``X`` = 24); ``$``/``*`` (and other specials) are never masked.

    Returns ``(masked_tokens, was_masked, mask_rate)`` where ``was_masked`` flags the
    corrupted positions (the MLM loss is computed only there) and ``mask_rate`` is the
    realized fraction of *maskable residues* masked — the quantity the >30% loss-drop
    rule (spec §7) checks.
    """
    seq_tokens = np.asarray(seq_tokens)
    maskable = residue_mask(seq_tokens, special_tokens)
    rate = float(rng.uniform(0.0, max_rate))
    was_masked = sample_mask_pattern(maskable, rate, rng)
    masked = seq_tokens.copy()
    masked[was_masked] = mask_token
    n_maskable = int(maskable.sum())
    mask_rate = float(was_masked.sum()) / n_maskable if n_maskable > 0 else 0.0
    return masked, was_masked, mask_rate


def sample_structure_mask(
    seq_tokens: npt.NDArray[np.integer],
    plddt: npt.NDArray[np.floating],
    atomx: npt.NDArray[np.floating],
    rng: np.random.Generator,
    max_rate: float = 1.0,
    special_tokens: tuple[int, ...] = SPECIAL_TOKENS,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], BoolArray]:
    """Noise the structure tracks (pLDDT + N/Cα/C coords) at 0–``max_rate`` (spec §8.2).

    Masked residues have their ``plddt`` and ``atomx`` set to ``np.nan``; copies are
    returned. ``atomb`` is recomputed downstream from the masked ``atomx``.
    """
    plddt = np.array(plddt, dtype=np.float32, copy=True)
    atomx = np.array(atomx, dtype=np.float32, copy=True)
    maskable = residue_mask(seq_tokens, special_tokens)
    rate = float(rng.uniform(0.0, max_rate))
    was_masked = sample_mask_pattern(maskable, rate, rng)
    plddt[was_masked] = np.nan
    atomx[was_masked] = np.nan
    return plddt, atomx, was_masked


def maybe_reverse(
    seq_tokens: npt.NDArray[np.integer],
    plddt: npt.NDArray[np.floating] | None,
    atomx: npt.NDArray[np.floating] | None,
    rng: np.random.Generator,
    p: float = 0.5,
    start_token: int = START_TOKEN,
    stop_token: int = STOP_TOKEN,
) -> tuple[
    npt.NDArray[np.integer],
    npt.NDArray[np.floating] | None,
    npt.NDArray[np.floating] | None,
    bool,
]:
    """Tranception-style sequence reversal (spec §8.3) with probability ``p``.

    Reverses residue order *between* a leading ``$`` and a trailing ``*`` (keeping those
    in place), reversing ``plddt``/``atomx`` in lockstep so per-residue structure stays
    aligned. The Cα–Cα distance matrix is order-invariant; the local backbone ``atomb``
    is recomputed downstream from the reversed ``atomx``. Returns the (possibly reversed)
    arrays plus a bool indicating whether reversal happened.
    """
    seq_tokens = np.asarray(seq_tokens)
    if rng.random() >= p:
        return seq_tokens, plddt, atomx, False
    length = seq_tokens.shape[0]
    lo = 1 if length > 0 and int(seq_tokens[0]) == start_token else 0
    hi = length - 1 if length > 0 and int(seq_tokens[-1]) == stop_token else length
    sl = slice(lo, hi)
    seq_tokens = seq_tokens.copy()
    seq_tokens[sl] = seq_tokens[sl][::-1]
    if plddt is not None:
        plddt = plddt.copy()
        plddt[sl] = plddt[sl][::-1]
    if atomx is not None:
        atomx = atomx.copy()
        atomx[sl] = atomx[sl][::-1]
    return seq_tokens, plddt, atomx, True
