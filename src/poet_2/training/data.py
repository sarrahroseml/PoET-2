"""Training-time data **consumer** + the input-format contract.

This module does **not** create, sample, or write any dataset — that's your separate
data-prep. It only (1) stipulates the on-disk format the trainer expects, (2) reads it,
and (3) applies the *training-time* transforms: fresh §8.2 masking + §8.3 reversal, then
``$``/``*`` wrapping and token-budget collation. Selection (which homologs form each
sample's context, and which is the held-out target) is **frozen by your prep**; masking is
re-sampled every visit (dynamic masking).

================================ Expected input format ================================

A directory of NumPy arrays (mmap-friendly), produced by your data-prep:

  pool_tokens.npy     uint8 (T,)    Flat concatenation of the UNIQUE homolog sequences,
                                    tokenized with `encode_residues` (Alphabet token ids,
                                    ungapped, uppercase, **without** ``$``/``*`` — the
                                    trainer adds start/stop). Dedup is recommended.
  pool_offsets.npy    int64 (P+1,)  Prefix-sum offsets; pooled sequence ``p`` is
                                    ``pool_tokens[off[p]:off[p+1]]``.

  recipe_target.npy      int64 (N,)    Pool id of each sample's held-out target sequence.
  recipe_ctx_ids.npy     int64 (C,)    Flattened pool ids of each sample's context members.
  recipe_ctx_offsets.npy int64 (N+1,)  Prefix offsets into recipe_ctx_ids; sample ``n``'s
                                       context ids are ``recipe_ctx_ids[off[n]:off[n+1]]``
                                       and must be **non-empty** (>= 1 member).

  meta.json (optional)  {"n_pool": P, "n_recipes": N, ...}  — informational only.

What your prep owns (the "frozen selection"): choosing the N samples (e.g. n_epochs ×
|corpus|, weighted ∝ 1/|family| or however you like), picking each sample's context
(token-budget subsampling) and held-out target, pre-shuffling the recipe order, and any
sharding policy. Masking config is a **train-time** argument here, never read from prep.

A worked example of producing this format lives in ``tests/test_collator.py``
(``_write_fixture``).

================================ Sample-dict schema ===================================

If you'd rather bypass the file format, produce sample dicts directly (see
:func:`augment_and_pack`) and feed them to :func:`collate_token_budget`. Each sample dict
holds, for the encoder context and the two decoder targets, the input tokens, clean target
tokens, and the masked-position indicators the loss needs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from poet_2.alphabet.sparse_uniref_cluster2 import Alphabet
from poet_2.alphabets import append_startstop
from poet_2.training.noise import sample_mask_pattern

_ALPHABET = Alphabet()
MASK_TOKEN: int = int(_ALPHABET.mask_token)  # 24, also the pad value
GAP_BYTE = b"-"


@dataclass
class CollatorConfig:
    """Train-time augmentation knobs (NOT prep/selection settings)."""

    seq_mask_max: float = 0.30  # context + target sequence-track masking rate ~ U(0, this)
    rate_cap: float = 0.30  # >this realized rate -> no MLM loss for that sequence (spec §7)
    reversal_p: float = 0.5  # Tranception-style reversal probability (spec §8.3)


def encode_residues(seq: bytes) -> np.ndarray:
    """Canonical tokenization for the pool: bytes -> ungapped, uppercased residue token
    ids (Alphabet ids, **no** ``$``/``*``). Use this in your prep so the pool matches what
    the trainer expects."""
    seq = seq.replace(GAP_BYTE, b"").upper()
    return _ALPHABET.encode(seq)


# ---------------------------------------------------- training-time augmentation


def _wrap(residues: np.ndarray) -> np.ndarray:
    return append_startstop(residues, _ALPHABET)


def _noise_and_wrap(
    residues: np.ndarray, rng: np.random.Generator, cfg: CollatorConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Sequence-track noise on one sequence's residues, then wrap with ``$``/``*``.

    Returns ``(input_tokens, target_tokens, was_masked, mask_rate)`` each of length
    ``len(residues) + 2``; ``was_masked`` is ``False`` at the ``$``/``*`` positions.
    """
    n = int(residues.shape[0])
    maskable = np.ones(n, dtype=bool)
    rate = float(rng.uniform(0.0, cfg.seq_mask_max))
    was = sample_mask_pattern(maskable, rate, rng)
    noised = residues.copy()
    noised[was] = MASK_TOKEN
    input_tokens = _wrap(noised)
    target_tokens = _wrap(residues)
    was_masked = np.concatenate(([False], was, [False]))
    mask_rate = float(was.sum()) / n if n > 0 else 0.0
    return input_tokens, target_tokens, was_masked, mask_rate


def augment_and_pack(
    context_residues: Sequence[np.ndarray],
    target_residues: np.ndarray,
    rng: np.random.Generator,
    cfg: CollatorConfig = CollatorConfig(),
) -> dict:
    """Apply fresh §8.3 reversal + §8.2 masking to a frozen selection, producing a sample.

    Inputs are ungapped residue-token arrays (no ``$``/``*``): a list of context members
    and one held-out target (used as both the masked MLM target and the clean CLM target).
    Output is the sample dict consumed by :func:`collate_token_budget`.
    """
    if rng.random() < cfg.reversal_p:
        context_residues = [r[::-1].copy() for r in context_residues]
        target_residues = target_residues[::-1].copy()

    ctx_inputs, ctx_targets, ctx_was = [], [], []
    for r in context_residues:
        inp, tgt, was, rate = _noise_and_wrap(r, rng, cfg)
        if rate > cfg.rate_cap:  # per-segment >30% rule applied here for the encoder
            was = np.zeros_like(was)
        ctx_inputs.append(inp)
        ctx_targets.append(tgt)
        ctx_was.append(was)

    mlm_inp, mlm_tgt, mlm_was, mlm_rate = _noise_and_wrap(target_residues, rng, cfg)
    clm_tok = _wrap(target_residues)  # CLM target is clean / un-noised

    return {
        "ctx_inputs": ctx_inputs,
        "ctx_targets": ctx_targets,
        "ctx_was": ctx_was,
        "mlm_input": mlm_inp,
        "mlm_target": mlm_tgt,
        "mlm_was": mlm_was,
        "mlm_rate": mlm_rate,
        "clm_input": clm_tok,
        "clm_target": clm_tok,
        # TODO(structure): attach plddt/atomx/atomb tracks (NaN-masked) when enabled.
    }


def sample_token_count(sample: dict) -> int:
    """Encoder + decoder token footprint of a sample, for token-budget batching."""
    ctx = sum(int(t.shape[0]) for t in sample["ctx_inputs"])
    return ctx + int(sample["mlm_input"].shape[0]) + int(sample["clm_input"].shape[0])


# -------------------------------------------------------------------- collation


def _pad_int(rows: list[np.ndarray], value: int) -> torch.Tensor:
    return pad_sequence(
        [torch.as_tensor(r, dtype=torch.long) for r in rows],
        batch_first=True,
        padding_value=value,
    )


def _pad_bool(rows: list[np.ndarray]) -> torch.Tensor:
    return pad_sequence(
        [torch.as_tensor(r, dtype=torch.bool) for r in rows],
        batch_first=True,
        padding_value=False,
    )


def collate_token_budget(samples: Sequence[dict]) -> dict[str, torch.Tensor]:
    """Pad a list of samples into a ``training_forward`` + ``total_loss`` batch dict.

    Encoder rows pack each family's context (``cat`` over its sequences); decoder rows are
    single sequences. Sequence pads use ``mask_token`` (== ignore_index), segment-size pads
    use 0, ``was_masked`` pads use ``False``.
    """
    B = len(samples)

    xs_rows, xs_tgt_rows, xs_was_rows, xs_seg_rows = [], [], [], []
    for s in samples:
        xs_rows.append(np.concatenate(s["ctx_inputs"]))
        xs_tgt_rows.append(np.concatenate(s["ctx_targets"]))
        xs_was_rows.append(np.concatenate(s["ctx_was"]))
        xs_seg_rows.append(np.array([t.shape[0] for t in s["ctx_inputs"]], dtype=np.int64))

    return {
        "xs": _pad_int(xs_rows, MASK_TOKEN),
        "xs_segment_sizes": _pad_int(xs_seg_rows, 0),
        "xs_targets": _pad_int(xs_tgt_rows, MASK_TOKEN),
        "xs_was_masked": _pad_bool(xs_was_rows),
        # encoder >30% rule already applied per segment in augment_and_pack -> pass zeros
        "xs_seq_mask_rate": torch.zeros(B, dtype=torch.float32),
        "mlm_ys": _pad_int([s["mlm_input"] for s in samples], MASK_TOKEN),
        "mlm_ys_segment_sizes": _pad_int(
            [np.array([s["mlm_input"].shape[0]], dtype=np.int64) for s in samples], 0
        ),
        "mlm_ys_targets": _pad_int([s["mlm_target"] for s in samples], MASK_TOKEN),
        "mlm_ys_was_masked": _pad_bool([s["mlm_was"] for s in samples]),
        "mlm_ys_seq_mask_rate": torch.tensor(
            [s["mlm_rate"] for s in samples], dtype=torch.float32
        ),
        "clm_ys": _pad_int([s["clm_input"] for s in samples], MASK_TOKEN),
        "clm_ys_segment_sizes": _pad_int(
            [np.array([s["clm_input"].shape[0]], dtype=np.int64) for s in samples], 0
        ),
        "clm_ys_targets": _pad_int([s["clm_target"] for s in samples], MASK_TOKEN),
    }


def batch_by_token_budget(
    samples: Iterable[dict], tokens_per_batch: int
) -> Iterator[list[dict]]:
    """Group a stream of samples into variable-size batches under a token budget."""
    group: list[dict] = []
    used = 0
    for s in samples:
        cost = sample_token_count(s)
        if group and used + cost > tokens_per_batch:
            yield group
            group, used = [], 0
        group.append(s)
        used += cost
    if group:
        yield group


# -------------------------------------------------------------- format reader

_POOL_TOKENS = "pool_tokens.npy"
_POOL_OFFSETS = "pool_offsets.npy"
_REC_TARGET = "recipe_target.npy"
_REC_CTX_IDS = "recipe_ctx_ids.npy"
_REC_CTX_OFF = "recipe_ctx_offsets.npy"
_META = "meta.json"


class MaterializedDataset(Dataset):
    """Map-style reader over the input format documented above (produced by your prep).

    ``__getitem__`` gathers a recipe's sequences from the mmap'd pool and applies **fresh**
    masking/reversal via :func:`augment_and_pack`, seeded by ``(seed, recipe_index)`` so a
    run is reproducible/resumable. Recipes are rank-sharded. ``cfg`` is the train-time
    masking config (independent of how the data was prepped).
    """

    def __init__(
        self,
        data_dir: str,
        cfg: CollatorConfig = CollatorConfig(),
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.dir = data_dir
        self.cfg = cfg
        self.seed = seed
        meta_path = os.path.join(data_dir, _META)
        self.meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        self.pool_tokens = np.load(os.path.join(data_dir, _POOL_TOKENS), mmap_mode="r")
        self.pool_offsets = np.load(os.path.join(data_dir, _POOL_OFFSETS))
        self.r_target = np.load(os.path.join(data_dir, _REC_TARGET))
        self.r_ctx_ids = np.load(os.path.join(data_dir, _REC_CTX_IDS), mmap_mode="r")
        self.r_ctx_off = np.load(os.path.join(data_dir, _REC_CTX_OFF))
        self.indices = np.arange(self.r_target.shape[0])[rank::world_size]
        self._n_tokens = self._compute_token_footprints()

    def _compute_token_footprints(self) -> np.ndarray:
        """Derive each recipe's token footprint from pool lengths (no masking needed)."""
        n = self.r_target.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        pool_len = np.diff(self.pool_offsets)
        ctx_tok = pool_len[np.asarray(self.r_ctx_ids)] + 2  # +$/* per context member
        ctx_sum = np.add.reduceat(ctx_tok, self.r_ctx_off[:-1])
        target_tok = 2 * (pool_len[self.r_target] + 2)  # mlm + clm targets
        return (ctx_sum + target_tok).astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _gather(self, pid: int) -> np.ndarray:
        lo, hi = int(self.pool_offsets[pid]), int(self.pool_offsets[pid + 1])
        return np.asarray(self.pool_tokens[lo:hi]).copy()

    def __getitem__(self, i: int) -> dict:
        ridx = int(self.indices[i])
        lo, hi = int(self.r_ctx_off[ridx]), int(self.r_ctx_off[ridx + 1])
        ctx_res = [self._gather(int(c)) for c in np.asarray(self.r_ctx_ids[lo:hi])]
        tgt_res = self._gather(int(self.r_target[ridx]))
        rng = np.random.default_rng([self.seed, ridx])  # fresh-but-reproducible masking
        return augment_and_pack(ctx_res, tgt_res, rng, self.cfg)

    def token_counts(self) -> np.ndarray:
        """Frozen per-sample token footprints for this rank's shard (for budget batching)."""
        return self._n_tokens[self.indices]

    def iter_samples(self) -> Iterator[dict]:
        for i in range(len(self)):
            yield self[i]
