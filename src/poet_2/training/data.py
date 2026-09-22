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

  sample_target.npy      int64 (N,)    Pool id of each sample's held-out target sequence.
  sample_ctx_ids.npy     int64 (C,)    Flattened pool ids of each sample's context members.
  sample_ctx_offsets.npy int64 (N+1,)  Prefix offsets into sample_ctx_ids; sample ``n``'s
                                       context ids are ``sample_ctx_ids[off[n]:off[n+1]]``
                                       and must be **non-empty** (>= 1 member).

  meta.json (optional)  {"n_pool": P, "n_samples": N, ...}  — informational only.

What your prep owns (the "frozen selection"): choosing the N samples (e.g. n_epochs ×
|corpus|, weighted ∝ 1/|family| or however you like), picking each sample's context
(token-budget subsampling) and held-out target, pre-shuffling the sample order, and any
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

from poet_2.alphabet.sparse_uniref_cluster2 import Alphabet, S3DiAlphabet
from poet_2.alphabets import append_startstop
from poet_2.training.noise import sample_mask_pattern

_ALPHABET = Alphabet()
_S3DI_ALPHABET = S3DiAlphabet()
MASK_TOKEN: int = int(_ALPHABET.mask_token)  # 24, also the pad value
S3DI_MASK: int = int(_S3DI_ALPHABET.mask_token)
GAP_BYTE = b"-"

_ATOMB_TRIU = torch.triu_indices(9, 9, offset=1)
N_ATOMB = 36


def _atomb_from_atomx(atomx: np.ndarray) -> torch.Tensor:
    """Compute pairwise backbone distances from (L, 3, 3) coordinates.
    Mirrors poet_2_helpers.atomb_from_atomx without the heavy model import."""
    if atomx.shape[0] <= 2:
        return torch.full((atomx.shape[0], N_ATOMB), float("nan"), dtype=torch.half)
    atomx = atomx[:, :3, :]
    left, center, right = atomx[:-2], atomx[1:-1], atomx[2:]
    atoms = np.concatenate((left, center, right), axis=1)  # (L-2, 9, 3)
    atoms_t = torch.from_numpy(atoms)
    distances = torch.cdist(atoms_t, atoms_t)
    distances = distances[:, _ATOMB_TRIU[0], _ATOMB_TRIU[1]].half()
    return torch.nn.functional.pad(distances, (0, 0, 1, 1), value=float("nan"))


@dataclass
class CollatorConfig:
    """Train-time augmentation knobs (NOT prep/selection settings)."""

    seq_mask_max: float = 0.30  # context + target sequence-track masking rate ~ U(0, this)
    rate_cap: float = 0.30  # >this realized rate -> no MLM loss for that sequence (spec §7)
    reversal_p: float = 0.5  # Tranception-style reversal probability (spec §8.3)
    struct_dropout: float = 0.5  # per-sequence probability of dropping structure to NaN
    ifq_p: float = 0.0  # probability of IFQ-aware training: insert masked-X target + structure
                         # as first context member and enable ref-value blending in CLM decoder


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


def _wrap_plddt(plddt: np.ndarray) -> np.ndarray:
    """Wrap plddt with NaN sentinels at $/* positions."""
    return np.concatenate(([np.nan], plddt, [np.nan])).astype(np.float32)


def _wrap_atomx(atomx: np.ndarray) -> np.ndarray:
    """Wrap atomx with NaN sentinels at $/* positions."""
    L = atomx.shape[0]
    out = np.full((L + 2, 3, 3), np.nan, dtype=np.float32)
    out[1:-1] = atomx
    return out


def _wrap_s3di(n: int) -> np.ndarray:
    """Create s3di track filled with mask token (no real 3Di computation)."""
    return np.full(n + 2, S3DI_MASK, dtype=np.uint8)


def augment_and_pack(
    context_residues: Sequence[np.ndarray],
    target_residues: np.ndarray,
    rng: np.random.Generator,
    cfg: CollatorConfig = CollatorConfig(),
    context_plddts: Sequence[np.ndarray] | None = None,
    context_atomxs: Sequence[np.ndarray] | None = None,
    target_plddt: np.ndarray | None = None,
    target_atomx: np.ndarray | None = None,
) -> dict:
    """Apply fresh §8.3 reversal + §8.2 masking to a frozen selection, producing a sample.

    Inputs are ungapped residue-token arrays (no ``$``/``*``): a list of context members
    and one held-out target (used as both the masked MLM target and the clean CLM target).
    When structure arrays (plddt/atomx) are provided, they are carried through reversal
    and wrapped with NaN/$/* sentinels.
    Output is the sample dict consumed by :func:`collate_token_budget`.
    """
    has_struct = context_plddts is not None

    if rng.random() < cfg.reversal_p:
        context_residues = [r[::-1].copy() for r in context_residues]
        target_residues = target_residues[::-1].copy()
        if has_struct:
            context_plddts = [p[::-1].copy() for p in context_plddts]
            context_atomxs = [a[::-1].copy() for a in context_atomxs]
            target_plddt = target_plddt[::-1].copy()
            target_atomx = target_atomx[::-1].copy()

    def _maybe_drop_struct(plddt: np.ndarray, atomx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-sequence structure dropout: NaN-out with probability struct_dropout."""
        if rng.random() < cfg.struct_dropout:
            return np.full_like(plddt, np.nan), np.full_like(atomx, np.nan)
        return plddt, atomx

    ctx_inputs, ctx_targets, ctx_was = [], [], []
    ctx_plddts_wrapped, ctx_atomxs_wrapped, ctx_s3dis_wrapped = [], [], []
    for idx, r in enumerate(context_residues):
        inp, tgt, was, rate = _noise_and_wrap(r, rng, cfg)
        if rate > cfg.rate_cap:
            was = np.zeros_like(was)
        ctx_inputs.append(inp)
        ctx_targets.append(tgt)
        ctx_was.append(was)
        if has_struct:
            p, a = _maybe_drop_struct(context_plddts[idx], context_atomxs[idx])
            ctx_plddts_wrapped.append(_wrap_plddt(p))
            ctx_atomxs_wrapped.append(_wrap_atomx(a))
            ctx_s3dis_wrapped.append(_wrap_s3di(len(r)))

    mlm_inp, mlm_tgt, mlm_was, mlm_rate = _noise_and_wrap(target_residues, rng, cfg)
    clm_tok = _wrap(target_residues)

    ifq_active = False
    if has_struct and cfg.ifq_p > 0 and rng.random() < cfg.ifq_p:
        target_has_struct = not np.all(np.isnan(target_plddt))
        if target_has_struct:
            ifq_active = True
            ifq_tokens = np.full_like(target_residues, MASK_TOKEN)
            ifq_inp = _wrap(ifq_tokens)
            ifq_tgt = _wrap(ifq_tokens)
            ifq_was = np.zeros(len(ifq_inp), dtype=bool)
            ctx_inputs.insert(0, ifq_inp)
            ctx_targets.insert(0, ifq_tgt)
            ctx_was.insert(0, ifq_was)
            ctx_plddts_wrapped.insert(0, _wrap_plddt(target_plddt.copy()))
            ctx_atomxs_wrapped.insert(0, _wrap_atomx(target_atomx.copy()))
            ctx_s3dis_wrapped.insert(0, _wrap_s3di(len(target_residues)))

    result = {
        "ctx_inputs": ctx_inputs,
        "ctx_targets": ctx_targets,
        "ctx_was": ctx_was,
        "mlm_input": mlm_inp,
        "mlm_target": mlm_tgt,
        "mlm_was": mlm_was,
        "mlm_rate": mlm_rate,
        "clm_input": clm_tok,
        "clm_target": clm_tok,
        "ifq_active": ifq_active,
    }

    if has_struct:
        tgt_p, tgt_a = _maybe_drop_struct(target_plddt, target_atomx)
        tgt_plddt_w = _wrap_plddt(tgt_p)
        tgt_atomx_w = _wrap_atomx(tgt_a)
        tgt_s3di_w = _wrap_s3di(len(target_residues))
        result.update({
            "ctx_plddts": ctx_plddts_wrapped,
            "ctx_atomxs": ctx_atomxs_wrapped,
            "ctx_s3dis": ctx_s3dis_wrapped,
            "mlm_plddt": tgt_plddt_w,
            "mlm_atomx": tgt_atomx_w,
            "mlm_s3di": tgt_s3di_w,
            "clm_plddt": tgt_plddt_w,
            "clm_atomx": tgt_atomx_w,
            "clm_s3di": tgt_s3di_w,
        })

    return result


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


def _pad_float(rows: list[np.ndarray], value: float = float("nan")) -> torch.Tensor:
    return pad_sequence(
        [torch.as_tensor(r, dtype=torch.float32) for r in rows],
        batch_first=True,
        padding_value=value,
    )


def _pad_atomx(rows: list[np.ndarray]) -> torch.Tensor:
    """Pad (L,3,3) atomx arrays to (B, max_L, 3, 3) with NaN."""
    max_len = max(r.shape[0] for r in rows)
    B = len(rows)
    out = torch.full((B, max_len, 3, 3), float("nan"), dtype=torch.float32)
    for i, r in enumerate(rows):
        out[i, : r.shape[0]] = torch.from_numpy(r)
    return out


def _pad_atomb(rows: list[torch.Tensor]) -> torch.Tensor:
    """Pad (L, 36) atomb tensors to (B, max_L, 36) with NaN."""
    return pad_sequence(rows, batch_first=True, padding_value=float("nan"))


def _pad_s3di(rows: list[np.ndarray]) -> torch.Tensor:
    return pad_sequence(
        [torch.as_tensor(r, dtype=torch.long) for r in rows],
        batch_first=True,
        padding_value=S3DI_MASK,
    )


def collate_token_budget(samples: Sequence[dict]) -> dict[str, torch.Tensor]:
    """Pad a list of samples into a ``training_forward`` + ``total_loss`` batch dict.

    Encoder rows pack each family's context (``cat`` over its sequences); decoder rows are
    single sequences. Sequence pads use ``mask_token`` (== ignore_index), segment-size pads
    use 0, ``was_masked`` pads use ``False``.
    """
    atomb_from_atomx = _atomb_from_atomx

    B = len(samples)
    has_struct = "ctx_plddts" in samples[0]

    xs_rows, xs_tgt_rows, xs_was_rows, xs_seg_rows = [], [], [], []
    xs_plddt_rows, xs_atomx_rows, xs_s3di_rows = [], [], []
    for s in samples:
        xs_rows.append(np.concatenate(s["ctx_inputs"]))
        xs_tgt_rows.append(np.concatenate(s["ctx_targets"]))
        xs_was_rows.append(np.concatenate(s["ctx_was"]))
        xs_seg_rows.append(np.array([t.shape[0] for t in s["ctx_inputs"]], dtype=np.int64))
        if has_struct:
            xs_plddt_rows.append(np.concatenate(s["ctx_plddts"]))
            xs_atomx_rows.append(np.concatenate(s["ctx_atomxs"]))
            xs_s3di_rows.append(np.concatenate(s["ctx_s3dis"]))

    batch = {
        "xs": _pad_int(xs_rows, MASK_TOKEN),
        "xs_segment_sizes": _pad_int(xs_seg_rows, 0),
        "xs_targets": _pad_int(xs_tgt_rows, MASK_TOKEN),
        "xs_was_masked": _pad_bool(xs_was_rows),
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

    ifq_flags = [s.get("ifq_active", False) for s in samples]
    if any(ifq_flags):
        batch["ifq_active"] = torch.tensor(ifq_flags, dtype=torch.bool)

    if has_struct:
        batch["xs_plddts"] = _pad_float(xs_plddt_rows)
        batch["xs_atomxs"] = _pad_atomx(xs_atomx_rows)
        batch["xs_atombs"] = _pad_atomb([atomb_from_atomx(a) for a in xs_atomx_rows])
        batch["xs_s3dis"] = _pad_s3di(xs_s3di_rows)
        for prefix, plddt_key, atomx_key, s3di_key in [
            ("mlm_ys", "mlm_plddt", "mlm_atomx", "mlm_s3di"),
            ("clm_ys", "clm_plddt", "clm_atomx", "clm_s3di"),
        ]:
            plddt_rows = [s[plddt_key] for s in samples]
            atomx_rows = [s[atomx_key] for s in samples]
            s3di_rows = [s[s3di_key] for s in samples]
            batch[f"{prefix}_plddts"] = _pad_float(plddt_rows)
            batch[f"{prefix}_atomxs"] = _pad_atomx(atomx_rows)
            batch[f"{prefix}_atombs"] = _pad_atomb([atomb_from_atomx(a) for a in atomx_rows])
            batch[f"{prefix}_s3dis"] = _pad_s3di(s3di_rows)

    return batch


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
_POOL_PLDDT = "pool_plddt.npy"
_POOL_ATOMX = "pool_atomx.npy"
_SAMPLE_TARGET = "sample_target.npy"
_SAMPLE_CTX_IDS = "sample_ctx_ids.npy"
_SAMPLE_CTX_OFF = "sample_ctx_offsets.npy"
_META = "meta.json"


class PoET2Dataset(Dataset):
    """Map-style reader over the input format documented above (produced by your prep).

    ``__getitem__`` gathers a sample's sequences from the mmap'd pool and applies **fresh**
    masking/reversal via :func:`augment_and_pack`, seeded by ``(seed, sample_index)`` so a
    run is reproducible/resumable. Samples are rank-sharded. ``cfg`` is the train-time
    masking config (independent of how the data was prepped).

    When ``pool_plddt.npy`` and ``pool_atomx.npy`` exist in ``data_dir``, structure tracks
    are loaded and passed through to ``augment_and_pack``.
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
        self.target_ids = np.load(os.path.join(data_dir, _SAMPLE_TARGET))
        self.ctx_ids = np.load(os.path.join(data_dir, _SAMPLE_CTX_IDS), mmap_mode="r")
        self.ctx_offsets = np.load(os.path.join(data_dir, _SAMPLE_CTX_OFF))
        self.indices = np.arange(self.target_ids.shape[0])[rank::world_size]
        self._n_tokens = self._compute_token_footprints()

        plddt_path = os.path.join(data_dir, _POOL_PLDDT)
        atomx_path = os.path.join(data_dir, _POOL_ATOMX)
        if os.path.exists(plddt_path) and os.path.exists(atomx_path):
            self.pool_plddt = np.load(plddt_path, mmap_mode="r")
            self.pool_atomx = np.load(atomx_path, mmap_mode="r")
            self.has_struct = True
        else:
            self.pool_plddt = None
            self.pool_atomx = None
            self.has_struct = False

    def _compute_token_footprints(self) -> np.ndarray:
        """Derive each sample's token footprint from pool lengths (no masking needed)."""
        n = self.target_ids.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        pool_len = np.diff(self.pool_offsets)
        ctx_tok = pool_len[np.asarray(self.ctx_ids)] + 2  # +$/* per context member
        ctx_sum = np.add.reduceat(ctx_tok, self.ctx_offsets[:-1])
        target_tok = 2 * (pool_len[self.target_ids] + 2)  # mlm + clm targets
        return (ctx_sum + target_tok).astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _gather(self, pid: int) -> np.ndarray:
        lo, hi = int(self.pool_offsets[pid]), int(self.pool_offsets[pid + 1])
        return np.asarray(self.pool_tokens[lo:hi]).copy()

    def _gather_plddt(self, pid: int) -> np.ndarray:
        lo, hi = int(self.pool_offsets[pid]), int(self.pool_offsets[pid + 1])
        return np.asarray(self.pool_plddt[lo:hi]).copy()

    def _gather_atomx(self, pid: int) -> np.ndarray:
        lo, hi = int(self.pool_offsets[pid]), int(self.pool_offsets[pid + 1])
        return np.asarray(self.pool_atomx[lo:hi]).copy()

    def __getitem__(self, i: int) -> dict:
        sidx = int(self.indices[i])
        lo, hi = int(self.ctx_offsets[sidx]), int(self.ctx_offsets[sidx + 1])
        ctx_pids = np.asarray(self.ctx_ids[lo:hi])
        ctx_res = [self._gather(int(c)) for c in ctx_pids]
        tgt_pid = int(self.target_ids[sidx])
        tgt_res = self._gather(tgt_pid)
        rng = np.random.default_rng([self.seed, sidx])

        struct_kwargs = {}
        if self.has_struct:
            struct_kwargs["context_plddts"] = [self._gather_plddt(int(c)) for c in ctx_pids]
            struct_kwargs["context_atomxs"] = [self._gather_atomx(int(c)) for c in ctx_pids]
            struct_kwargs["target_plddt"] = self._gather_plddt(tgt_pid)
            struct_kwargs["target_atomx"] = self._gather_atomx(tgt_pid)

        return augment_and_pack(ctx_res, tgt_res, rng, self.cfg, **struct_kwargs)

    def token_counts(self) -> np.ndarray:
        """Frozen per-sample token footprints for this rank's shard (for budget batching)."""
        return self._n_tokens[self.indices]

    def iter_samples(self) -> Iterator[dict]:
        for i in range(len(self)):
            yield self[i]
