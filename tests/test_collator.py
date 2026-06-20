"""CPU unit tests for the training-time data consumer (poet_2.training.data).

The module creates no dataset; it reads a documented format and applies fresh masking.
``_write_fixture`` below is a minimal reference implementation of producing that format
(what your separate data-prep is responsible for).
"""

import json

import numpy as np
import torch

from poet_2.training import losses
from poet_2.training.data import (
    MASK_TOKEN,
    CollatorConfig,
    MaterializedDataset,
    _noise_and_wrap,
    augment_and_pack,
    batch_by_token_budget,
    collate_token_budget,
    encode_residues,
    sample_token_count,
)
from poet_2.training.noise import START_TOKEN, STOP_TOKEN

FAMILIES = [
    [b"MKTAYIAKQRQISFVK", b"MRTAYIAKQSQISFVR", b"MKTAYIAKQRQISFVL", b"MKSAYIAKQRQILFVK"],
    [b"GGGAAACCCWWWYYY", b"GGGAAACCDWWWYYF", b"GGGAAGCCCWWWYYY"],
    [b"PLLTTIGGPKELTAF", b"PLLTTVGGPKELSAF"],
]


def _raw(family):
    """family bytes -> (context_residues, target_residues): target = member 0."""
    res = [encode_residues(s) for s in family]
    return res[1:] or [res[0]], res[0]


# ----------------------------------------------------------- augmentation


def test_encode_strips_gaps_and_uppercases():
    assert np.array_equal(encode_residues(b"mk-t-ay"), encode_residues(b"MKTAY"))


def test_noise_and_wrap_invariants():
    res = encode_residues(b"MKTAYIAKQRQISFVK")
    inp, tgt, was, rate = _noise_and_wrap(res, np.random.default_rng(0), CollatorConfig())
    L = res.shape[0]
    assert inp.shape[0] == tgt.shape[0] == was.shape[0] == L + 2
    assert inp[0] == START_TOKEN and inp[-1] == STOP_TOKEN
    assert not was[0] and not was[-1]
    assert np.array_equal(tgt[1:-1], res)
    masked = was[1:-1]
    assert np.all(inp[1:-1][masked] == MASK_TOKEN)
    assert np.array_equal(inp[1:-1][~masked], res[~masked])
    assert np.isclose(rate, masked.sum() / L)


def test_augment_and_pack_clean_clm_and_mask_alignment():
    ctx, tgt = _raw(FAMILIES[0])
    s = augment_and_pack(ctx, tgt, np.random.default_rng(1), CollatorConfig(reversal_p=0.0))
    assert len(s["ctx_inputs"]) == len(ctx)
    assert np.array_equal(s["clm_input"], s["clm_target"])  # CLM is clean
    assert not np.any(s["clm_input"] == MASK_TOKEN)
    assert np.array_equal(s["mlm_target"][~s["mlm_was"]], s["mlm_input"][~s["mlm_was"]])
    assert np.all(s["mlm_input"][s["mlm_was"]] == MASK_TOKEN)


def test_augment_and_pack_reversal():
    res = encode_residues(b"MKTAYIAKQR")
    s = augment_and_pack([res], res, np.random.default_rng(3), CollatorConfig(reversal_p=1.0))
    assert np.array_equal(s["clm_target"][1:-1], res[::-1])


def test_per_segment_rate_cap_holds():
    ctx, tgt = _raw(FAMILIES[0])
    cfg = CollatorConfig(seq_mask_max=0.30, rate_cap=0.30)
    for seed in range(200):
        s = augment_and_pack(ctx, tgt, np.random.default_rng(seed), cfg)
        for was in s["ctx_was"]:
            frac = was[1:-1].sum() / max(1, was.shape[0] - 2)
            assert frac <= cfg.rate_cap + 1e-9


# ----------------------------------------------------------- collation


def _samples(seed=7):
    rng = np.random.default_rng(seed)
    return [augment_and_pack(*_raw(f), rng, CollatorConfig(reversal_p=0.0)) for f in FAMILIES]


def test_collate_shapes_and_pad_values():
    b = collate_token_budget(_samples())
    B = len(FAMILIES)
    assert b["xs"].shape[0] == B and b["xs"].dtype == torch.long
    assert b["xs_targets"].shape == b["xs"].shape == b["xs_was_masked"].shape
    assert torch.equal(b["xs_seq_mask_rate"], torch.zeros(B))
    m = b["xs_was_masked"]
    assert torch.all(b["xs"][m] == MASK_TOKEN)
    assert torch.all(b["xs_targets"][m] != MASK_TOKEN)
    real = b["xs"] != MASK_TOKEN
    assert torch.all(b["xs"][real] == b["xs_targets"][real])


def test_batch_by_token_budget_respects_budget():
    rng = np.random.default_rng(0)
    samples = [augment_and_pack(*_raw(FAMILIES[i % len(FAMILIES)]), rng, CollatorConfig())
               for i in range(20)]
    groups = list(batch_by_token_budget(samples, tokens_per_batch=200))
    assert sum(len(g) for g in groups) == len(samples)
    for g in groups:
        if len(g) > 1:
            assert sum(sample_token_count(s) for s in g) <= 200 + max(
                sample_token_count(s) for s in g
            )


def test_collated_loss_tensors_feed_total_loss():
    b = collate_token_budget(_samples(seed=2))
    V = 2 * losses.AA_VOCAB
    xs = torch.randn(*b["xs"].shape, V, requires_grad=True)
    mlm = torch.randn(*b["mlm_ys"].shape, V, requires_grad=True)
    clm = torch.randn(*b["clm_ys"].shape, V, requires_grad=True)
    loss, logs = losses.total_loss(xs, mlm, clm, b)
    assert torch.isfinite(loss)
    assert logs["n_L_clm_dec"] > 0
    loss.backward()
    assert xs.grad is not None and torch.isfinite(xs.grad).all()


# ----------------------------------------------- input-format reader (MaterializedDataset)


def _write_fixture(d, families) -> str:
    """Reference: produce the expected input format. (Your real data-prep does this, with
    its own family weighting / context subsampling / shuffling.)"""
    pool_index, pool = {}, []
    rec_ctx_ids, rec_ctx_off, rec_target = [], [0], []
    for fam in families:
        ids = []
        for s in fam:
            r = encode_residues(s)
            key = r.tobytes()
            if key not in pool_index:
                pool_index[key] = len(pool)
                pool.append(r)
            ids.append(pool_index[key])
        target, ctx = ids[0], (ids[1:] or [ids[0]])  # held-out target = member 0
        rec_target.append(target)
        rec_ctx_ids.extend(ctx)
        rec_ctx_off.append(len(rec_ctx_ids))
    lengths = np.array([len(p) for p in pool], dtype=np.int64)
    np.save(f"{d}/pool_tokens.npy", np.concatenate(pool).astype(np.uint8))
    np.save(f"{d}/pool_offsets.npy", np.concatenate(([0], np.cumsum(lengths))).astype(np.int64))
    np.save(f"{d}/recipe_target.npy", np.array(rec_target, dtype=np.int64))
    np.save(f"{d}/recipe_ctx_ids.npy", np.array(rec_ctx_ids, dtype=np.int64))
    np.save(f"{d}/recipe_ctx_offsets.npy", np.array(rec_ctx_off, dtype=np.int64))
    json.dump({"n_pool": len(pool), "n_recipes": len(rec_target)}, open(f"{d}/meta.json", "w"))
    return str(d)


def _unique_keys(families):
    return {encode_residues(s).tobytes() for fam in families for s in fam}


def test_reader_len_and_meta(tmp_path):
    ds = MaterializedDataset(_write_fixture(tmp_path, FAMILIES))
    assert len(ds) == len(FAMILIES)
    assert ds.meta["n_pool"] == len(_unique_keys(FAMILIES))


def test_reader_getitem_token_count_and_clean_clm(tmp_path):
    ds = MaterializedDataset(_write_fixture(tmp_path, FAMILIES))
    for i in range(len(ds)):
        s = ds[i]
        assert len(s["ctx_inputs"]) >= 1
        assert np.array_equal(s["clm_input"], s["clm_target"])
        # derived footprint matches the realized sample (masking/reversal preserve length)
        assert sample_token_count(s) == int(ds.token_counts()[i])


def test_reader_pool_roundtrip(tmp_path):
    ds = MaterializedDataset(_write_fixture(tmp_path, FAMILIES))
    keys = _unique_keys(FAMILIES)
    for pid in range(ds.pool_offsets.shape[0] - 1):
        assert ds._gather(pid).astype(np.uint8).tobytes() in keys


def test_reader_sharding_partitions_recipes(tmp_path):
    d = _write_fixture(tmp_path, FAMILIES)
    r0 = MaterializedDataset(d, rank=0, world_size=2).indices
    r1 = MaterializedDataset(d, rank=1, world_size=2).indices
    assert set(r0.tolist()) | set(r1.tolist()) == set(range(len(FAMILIES)))
    assert set(r0.tolist()) & set(r1.tolist()) == set()


def test_reader_masking_is_fresh_but_reproducible(tmp_path):
    d = _write_fixture(tmp_path, FAMILIES)
    # same (seed, index) -> identical masking
    a, b = MaterializedDataset(d, seed=0)[1], MaterializedDataset(d, seed=0)[1]
    assert np.array_equal(a["mlm_input"], b["mlm_input"])
    # different dataset seed -> different masking on the same frozen selection
    other = MaterializedDataset(d, seed=999)
    assert any(
        not np.array_equal(MaterializedDataset(d, seed=0)[i]["mlm_input"], other[i]["mlm_input"])
        for i in range(len(other))
    )


def test_reader_feeds_losses(tmp_path):
    ds = MaterializedDataset(_write_fixture(tmp_path, FAMILIES))
    b = collate_token_budget([ds[i] for i in range(len(ds))])
    V = 2 * losses.AA_VOCAB
    xs = torch.randn(*b["xs"].shape, V, requires_grad=True)
    mlm = torch.randn(*b["mlm_ys"].shape, V, requires_grad=True)
    clm = torch.randn(*b["clm_ys"].shape, V, requires_grad=True)
    loss, _ = losses.total_loss(xs, mlm, clm, b)
    assert torch.isfinite(loss)
    loss.backward()
    assert xs.grad is not None
