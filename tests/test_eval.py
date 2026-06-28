"""CPU unit tests for the model-free parts of the eval hook (poet_2.training.eval).

The scoring itself (compute_memory/score_sequences) needs the model and runs on the GPU
box; here we test a3m/fasta parsing, context selection, length adjustment, and Spearman,
plus an end-to-end parse on the shipped BLAT data.
"""

import math
import os

import numpy as np

from poet_2.training import eval as ev

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_ungapped_and_match_columns():
    assert ev.ungapped(b"ab-CD-e") == b"ABCDE"
    assert ev.match_columns(b"AbC-d") == b"AC-"  # drop lowercase inserts, keep upper + '-'


def test_identity_to_query():
    q = b"ABCD"
    assert ev.identity_to_query(b"ABCD", q) == 1.0
    assert ev.identity_to_query(b"ABXD", q) == 0.75  # one mismatch / 4
    # gap columns in the query are excluded from the denominator
    assert ev.identity_to_query(b"AZCD", b"A-CD") == 1.0  # only A,C,D count, all match


def test_select_context_filters_dedups_and_strips_gaps():
    names = [b"query", b"h_identical", b"h2", b"h2dup", b"h3"]
    rows = [b"ABCDEFGH", b"ABCDEFGH", b"ABCDEFGG", b"ABCDEFGG", b"XYCDEFGH"]
    ctx = ev.select_context(names, rows, max_similarity=0.9, max_tokens=10_000, seed=0)
    # identical-to-query homolog excluded (identity 1.0 > 0.9); h2/h2dup deduped
    assert b"ABCDEFGH" not in ctx
    assert len(ctx) == 2
    assert all(b"-" not in s and s == s.upper() for s in ctx)


def test_select_context_token_budget():
    names = [b"q"] + [f"h{i}".encode() for i in range(5)]
    rows = [b"AAAAAAAA"] + [bytes([66 + i]) * 8 for i in range(5)]  # distinct 8-mers
    ctx = ev.select_context(names, rows, max_similarity=1.0, max_tokens=10, seed=1)
    assert len(ctx) == 1  # each costs 8+2=10; only one fits


def test_length_adjusted():
    adj = ev.length_adjusted([0.0, -10.0], [10, 10], alpha=1.96)
    assert adj[0] == 19.6 and math.isclose(adj[1], 9.6)


def test_spearman():
    assert ev.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert ev.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert math.isnan(ev.spearman([1, 1, 1], [1, 2, 3]))  # constant -> undefined


def test_read_labels_and_variants(tmp_path):
    csv_p = tmp_path / "dms.csv"
    csv_p.write_text("mutant,DMS_score\nA1B,0.5\nC2D,-1.0\n")
    assert np.allclose(ev.read_labels(str(csv_p)), [0.5, -1.0])
    fa = tmp_path / "v.fasta"
    fa.write_text(">v1\nABCD\n>v2\nEF-gh\n")
    assert ev.read_variant_seqs(str(fa)) == [b"ABCD", b"EFGH"]


def test_eval_parsing_on_shipped_blat_data():
    a3m = os.path.join(REPO, "data", "BLAT_ECOLX_ColabFold_2202.a3m")
    variants = os.path.join(REPO, "data", "BLAT_ECOLX_Jacquier_2013_variants.fasta")
    csv_p = os.path.join(REPO, "data", "BLAT_ECOLX_Jacquier_2013.csv")
    if not (os.path.exists(a3m) and os.path.exists(variants) and os.path.exists(csv_p)):
        import pytest

        pytest.skip("shipped BLAT data not present")
    names, rows = ev._read(a3m, upper=False)
    ctx = ev.select_context(names, rows, max_similarity=0.95, max_tokens=6144, seed=0)
    assert len(ctx) > 0 and all(b"-" not in s and s == s.upper() for s in ctx)
    assert sum(len(s) + 2 for s in ctx) <= 6144 + max(len(s) + 2 for s in ctx)
    variants_seqs = ev.read_variant_seqs(variants)
    labels = ev.read_labels(csv_p, "DMS_score")
    assert len(variants_seqs) > 0 and np.isfinite(labels).all()
    assert ev.spearman(labels, labels) == 1.0
