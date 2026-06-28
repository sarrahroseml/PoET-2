"""M5: lightweight zero-shot eval hook to track regression during training (spec §9).

PoET-2 is alignment-free — it scores from a set of *ungapped* homolog sequences. The a3m
alignment is used only to *select* the context by identity to the WT (the ColabFold
max-similarity protocol); a plain homolog FASTA also works with similarity filtering off.
Variant scoring needs no alignment (full sequences vs WT, with the α·len adjustment).

Model-free parts (a3m parsing, context selection, length adjustment, Spearman) are
unit-tested locally. The scoring reuses ``poet_2_helpers.compute_memory`` +
``score_sequences`` via a lazy import (needs flash_attn / the GPU box). This is a single
prompt — not the full 15-combo §9 ensembling — enough to track training.
"""

from __future__ import annotations

import csv

import numpy as np

from poet_2.fasta import parse_stream

ALPHA = 1.96  # length adjustment (spec §9)


# ----------------------------------------------------------- a3m / fasta (model-free)


def _read(path: str, upper: bool) -> tuple[list[bytes], list[bytes]]:
    with open(path, "rb") as f:
        names, seqs = [], []
        for name, seq in parse_stream(f, upper=upper):
            names.append(name)
            seqs.append(seq)
    return names, seqs


def ungapped(seq: bytes) -> bytes:
    """Full homolog sequence the model sees: drop gaps, uppercase inserts."""
    return seq.replace(b"-", b"").upper()


def match_columns(seq: bytes) -> bytes:
    """a3m match-state string: drop lowercase inserts, keep uppercase + '-' (query-aligned)."""
    return bytes(c for c in seq if not (97 <= c <= 122))


def identity_to_query(row: bytes, query_match: bytes) -> float:
    """Fractional identity over non-gap query columns (row given as a raw a3m line)."""
    q = np.frombuffer(query_match, dtype=np.uint8)
    r = np.frombuffer(match_columns(row), dtype=np.uint8)
    n = min(q.shape[0], r.shape[0])
    if n == 0:
        return 0.0
    q, r = q[:n], r[:n]
    nongap = q != ord("-")
    denom = int(nongap.sum())
    return float(((r == q) & nongap).sum() / denom) if denom else 0.0


def select_context(
    names: list[bytes],
    rows: list[bytes],
    max_similarity: float = 1.0,
    max_tokens: int = 6144,
    seed: int = 0,
) -> list[bytes]:
    """Pick ungapped context homologs: keep those with identity-to-query <= max_similarity
    (set <1.0 to diversify, §9), dedup, shuffle, pack under a token budget."""
    query_match = match_columns(rows[0]) if rows else b""
    rng = np.random.default_rng(seed)
    seen: set[bytes] = set()
    uniq: list[bytes] = []
    for row in rows[1:]:
        if query_match and identity_to_query(row, query_match) > max_similarity:
            continue
        s = ungapped(row)
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
    order = rng.permutation(len(uniq))
    out, used = [], 0
    for i in order:
        s = uniq[int(i)]
        cost = len(s) + 2
        if out and used + cost > max_tokens:
            continue
        out.append(s)
        used += cost
        if used >= max_tokens:
            break
    return out


def read_variant_seqs(path: str) -> list[bytes]:
    return [ungapped(s) for s in _read(path, upper=True)[1]]


def read_labels(csv_path: str, col: str = "DMS_score") -> np.ndarray:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r[col]) for r in rows], dtype=float)


def length_adjusted(logps, lengths, alpha: float = ALPHA) -> np.ndarray:
    return np.asarray(logps, dtype=float) + alpha * np.asarray(lengths, dtype=float)


def spearman(x, y) -> float:
    """Spearman ρ (Pearson on ranks); ties broken arbitrarily — fine for tracking."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape[0] < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")  # undefined for constant input
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


# ----------------------------------------------------------- scoring (needs the model)


def zero_shot_adjusted_lls(model, context_seqs, query_seqs, alpha: float = ALPHA):
    """Length-adjusted log-likelihoods of each query sequence under the homolog context."""
    from poet_2.models.poet_2_helpers import compute_memory, score_sequences  # lazy

    memory, _ = compute_memory(model, [context_seqs])  # single prompt
    logps = score_sequences(model, memory, query_seqs).float().cpu().numpy()
    lengths = [len(s) for s in query_seqs]
    return length_adjusted(logps, lengths, alpha)


def evaluate(
    model,
    a3m_path: str,
    variants_fasta: str,
    *,
    wt_sequence: str | bytes | None = None,
    labels_csv: str | None = None,
    label_col: str = "DMS_score",
    alpha: float = ALPHA,
    max_similarity: float = 1.0,
    context_tokens: int = 6144,
    seed: int = 0,
) -> tuple[dict, np.ndarray]:
    """Zero-shot evaluation for the training hook. Returns ``(metrics, scores)``.

    ``scores`` are WT-relative LLRs if ``wt_sequence`` is given, else raw adjusted LLs.
    ``metrics["spearman"]`` is present when ``labels_csv`` is provided.
    """
    import torch

    names, rows = _read(a3m_path, upper=False)
    context = select_context(names, rows, max_similarity, context_tokens, seed)
    variants = read_variant_seqs(variants_fasta)
    wt = wt_sequence.encode() if isinstance(wt_sequence, str) else wt_sequence
    query = ([wt] + variants) if wt is not None else variants

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            adj = zero_shot_adjusted_lls(model, context, query, alpha)
    finally:
        if was_training:
            model.train()

    scores = (adj[1:] - adj[0]) if wt is not None else adj
    metrics: dict = {"n_variants": len(variants), "n_context": len(context)}
    if labels_csv is not None:
        labels = read_labels(labels_csv, label_col)
        m = min(scores.shape[0], labels.shape[0])
        metrics["spearman"] = spearman(scores[:m], labels[:m])
    return metrics, scores
