#!/usr/bin/env python3
"""Extract unique context sequences selected by the eval pipeline for all DMS datasets.

Outputs a FASTA file suitable for structure prediction (Protenix).
Uses the same select_context logic as eval.py to ensure exact match.
"""

from __future__ import annotations

import argparse
import hashlib
import sys

sys.path.insert(0, "src")
from poet_2.training.eval import discover_dms_suite, _read, select_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dms-dir", default="data/evals")
    parser.add_argument("--out", default="data/eval_context_seqs.fasta")
    parser.add_argument("--max-similarity", type=float, default=1.0)
    parser.add_argument("--context-tokens", type=int, default=6144)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    suite = discover_dms_suite(args.dms_dir)
    print(f"Found {len(suite)} DMS datasets", flush=True)

    unique_seqs: dict[bytes, str] = {}  # seq -> first DMS name seen
    total_selected = 0

    for entry in suite:
        name = entry["name"]
        names, rows = _read(entry["a2m"], upper=False)
        ctx_names, ctx_seqs = select_context(
            names, rows, args.max_similarity, args.context_tokens, args.seed,
        )
        for seq in ctx_seqs:
            if seq not in unique_seqs:
                unique_seqs[seq] = name
        total_selected += len(ctx_seqs)
        print(f"  {name}: {len(ctx_seqs)} context seqs", flush=True)

    # Also include WT sequences (query row 0) — they already have structures
    # from viral_dms_structures/ but include them for completeness
    wt_seqs = set()
    for entry in suite:
        names, rows = _read(entry["a2m"], upper=False)
        if rows:
            wt = rows[0].replace(b"-", b"").upper()
            wt_seqs.add(wt)

    print(f"\nTotal context sequences selected: {total_selected}")
    print(f"Unique context sequences: {len(unique_seqs)}")
    print(f"WT sequences (already have structures): {len(wt_seqs)}")

    # Remove WTs that already have structures
    to_fold = {s: d for s, d in unique_seqs.items() if s not in wt_seqs}
    print(f"Sequences to fold (excluding WTs): {len(to_fold)}")

    with open(args.out, "w") as f:
        for seq, dms_name in sorted(to_fold.items(), key=lambda x: x[1]):
            seq_hash = hashlib.md5(seq).hexdigest()[:12]
            f.write(f">ctx_{seq_hash}_{dms_name}\n")
            f.write(seq.decode() + "\n")

    print(f"\nWrote {len(to_fold)} sequences to {args.out}")

    avg_len = sum(len(s) for s in to_fold) / max(1, len(to_fold))
    total_res = sum(len(s) for s in to_fold)
    print(f"Average length: {avg_len:.0f} aa")
    print(f"Total residues: {total_res:,}")
    print(f"Estimated GPU-hours (Protenix): {total_res / 200 / 60:.0f}")


if __name__ == "__main__":
    main()
