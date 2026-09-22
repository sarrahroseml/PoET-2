#!/usr/bin/env python3
"""Prepare Protenix input JSONs for eval context sequences.

Reads the FASTA produced by extract_eval_context_seqs.py and creates
batched JSON files for Protenix single-sequence folding (no MSA).
Each batch JSON is a list of prediction entries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, "src")
from poet_2.fasta import parse_stream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", default="data/eval_context_seqs.fasta")
    parser.add_argument("--out-dir", default="data/eval_context_fold/inputs")
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    with open(args.fasta, "rb") as f:
        entries = list(parse_stream(f, upper=True))

    print(f"Read {len(entries)} sequences from {args.fasta}")

    os.makedirs(args.out_dir, exist_ok=True)

    batch_id = 0
    for start in range(0, len(entries), args.batch_size):
        chunk = entries[start:start + args.batch_size]
        batch_entries = []
        for name, seq in chunk:
            batch_entries.append({
                "name": name.decode().split()[0],
                "sequences": [
                    {
                        "proteinChain": {
                            "count": 1,
                            "sequence": seq.decode(),
                        },
                    }
                ],
            })

        out_path = os.path.join(args.out_dir, f"batch_{batch_id}.json")
        with open(out_path, "w") as f:
            json.dump(batch_entries, f, indent=2)

        batch_id += 1

    print(f"Created {batch_id} batches in {args.out_dir}")
    print(f"SLURM array range: 0-{batch_id - 1}")


if __name__ == "__main__":
    main()
