"""Split all_seqs.fasta into chunks for parallel colabfold_search.

Each chunk becomes one FASTA file processed by a SLURM array task.
colabfold_search is most efficient with batched queries, so we use
larger batches than for Protenix folding.

Usage:
    python scripts/prepare_msa_batches.py \
        --fasta data/all_seqs.fasta \
        --out-dir data/msa_jobs \
        --batch-size 500
"""

import argparse
import os


def parse_fasta(path):
    header = None
    seq_parts = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield header.split()[0], "".join(seq_parts)
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        yield header.split()[0], "".join(seq_parts)


def main():
    p = argparse.ArgumentParser(description="Split FASTA into MSA search batches")
    p.add_argument("--fasta", required=True)
    p.add_argument("--out-dir", default="data/msa_jobs")
    p.add_argument("--batch-size", type=int, default=500,
                   help="Sequences per colabfold_search batch")
    args = p.parse_args()

    queries_dir = os.path.join(args.out_dir, "queries")
    results_dir = os.path.join(args.out_dir, "results")
    logs_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(queries_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    seqs = list(parse_fasta(args.fasta))
    print(f"Read {len(seqs)} sequences from {args.fasta}")

    n_batches = 0
    for start in range(0, len(seqs), args.batch_size):
        batch = seqs[start:start + args.batch_size]
        batch_path = os.path.join(queries_dir, f"batch_{n_batches}.fasta")
        with open(batch_path, "w") as f:
            for seq_id, seq in batch:
                safe_id = seq_id.replace("|", "_").replace("/", "_")
                f.write(f">{safe_id}\n{seq}\n")
        n_batches += 1

    print(f"Wrote {n_batches} batch FASTA files to {queries_dir}")
    print(f"SLURM array range: 0-{n_batches - 1}")


if __name__ == "__main__":
    main()
