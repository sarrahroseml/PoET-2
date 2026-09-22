"""Prepare Protenix v2 inputs for all sequences.

Reads all_seqs.fasta and generates batched Protenix JSON input files.

Usage:
    python scripts/prepare_protenix.py \
        --fasta data/all_seqs.fasta \
        --out-dir data/protenix_jobs/ \
        --batch-size 50
"""

import argparse
import json
import os
import subprocess


def needs_folding(seq_id):
    """Return True — fold everything with Protenix."""
    return True


def parse_fasta(path):
    if path.startswith("gs://"):
        proc = subprocess.Popen(
            ["gsutil", "cat", path], stdout=subprocess.PIPE, universal_newlines=True
        )
        fh = proc.stdout
    else:
        fh = open(path)

    header = None
    seq_parts = []
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
    fh.close()


def make_protenix_json(seq_id, sequence):
    """Build a single Protenix v2 input entry."""
    safe_name = seq_id.replace("|", "_").replace("/", "_").replace(" ", "_")
    return {
        "name": safe_name,
        "covalent_bonds": [],
        "sequences": [
            {
                "proteinChain": {
                    "count": 1,
                    "sequence": sequence,
                }
            }
        ],
    }


def main():
    p = argparse.ArgumentParser(description="Prepare Protenix v2 folding jobs")
    p.add_argument("--fasta", required=True, help="all_seqs.fasta (local or gs://)")
    p.add_argument("--out-dir", required=True, help="Output directory for Protenix jobs")
    p.add_argument("--batch-size", type=int, default=50,
                   help="Sequences per Protenix JSON batch (default: 50)")
    args = p.parse_args()

    inputs_dir = os.path.join(args.out_dir, "inputs")
    logs_dir = os.path.join(args.out_dir, "logs")
    outputs_dir = os.path.join(args.out_dir, "outputs")
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    to_fold = []
    print("Scanning FASTA for all sequences...")
    for seq_id, seq in parse_fasta(args.fasta):
        to_fold.append((seq_id, seq))

    print("  Found %d sequences" % len(to_fold))

    if not to_fold:
        print("Nothing to fold!")
        return

    # Write ID list
    id_list_path = os.path.join(args.out_dir, "sequences_to_fold.txt")
    with open(id_list_path, "w") as f:
        for seq_id, seq in to_fold:
            f.write("%s\t%d\n" % (seq_id, len(seq)))
    print("  ID list: %s" % id_list_path)

    # Write batched JSON inputs
    n_batches = 0
    for batch_start in range(0, len(to_fold), args.batch_size):
        batch = to_fold[batch_start:batch_start + args.batch_size]
        entries = [make_protenix_json(sid, seq) for sid, seq in batch]
        batch_path = os.path.join(inputs_dir, "batch_%d.json" % n_batches)
        with open(batch_path, "w") as f:
            json.dump(entries, f, indent=2)
        n_batches += 1

    print("  Wrote %d batch JSON files to %s" % (n_batches, inputs_dir))
    print("\nUpdate slurm/protenix_fold.slurm array range to 0-%d" % (n_batches - 1))
    print("Then submit: sbatch slurm/protenix_fold.slurm")
    print("\nEstimated: ~%.0f GPU-hours total, ~%.1f wall-hours with 96 GPUs" % (
        len(to_fold) * 30.0 / 3600,
        len(to_fold) * 30.0 / 3600 / 96,
    ))


if __name__ == "__main__":
    main()
