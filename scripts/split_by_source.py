"""Split all_seqs.fasta into per-source-database FASTAs for structure fetching.

Reads the master FASTA (local or from GCS) and writes one FASTA per structure
source so you can fetch structures from each DB independently.

Usage:
    python scripts/split_by_source.py \
        --fasta data/all_seqs.fasta \
        --out-dir data/by_source/

If --fasta starts with gs://, fetches via gsutil.

Output files:
    viralaf2.fasta    — IDs with |viralaf2 suffix (accession as header)
    nomburg.fasta     — IDs with |nomburg suffix (full name as header)
    viro3d.fasta      — IDs with |viro3d suffix
    bfvd.fasta        — IDs with |bfvd suffix
    af2.fasta         — IDs with |af2 suffix
    uniprot.fasta     — tr|...|... and sp|...|... IDs (accession as header, for AFDB)
    uniref100.fasta   — UniRef100_... IDs (seed accession as header, for AFDB)
    no_structure.fasta — everything else (metagenome/SRA, needs Protenix)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import defaultdict


DB_SUFFIXES = ["viralaf2", "nomburg", "viro3d", "bfvd", "af2"]


def classify_id(seq_id: str) -> tuple[str, str]:
    """Return (source_db, lookup_key) for a FASTA header's first token."""
    for suffix in DB_SUFFIXES:
        if seq_id.endswith(f"|{suffix}"):
            bare = seq_id[: -(len(suffix) + 1)]
            return suffix, bare

    if seq_id.startswith("tr|") or seq_id.startswith("sp|"):
        parts = seq_id.split("|")
        accession = parts[1] if len(parts) >= 2 else seq_id
        return "uniprot", accession

    if seq_id.startswith("UniRef100_"):
        seed = seq_id.replace("UniRef100_", "")
        return "uniref100", seed

    return "no_structure", seq_id


def parse_fasta(path: str):
    """Yield (header_line, sequence) from a FASTA file or GCS path."""
    if path.startswith("gs://"):
        proc = subprocess.Popen(
            ["gsutil", "cat", path], stdout=subprocess.PIPE, text=True
        )
        fh = proc.stdout
    else:
        fh = open(path)

    header = None
    seq_parts: list[str] = []
    for line in fh:
        line = line.rstrip("\n")
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(seq_parts)
            header = line[1:]
            seq_parts = []
        else:
            seq_parts.append(line)
    if header is not None:
        yield header, "".join(seq_parts)
    fh.close()


def main():
    p = argparse.ArgumentParser(description="Split FASTA by structure source DB")
    p.add_argument("--fasta", required=True, help="Path to all_seqs.fasta (local or gs://)")
    p.add_argument("--out-dir", required=True, help="Output directory for per-DB FASTAs")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    counts: dict[str, int] = defaultdict(int)
    handles: dict[str, object] = {}

    for header, seq in parse_fasta(args.fasta):
        seq_id = header.split()[0]
        source, lookup_key = classify_id(seq_id)
        counts[source] += 1

        if source not in handles:
            handles[source] = open(os.path.join(args.out_dir, f"{source}.fasta"), "w")

        fh = handles[source]
        fh.write(f">{lookup_key}\n{seq}\n")

    for fh in handles.values():
        fh.close()

    print("Sequences per source:")
    for source in sorted(counts, key=counts.get, reverse=True):
        path = os.path.join(args.out_dir, f"{source}.fasta")
        print(f"  {source:20s} {counts[source]:>8,d}  → {path}")


if __name__ == "__main__":
    main()
