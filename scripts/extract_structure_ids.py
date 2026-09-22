"""Extract per-DB lookup ID lists from all_seqs.fasta.

Writes simple text files (one ID per line) that you can take to the structure
server and use to pull PDBs.

Usage:
    python scripts/extract_structure_ids.py \
        --fasta data/all_seqs.fasta \
        --out-dir data/structure_ids/

Output:
    viralaf2_ids.txt   -- bare accessions, e.g. NP_056651
    nomburg_ids.txt    -- full names, e.g. E6__YP_009182324__Rattus_...__1756445
    viro3d_ids.txt     -- bare IDs, e.g. AAV34155.1.1.2_7162
"""

import argparse
import os
import subprocess
from collections import defaultdict


DB_SUFFIXES = ["viralaf2", "nomburg", "viro3d"]


def parse_fasta_ids(path):
    if path.startswith("gs://"):
        proc = subprocess.Popen(
            ["gsutil", "cat", path], stdout=subprocess.PIPE, universal_newlines=True
        )
        fh = proc.stdout
    else:
        fh = open(path)

    for line in fh:
        if line.startswith(">"):
            yield line[1:].rstrip("\n").split()[0]
    fh.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta", required=True, help="all_seqs.fasta (local or gs://)")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ids = defaultdict(list)

    for seq_id in parse_fasta_ids(args.fasta):
        for suffix in DB_SUFFIXES:
            if seq_id.endswith("|" + suffix):
                bare = seq_id[:-(len(suffix) + 1)]
                ids[suffix].append(bare)
                break

    for suffix in DB_SUFFIXES:
        out_path = os.path.join(args.out_dir, suffix + "_ids.txt")
        with open(out_path, "w") as f:
            for lookup_id in ids[suffix]:
                f.write(lookup_id + "\n")
        print("  %-12s  %6d IDs -> %s" % (suffix, len(ids[suffix]), out_path))


if __name__ == "__main__":
    main()
