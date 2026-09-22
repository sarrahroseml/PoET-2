"""Inject precomputed MSA paths into Protenix batch JSON files.

After colabfold_search produces .a3m files, this script updates the
Protenix input JSONs so that each proteinChain has pairedMsaPath and
unpairedMsaPath pointing to the local A3M files. Protenix will then
skip the remote MSA server entirely.

colabfold_search with a single-chain query produces one .a3m file per
query sequence. For Protenix, we set unpairedMsaPath to the A3M and
pairedMsaPath to an empty A3M (single-chain, no pairing needed).

Usage:
    python scripts/inject_msa_paths.py \
        --msa-dir data/msa_jobs/results \
        --protenix-dir data/protenix_jobs/inputs \
        --out-dir data/protenix_jobs/inputs_msa
"""

import argparse
import json
import os
from pathlib import Path


def find_a3m_files(msa_dir):
    """Build a map of sequence_id -> a3m_path from colabfold_search results.

    colabfold_search names output files using the full FASTA header with
    spaces replaced by underscores, e.g. "AMPV_F_sources_jackhmmer_families_...".
    The Protenix JSON uses only the short ID ("AMPV_F").  We extract the
    short ID by splitting on "_sources_" and also store the full stem so
    both lookup styles work.
    """
    a3m_map = {}
    for batch_dir in sorted(Path(msa_dir).glob("batch_*")):
        for a3m_file in batch_dir.glob("*.a3m"):
            full_stem = a3m_file.stem
            a3m_path = str(a3m_file.resolve())
            a3m_map[full_stem] = a3m_path
            short_id = full_stem.split("_sources_")[0] if "_sources_" in full_stem else full_stem
            if short_id not in a3m_map:
                a3m_map[short_id] = a3m_path
    return a3m_map


def make_empty_a3m(out_dir, seq_id, sequence):
    """Create a minimal A3M file with just the query sequence (for pairing)."""
    path = os.path.join(out_dir, f"{seq_id}_paired.a3m")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(f">query\n{sequence}\n")
    return os.path.abspath(path)


def main():
    p = argparse.ArgumentParser(description="Inject MSA paths into Protenix JSONs")
    p.add_argument("--msa-dir", required=True,
                   help="Directory with colabfold_search results (data/msa_jobs/results)")
    p.add_argument("--protenix-dir", required=True,
                   help="Directory with original Protenix batch JSONs")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for updated JSONs")
    p.add_argument("--empty-a3m-dir", default=None,
                   help="Directory for empty paired A3M files (default: <out-dir>/paired_a3ms)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    empty_dir = args.empty_a3m_dir or os.path.join(args.out_dir, "paired_a3ms")
    os.makedirs(empty_dir, exist_ok=True)

    print("Scanning MSA results...")
    a3m_map = find_a3m_files(args.msa_dir)
    print(f"  Found {len(a3m_map)} A3M files")

    found = 0
    missing = 0
    total = 0

    for json_file in sorted(Path(args.protenix_dir).glob("batch_*.json")):
        with open(json_file) as f:
            entries = json.load(f)

        for entry in entries:
            name = entry["name"]
            total += 1
            if name in a3m_map:
                seq = entry["sequences"][0]["proteinChain"]["sequence"]
                entry["sequences"][0]["proteinChain"]["unpairedMsaPath"] = a3m_map[name]
                entry["sequences"][0]["proteinChain"]["pairedMsaPath"] = make_empty_a3m(
                    empty_dir, name, seq
                )
                found += 1
            else:
                missing += 1

        out_path = os.path.join(args.out_dir, json_file.name)
        with open(out_path, "w") as f:
            json.dump(entries, f, indent=2)

    print(f"\nProcessed {total} entries: {found} with MSA, {missing} without")
    print(f"Updated JSONs written to {args.out_dir}")
    if missing:
        print(f"WARNING: {missing} sequences have no MSA — Protenix will use remote server for those")


if __name__ == "__main__":
    main()
