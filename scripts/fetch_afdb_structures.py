"""Download structures from AlphaFold DB for sequences not covered by local DBs.

Handles: uniprot (tr|X|...), uniref100 (UniRef100_X), af2 (AF-X|af2), bfvd (X|bfvd).
Downloads PDB files in parallel via the AFDB REST API.

Usage:
    python scripts/fetch_afdb_structures.py \
        --fasta data/all_seqs.fasta \
        --out-dir data/structures/afdb/ \
        --workers 32

Can also run as a SLURM job (see slurm/fetch_structures.slurm).
"""

import argparse
import os
import subprocess
import sys
import time
from collections import defaultdict
from multiprocessing.pool import ThreadPool

try:
    from urllib.request import urlretrieve, urlopen
    from urllib.error import HTTPError, URLError
except ImportError:
    from urllib import urlretrieve

AFDB_URLS = [
    "https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v6.pdb",
    "https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v4.pdb",
]


def parse_fasta(path):
    """Yield (seq_id, full_header, sequence) from a FASTA file or GCS path."""
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
                yield header.split()[0], header, "".join(seq_parts)
            header = line[1:]
            seq_parts = []
        else:
            seq_parts.append(line)
    if header is not None:
        yield header.split()[0], header, "".join(seq_parts)
    fh.close()


def extract_afdb_accession(seq_id):
    """Return (category, afdb_accession) or None if not fetchable from AFDB."""
    if seq_id.endswith("|af2"):
        bare = seq_id[:-(len("af2") + 1)]
        if bare.startswith("AF-") and "-F1-" in bare:
            accession = bare.split("-")[1]
            return "af2", accession
        return "af2", bare

    if seq_id.endswith("|bfvd"):
        bare = seq_id[:-(len("bfvd") + 1)]
        return "bfvd", bare

    if seq_id.startswith("tr|") or seq_id.startswith("sp|"):
        parts = seq_id.split("|")
        if len(parts) >= 2:
            return "uniprot", parts[1]

    if seq_id.startswith("UniRef100_"):
        seed = seq_id[len("UniRef100_"):]
        return "uniref100", seed

    return None


def download_one(args):
    """Download a single PDB from AFDB. Returns (seq_id, accession, success, path_or_error)."""
    seq_id, accession, out_dir = args
    out_path = os.path.join(out_dir, accession + ".pdb")
    if os.path.exists(out_path):
        return (seq_id, accession, True, out_path)

    last_error = None
    for url_template in AFDB_URLS:
        url = url_template.format(accession=accession)
        for attempt in range(2):
            try:
                urlretrieve(url, out_path)
                return (seq_id, accession, True, out_path)
            except Exception as e:
                last_error = e
                if attempt < 1:
                    time.sleep(0.5)
    # Clean up partial download on failure
    if os.path.exists(out_path):
        os.remove(out_path)
    return (seq_id, accession, False, str(last_error))


def main():
    p = argparse.ArgumentParser(description="Fetch structures from AlphaFold DB")
    p.add_argument("--fasta", required=True, help="all_seqs.fasta (local or gs://)")
    p.add_argument("--out-dir", required=True, help="Output directory for PDB files")
    p.add_argument("--workers", type=int, default=32, help="Parallel download threads")
    p.add_argument("--categories", default="uniprot,uniref100,af2,bfvd",
                   help="Comma-separated categories to fetch (default: all)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    wanted = set(args.categories.split(","))

    tasks = []
    by_cat = defaultdict(int)
    seen = set()

    print("Scanning FASTA for AFDB-fetchable sequences...")
    for seq_id, header, seq in parse_fasta(args.fasta):
        result = extract_afdb_accession(seq_id)
        if result is None:
            continue
        cat, accession = result
        if cat not in wanted:
            continue
        if accession in seen:
            continue
        seen.add(accession)
        by_cat[cat] += 1
        tasks.append((seq_id, accession, args.out_dir))

    print("Sequences to fetch:")
    for cat in sorted(by_cat):
        print("  %-12s  %6d" % (cat, by_cat[cat]))
    print("  %-12s  %6d" % ("TOTAL", len(tasks)))

    already = sum(1 for _, acc, _ in tasks
                  if os.path.exists(os.path.join(args.out_dir, acc + ".pdb")))
    if already:
        print("  (already downloaded: %d, remaining: %d)" % (already, len(tasks) - already))

    print("\nDownloading with %d workers..." % args.workers)
    success = 0
    failed = 0
    failed_ids = []
    pool = ThreadPool(args.workers)

    for i, result in enumerate(pool.imap_unordered(download_one, tasks)):
        seq_id, accession, ok, detail = result
        if ok:
            success += 1
        else:
            failed += 1
            failed_ids.append((seq_id, accession, detail))

        if (i + 1) % 1000 == 0 or (i + 1) == len(tasks):
            print("  %d / %d  (ok: %d, failed: %d)" % (i + 1, len(tasks), success, failed))
            sys.stdout.flush()

    pool.close()
    pool.join()

    # Write manifest
    manifest_path = os.path.join(args.out_dir, "manifest.tsv")
    with open(manifest_path, "w") as f:
        f.write("seq_id\taccession\tpdb_path\n")
        for seq_id, accession, _ in tasks:
            pdb_path = os.path.join(args.out_dir, accession + ".pdb")
            if os.path.exists(pdb_path):
                f.write("%s\t%s\t%s\n" % (seq_id, accession, pdb_path))

    # Write failed list
    if failed_ids:
        failed_path = os.path.join(args.out_dir, "failed.tsv")
        with open(failed_path, "w") as f:
            for seq_id, accession, error in failed_ids:
                f.write("%s\t%s\t%s\n" % (seq_id, accession, error))
        print("\nFailed downloads: %d (see %s)" % (failed, failed_path))

    print("\nDone: %d downloaded, %d failed, manifest at %s" % (success, failed, manifest_path))


if __name__ == "__main__":
    main()
