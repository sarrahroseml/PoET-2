"""Convert protenix CIF + AFDB PDB structures into per-sequence NPZ files.

For each sequence ID in the FASTA, finds the corresponding structure file and
extracts backbone coordinates (N/CA/C) and per-residue pLDDT. Saves one NPZ
per sequence: {plddt: (L,) float32, atomx: (L,3,3) float32}.

Usage:
    pixi run python scripts/structures_to_npz.py \
        --fasta data/all_seqs.fasta \
        --protenix-dir data/protenix_jobs/outputs \
        --afdb-dir data/structures/afdb \
        --out data/structures/npz \
        --workers 16
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def fasta_ids(path: str) -> list[str]:
    ids = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                ids.append(line[1:].split()[0])
    return ids


def fasta_id_to_protenix_folder(fasta_id: str) -> str:
    return fasta_id.replace("|", "_")


def _sanitize_for_protenix(name: str) -> str:
    """Protenix replaces : and + with _ in folder names."""
    return name.replace("|", "_").replace(":", "_").replace("+", "_")


def find_protenix_cif(protenix_dir: str, folder_name: str) -> str | None:
    for batch in os.listdir(protenix_dir):
        cif_dir = os.path.join(protenix_dir, batch, folder_name, "seed_101", "predictions")
        if os.path.isdir(cif_dir):
            sample0 = os.path.join(cif_dir, f"{folder_name}_sample_0.cif")
            if os.path.isfile(sample0):
                return sample0
    return None


def find_afdb_pdb(afdb_dir: str, fasta_id: str) -> str | None:
    if not fasta_id.startswith("AF-"):
        return None
    accession = fasta_id.split("-")[1]
    pdb = os.path.join(afdb_dir, f"{accession}.pdb")
    return pdb if os.path.isfile(pdb) else None


def convert_one(struct_path: str, out_path: str) -> bool:
    try:
        from openprotein.protein import Protein
        p = Protein.from_filepath(struct_path, chain_id="A")
        atomx = p.coordinates[:, :3].astype(np.float32)
        plddt = p.plddt.astype(np.float32)
        np.savez_compressed(out_path, plddt=plddt, atomx=atomx)
        return True
    except Exception as e:
        print(f"  WARN: {struct_path}: {e}", flush=True)
        return False


def build_work_list(
    fasta_path: str,
    protenix_dir: str | None,
    afdb_dir: str | None,
    out_dir: str,
) -> list[tuple[str, str, str]]:
    """Returns list of (fasta_id, struct_path, out_path)."""
    ids = fasta_ids(fasta_path)
    print(f"FASTA: {len(ids)} sequences", flush=True)

    protenix_index: dict[str, str] | None = None
    if protenix_dir and os.path.isdir(protenix_dir):
        print(f"Indexing protenix outputs in {protenix_dir} (using find) ...", flush=True)
        import subprocess
        protenix_index = {}
        result = subprocess.run(
            ["find", protenix_dir, "-name", "*_sample_0.cif", "-path", "*/predictions/*"],
            capture_output=True, text=True, timeout=1800,
        )
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            folder = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(line))))
            protenix_index[folder] = line
        print(f"  {len(protenix_index)} protenix structures indexed", flush=True)

    work = []
    found_protenix = 0
    found_afdb = 0
    for fid in ids:
        safe_name = fid.replace("|", "_").replace("/", "_").replace(":", "_").replace("+", "_")
        out_path = os.path.join(out_dir, f"{safe_name}.npz")
        if os.path.exists(out_path):
            continue

        struct_path = None
        if protenix_index is not None:
            folder = fasta_id_to_protenix_folder(fid)
            if folder in protenix_index:
                struct_path = protenix_index[folder]
                found_protenix += 1
            else:
                sanitized = _sanitize_for_protenix(fid)
                if sanitized in protenix_index:
                    struct_path = protenix_index[sanitized]
                    found_protenix += 1

        if struct_path is None and afdb_dir:
            struct_path = find_afdb_pdb(afdb_dir, fid)
            if struct_path:
                found_afdb += 1

        if struct_path is not None:
            work.append((fid, struct_path, out_path))

    total = found_protenix + found_afdb
    print(f"  {total} structures found ({found_protenix} protenix, {found_afdb} AFDB)", flush=True)
    print(f"  {len(work)} to convert (skipping existing NPZ)", flush=True)
    return work


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta", required=True)
    p.add_argument("--protenix-dir", default=None)
    p.add_argument("--afdb-dir", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    work = build_work_list(args.fasta, args.protenix_dir, args.afdb_dir, args.out)

    if not work:
        print("Nothing to convert.", flush=True)
        return

    done = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(convert_one, sp, op): fid for fid, sp, op in work}
        for fut in as_completed(futs):
            ok = fut.result()
            if ok:
                done += 1
            else:
                failed += 1
            if (done + failed) % 5000 == 0:
                print(f"  progress: {done + failed}/{len(work)} ({failed} failed)", flush=True)

    print(f"Done: {done} converted, {failed} failed out of {len(work)} total", flush=True)


if __name__ == "__main__":
    main()
