#!/usr/bin/env python3
"""Convert Protenix CIF outputs for eval context sequences to NPZ files.

Reads the FASTA of context sequences (for name→seq mapping) and the
Protenix output directories. Produces one NPZ per sequence named by
the FASTA header (which includes the md5 hash for lookup by eval.py).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, "src")


def _find_best_cif(output_dir: str, name: str) -> str | None:
    """Find the best-ranked CIF file for a given sequence name in Protenix outputs."""
    # Protenix outputs: batch_N/seed_101/<name>/seed_101/<name>_sample_0.cif (or similar)
    result = subprocess.run(
        ["find", output_dir, "-name", f"{name}*.cif", "-type", "f"],
        capture_output=True, text=True, timeout=60,
    )
    cifs = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
    if not cifs:
        return None
    # Prefer sample_0 (highest confidence)
    for c in sorted(cifs):
        if "sample_0" in c or "rank_0" in c:
            return c
    return sorted(cifs)[0]


def _convert_one(args_tuple):
    name, cif_path, out_path = args_tuple
    try:
        from openprotein.protein import Protein
        p = Protein.from_filepath(cif_path, chain_id="A")
        coords = p.coordinates[:, :3].copy()  # (L, 3, 3) backbone N/CA/C
        plddt = p.plddt.copy()
        np.savez_compressed(out_path, plddt=plddt.astype(np.float32),
                           atomx=coords.astype(np.float32))
        return name, True
    except Exception as e:
        return name, f"error: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", default="data/eval_context_seqs.fasta")
    parser.add_argument("--protenix-dir", default="data/eval_context_fold/outputs")
    parser.add_argument("--out-dir", default="data/eval_context_structures")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Overwrite existing NPZ files")
    args = parser.parse_args()

    from poet_2.fasta import parse_stream
    with open(args.fasta, "rb") as f:
        entries = list(parse_stream(f, upper=True))

    print(f"Read {len(entries)} sequences from {args.fasta}")
    os.makedirs(args.out_dir, exist_ok=True)

    # Build index of all CIF files
    print("Indexing Protenix outputs...", flush=True)
    result = subprocess.run(
        ["find", args.protenix_dir, "-name", "*.cif", "-type", "f"],
        capture_output=True, text=True, timeout=300,
    )
    all_cifs = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
    print(f"Found {len(all_cifs)} CIF files", flush=True)

    # Map name → best CIF path (by ranking_score from confidence JSON)
    import json as _json
    cif_by_name: dict[str, str] = {}
    score_by_name: dict[str, float] = {}
    for cif in all_cifs:
        basename = os.path.basename(cif).replace(".cif", "")
        parts = basename.rsplit("_sample_", 1)
        seq_name = parts[0]
        conf_json = cif.replace(".cif", "").replace(
            basename, seq_name + "_summary_confidence_" + "sample_" + parts[1]
        ) + ".json" if len(parts) > 1 else None
        ranking = -1.0
        if conf_json and os.path.exists(conf_json):
            try:
                with open(conf_json) as jf:
                    ranking = _json.load(jf).get("ranking_score", -1.0)
            except Exception:
                pass
        if seq_name not in cif_by_name or ranking > score_by_name.get(seq_name, -1.0):
            cif_by_name[seq_name] = cif
            score_by_name[seq_name] = ranking

    # Prepare conversion tasks
    tasks = []
    missing = 0
    for name, seq in entries:
        header = name.decode().split()[0]
        cif = cif_by_name.get(header)
        if cif is None:
            missing += 1
            continue
        out_path = os.path.join(args.out_dir, f"{header}.npz")
        if os.path.exists(out_path) and not args.force:
            continue
        tasks.append((header, cif, out_path))

    print(f"Matched: {len(entries) - missing}/{len(entries)}, to convert: {len(tasks)}, "
          f"already done: {len(entries) - missing - len(tasks)}", flush=True)

    if tasks:
        ok, fail = 0, 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for name, result in pool.map(_convert_one, tasks):
                if result is True:
                    ok += 1
                else:
                    fail += 1
                    print(f"  FAIL: {name}: {result}", flush=True)
        print(f"Converted {ok} NPZ files, {fail} failures", flush=True)

    total = len([f for f in os.listdir(args.out_dir) if f.endswith(".npz")])
    print(f"Total NPZ files in {args.out_dir}: {total}")


if __name__ == "__main__":
    main()
