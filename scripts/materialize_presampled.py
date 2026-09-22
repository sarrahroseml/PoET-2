"""Convert a presampled TSV + FASTA pool into the NumPy format expected by PoET2Dataset.

The TSV has no header; each row is tab-separated:
    target_id\tcontext_id1\tcontext_id2\t...

The FASTA contains all referenced sequences, keyed by the text before the first space
in each header line (e.g. ">AMPV_F sources=..." → key "AMPV_F").

Usage:
    pixi run python scripts/materialize_presampled.py \
        --fasta data/all_seqs.fasta \
        --tsv data/presampled_diversity.tsv \
        --out data/gitignore/materialized/diversity
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from poet_2.training.data import encode_residues


def parse_fasta(path: str) -> dict[str, bytes]:
    """Parse FASTA into {id: sequence} dict. ID = header text before first space."""
    seqs: dict[str, bytes] = {}
    current_id = None
    parts: list[bytes] = []
    with open(path, "rb") as f:
        for line in f:
            line = line.rstrip(b"\n\r")
            if line.startswith(b">"):
                if current_id is not None:
                    seqs[current_id] = b"".join(parts)
                header = line[1:]
                current_id = header.split(b" ", 1)[0].decode()
                parts = []
            elif current_id is not None:
                parts.append(line)
    if current_id is not None:
        seqs[current_id] = b"".join(parts)
    return seqs


def parse_tsv(path: str) -> list[tuple[str, list[str]]]:
    """Parse presampled TSV. Returns [(target_id, [ctx_id, ...]), ...]."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t")
            target = parts[0]
            context = parts[1:] if len(parts) > 1 else [target]
            samples.append((target, context))
    return samples


def materialize(fasta_path: str, tsv_path: str, out_dir: str, struct_dir: str | None = None) -> None:
    print(f"Parsing FASTA: {fasta_path}")
    id_to_seq = parse_fasta(fasta_path)
    print(f"  {len(id_to_seq)} sequences loaded")

    print(f"Parsing TSV: {tsv_path}")
    samples = parse_tsv(tsv_path)
    print(f"  {len(samples)} samples loaded")

    # Build struct NPZ lookup if struct_dir provided
    struct_npz: dict[str, str] | None = None
    if struct_dir and os.path.isdir(struct_dir):
        print(f"Indexing structure NPZs in {struct_dir}")
        struct_npz = {}
        for f in os.listdir(struct_dir):
            if f.endswith(".npz"):
                struct_npz[f[:-4]] = os.path.join(struct_dir, f)
        print(f"  {len(struct_npz)} NPZ files found")

    pool_index: dict[bytes, int] = {}
    pool: list[np.ndarray] = []
    pool_plddt: list[np.ndarray] = []
    pool_atomx: list[np.ndarray] = []
    has_struct: list[bool] = []
    id_for_pool: list[str] = []

    def get_pool_id(seq_id: str) -> int | None:
        raw = id_to_seq.get(seq_id)
        if raw is None:
            return None
        encoded = encode_residues(raw)
        key = encoded.tobytes()
        if key not in pool_index:
            pool_index[key] = len(pool)
            pool.append(encoded)
            L = len(encoded)
            safe_name = seq_id.replace("|", "_").replace("/", "_").replace(":", "_").replace("+", "_")
            npz_path = struct_npz.get(safe_name) if struct_npz else None
            if npz_path is not None:
                try:
                    d = np.load(npz_path)
                    p, a = d["plddt"], d["atomx"]
                    if p.shape[0] == L and a.shape == (L, 3, 3):
                        pool_plddt.append(p.astype(np.float32))
                        pool_atomx.append(a.astype(np.float32))
                        has_struct.append(True)
                    else:
                        pool_plddt.append(np.full(L, np.nan, dtype=np.float32))
                        pool_atomx.append(np.full((L, 3, 3), np.nan, dtype=np.float32))
                        has_struct.append(False)
                except Exception:
                    pool_plddt.append(np.full(L, np.nan, dtype=np.float32))
                    pool_atomx.append(np.full((L, 3, 3), np.nan, dtype=np.float32))
                    has_struct.append(False)
            else:
                pool_plddt.append(np.full(L, np.nan, dtype=np.float32))
                pool_atomx.append(np.full((L, 3, 3), np.nan, dtype=np.float32))
                has_struct.append(False)
            id_for_pool.append(seq_id)
        return pool_index[key]

    sample_targets: list[int] = []
    sample_ctx_ids: list[int] = []
    sample_ctx_offsets: list[int] = [0]
    skipped = 0

    for i, (target_id, ctx_ids) in enumerate(samples):
        if (i + 1) % 50000 == 0:
            print(f"  processed {i + 1}/{len(samples)} samples, pool={len(pool)}")

        tgt_pid = get_pool_id(target_id)
        if tgt_pid is None:
            skipped += 1
            continue

        ctx_pids = []
        for cid in ctx_ids:
            pid = get_pool_id(cid)
            if pid is not None:
                ctx_pids.append(pid)

        if not ctx_pids:
            ctx_pids = [tgt_pid]

        sample_targets.append(tgt_pid)
        sample_ctx_ids.extend(ctx_pids)
        sample_ctx_offsets.append(len(sample_ctx_ids))

    print(f"  skipped {skipped} samples (missing target in FASTA)")
    print(f"  final: {len(sample_targets)} samples, {len(pool)} unique sequences")

    os.makedirs(out_dir, exist_ok=True)

    pool_tokens = np.concatenate(pool).astype(np.uint8) if pool else np.array([], dtype=np.uint8)
    lengths = np.array([len(p) for p in pool], dtype=np.int64)
    pool_offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)

    np.save(os.path.join(out_dir, "pool_tokens.npy"), pool_tokens)
    np.save(os.path.join(out_dir, "pool_offsets.npy"), pool_offsets)
    np.save(os.path.join(out_dir, "sample_target.npy"), np.array(sample_targets, dtype=np.int64))
    np.save(os.path.join(out_dir, "sample_ctx_ids.npy"), np.array(sample_ctx_ids, dtype=np.int64))
    np.save(os.path.join(out_dir, "sample_ctx_offsets.npy"), np.array(sample_ctx_offsets, dtype=np.int64))

    n_with_struct = sum(has_struct)
    if struct_npz is not None and pool_plddt:
        flat_plddt = np.concatenate(pool_plddt).astype(np.float32)
        flat_atomx = np.concatenate(pool_atomx).reshape(-1, 3, 3).astype(np.float32)
        np.save(os.path.join(out_dir, "pool_plddt.npy"), flat_plddt)
        np.save(os.path.join(out_dir, "pool_atomx.npy"), flat_atomx)
        print(f"  structure coverage: {n_with_struct}/{len(pool)} sequences ({100*n_with_struct/len(pool):.1f}%)")

    json.dump(
        {
            "n_pool": len(pool),
            "n_samples": len(sample_targets),
            "skipped": skipped,
            "n_with_struct": n_with_struct,
            "source_tsv": os.path.basename(tsv_path),
            "source_fasta": os.path.basename(fasta_path),
        },
        open(os.path.join(out_dir, "meta.json"), "w"),
        indent=2,
    )
    print(f"Wrote {out_dir}/")


def main():
    p = argparse.ArgumentParser(description="Materialize presampled TSV into PoET2Dataset format")
    p.add_argument("--fasta", required=True, help="Path to all_seqs.fasta")
    p.add_argument("--tsv", required=True, help="Path to presampled TSV")
    p.add_argument("--out", required=True, help="Output directory for NumPy arrays")
    p.add_argument("--struct-dir", default=None, help="Directory of per-sequence NPZ structure files")
    args = p.parse_args()
    materialize(args.fasta, args.tsv, args.out, struct_dir=args.struct_dir)


if __name__ == "__main__":
    main()
