"""Generate a small synthetic dataset for smoke-testing the training loop (A3).

Produces the 6-file format expected by PoET2Dataset: pool_tokens, pool_offsets,
sample_target, sample_ctx_ids, sample_ctx_offsets, meta.json.

Usage:
    pixi run --no-lockfile-update python scripts/make_smoke_dataset.py \
        --out data/gitignore/smoke_dataset

    # Then feed it to the trainer:
    pixi run --no-lockfile-update python -m poet_2.training.train \
        --data-dir data/gitignore/smoke_dataset --overfit-one-batch --total-steps 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from poet_2.training.data import encode_residues

AA = b"ARNDCQEGHILKMFPSTWYV"


def random_seq(rng: np.random.Generator, length: int) -> bytes:
    return rng.choice(list(AA), size=length).astype(np.uint8).tobytes()


def mutate(rng: np.random.Generator, seq: bytes, n_muts: int) -> bytes:
    s = bytearray(seq)
    positions = rng.choice(len(s), size=min(n_muts, len(s)), replace=False)
    for p in positions:
        s[p] = int(rng.choice(list(AA)))
    return bytes(s)


def make_families(
    rng: np.random.Generator,
    n_families: int = 5,
    members_per_family: int = 8,
    min_len: int = 40,
    max_len: int = 120,
) -> list[list[bytes]]:
    families = []
    for _ in range(n_families):
        length = rng.integers(min_len, max_len + 1)
        ancestor = random_seq(rng, length)
        n_muts = max(1, length // 5)
        family = [ancestor] + [mutate(rng, ancestor, n_muts) for _ in range(members_per_family - 1)]
        families.append(family)
    return families


def write_dataset(out_dir: str, families: list[list[bytes]], n_epochs: int = 10) -> None:
    os.makedirs(out_dir, exist_ok=True)

    pool_index: dict[bytes, int] = {}
    pool: list[np.ndarray] = []

    sample_targets: list[int] = []
    sample_ctx_ids: list[int] = []
    sample_ctx_offsets: list[int] = [0]

    for fam in families:
        ids = []
        for seq in fam:
            r = encode_residues(seq)
            key = r.tobytes()
            if key not in pool_index:
                pool_index[key] = len(pool)
                pool.append(r)
            ids.append(pool_index[key])

        for _ in range(n_epochs):
            for i, target_id in enumerate(ids):
                context = [pid for j, pid in enumerate(ids) if j != i]
                if not context:
                    context = [target_id]
                sample_targets.append(target_id)
                sample_ctx_ids.extend(context)
                sample_ctx_offsets.append(len(sample_ctx_ids))

    lengths = np.array([len(p) for p in pool], dtype=np.int64)
    np.save(f"{out_dir}/pool_tokens.npy", np.concatenate(pool).astype(np.uint8))
    np.save(f"{out_dir}/pool_offsets.npy", np.concatenate(([0], np.cumsum(lengths))).astype(np.int64))
    np.save(f"{out_dir}/sample_target.npy", np.array(sample_targets, dtype=np.int64))
    np.save(f"{out_dir}/sample_ctx_ids.npy", np.array(sample_ctx_ids, dtype=np.int64))
    np.save(f"{out_dir}/sample_ctx_offsets.npy", np.array(sample_ctx_offsets, dtype=np.int64))
    json.dump(
        {"n_pool": len(pool), "n_samples": len(sample_targets)},
        open(f"{out_dir}/meta.json", "w"),
    )
    print(f"Wrote {out_dir}/  pool={len(pool)} seqs, samples={len(sample_targets)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Smoke dataset for training loop validation")
    p.add_argument("--out", default="data/gitignore/smoke_dataset", help="Output directory")
    p.add_argument("--n-families", type=int, default=5)
    p.add_argument("--members", type=int, default=8, help="Members per family")
    p.add_argument("--n-epochs", type=int, default=10, help="Epoch repeats per sample")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    families = make_families(rng, n_families=args.n_families, members_per_family=args.members)
    write_dataset(args.out, families, n_epochs=args.n_epochs)


if __name__ == "__main__":
    main()
