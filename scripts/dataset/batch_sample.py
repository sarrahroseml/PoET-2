from pathlib import Path
import random

from sample import (
    build_sampling_pool,
    compute_sampling_records,
    write_uniref100_accession,
    UniRefError
)

INPUT = Path("/n/groups/marks/projects/viral_plm/models/PoET-2/data/u90_acc.txt")
OUTPUT_DIR = Path("/n/groups/marks/projects/viral_plm/models/PoET-2/data/h_seqs")
SAMPLE_SIZE = 100          # tweak per run
WORKERS = 6
SEED = 123
OVERWRITE = False

def main() -> None:
    rng = random.Random(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    with INPUT.open() as handle:
        for line in handle:
            seed = line.strip()
            if not seed or seed.startswith("#"):
                continue
            out_path = OUTPUT_DIR / f"{seed}.txt"
            if out_path.exists() and not OVERWRITE: 
                print(f"{seed} already exists, skipping")
                continue
            try:
                uniref50_id, clusters = build_sampling_pool(seed, max_workers=WORKERS)
            except UniRefError as exc: 
                print(f"Warning: skipping {seed} ({exc})")
                continue

            records = compute_sampling_records(clusters)

            sampled = rng.choices(
                records,
                weights=[rec.probability for rec in records],
                k=SAMPLE_SIZE,
            )
            
            write_uniref100_accession(sampled, out_path)
            print(f"{seed}: wrote {SAMPLE_SIZE} UniRef100 IDs (parent {uniref50_id})")

if __name__ == "__main__":
    main()
