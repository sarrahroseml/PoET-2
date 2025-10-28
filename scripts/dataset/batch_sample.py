from pathlib import Path
import random

from scripts.dataset.sample import (
    build_sampling_pool,
    compute_sampling_records,
    write_uniref100_accession,
)

INPUT = Path("uniref90_ids.txt")
OUTPUT_DIR = Path("out")
SAMPLE_SIZE = 100          # tweak per run
WORKERS = 6
SEED = 123

def main() -> None:
    rng = random.Random(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    with INPUT.open() as handle:
        for line in handle:
            seed = line.strip()
            if not seed or seed.startswith("#"):
                continue

            uniref50_id, clusters = build_sampling_pool(seed, max_workers=WORKERS)
            records = compute_sampling_records(clusters)

            sampled = rng.choices(
                records,
                weights=[rec.probability for rec in records],
                k=SAMPLE_SIZE,
            )
            out_path = OUTPUT_DIR / f"{acc}-accessions.txt"
            write_uniref100_accession(sampled, out_path)
            print(f"{seed}: wrote {SAMPLE_SIZE} UniRef100 IDs (parent {uniref50_id})")

if __name__ == "__main__":
    main()
