import random
from pathlib import Path

from sample import (
    build_sampling_pool,
    compute_sampling_records,
    write_fasta,
    sample_until_token_limit,
)

INPUT = Path("uniref90_ids.txt")
OUTPUT_DIR = Path("out")
TOKEN_LIMIT = 0    
SEED = 123         

def main() -> None:
    rng = random.Random(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with INPUT.open() as handle:
        for line in handle:
            acc = line.strip()
            if not acc or acc.startswith("#"):
                continue

            uniref50_id, clusters = build_sampling_pool(acc)
            records = compute_sampling_records(clusters)

            if TOKEN_LIMIT:
                sampled, used = sample_until_token_limit(records, TOKEN_LIMIT, rng)
                write_fasta(sampled, OUTPUT_DIR / f"{acc}.fasta")
                print(f"{acc}: sampled {len(sampled)} records ({used} tokens) from {uniref50_id}")
            else:
                write_fasta(records, OUTPUT_DIR / f"{acc}.fasta")
                print(f"{acc}: wrote all {len(records)} UniRef100 records from {uniref50_id}")

if __name__ == "__main__":
    main()
