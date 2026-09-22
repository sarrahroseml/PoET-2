#!/bin/bash
# Download ~200K structures from AlphaFold DB using curl in parallel.
# CPU-only job, just HTTP downloads.
#
# Submit from repo root:  sbatch slurm/fetch_afdb.slurm

#SBATCH --job-name=fetch-afdb
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=fetch-afdb-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

OUT_DIR="data/structures/afdb"
URL_FILE="data/structure_ids/afdb_urls.tsv"
mkdir -p "$OUT_DIR"

echo "host: $(hostname)  start: $(date)"
echo "URLs to fetch: $(wc -l < "$URL_FILE")"

# Download function: skip if already exists
download_one() {
    local acc="$1"
    local url="$2"
    local out="$OUT_DIR/${acc}.pdb"
    if [ -f "$out" ]; then
        return 0
    fi
    curl -sf --retry 2 --retry-delay 1 -o "$out" "$url" 2>/dev/null || rm -f "$out"
}
export -f download_one
export OUT_DIR

# Run 32 parallel downloads
cat "$URL_FILE" | xargs -P 32 -I{} bash -c 'download_one $(echo "{}" | tr "\t" " ")'

# Count results
TOTAL=$(wc -l < "$URL_FILE")
DOWNLOADED=$(find "$OUT_DIR" -name "*.pdb" | wc -l)
echo "done: $(date)  downloaded: $DOWNLOADED / $TOTAL"
