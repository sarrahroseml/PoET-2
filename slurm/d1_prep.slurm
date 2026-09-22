#!/bin/bash
# Convert d1_logan protenix structures to NPZ, combine FASTAs, materialize datasets.
#SBATCH --job-name=d1-prep
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=d1-prep-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

echo "host: $(hostname)"
echo "start: $(date)"

# 1. Convert protenix CIF → NPZ (adds to existing data/structures/npz/)
echo "=== Step 1: Converting protenix structures to NPZ ==="
pixi run --frozen python scripts/structures_to_npz.py \
    --fasta data/d1_logan/all_new_seqs.fasta \
    --protenix-dir data/d1_logan/protenix_jobs/outputs \
    --out data/structures/npz \
    --workers 16

# 2. Combine FASTAs (base + d1_logan)
echo "=== Step 2: Combining FASTA files ==="
COMBINED=data/d1_logan/all_seqs_combined.fasta
cat data/all_seqs.fasta data/d1_logan/all_new_seqs.fasta > "$COMBINED"
echo "Combined FASTA: $(grep -c '^>' "$COMBINED") sequences"

# 3. Decompress TSVs
echo "=== Step 3: Decompressing TSVs ==="
zcat data/d1_logan/presampled_diversity_100ep.tsv.gz > data/d1_logan/presampled_diversity_100ep.tsv
zcat data/d1_logan/presampled_weighted_100ep.tsv.gz > data/d1_logan/presampled_weighted_100ep.tsv
echo "Diversity: $(wc -l < data/d1_logan/presampled_diversity_100ep.tsv) samples"
echo "Weighted:  $(wc -l < data/d1_logan/presampled_weighted_100ep.tsv) samples"

# 4. Materialize with structures
echo "=== Step 4a: Materializing diversity dataset (with structures) ==="
pixi run --frozen python scripts/materialize_presampled.py \
    --fasta "$COMBINED" \
    --tsv data/d1_logan/presampled_diversity_100ep.tsv \
    --struct-dir data/structures/npz \
    --out data/gitignore/materialized/d1_diversity_struct

echo "=== Step 4b: Materializing weighted dataset (with structures) ==="
pixi run --frozen python scripts/materialize_presampled.py \
    --fasta "$COMBINED" \
    --tsv data/d1_logan/presampled_weighted_100ep.tsv \
    --struct-dir data/structures/npz \
    --out data/gitignore/materialized/d1_weighted_struct

echo "=== Done ==="
echo "end: $(date)"
ls -la data/gitignore/materialized/d1_diversity_struct/
ls -la data/gitignore/materialized/d1_weighted_struct/
