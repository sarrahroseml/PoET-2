#!/bin/bash
# Materialize training data WITH structure arrays (pool_plddt + pool_atomx).
# Run AFTER structures_to_npz.slurm completes.
# Submit: sbatch slurm/materialize_struct.slurm

#SBATCH --job-name=materialize-struct
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=materialize-struct-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

echo "host: $(hostname)"
echo "start: $(date)"

# Count available NPZ structures
echo "NPZ structures available:"
find data/structures/npz -name "*.npz" -maxdepth 1 | wc -l

pixi run --frozen python scripts/materialize_presampled.py \
  --fasta data/all_seqs.fasta \
  --tsv data/presampled_weighted.tsv \
  --struct-dir data/structures/npz \
  --out data/gitignore/materialized/presampled_weighted_struct

echo "done: $(date)"
