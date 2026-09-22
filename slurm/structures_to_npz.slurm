#!/bin/bash
# Convert protenix CIF + AFDB PDB structures to per-sequence NPZ files.
# Submit: sbatch slurm/structures_to_npz.slurm

#SBATCH --job-name=struct-to-npz
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=struct-to-npz-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

echo "host: $(hostname)"
echo "start: $(date)"

pixi run --frozen python scripts/structures_to_npz.py \
  --fasta data/all_seqs.fasta \
  --protenix-dir data/protenix_jobs/outputs \
  --afdb-dir data/structures/afdb \
  --out data/structures/npz \
  --workers 32

echo "done: $(date)"
