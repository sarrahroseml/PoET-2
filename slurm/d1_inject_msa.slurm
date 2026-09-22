#!/bin/bash
# Inject D1 MSA paths into Protenix JSONs. Run after d1_msa_search completes.

#SBATCH --job-name=d1-inject
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=d1-inject-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

echo "Start: $(date)"
pixi run --frozen python scripts/inject_d1_msa.py
echo "Done: $(date)"
