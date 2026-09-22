#!/bin/bash
# Evaluate top D1 checkpoints with full struct/IFQ/ensemble scoring.
# Submit: sbatch slurm/eval_d1_struct.slurm

#SBATCH --job-name=eval-d1-struct
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=eval-d1-struct-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

echo "host: $(hostname)"
echo "start: $(date)"
nvidia-smi -L || true

pixi run --frozen python scripts/eval_d1_struct.py
