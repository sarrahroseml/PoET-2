#!/bin/bash
# Evaluate pretrained + best checkpoint with struct/ifq scoring modes.
# Submit: sbatch slurm/eval_struct.slurm

#SBATCH --job-name=poet2-struct-eval
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=eval-struct-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

echo "host: $(hostname)"
echo "start: $(date)"
nvidia-smi -L || true

pixi run --frozen python scripts/eval_struct_modes.py
