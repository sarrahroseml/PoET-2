#!/bin/bash
# Evaluate 4 checkpoints on all 45 DMSes, report per-DMS Spearman with IID/OOD split.
# Each checkpoint evals ~45 DMSes sequentially. ~15 min per checkpoint, ~1 hour total.
# Submit: sbatch slurm/eval_per_dms.slurm

#SBATCH --job-name=poet2-perdms
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=eval-perdms-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

echo "host: $(hostname)"
echo "start: $(date)"
nvidia-smi -L || true

pixi run --frozen python scripts/eval_per_dms.py

echo "done: $(date)"
