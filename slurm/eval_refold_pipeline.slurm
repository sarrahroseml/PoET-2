#!/bin/bash
# After MSA search completes: inject MSA paths, clear old outputs, re-fold, convert to NPZ.
# Submit with: sbatch --dependency=afterok:MSA_JOB_ID slurm/eval_refold_pipeline.slurm

#SBATCH --job-name=eval-inject
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=eval-inject-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

echo "=== Injecting MSA paths into eval context JSONs ==="
echo "Start: $(date)"

pixi run --frozen python scripts/inject_eval_msa.py

echo "=== Clearing old eval fold outputs ==="
rm -rf data/eval_context_fold/outputs/batch_*

echo "=== Done at $(date) ==="
echo "Now submit: sbatch slurm/fold_eval_context.slurm"
