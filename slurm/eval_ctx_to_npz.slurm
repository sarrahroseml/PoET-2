#!/bin/bash
# Convert Protenix CIF outputs for eval context sequences to NPZ.
# Run after fold_eval_context.slurm completes.
# Submit: sbatch --dependency=afterok:FOLD_JOB_ID slurm/eval_ctx_to_npz.slurm

#SBATCH --job-name=eval-ctx-npz
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=eval-ctx-npz-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

echo "host: $(hostname)"
echo "start: $(date)"

pixi run --frozen python scripts/eval_context_to_npz.py \
  --fasta data/eval_context_seqs.fasta \
  --protenix-dir data/eval_context_fold/outputs \
  --out-dir data/eval_context_structures \
  --workers 8 \
  --force

echo "done: $(date)"
