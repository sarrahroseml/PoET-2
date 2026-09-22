#!/bin/bash
# Evaluate pretrained checkpoint (step 0) on all 45 DMSes.
# Uses --total-steps 1 so it runs the step-0 eval then exits quickly.
# Submit: sbatch slurm/eval_baseline.slurm

#SBATCH --job-name=poet2-eval0
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=eval-baseline-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

echo "host: $(hostname)"
nvidia-smi -L || true

pixi run --frozen python -m poet_2.training.train \
  --data-dir data/gitignore/materialized/presampled_weighted \
  --checkpoint data/gitignore/models/poet-2.ckpt \
  --out-dir data/gitignore/checkpoints/baseline-eval \
  --dtype bf16 \
  --total-steps 1 \
  --eval-every 1 \
  --eval-dms-dir data/evals \
  --log-every 1
