#!/bin/bash
#SBATCH --job-name=poet2-m2
#SBATCH --partition=gpu_quad
#SBATCH --gres=gpu:1               
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=poet2-m2-%j.out

set -euo pipefail

# Run from wherever the job was submitted (the repo root if you `sbatch slurm/test_m2.slurm`).
cd "${SLURM_SUBMIT_DIR:-.}"

echo "host: $(hostname)"
nvidia-smi || echo "WARNING: nvidia-smi failed -- no GPU visible; flash_attn import may fail"

# M2: standalone runner (no pytest dependency). Prints PASS/FAIL per check, exits nonzero on fail.
pixi run --no-lockfile-update python scripts/check_m2.py

