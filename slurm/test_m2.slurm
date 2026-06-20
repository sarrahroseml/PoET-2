#!/bin/bash
# Run the M2 model-integration tests (training_forward vs stock forward) on a GPU node.
#
# M2 builds a tiny model in-process, so it needs the pixi env (for flash_attn) but NOT the
# released checkpoint -- you do not need `make download_model`. The test runs the model on
# CPU; the GPU is only required so `flash_attn` imports.
#
# Submit from the repo root:   sbatch slurm/test_m2.slurm
#
# Adjust the #SBATCH lines below to your cluster (partition / account names vary).

#SBATCH --job-name=poet2-m2
#SBATCH --partition=gpu            # <-- your GPU partition
#SBATCH --gres=gpu:1               # 1 GPU, only so flash_attn imports
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=poet2-m2-%j.out
# #SBATCH --account=<your-account> # uncomment if your cluster requires an account

set -euo pipefail

# Run from wherever the job was submitted (the repo root if you `sbatch slurm/test_m2.slurm`).
cd "${SLURM_SUBMIT_DIR:-.}"

# If your COMPUTE nodes have internet, uncomment these (otherwise run them once on the login
# node before submitting, since `pixi install` / pip need to download packages):
# pixi install
# pixi run --no-lockfile-update pip install pytest

echo "host: $(hostname)"
nvidia-smi || echo "WARNING: nvidia-smi failed -- no GPU visible; flash_attn import may fail"

# M2 only (the make-or-break equivalence check + shapes + backward):
pixi run --no-lockfile-update python -m pytest tests/test_training_forward.py -v

# To validate the whole stack in the real env instead (M1 + collator + M2; nothing skips
# once flash_attn is importable), swap the line above for:
# pixi run --no-lockfile-update python -m pytest tests/ -q
