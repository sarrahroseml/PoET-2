#!/bin/bash
# M3 continued-training run (single-node, multi-GPU via torchrun).
#
# Prereqs:
#   * pixi env built (login node): pixi install
#   * model checkpoint:            make download_model   (-> data/gitignore/models/poet-2.ckpt)
#   * a materialized dataset dir from your data-prep (see poet_2.training.data format)
#
# Submit from the repo root:  sbatch slurm/train.slurm
# Edit the #SBATCH lines + DATA_DIR for your cluster.

#SBATCH --job-name=poet2-train
#SBATCH --partition=gpu_quad        # <-- your GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # torchrun spawns one proc per GPU itself
#SBATCH --gres=gpu:4                # <-- GPUs per node
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=poet2-train-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

DATA_DIR=/path/to/materialized_dataset   # <-- your prep output
NGPU=${SLURM_GPUS_ON_NODE:-4}

# If torch can't init CUDA on your nodes (the "No CUDA GPUs are available" issue we hit),
# uncomment whichever workaround applies once you've found it, e.g.:
# export LD_PRELOAD=/usr/lib64/libcuda.so.1

echo "host: $(hostname)  ngpu: ${NGPU}"
nvidia-smi -L || true

pixi run --no-lockfile-update torchrun --standalone --nproc_per_node="${NGPU}" \
  -m poet_2.training.train \
  --data-dir "${DATA_DIR}" \
  --checkpoint data/gitignore/models/poet-2.ckpt \
  --out-dir data/gitignore/checkpoints \
  --dtype bf16 \
  --optimizer adamw \
  --peak-lr 3e-4 \
  --warmup-steps 200 \
  --schedule cosine \
  --tokens-per-gpu 45056 \
  --grad-accum 2 \
  --ddp-mode find_unused \
  --save-every 1000 --log-every 10

# --- M3 validation gate: overfit one batch on a SINGLE GPU first ---
# Loss should fall toward ~0 within ~100 steps if training_forward + losses + optimizer
# are wired correctly. Run interactively (1 GPU) before a full multi-GPU run:
#
#   pixi run --no-lockfile-update python -m poet_2.training.train \
#     --data-dir "${DATA_DIR}" --overfit-one-batch --total-steps 100 \
#     --tokens-per-gpu 8192 --log-every 5 --save-every 0
