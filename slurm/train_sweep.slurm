#!/bin/bash
# Hyperparameter sweep for continued training on presampled_weighted.
# Submit: sbatch --export=LR=1e-4,MASK=0.50 slurm/train_sweep.slurm

#SBATCH --job-name=poet2-sweep
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=sweep-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

LR="${LR:?Set LR via --export=LR=...}"
MASK="${MASK:?Set MASK via --export=MASK=...}"
DATA_DIR=data/gitignore/materialized/presampled_weighted
TAG="lr${LR}-mask${MASK}"
OUT_DIR="data/gitignore/checkpoints/sweep-${TAG}"
NGPU=${SLURM_GPUS_ON_NODE:-4}

echo "host: $(hostname)  ngpu: ${NGPU}  lr: ${LR}  mask: ${MASK}"
nvidia-smi -L || true

pixi run --frozen torchrun --standalone --nproc_per_node="${NGPU}" \
  -m poet_2.training.train \
  --data-dir "${DATA_DIR}" \
  --checkpoint data/gitignore/models/poet-2.ckpt \
  --out-dir "${OUT_DIR}" \
  --dtype bf16 \
  --optimizer adamw \
  --peak-lr "${LR}" \
  --warmup-steps 200 \
  --schedule cosine \
  --tokens-per-gpu 45056 \
  --grad-accum 2 \
  --ddp-mode static_graph \
  --seq-mask-max "${MASK}" \
  --rate-cap "${MASK}" \
  --save-every 1000 --log-every 10 \
  --eval-every 500 \
  --eval-dms-dir data/evals \
  --wandb-project poet2-viral \
  --wandb-run "sweep-${TAG}"
