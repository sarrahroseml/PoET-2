#!/bin/bash
# Structure-conditioned continued training on the DIVERSITY dataset.
# Submit: sbatch slurm/train_struct_diversity.slurm

#SBATCH --job-name=poet2-struct-div
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=struct-div-train-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

DATA_DIR=data/gitignore/materialized/presampled_diversity_struct
OUT_DIR=data/gitignore/checkpoints/struct-div-lr1e-4
NGPU=${SLURM_GPUS_ON_NODE:-4}

echo "host: $(hostname)  ngpu: ${NGPU}"
nvidia-smi -L || true

if [ ! -f "${DATA_DIR}/pool_plddt.npy" ]; then
    echo "ERROR: ${DATA_DIR}/pool_plddt.npy not found. Materialize first."
    exit 1
fi

pixi run --frozen torchrun --standalone --nproc_per_node="${NGPU}" \
  -m poet_2.training.train \
  --data-dir "${DATA_DIR}" \
  --checkpoint data/gitignore/models/poet-2.ckpt \
  --out-dir "${OUT_DIR}" \
  --dtype bf16 \
  --optimizer adamw \
  --peak-lr 1e-4 \
  --warmup-steps 200 \
  --schedule cosine \
  --tokens-per-gpu 45056 \
  --grad-accum 2 \
  --ddp-mode static_graph \
  --seq-mask-max 0.30 \
  --rate-cap 0.30 \
  --save-every 1000 --log-every 10 \
  --eval-every 500 \
  --eval-dms-dir data/evals \
  --wandb-project poet2-viral \
  --wandb-run struct-div-lr1e-4
