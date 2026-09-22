#!/bin/bash
# Hyperparameter sweep for structure-conditioned training on diversity dataset.
# Array job: each task runs a different (LR, mask_rate) combination.
# Submit: sbatch slurm/sweep_struct_div.slurm

#SBATCH --job-name=sweep-sdiv
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-8
#SBATCH --output=sweep-sdiv-%a-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

DATA_DIR=data/gitignore/materialized/presampled_diversity_struct

# Sweep grid:
# LR: 3e-5, 5e-5, 1e-4
# Mask: 0.00, 0.15, 0.30
# Total: 9 combinations
LRS=(3e-5 5e-5 1e-4 3e-5 5e-5 1e-4 3e-5 5e-5 1e-4)
MASKS=(0.00 0.00 0.00 0.15 0.15 0.15 0.30 0.30 0.30)

IDX=$SLURM_ARRAY_TASK_ID
LR=${LRS[$IDX]}
MASK=${MASKS[$IDX]}

# Convert mask to string for dir name (replace . with p)
MASK_STR=$(echo $MASK | tr '.' 'p')
RUN_NAME="sdiv-lr${LR}-mask${MASK_STR}"
OUT_DIR="data/gitignore/checkpoints/${RUN_NAME}"

NGPU=${SLURM_GPUS_ON_NODE:-4}

echo "host: $(hostname)  ngpu: ${NGPU}"
echo "Run: ${RUN_NAME}  LR=${LR}  mask=${MASK}"
nvidia-smi -L || true

if [ ! -f "${DATA_DIR}/pool_plddt.npy" ]; then
    echo "ERROR: ${DATA_DIR}/pool_plddt.npy not found."
    exit 1
fi

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
  --wandb-run "${RUN_NAME}"
