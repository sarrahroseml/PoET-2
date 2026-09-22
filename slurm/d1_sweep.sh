#!/bin/bash
# Launch hyperparameter sweep for d1_logan training with structures.
# Usage: bash slurm/d1_sweep.sh [--dep JOB_ID]
#   --dep: optional SLURM dependency (e.g. d1_prep job ID)
#
# Sweep grid (16 configs):
#   dataset:        diversity, weighted
#   lr:             1e-4, 5e-5
#   seq_mask_max:   0.00, 0.15
#   struct_dropout: 0.0, 0.5

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

DEP=""
if [[ "${1:-}" == "--dep" ]]; then
    DEP="--dependency=afterok:$2"
fi

DATASETS=(
    "d1_div:${REPO_DIR}/data/gitignore/materialized/d1_diversity_struct"
    "d1_wgt:${REPO_DIR}/data/gitignore/materialized/d1_weighted_struct"
)
LRS=(1e-4 5e-5)
MASKS=(0.00 0.15)
STRUCT_DROPS=(0.0 0.5)

TOTAL_STEPS=10000
EVAL_EVERY=2000
SAVE_EVERY=2000
WARMUP=200
NGPU=4
GRAD_ACCUM=2

for ds_entry in "${DATASETS[@]}"; do
    DS_NAME="${ds_entry%%:*}"
    DS_PATH="${ds_entry##*:}"
    for LR in "${LRS[@]}"; do
        for MASK in "${MASKS[@]}"; do
            for SDROP in "${STRUCT_DROPS[@]}"; do
                RUN_NAME="${DS_NAME}-lr${LR}-mask${MASK}-sdrop${SDROP}"
                OUT_DIR="${REPO_DIR}/data/gitignore/checkpoints/${RUN_NAME}"

                # Write a temporary SLURM script
                SCRIPT=$(mktemp /tmp/d1_train_XXXXXX.slurm)
                cat > "$SCRIPT" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name=d1-${RUN_NAME}
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${NGPU}
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=${REPO_DIR}/d1-train-${RUN_NAME}-%j.out

set -euo pipefail
cd "${REPO_DIR}"
echo "host: \$(hostname)  run: ${RUN_NAME}"
nvidia-smi -L || true

pixi run --frozen torchrun --standalone --nproc_per_node=${NGPU} \\
  -m poet_2.training.train \\
  --data-dir "${DS_PATH}" \\
  --checkpoint data/gitignore/models/poet-2.ckpt \\
  --out-dir "${OUT_DIR}" \\
  --dtype bf16 \\
  --optimizer adamw \\
  --peak-lr ${LR} \\
  --warmup-steps ${WARMUP} \\
  --schedule cosine \\
  --total-steps ${TOTAL_STEPS} \\
  --tokens-per-gpu 45056 \\
  --grad-accum ${GRAD_ACCUM} \\
  --seq-mask-max ${MASK} \\
  --struct-dropout ${SDROP} \\
  --ddp-mode find_unused \\
  --save-every ${SAVE_EVERY} --log-every 10 \\
  --eval-every ${EVAL_EVERY} \\
  --eval-dms-dir data/evals \\
  --eval-seq-only \\
  --wandb-project poet2-d1-sweep \\
  --wandb-run ${RUN_NAME}
SLURM_EOF

                echo "Submitting: ${RUN_NAME}"
                sbatch ${DEP} "$SCRIPT"
                rm "$SCRIPT"
            done
        done
    done
done

echo "All sweep jobs submitted."
