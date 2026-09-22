#!/bin/bash
# Similarity-filtered training: train on cluster-representative-only datasets
# to reduce redundancy. Tests 95% and 90% identity thresholds with LRs 2e-5 and 5e-5.
#
# Sweep:
#   sim{0.95, 0.90} × LR{2e-5, 5e-5}  (4 runs)
#
# Usage: bash slurm/d1_simfilter_train.sh

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

NGPU=4
GRAD_ACCUM=2
WARMUP=200
TOTAL_STEPS=10000
EVAL_EVERY=2000

declare -a RUNS=(
    "sim095 2e-5"
    "sim095 5e-5"
    "sim090 2e-5"
    "sim090 5e-5"
)

for entry in "${RUNS[@]}"; do
    read -r SIM LR <<< "$entry"
    DS_PATH="${REPO_DIR}/data/gitignore/materialized/d1_diversity_struct_${SIM}"
    RUN_NAME="d1_${SIM}-lr${LR}-sdrop0.0"
    OUT_DIR="${REPO_DIR}/data/gitignore/checkpoints/${RUN_NAME}"

    if [ ! -d "$DS_PATH" ]; then
        echo "SKIP: dataset not found: $DS_PATH"
        continue
    fi

    SCRIPT=$(mktemp /tmp/d1_sim_XXXXXX.slurm)
    cat > "$SCRIPT" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name=${RUN_NAME}
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${NGPU}
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=${REPO_DIR}/d1-${SIM}-${RUN_NAME}-%j.out

set -euo pipefail
cd "${REPO_DIR}"
echo "host: \$(hostname)  run: ${RUN_NAME}  dataset: ${DS_PATH}"
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
  --seq-mask-max 0.00 \\
  --struct-dropout 0.0 \\
  --ddp-mode find_unused \\
  --save-every 2000 --log-every 10 \\
  --eval-every ${EVAL_EVERY} \\
  --eval-dms-dir data/evals \\
  --eval-all-modes --eval-skip-ensemble \\
  --wandb-project poet2-d1-sweep \\
  --wandb-run ${RUN_NAME}
SLURM_EOF

    echo "Submitting: ${RUN_NAME}"
    sbatch "$SCRIPT"
    rm "$SCRIPT"
done

echo "All sim-filter jobs submitted."
