#!/bin/bash
# Mid-range LR sweep: explore the sweet spot between 1e-6 (no learning)
# and 5e-5 (IFQ degrades -0.029). All with mask=0.00 (CLM-only, best for seq_only)
# and all-mode eval tracking (seq_only + struct + IFQ).
#
# Also try the best IFQ-preserving config from Phase 1.5 (sdrop=0.2, lr5e-5).
#
# Sweep:
#   LR {1e-5, 2e-5, 3e-5} × sdrop=0.0    (3 runs - LR exploration)
#   LR 5e-5 × sdrop=0.2                    (1 run - best IFQ-preserving combo)
#
# Usage: bash slurm/d1_midlr_train.sh

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

DS_PATH="${REPO_DIR}/data/gitignore/materialized/d1_diversity_struct"
NGPU=4
GRAD_ACCUM=2
WARMUP=200
TOTAL_STEPS=10000
EVAL_EVERY=2000

declare -a RUNS=(
    "1e-5 0.0"
    "2e-5 0.0"
    "3e-5 0.0"
    "5e-5 0.2"
)

for entry in "${RUNS[@]}"; do
    read -r LR SDROP <<< "$entry"
    RUN_NAME="d1_mid-lr${LR}-sdrop${SDROP}"
    OUT_DIR="${REPO_DIR}/data/gitignore/checkpoints/${RUN_NAME}"

    SCRIPT=$(mktemp /tmp/d1_mid_XXXXXX.slurm)
    cat > "$SCRIPT" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name=mid-${RUN_NAME}
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${NGPU}
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=${REPO_DIR}/d1-mid-${RUN_NAME}-%j.out

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
  --seq-mask-max 0.00 \\
  --struct-dropout ${SDROP} \\
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

echo "All mid-LR jobs submitted."
