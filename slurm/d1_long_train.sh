#!/bin/bash
# Longer training runs on the best D1 config (diversity, mask=0.15, sdrop=0.5)
# with lower learning rates to see if more steps helps.
#
# Also includes an IFQ-preserving sweep: low struct_dropout to keep the model
# reliant on structure, which should preserve IFQ eval capability.
#
# Usage: bash slurm/d1_long_train.sh

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

DS_PATH="${REPO_DIR}/data/gitignore/materialized/d1_diversity_struct"
NGPU=4
GRAD_ACCUM=2
WARMUP=500

# ── Experiment A: Longer training (50k steps) on best config ──
# Best D1 config was lr=1e-4, mask=0.15, sdrop=0.5 at 10k steps.
# Try longer with lower LR.

for LR in 2e-5 1e-5; do
    RUN_NAME="d1_long-lr${LR}-mask0.15-sdrop0.5"
    OUT_DIR="${REPO_DIR}/data/gitignore/checkpoints/${RUN_NAME}"

    SCRIPT=$(mktemp /tmp/d1_long_XXXXXX.slurm)
    cat > "$SCRIPT" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name=d1l-${RUN_NAME}
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${NGPU}
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=${REPO_DIR}/d1-long-${RUN_NAME}-%j.out

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
  --total-steps 50000 \\
  --tokens-per-gpu 45056 \\
  --grad-accum ${GRAD_ACCUM} \\
  --seq-mask-max 0.15 \\
  --struct-dropout 0.5 \\
  --ddp-mode find_unused \\
  --save-every 5000 --log-every 10 \\
  --eval-every 5000 \\
  --eval-dms-dir data/evals \\
  --eval-seq-only \\
  --wandb-project poet2-d1-sweep \\
  --wandb-run ${RUN_NAME}
SLURM_EOF

    echo "Submitting: ${RUN_NAME} (50k steps)"
    sbatch "$SCRIPT"
    rm "$SCRIPT"
done

# ── Experiment B: IFQ-preserving training ──
# Low struct_dropout (0.0, 0.2) so the model keeps relying on structures.
# This should preserve the IFQ eval pathway.
# Also try mask=0 vs mask=0.05 (small MLM component helps IFQ-like decoding).

for SDROP in 0.0 0.2; do
    for MASK in 0.00 0.05; do
        RUN_NAME="d1_ifq-lr1e-4-mask${MASK}-sdrop${SDROP}"
        OUT_DIR="${REPO_DIR}/data/gitignore/checkpoints/${RUN_NAME}"

        SCRIPT=$(mktemp /tmp/d1_ifq_XXXXXX.slurm)
        cat > "$SCRIPT" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name=d1i-${RUN_NAME}
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${NGPU}
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=${REPO_DIR}/d1-ifq-${RUN_NAME}-%j.out

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
  --peak-lr 1e-4 \\
  --warmup-steps 200 \\
  --schedule cosine \\
  --total-steps 10000 \\
  --tokens-per-gpu 45056 \\
  --grad-accum ${GRAD_ACCUM} \\
  --seq-mask-max ${MASK} \\
  --struct-dropout ${SDROP} \\
  --ddp-mode find_unused \\
  --save-every 2000 --log-every 10 \\
  --eval-every 2000 \\
  --eval-dms-dir data/evals \\
  --eval-all-modes --eval-skip-ensemble \\
  --wandb-project poet2-d1-sweep \\
  --wandb-run ${RUN_NAME}
SLURM_EOF

        echo "Submitting: ${RUN_NAME} (10k steps, IFQ-preserving)"
        sbatch "$SCRIPT"
        rm "$SCRIPT"
    done
done

echo "All jobs submitted."
