#!/bin/bash
# LoRA fine-tuning sweep + ultra-low LR baselines.
#
# Freeze all pretrained weights, train only LoRA adapters on encoder Q/V
# projections. The pretrained IFQ pathway stays intact since decoder base
# weights are frozen and LoRA only modifies the encoder's forward path.
#
# Sweep:
#   LoRA:    rank {8, 16} × LR {5e-4, 1e-3}     (4 runs)
#   Low-LR:  full fine-tune LR {1e-6, 5e-6}      (2 runs)
#
# All runs: mask=0.00 (CLM only), struct_dropout=0.0, 10k steps
#
# Usage: bash slurm/d1_lora_train.sh

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

DS_PATH="${REPO_DIR}/data/gitignore/materialized/d1_diversity_struct"
NGPU=4
GRAD_ACCUM=2
WARMUP=200
TOTAL_STEPS=10000
EVAL_EVERY=2000

# --- LoRA runs ---
for RANK in 8 16; do
    for LR in 5e-4 1e-3; do
        RUN_NAME="d1_lora-r${RANK}-lr${LR}"
        OUT_DIR="${REPO_DIR}/data/gitignore/checkpoints/${RUN_NAME}"

        SCRIPT=$(mktemp /tmp/d1_lora_XXXXXX.slurm)
        cat > "$SCRIPT" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name=lora-${RUN_NAME}
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${NGPU}
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=${REPO_DIR}/d1-lora-${RUN_NAME}-%j.out

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
  --struct-dropout 0.0 \\
  --lora-rank ${RANK} \\
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
done

# --- Ultra-low LR full fine-tune baselines ---
for LR in 1e-6 5e-6; do
    RUN_NAME="d1_lowlr-lr${LR}"
    OUT_DIR="${REPO_DIR}/data/gitignore/checkpoints/${RUN_NAME}"

    SCRIPT=$(mktemp /tmp/d1_lowlr_XXXXXX.slurm)
    cat > "$SCRIPT" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name=lowlr-${RUN_NAME}
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${NGPU}
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=${REPO_DIR}/d1-lowlr-${RUN_NAME}-%j.out

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

echo "All LoRA + low-LR jobs submitted."
