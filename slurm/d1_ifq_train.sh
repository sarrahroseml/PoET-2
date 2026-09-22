#!/bin/bash
# IFQ-aware training: inserts masked-X target + structure as first context member
# and enables ref-value blending in the CLM decoder, exercising the IFQ pathway
# during training to prevent degradation.
#
# Sweep: ifq_p × struct_dropout × mask
# All runs track both seq_only AND IFQ spearman during training (skip ensemble).
#
# Usage: bash slurm/d1_ifq_train.sh

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

DS_PATH="${REPO_DIR}/data/gitignore/materialized/d1_diversity_struct"
NGPU=4
GRAD_ACCUM=2
WARMUP=200
TOTAL_STEPS=10000
EVAL_EVERY=2000

# Sweep grid:
#   ifq_p:          0.3, 0.5 (fraction of samples with IFQ training)
#   struct_dropout:  0.0, 0.2 (keep structures visible)
#   seq_mask_max:    0.00 (CLM-only for best seq_only)
#   LR:              1e-4

for IFQ_P in 0.3 0.5; do
    for SDROP in 0.0 0.2; do
        RUN_NAME="d1_ifqt-ifq${IFQ_P}-mask0.00-sdrop${SDROP}"
        OUT_DIR="${REPO_DIR}/data/gitignore/checkpoints/${RUN_NAME}"

        SCRIPT=$(mktemp /tmp/d1_ifqt_XXXXXX.slurm)
        cat > "$SCRIPT" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name=ifqt-${RUN_NAME}
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${NGPU}
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=${REPO_DIR}/d1-ifqt-${RUN_NAME}-%j.out

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
  --warmup-steps ${WARMUP} \\
  --schedule cosine \\
  --total-steps ${TOTAL_STEPS} \\
  --tokens-per-gpu 45056 \\
  --grad-accum ${GRAD_ACCUM} \\
  --seq-mask-max 0.00 \\
  --struct-dropout ${SDROP} \\
  --ifq-p ${IFQ_P} \\
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

echo "All IFQ-aware training jobs submitted."
