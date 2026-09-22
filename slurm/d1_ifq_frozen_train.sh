#!/bin/bash
# IFQ-objective training on a FROZEN encoder (experiment #3).
#
# Every past ifq_p run had TIED weights, so IFQ training also corrupted the
# encoder that IFQ reads from -> self-defeating (d1_ifq / d1_ifqt all hurt IFQ).
# Here we untie + freeze the encoder, then train ONLY the decoders with high
# ifq_p. The decoder gets more practice at the structure->sequence (inverse
# folding) readout while the encoder's structure representation stays fixed.
#
# struct_dropout=0.0 so structures are always present (IFQ needs them).
# pLDDT diagnostic: 58% of training residues are >=70 confidence -> sensible.
#
# Sweep:
#   ifq_p 0.5, lr 1e-5  : half of samples exercise IFQ readout, safe LR
#   ifq_p 1.0, lr 1e-5  : every sample is IFQ-style, safe LR
#   ifq_p 1.0, lr 5e-5  : does frozen encoder make higher LR safe for IFQ?
#
# Usage: bash slurm/d1_ifq_frozen_train.sh

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

DS_PATH="${REPO_DIR}/data/gitignore/materialized/d1_diversity_struct"
NGPU=4
GRAD_ACCUM=2
WARMUP=200
TOTAL_STEPS=10000
EVAL_EVERY=2000

declare -a RUNS=(
    "0.5 1e-5"
    "1.0 1e-5"
    "1.0 5e-5"
)

for entry in "${RUNS[@]}"; do
    read -r IFQP LR <<< "$entry"
    RUN_NAME="d1_ifqfz-ifq${IFQP}-lr${LR}"
    OUT_DIR="${REPO_DIR}/data/gitignore/checkpoints/${RUN_NAME}"

    SCRIPT=$(mktemp /tmp/d1_ifqfz_XXXXXX.slurm)
    cat > "$SCRIPT" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name=ifqfz-${RUN_NAME}
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${NGPU}
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=${REPO_DIR}/d1-ifqfz-${RUN_NAME}-%j.out

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
  --ifq-p ${IFQP} \\
  --freeze-encoder \\
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

echo "All IFQ-frozen-encoder jobs submitted."
