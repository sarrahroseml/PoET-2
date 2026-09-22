#!/bin/bash
# Score pretrained PoET (v1) on all 45 viral DMS eval datasets.
# Submit: sbatch slurm/eval_poet1_baseline.slurm
# Debug:  sbatch slurm/eval_poet1_baseline.slurm --debug

#SBATCH --job-name=poet1-eval
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=eval-poet1-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

echo "host: $(hostname)"
nvidia-smi -L || true

# Download PoET checkpoint if missing
CKPT="data/poet.ckpt"
if [ ! -f "$CKPT" ]; then
    echo "Downloading PoET checkpoint..."
    mkdir -p data
    wget -q -c "https://zenodo.org/records/10061322/files/poet.ckpt?download=1" -O "$CKPT"
    echo "Downloaded to $CKPT"
fi

pixi run --frozen python scripts/eval_poet1_baseline.py \
    --ckpt_path "$CKPT" \
    --dms_dir data/evals \
    --output_dir data/gitignore/outputs/poet1_baseline \
    --batch_size 8 \
    "$@"
