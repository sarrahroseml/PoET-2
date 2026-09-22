#!/bin/bash
# Fold all ~210K sequences with Protenix v2 using local MSA.
# SLURM array job — one GPU per batch of 50 sequences.
#
# Prerequisites:
#   1. MSA search complete: data/msa_jobs/outputs/ has .a3m files
#   2. Inject MSA paths: python scripts/inject_msa_paths.py --msa-dir data/msa_jobs/outputs --protenix-dir data/protenix_jobs/inputs --out-dir data/protenix_jobs/inputs_msa
#   3. Submit: sbatch slurm/protenix_fold.slurm

#SBATCH --job-name=protenix-fold
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=0-1247%12
#SBATCH --output=data/protenix_jobs/logs/batch_%a.out
#SBATCH --error=data/protenix_jobs/logs/batch_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

module load python/3.12.5-fasrc01
module load cuda/12.4.1-fasrc01
module load gcc/12.2.0-fasrc01
source data/protenix_env/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"
export PROTENIX_ROOT_DIR="/n/netscratch/marks_lab/Lab/nyoussef/final_poet/PoET-2/data/protenix_cache"

BATCH_ID=$SLURM_ARRAY_TASK_ID
INPUT_JSON="data/protenix_jobs/inputs_msa_rest_prioritized/batch_${BATCH_ID}.json"
OUTPUT_DIR="data/protenix_jobs/outputs/batch_${BATCH_ID}"

if [ ! -f "$INPUT_JSON" ]; then
    echo "No input file for batch $BATCH_ID — skipping"
    exit 0
fi

echo "host: $(hostname)  gpu: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "Processing batch $BATCH_ID: $INPUT_JSON"
echo "Start: $(date)"
mkdir -p "$OUTPUT_DIR"

protenix pred \
    -i "$INPUT_JSON" \
    -o "$OUTPUT_DIR" \
    -n protenix_base_default_v1.0.0 \
    --use_default_params true \
    --use_msa True \
    -s 101

echo "Done batch $BATCH_ID at $(date)"
