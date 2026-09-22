#!/bin/bash
# Fold remaining sequences on gpu_requeue (preemptible) partition.
# Companion to protenix_fold.slurm — handles the second half of batches.
# Preempted jobs are automatically requeued by SLURM.

#SBATCH --job-name=protenix-rq
#SBATCH --account=marks_lab
#SBATCH --partition=gpu_requeue
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --array=751-1501%32
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
INPUT_JSON="data/protenix_jobs/inputs_msa_rest/batch_${BATCH_ID}.json"
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
