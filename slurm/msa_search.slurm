#!/bin/bash
# Run colabfold_search locally against UniRef30 for MSA generation.
# Array job — one task per FASTA batch (1000 seqs each).
#
# Prerequisites:
#   1. Set up UniRef30: sbatch slurm/setup_colabfold_db.slurm
#   2. Split FASTA: python scripts/split_fasta_batches.py --fasta data/all_seqs.fasta --out-dir data/msa_jobs/inputs --batch-size 1000
#   3. Submit: sbatch slurm/msa_search.slurm

#SBATCH --job-name=msa-search
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --array=0-210%16
#SBATCH --output=data/msa_jobs/logs/batch_%a.out
#SBATCH --error=data/msa_jobs/logs/batch_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

module load python/3.12.5-fasrc01
source data/protenix_env/bin/activate

export PATH="$(pwd)/tools/mmseqs/bin:$PATH"

BATCH_ID=$SLURM_ARRAY_TASK_ID
INPUT_FASTA="data/msa_jobs/inputs/batch_${BATCH_ID}.fasta"
OUTPUT_DIR="data/msa_jobs/outputs/batch_${BATCH_ID}"
DB_DIR="data/msa_db"

if [ ! -f "$INPUT_FASTA" ]; then
    echo "No input file for batch $BATCH_ID — skipping"
    exit 0
fi

echo "host: $(hostname)"
echo "Processing batch $BATCH_ID: $INPUT_FASTA ($(grep -c '^>' "$INPUT_FASTA") sequences)"
echo "Start: $(date)"
mkdir -p "$OUTPUT_DIR"

colabfold_search \
    "$INPUT_FASTA" \
    "$DB_DIR" \
    "$OUTPUT_DIR" \
    --db1 uniref30_2302_db \
    --use-env 0 \
    --use-templates 0 \
    --mmseqs "$(pwd)/tools/mmseqs/bin/mmseqs" \
    --threads "$SLURM_CPUS_PER_TASK" \
    --db-load-mode 2

echo "Done batch $BATCH_ID at $(date)"
echo "Results:"
ls "$OUTPUT_DIR"/*.a3m 2>/dev/null | wc -l
echo "a3m files generated"
