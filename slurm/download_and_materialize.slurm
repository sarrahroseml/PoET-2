#!/bin/bash
# Download presampled TSVs from GCS and materialize into PoET2Dataset format.
# Run on a CPU node (no GPU needed). Submit: sbatch slurm/download_and_materialize.slurm

#SBATCH --job-name=materialize
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=materialize-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

GCLOUD="$HOME/google-cloud-sdk/bin/gcloud"
GCS_BUCKET="gs://viralplm/parsed/jackhmmer_foldseek_combined"
FASTA="data/all_seqs.fasta"
DATA_DIR="data/gitignore/materialized"

mkdir -p data "${DATA_DIR}"

# Download TSVs if not already present
for NAME in presampled_diversity presampled_weighted; do
    TSV="data/${NAME}.tsv"
    if [ ! -f "${TSV}" ]; then
        echo "Downloading ${NAME}.tsv ..."
        "${GCLOUD}" storage cp "${GCS_BUCKET}/${NAME}.tsv" "${TSV}"
    else
        echo "${TSV} already exists, skipping download"
    fi
done

# Materialize each dataset
for NAME in presampled_diversity presampled_weighted; do
    TSV="data/${NAME}.tsv"
    OUT="${DATA_DIR}/${NAME}"
    if [ -f "${OUT}/meta.json" ]; then
        echo "${OUT} already materialized, skipping"
        continue
    fi
    echo "Materializing ${NAME} ..."
    pixi run python scripts/materialize_presampled.py \
        --fasta "${FASTA}" \
        --tsv "${TSV}" \
        --out "${OUT}"
done

echo "Done. Materialized datasets:"
for NAME in presampled_diversity presampled_weighted; do
    cat "${DATA_DIR}/${NAME}/meta.json" 2>/dev/null || echo "  ${NAME}: not found"
done
