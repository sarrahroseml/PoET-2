#!/bin/bash
# Extract and index the UniRef30 database after download completes.
# Run after uniref30_2302.tar.gz has been downloaded to data/msa_db/.
#
# Submit: sbatch slurm/setup_colabfold_db.slurm

#SBATCH --job-name=uniref30-setup
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --output=uniref30-setup-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

bash scripts/setup_uniref30.sh data/msa_db
