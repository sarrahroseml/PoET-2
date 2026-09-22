#!/bin/bash
#SBATCH --job-name=ifq-sweep
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=ifq-sweep-%j.out

set -euo pipefail
cd /n/netscratch/marks_lab/Lab/nyoussef/final_poet/PoET-2
echo "host: $(hostname)"
nvidia-smi -L || true

pixi run --frozen python3 scripts/eval_ifq_sweep.py
