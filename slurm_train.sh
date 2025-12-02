#!/bin/bash
#SBATCH --job-name=poet2-train
#SBATCH --output=logs/poet2-train-%j.out
#SBATCH --error=logs/poet2-train-%j.err
#SBATCH --partition=gpu_quad
#SBATCH --reservation=marks_lab
#SBATCH --gres=gpu:2                 
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00             

# Example submission:
#sbatch -p gpu_quad --reservation=marks_lab poet2_train.sbatch
# Make sure pixi is on PATH (adjust if your pixi is elsewhere)
export PATH="$HOME/.pixi/bin:$PATH"

# Ensure environment is installed 
pixi install --frozen

# Environment variables that were set inside the Modal container
export POET2_DATA_ROOT="/n/groups/marks/projects/viral_plm/models/PoET-2/data/train_set"        
export POET2_CHECKPOINT="/n/groups/marks/projects/viral_plm/models/PoET-2/data/gitignore/models/poet-2.ckpt"          
export POET2_DEVICE="cuda"

export POET2_WANDB_PROJECT="poet2"
export POET2_WANDB_RUN_NAME="slurm-run-test-final"                        
export SAVE_DIR="/n/groups/marks/projects/viral_plm/models/PoET-2/data/saved_checkpoints/more_use_checkpoints"     

export WANDB_API_KEY="79342dee142e65a38d0f7de963d5090c7f79f93b"

echo "Starting training at $(date)"
echo "Running: pixi run python -m scripts.train.train"

# Use srun to bind the job to the allocated GPU/CPUs
srun pixi run python -m scripts.train.train

echo "Finished at $(date)"
