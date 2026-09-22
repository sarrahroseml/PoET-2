#!/bin/bash
#SBATCH --job-name=fix-ptxas
#SBATCH --partition=kempner_h100_priority
#SBATCH --account=kempner_marks_lab
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=fix-ptxas-%j.out

cd "${SLURM_SUBMIT_DIR:-.}"

# Load CUDA 12.9 to get an sm_90a-capable ptxas
module load cuda/12.9.1-fasrc01
SYSTEM_PTXAS=$(which ptxas)
echo "System ptxas: $SYSTEM_PTXAS"
ptxas --version

echo ""
echo "=== Test 1: override via pixi run bash -c ==="
pixi run --frozen bash -c "export TRITON_PTXAS_PATH=$SYSTEM_PTXAS && python -c \"
import os
print(f'TRITON_PTXAS_PATH = {os.environ.get(\\\"TRITON_PTXAS_PATH\\\", \\\"(unset)\\\")}')
import torch
print(f'torch {torch.__version__}, cuda {torch.cuda.is_available()}')
x = torch.randn(2, 4, device='cuda', dtype=torch.bfloat16)
import sys; sys.path.insert(0, 'src')
from poet_2.models.modules.norm import RMSNorm
rn = RMSNorm(4).cuda().bfloat16()
y = rn(x)
print(f'RMSNorm Triton kernel compiled OK: {y.shape}')
print('SUCCESS')
\"" 2>&1
