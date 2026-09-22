#!/bin/bash
#SBATCH --job-name=find-ptxas
#SBATCH --partition=kempner_h100_priority
#SBATCH --account=kempner_marks_lab
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=find-ptxas-%j.out

set -euo pipefail

echo "=== System CUDA ==="
nvidia-smi | head -4
which ptxas 2>/dev/null && ptxas --version || echo "ptxas not in PATH"

echo ""
echo "=== Module system ==="
module avail cuda 2>&1 | head -20

echo ""
echo "=== Find ptxas on system ==="
find /usr/local -name ptxas 2>/dev/null | head -5
find /opt -name ptxas 2>/dev/null | head -5
find /n/sw -name ptxas 2>/dev/null | head -5
ls /usr/local/cuda*/bin/ptxas 2>/dev/null

echo ""
echo "=== Try module load ==="
module load cuda/12.4.0-fasrc01 2>/dev/null && which ptxas && ptxas --version || \
module load cuda/12.2.0-fasrc01 2>/dev/null && which ptxas && ptxas --version || \
module load cuda 2>/dev/null && which ptxas && ptxas --version || \
echo "No cuda module found"

echo ""
echo "=== Test unset TRITON_PTXAS_PATH ==="
unset TRITON_PTXAS_PATH
pixi run --frozen python -c "
import os
print(f'TRITON_PTXAS_PATH = {os.environ.get(\"TRITON_PTXAS_PATH\", \"(unset)\")}')
import torch
x = torch.randn(2, 2, device='cuda', dtype=torch.bfloat16)
# trigger triton compilation
from poet_2.models.modules.norm import RMSNorm
rn = RMSNorm(4).cuda().bfloat16()
y = rn(torch.randn(1, 4, device='cuda', dtype=torch.bfloat16))
print(f'RMSNorm output: {y.shape} - Triton compiled OK')
"
