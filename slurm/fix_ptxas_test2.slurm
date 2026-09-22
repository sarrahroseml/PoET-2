#!/bin/bash
#SBATCH --job-name=fix-ptxas2
#SBATCH --partition=kempner_h100_priority
#SBATCH --account=kempner_marks_lab
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=fix-ptxas2-%j.out

cd "${SLURM_SUBMIT_DIR:-.}"

# Test with cuda 12.4 ptxas (don't load full module — just point to the ptxas binary)
PTXAS_124=/n/sw/helmod-rocky8/apps/Core/cuda/12.4.1-fasrc01/cuda/bin/ptxas
echo "Using ptxas: $PTXAS_124"
$PTXAS_124 --version 2>&1 || echo "ptxas 12.4 not found, trying 12.2"

# Fallback to 12.2
PTXAS_122=/n/sw/helmod-rocky8/apps/Core/cuda/12.2.0-fasrc01/cuda/bin/ptxas
echo "Fallback ptxas: $PTXAS_122"
$PTXAS_122 --version 2>&1

# Test: sm_90a support check
echo ""
echo "=== sm_90a support check ==="
echo ".version 7.0" > /tmp/test_sm90.ptx
echo ".target sm_90a" >> /tmp/test_sm90.ptx
echo ".address_size 64" >> /tmp/test_sm90.ptx
$PTXAS_124 --gpu-name=sm_90a /tmp/test_sm90.ptx -o /dev/null 2>&1 && echo "12.4 ptxas supports sm_90a" || echo "12.4 ptxas does NOT support sm_90a"
$PTXAS_122 --gpu-name=sm_90a /tmp/test_sm90.ptx -o /dev/null 2>&1 && echo "12.2 ptxas supports sm_90a" || echo "12.2 ptxas does NOT support sm_90a"

echo ""
echo "=== Test with TRITON_PTXAS_PATH override (no module load) ==="
# Only set TRITON_PTXAS_PATH inside pixi run, don't load any cuda module
pixi run --frozen bash -c "
export TRITON_PTXAS_PATH=$PTXAS_124
python -c \"
import os
print(f'TRITON_PTXAS_PATH = {os.environ.get(\\\"TRITON_PTXAS_PATH\\\", \\\"(unset)\\\")}')
import torch
print(f'torch {torch.__version__}, cuda ver={torch.version.cuda}')
x = torch.randn(2, 4, device='cuda', dtype=torch.bfloat16)
import sys; sys.path.insert(0, 'src')
from poet_2.models.modules.norm import RMSNorm
rn = RMSNorm(4).cuda().bfloat16()
y = rn(x)
print(f'RMSNorm Triton kernel compiled OK: {y.shape}')
print('SUCCESS')
\"
" 2>&1
