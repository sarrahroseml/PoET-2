#!/bin/bash
#SBATCH --job-name=find-ptxas2
#SBATCH --partition=kempner_h100_priority
#SBATCH --account=kempner_marks_lab
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=find-ptxas2-%j.out

cd "${SLURM_SUBMIT_DIR:-.}"

echo "=== nvidia-smi ==="
nvidia-smi | head -6

echo ""
echo "=== which ptxas ==="
which ptxas 2>/dev/null || echo "ptxas not in PATH"

echo ""
echo "=== module avail cuda ==="
module avail cuda 2>&1 | head -20 || true

echo ""
echo "=== find ptxas ==="
find /usr/local -name ptxas 2>/dev/null | head -5 || true
ls /usr/local/cuda*/bin/ptxas 2>/dev/null || true

echo ""
echo "=== Try loading cuda modules ==="
for v in cuda/12.6.1-fasrc01 cuda/12.4.0-fasrc01 cuda/12.2.0-fasrc01 cuda; do
    echo "Trying: module load $v"
    if module load $v 2>/dev/null; then
        which ptxas && ptxas --version
        break
    fi
done

echo ""
echo "=== Test with unset TRITON_PTXAS_PATH ==="
unset TRITON_PTXAS_PATH
pixi run --frozen python -c "
import os
print(f'TRITON_PTXAS_PATH = {os.environ.get(\"TRITON_PTXAS_PATH\", \"(unset)\")}')
import torch
print(f'torch {torch.__version__}, cuda {torch.cuda.is_available()}')
x = torch.randn(2, 4, device='cuda', dtype=torch.bfloat16)
# trigger Triton RMSNorm kernel compilation
import sys; sys.path.insert(0, 'src')
from poet_2.models.modules.norm import RMSNorm
rn = RMSNorm(4).cuda().bfloat16()
y = rn(x)
print(f'RMSNorm Triton kernel compiled OK: {y.shape}')
print('SUCCESS')
" 2>&1 || echo "FAILED with unset TRITON_PTXAS_PATH"
