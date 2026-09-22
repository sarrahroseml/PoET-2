#!/bin/bash
#SBATCH --job-name=poet2-a1a2
#SBATCH --partition=kempner_h100_priority
#SBATCH --account=kempner_marks_lab
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=poet2-a1a2-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

echo "=== A1: CUDA Probe ==="
echo "host: $(hostname)"
nvidia-smi -L || echo "WARNING: nvidia-smi failed"
pixi run --frozen python -c "
import torch
print(f'torch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Capability: {torch.cuda.get_device_capability(0)}')
    x = torch.randn(2, 2, device='cuda')
    print(f'GPU tensor test: {x.sum().item():.4f}')
    print('A1 PASSED')
else:
    print('A1 FAILED: No CUDA')
    exit(1)
"

echo ""
echo "=== A2: M2 Equivalence ==="
pixi run --frozen python scripts/check_m2.py
