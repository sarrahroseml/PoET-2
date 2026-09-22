#!/bin/bash
# Build Protenix CUDA extensions on a GPU node (one-time setup).
# Submit: sbatch slurm/build_protenix_ext.slurm

#SBATCH --job-name=protenix-build
#SBATCH --account=kempner_marks_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=protenix-build-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

module load python/3.12.5-fasrc01
module load cuda/12.4.1-fasrc01
module load gcc/12.2.0-fasrc01
source data/protenix_env/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"

echo "host: $(hostname)  gpu: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "CUDA_HOME: $CUDA_HOME"
echo "Python: $(python3 --version)"
echo "torch cuda: $(python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"

echo ""
echo "=== Building CUDA extensions via import ==="
python3 -c "
from protenix.model.layer_norm.layer_norm import FusedLayerNorm
print('FusedLayerNorm imported OK')

# Check where the .so landed
import os, glob
pkg_dir = os.path.dirname(os.path.abspath(FusedLayerNorm.__module__.replace('.', '/') + '.py')) if False else None
layer_norm_dir = os.path.join(os.path.dirname(__import__('protenix.model.layer_norm.layer_norm', fromlist=['']).__file__))
sos = glob.glob(os.path.join(layer_norm_dir, '*.so'))
print('Package dir:', layer_norm_dir)
print('.so files:', sos)

# Also check torch cache
import torch.utils.cpp_extension as ext
cache = ext._get_build_directory('fast_layer_norm_cuda_v2', False)
print('Torch cache dir:', cache)
cache_sos = glob.glob(os.path.join(cache, '*.so'))
print('Cache .so files:', cache_sos)
"

echo ""
echo "=== Quick smoke test: run protenix on a tiny input ==="
# Create a minimal test input
python3 -c "
import json, tempfile, os
test = [{'name': 'test_build', 'covalent_bonds': [], 'sequences': [{'proteinChain': {'count': 1, 'sequence': 'MKTAYIAKQRQISFVK'}}]}]
path = 'data/protenix_jobs/test_build.json'
with open(path, 'w') as f:
    json.dump(test, f)
print('Test input:', path)
"

protenix pred \
    -i data/protenix_jobs/test_build.json \
    -o data/protenix_jobs/test_build_out \
    -n protenix_base_default_v1.0.0 \
    --use_default_params true \
    -s 101 2>&1 | tail -20

echo ""
echo "Build complete at $(date)"
