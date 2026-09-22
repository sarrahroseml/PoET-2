#!/bin/bash
#SBATCH --job-name=eval-af2
#SBATCH --partition=kempner_h100_priority
#SBATCH --account=kempner_marks_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=eval-af2-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

echo "host: $(hostname)"
nvidia-smi -L

WT='MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW'

echo "=== Testing all four eval modes ==="
pixi run --frozen python -c "
import torch
from poet_2.models.poet_2_helpers import load_model
from poet_2.training.eval import evaluate

model = load_model('data/gitignore/models/poet-2.ckpt', device=torch.device('cuda'))
metrics, scores = evaluate(
    model,
    'data/BLAT_ECOLX_ColabFold_2202.a3m',
    'data/BLAT_ECOLX_Jacquier_2013_variants.fasta',
    wt_sequence='${WT}',
    wt_structure_path='data/BLAT_ECOLX.pdb',
    af2_cache_folder='data/gitignore/cache/AF2',
    labels_csv='data/BLAT_ECOLX_Jacquier_2013.csv',
    max_similarity=0.95,
)
print(f'Full metrics: {metrics}')
print()
print(f'  spearman (seq only):            {metrics.get(\"spearman\", \"N/A\")}')
print(f'  spearman_struct (WT struct ctx): {metrics.get(\"spearman_struct\", \"N/A\")}')
print(f'  spearman_ifq (inverse folding):  {metrics.get(\"spearman_ifq\", \"N/A\")}')
print(f'  spearman_af2 (AF2 + WT struct):  {metrics.get(\"spearman_af2\", \"N/A\")}')
print(f'  n_af2_structures:                {metrics.get(\"n_af2_structures\", \"N/A\")}')
"

echo ""
echo "Eval AF2 test DONE"
