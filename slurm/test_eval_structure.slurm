#!/bin/bash
#SBATCH --job-name=eval-struct
#SBATCH --partition=kempner_h100_priority
#SBATCH --account=kempner_marks_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=eval-struct-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

echo "host: $(hostname)"
nvidia-smi -L

WT='MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW'

echo ""
echo "=== Eval WITHOUT WT structure (baseline) ==="
pixi run --frozen python -c "
import torch; from poet_2.models.poet_2_helpers import load_model
from poet_2.training.eval import evaluate

model = load_model('data/gitignore/models/poet-2.ckpt', device=torch.device('cuda'))
metrics, scores = evaluate(
    model,
    'data/BLAT_ECOLX_ColabFold_2202.a3m',
    'data/BLAT_ECOLX_Jacquier_2013_variants.fasta',
    wt_sequence='${WT}',
    labels_csv='data/BLAT_ECOLX_Jacquier_2013.csv',
    max_similarity=0.95,
)
print(f'No structure: {metrics}')
"

echo ""
echo "=== Eval WITH WT structure (inverse-folding query) ==="
pixi run --frozen python -c "
import torch; from poet_2.models.poet_2_helpers import load_model
from poet_2.training.eval import evaluate

model = load_model('data/gitignore/models/poet-2.ckpt', device=torch.device('cuda'))
metrics, scores = evaluate(
    model,
    'data/BLAT_ECOLX_ColabFold_2202.a3m',
    'data/BLAT_ECOLX_Jacquier_2013_variants.fasta',
    wt_sequence='${WT}',
    wt_structure_path='data/BLAT_ECOLX.pdb',
    labels_csv='data/BLAT_ECOLX_Jacquier_2013.csv',
    max_similarity=0.95,
)
print(f'With structure: {metrics}')
"

echo ""
echo "Eval structure test DONE"
