#!/bin/bash
#SBATCH --job-name=eval-viral
#SBATCH --partition=kempner_h100_priority
#SBATCH --account=kempner_marks_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=eval-viral-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

echo "host: $(hostname)"

echo "=== Testing eval on IAV_H1_HA_Doud ==="
pixi run --frozen python -c "
import torch
from poet_2.models.poet_2_helpers import load_model
from poet_2.training.eval import evaluate, _read, ungapped

# Extract WT from alignment (first sequence, ungapped)
_, rows = _read('data/evals/alignments/IAV_H1_HA_Doud.a2m', upper=False)
wt_seq = ungapped(rows[0]).decode()
print(f'WT length: {len(wt_seq)}')

model = load_model('data/gitignore/models/poet-2.ckpt', device=torch.device('cuda'))
metrics, scores = evaluate(
    model,
    'data/evals/alignments/IAV_H1_HA_Doud.a2m',
    'data/evals/viral_dms_substitutions/IAV_H1_HA_Doud_dms.csv',
    wt_sequence=wt_seq,
    wt_structure_path='data/evals/viral_dms_structures/IAV_H1_HA_Doud.pdb',
    labels_csv='data/evals/viral_dms_substitutions/IAV_H1_HA_Doud_dms.csv',
    max_similarity=0.95,
)
print(f'Results: {metrics}')
print()
print(f'  spearman (seq only):            {metrics.get(\"spearman\", \"N/A\")}')
print(f'  spearman_struct (WT struct ctx): {metrics.get(\"spearman_struct\", \"N/A\")}')
print(f'  spearman_ifq (inverse folding):  {metrics.get(\"spearman_ifq\", \"N/A\")}')
print(f'  spearman_af2 (AF2 + WT struct):  {metrics.get(\"spearman_af2\", \"N/A\")}')
print(f'  n_af2_structures:                {metrics.get(\"n_af2_structures\", \"N/A\")}')
"

echo ""
echo "Eval viral test DONE"
