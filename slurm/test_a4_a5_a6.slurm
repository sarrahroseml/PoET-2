#!/bin/bash
#SBATCH --job-name=poet2-a4a5a6
#SBATCH --partition=kempner_h100_priority
#SBATCH --account=kempner_marks_lab
#SBATCH --qos=kemp_gpu16_id32
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=poet2-a4a5a6-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

echo "host: $(hostname)"
nvidia-smi -L

echo ""
echo "=== A4: Overfit-One-Batch (loss should fall toward ~0) ==="
pixi run --frozen python -m poet_2.training.train \
  --data-dir data/gitignore/smoke_dataset \
  --checkpoint data/gitignore/models/poet-2.ckpt \
  --overfit-one-batch --total-steps 100 \
  --tokens-per-gpu 8192 --log-every 5 --save-every 0

echo ""
echo "=== A5: Online Eval Hook (scoring path + Spearman) ==="
pixi run --frozen python -m poet_2.training.train \
  --data-dir data/gitignore/smoke_dataset \
  --checkpoint data/gitignore/models/poet-2.ckpt \
  --total-steps 10 --tokens-per-gpu 8192 --log-every 1 --save-every 0 \
  --eval-every 5 \
  --eval-a3m data/BLAT_ECOLX_ColabFold_2202.a3m \
  --eval-variants data/BLAT_ECOLX_Jacquier_2013_variants.fasta \
  --eval-wt 'MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW' \
  --eval-labels data/BLAT_ECOLX_Jacquier_2013.csv \
  --eval-max-similarity 0.95

echo ""
echo "=== A6: Checkpoint Round-Trip (save + reload + score) ==="
pixi run --frozen python -m poet_2.training.train \
  --data-dir data/gitignore/smoke_dataset \
  --checkpoint data/gitignore/models/poet-2.ckpt \
  --total-steps 20 --tokens-per-gpu 8192 --save-every 10

echo "--- Scoring with saved checkpoint via score.py ---"
pixi run --frozen python scripts/score.py \
  --checkpoint data/gitignore/checkpoints/final.ckpt \
  --wt_sequence 'MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW' \
  --msa_a3m_path data/BLAT_ECOLX_ColabFold_2202.a3m \
  --variants_fasta_path data/BLAT_ECOLX_Jacquier_2013_variants.fasta \
  --output_npy_path data/gitignore/outputs/roundtrip_test.npy \
  --context_length 6144 \
  --max_similarity 0.95 \
  --batch_size 32

echo ""
echo "All A4-A6 tests complete."
