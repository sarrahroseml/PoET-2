#!/bin/bash
#SBATCH --job-name=poet2-a7
#SBATCH --partition=kempner_h100_priority
#SBATCH --account=kempner_marks_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=poet2-a7-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

NGPU=${SLURM_GPUS_ON_NODE:-4}
echo "host: $(hostname)  ngpu: ${NGPU}"
nvidia-smi -L

echo ""
echo "=== A7: Multi-GPU DDP Training ==="
pixi run --frozen torchrun --standalone --nproc_per_node="${NGPU}" \
  -m poet_2.training.train \
  --data-dir data/gitignore/smoke_dataset \
  --checkpoint data/gitignore/models/poet-2.ckpt \
  --out-dir data/gitignore/checkpoints_ddp \
  --dtype bf16 \
  --optimizer adamw \
  --peak-lr 3e-4 \
  --warmup-steps 10 \
  --schedule cosine \
  --tokens-per-gpu 8192 \
  --grad-accum 2 \
  --ddp-mode static_graph \
  --total-steps 50 \
  --save-every 25 --log-every 5

echo ""
echo "=== Verify DDP checkpoint loads in single-GPU score.py ==="
pixi run --frozen python scripts/score.py \
  --checkpoint data/gitignore/checkpoints_ddp/final.ckpt \
  --wt_sequence 'MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW' \
  --msa_a3m_path data/BLAT_ECOLX_ColabFold_2202.a3m \
  --variants_fasta_path data/BLAT_ECOLX_Jacquier_2013_variants.fasta \
  --output_npy_path data/gitignore/outputs/ddp_roundtrip_test.npy \
  --context_length 6144 \
  --max_similarity 0.95 \
  --batch_size 32

echo ""
echo "A7 PASSED"
