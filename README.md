# PoET-2

This repository contains inference code for [PoET-2](https://arxiv.org/abs/2508.04724),
a multimodal, retrieval-augmented protein language model for state-of-the-art variant
effect prediction and controllable protein sequence generation.

## Environment Setup

1. Be on a machine with a NVIDIA GPU.
1. Install [pixi](https://pixi.sh/dev/installation/), a package management tool.
1. Install the Python environment by running `pixi install` at the root of this
   repository.

   - To run a script using the Python environment, run
     `pixi run --no-lockfile-update python path/to/script.py`

1. Run `make download_model` to download the model weights (~400MB). The model weights
   will be located at `data/gitignore/models/poet-2.ckpt`. Please note the
   [license](#License).

## Examples

### Zero-shot variant effect prediction

Use the script `scripts/score.py` to predict the functional effects of protein variants. It scores variants by ensembling predictions across different prompting strategies that leverage PoET-2's multimodality.

These strategies include using prompts with and without predicted structures for homologs, and optionally conditioning on the wild-type structure via an "inverse-folding" query. For more details, see the [paper](https://arxiv.org/abs/2508.04724).

#### Primary Input Arguments

*   `--checkpoint`: Path to the PoET-2 model checkpoint file. This defaults to
  `data/gitignore/models/poet-2.ckpt`, which is the location where the model is
  placed after running `make download_model`.
*   `--wt_sequence`: The amino acid sequence of the wild-type (WT) protein. If
 provided, the script will report scores as log-likelihood ratios relative to the WT.
 Otherwise, the script will report raw log-likelihoods.
*   `--msa_a3m_path`: Path to a multiple sequence alignment (MSA) of homologs for the
 WT protein in A3M format. The model uses this MSA to construct the prompt
 context.
*   `--wt_structure_path`: Path to the WT protein's structure file (e.g. PDB or CIF).
*   `--variants_fasta_path`: Path to a FASTA file containing all the variant
 sequences to be scored.
*   `--output_npy_path`: The path where the final scores will be saved as a NumPy
 array (`.npy` file).
*   `--AF2_cache_folder`: A directory to cache protein structures downloaded from the
 AlphaFold DB; see the following section for more information. Defaults to
 `data/gitignore/cache/AF2`.

#### Structure Downloading and MSA Header Format

The script automatically downloads structures from the AlphaFold DB using UniProt
accession IDs extracted from the sequence headers in the MSA. For this to work, the
UniProt ID must be the last part of an underscore-separated string in the header (e.g.
from `>UniRef90_A0A1B2C3D4`, it extracts `A0A1B2C3D4`). Any text following a tab
character is ignored. This format is compatible with standard MSA generation tools like
ColabFold MMseqs2.

Downloaded structures are saved to the `--AF2_cache_folder` to avoid re-downloading
them on subsequent runs.

#### Example Usage

To use the scoring script to score all variants in the `BLAT_ECOLX_Jacquier_2013`
dataset from ProteinGym, we can run the following command:

```
pixi run --no-lockfile-update python scripts/score.py \
    --wt_sequence 'MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW' \
    --msa_a3m_path 'data/BLAT_ECOLX_ColabFold_2202.a3m' \
    --wt_structure_path 'data/BLAT_ECOLX.pdb' \
    --variants_fasta_path 'data/BLAT_ECOLX_Jacquier_2013_variants.fasta' \
    --output_npy_path 'data/gitignore/outputs/BLAT_ECOLX_Jacquier_2013_variants.npy'
```

This command uses the provided MSA and WT structure to score all sequences in the
variants FASTA file. The output scores will be saved as a NumPy array to the specified
output path. The resulting scores should have a Spearman correlation of ~0.7 with the
experimental fitness values, which can be found in the `DMS_score` column of 
`data/BLAT_ECOLX_Jacquier_2013.csv`.

#### Additional Flags

Downloading structures for all proteins in the prompt context can be time-consuming.
If you want to prepare the prompts without running the full GPU-based scoring
pipeline, you can use the `--sample_prompts_only` flag. This will perform all
non-GPU steps, including sampling MSA contexts and fetching structures, and then
exit. This can be useful for debugging or pre-processing on a machine without a GPU.

## License

The following table lists the licenses for the components of this project. Please review
the terms for each component carefully.

| Component       | License                                                             |
|-----------------|---------------------------------------------------------------------|
| **Source Code </br> (excludes model weights)**   | [Apache License 2.0](LICENSE) |
| **Model Weights** | [PoET Non-Commercial License Agreement](MODEL_LICENSE.md)               |

For commercial use of the model weights, please reach out to us at contact@ne47.bio.

## Third-Party Components

This repository includes code from third-party projects:

- Code in `src/poet_2/models/modules/norm.py` is adapted from [Mamba](https://github.com/state-spaces/mamba)
  and is licensed under the Apache License, Version 2.0.
- Code in `src/poet_2/models/modules/glu.py` for the class `GLU` is adapted from
  [x-formers](https://github.com/lucidrains/x-transformers) and licensed under the MIT
  License.
- Code in `src/poet_2/models/poet_2.py` (specifically the `decode` function) is adapted
  from [FlashAttention](https://github.com/Dao-AILab/flash-attention) and licensed under
  the BSD 3-Clause License.
- Code in `src/poet_2/models/modules/packed_sequence.py` (specifically `unpad_input` and
  `pad_input`) is adapted from [FlashAttention](https://github.com/Dao-AILab/flash-attention)
  and licensed under the BSD 3-Clause License.
- Code in `src/poet_2/models/modules/attention_flash_fused_bias.py` is adapted from
  [TurboT5](https://github.com/Knowledgator/TurboT5) and is licensed under the Apache
  License, Version 2.0.

Copies of the applicable third-party licenses are available in the
`third_party_licenses/` directory.

Portions of the above files that the original licenses require to remain under those
licenses continue to be governed by them; everything else in these files, and in the
rest of the repository, is covered by this repository’s license(s).

## Citation

You may cite the paper as

```
@misc{truong2025understandingproteinfunctionmultimodal,
      title={Understanding protein function with a multimodal retrieval-augmented foundation model}, 
      author={Timothy Fei Truong Jr and Tristan Bepler},
      year={2025},
      eprint={2508.04724},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM},
      url={https://arxiv.org/abs/2508.04724}, 
}
```
