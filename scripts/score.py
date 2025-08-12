import argparse
import gzip
import itertools
import multiprocessing
import os
import requests
import tempfile
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import Sequence, cast

import numpy as np
import numpy.typing as npt
import torch

from tqdm import tqdm

from openprotein.protein import Protein

from poet_2.alphabet.sparse_uniref_cluster2 import Alphabet
from poet_2.fasta import parse_stream
from poet_2.msa.sampling import MSASampler, NeighborsSampler
import poet_2.models.poet_2_helpers as helpers

from utils import (
    get_numpy_seed,
    get_names_and_seqs_from_fastalike,
    get_encoded_msa_from_a3m_seqs,
    hash_of_list,
)


FETCH_NUM_WORKERS = min(
    int(os.environ.get("FETCH_NUM_WORKERS", "16")), multiprocessing.cpu_count()
)
PARSE_NUM_WORKERS = min(
    int(os.environ.get("PARSE_NUM_WORKERS", "16")), multiprocessing.cpu_count()
)
VERBOSE = os.environ.get("VERBOSE", "1") == "1"
DEBUG = os.environ.get("DEBUG", "0") == "1"


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Zero-shot variant-effect prediction with PoET-2",
    )
    parser.add_argument("--checkpoint", type=Path, default="data/gitignore/models/poet-2.ckpt")
    parser.add_argument(
        "--wt_sequence",
        type=str,
        default="MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW",
    )
    parser.add_argument(
        "--msa_a3m_path", type=Path, default="data/BLAT_ECOLX_ColabFold_2202.a3m"
    )
    parser.add_argument("--wt_structure_path", type=Path, default="data/BLAT_ECOLX.pdb")
    parser.add_argument(
        "--variants_fasta_path",
        type=Path,
        default="data/BLAT_ECOLX_Jacquier_2013_variants.fasta",
    )
    parser.add_argument(
        "--output_npy_path",
        type=Path,
        default="data/gitignore/outputs/BLAT_ECOLX_Jacquier_2013_variants.npy",
    )
    parser.add_argument(
        "--AF2_cache_folder", type=Path, default="data/gitignore/cache/AF2"
    )
    parser.add_argument("--theta", type=float, default=0.2)
    parser.add_argument(
        "--context_length", type=int, nargs="+", default=[6144, 12288, 24576]
    )
    parser.add_argument(
        "--max_similarity", type=float, nargs="+", default=[1.0, 0.95, 0.90, 0.70, 0.50]
    )
    parser.add_argument(
        "--structure_in_context", type=int, nargs="+", default=[1], choices=[0, 1]
    )
    parser.add_argument(
        "--inverse_folding_query", type=int, nargs="+", default=[0, 1], choices=[0, 1]
    )
    parser.add_argument("--batch_size", type=int, default=8 if DEBUG else 128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample_prompts_only", action="store_true")

    args = parser.parse_args()

    for boolean_list_arg_values in [
        args.structure_in_context,
        args.inverse_folding_query,
    ]:
        assert len(set(boolean_list_arg_values)) == len(
            boolean_list_arg_values
        ), "should not have duplicate values"
    args.structure_in_context = [bool(x) for x in args.structure_in_context]
    args.inverse_folding_query = [bool(x) for x in args.inverse_folding_query]
    args.seed = np.random.RandomState(args.seed).randint(2**31)
    return args


def sample_context(
    msa_names: list[bytes],
    msa_sequences: list[bytes],
    msa: npt.NDArray[np.uint8],
    theta: float,
    max_tokens: int,
    max_similarity: float,
    structure_in_context: bool,  # NB: unused here as structures are fetched later
    inverse_folding_query: bool,
    seed: int,
    sampling_weights_cache_dir: Path,
) -> list[Protein]:
    alphabet = Alphabet()
    sampler = MSASampler(
        method=NeighborsSampler(can_use_torch=False, theta=theta),
        force_include_first=inverse_folding_query,
        max_similarity=max_similarity,
    )
    sample_idxs = sampler.get_sample_idxs(
        msa=msa,
        gap_token=alphabet.gap_token,
        seed=seed,
        result_cache_dir=sampling_weights_cache_dir,
    )
    if inverse_folding_query:
        # NB: don't include wt in context
        assert sample_idxs[0] == 0
        sample_idxs = sample_idxs[1:]
    prompt: list[Protein] = []
    total_tokens = 0
    for idx in cast(Sequence[int], sample_idxs):
        sequence = msa_sequences[idx].upper().translate(None, delete=b"-")
        this_n_tokens = len(sequence) + 2
        if this_n_tokens + total_tokens > max_tokens:
            break
        prompt.append(Protein(name=msa_names[idx], sequence=sequence))
        total_tokens += this_n_tokens
    # shuffle order
    rng = np.random.RandomState(get_numpy_seed(f"{seed+1}"))
    return [prompt[i] for i in rng.permutation(len(prompt))]


def _fetch_structure(name: str, cache_dir: Path) -> str:
    cache_path = cache_dir / f"{name}.cif.gz"
    if cache_path.is_file():
        return name

    url = f"https://alphafold.ebi.ac.uk/files/AF-{name}-F1-model_v4.cif"
    response = requests.get(url)
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            cache_path.touch()
            return name
        else:
            raise
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=cache_dir) as tmp:
        with gzip.GzipFile(fileobj=tmp, mode="wb") as gz:
            gz.write(response.content)
        temp_path = Path(tmp.name)
    try:
        temp_path.rename(cache_path)
    except FileExistsError:
        # Another process beat us to writing the file
        temp_path.unlink()  # Clean up our temp file
    return name


def fetch_structures(proteins: list[Protein], cache_dir: Path) -> dict[str, Protein]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    proteins_with_structure: dict[str, Protein] = {}
    with ThreadPoolExecutor(max_workers=FETCH_NUM_WORKERS) as executor:
        seen_names: set[str] = set()
        futures: list[Future] = []
        for protein in proteins:
            assert protein.name is not None
            if protein.name.startswith("UPI"):
                continue
            if protein.name in seen_names:
                continue
            future = executor.submit(
                _fetch_structure, name=protein.name, cache_dir=cache_dir
            )
            seen_names.add(protein.name)
            futures.append(future)
        with ProcessPoolExecutor(max_workers=PARSE_NUM_WORKERS) as pool:
            parse_futures = []
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                mininterval=1.0,
                desc="Downloading structures...",
            ):
                name = future.result()
                cache_path = cache_dir / f"{name}.cif.gz"
                if cache_path.stat().st_size == 0:
                    continue
                future = pool.submit(
                    Protein.from_filepath, path=cache_path, chain_id="A"
                )
                parse_futures.append(future)
            for future in tqdm(
                as_completed(parse_futures),
                total=len(parse_futures),
                mininterval=1.0,
                desc="Parsing structures...",
            ):
                protein = future.result()
                assert isinstance(protein, Protein)
                assert protein.name is not None
                assert protein.name.endswith(".cif")
                name = protein.name.removesuffix(".cif")
                protein.name = name
                proteins_with_structure[name] = protein
    return proteins_with_structure


@torch.inference_mode()
def _main(tmpdir: Path):
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ###########################################################################
    # Variants to score
    ###########################################################################
    with open(args.variants_fasta_path, "rb") as f:
        variants = [seq.decode() for _, seq in parse_stream(f)]
    if args.wt_sequence is not None:
        variants.append(args.wt_sequence)  # WT last

    ###########################################################################
    # MSA processing
    ###########################################################################
    msa_names, msa_sequences = get_names_and_seqs_from_fastalike(args.msa_a3m_path)
    msa_names = [
        n.split(b"\t", 1)[0].split(b"_")[-1] for n in msa_names
    ]  # retain UniProt accession
    msa = get_encoded_msa_from_a3m_seqs(
        msa_sequences=msa_sequences, alphabet=Alphabet()
    )

    ###########################################################################
    # Load query
    ###########################################################################
    query = None
    if args.wt_structure_path is not None:
        query = Protein.from_filepath(args.wt_structure_path, chain_id="A")
        query.sequence = b"X" * len(query)  # mask sequence

    ###########################################################################
    # Prompt ensemble generation
    ###########################################################################
    prompts: dict[
        tuple[int, float, bool, bool, int], tuple[list[Protein], Protein | None]
    ] = {}
    for (
        max_tokens,
        max_similarity,
        structure_in_context,
        inverse_folding_query,
    ) in itertools.product(
        args.context_length,
        args.max_similarity,
        args.structure_in_context,
        args.inverse_folding_query,
    ):
        seed_offset = 0
        while (
            key := (
                max_tokens,
                max_similarity,
                structure_in_context,
                inverse_folding_query,
                seed_offset,
            )
        ) in prompts.keys():
            seed_offset += 1
        name = f"{max_tokens}_{max_similarity}_{structure_in_context}_{inverse_folding_query}"
        name = f"{name}_seed={args.seed + seed_offset}"
        context = sample_context(
            msa_names=msa_names,
            msa_sequences=msa_sequences,
            msa=msa,
            theta=args.theta,
            max_tokens=max_tokens,
            max_similarity=max_similarity,
            structure_in_context=structure_in_context,
            inverse_folding_query=inverse_folding_query,
            seed=get_numpy_seed(name),
            sampling_weights_cache_dir=tmpdir,
        )
        prompts[key] = (context, query if inverse_folding_query else None)
        if VERBOSE:
            context_hash = hash_of_list([c.sequence for c in context])
            print(f"Sampled prompt {name}: {context_hash}")
    # fetch structures
    need_structure_proteins: list[Protein] = []
    for (_, _, structure_in_context, _, _), (context, _) in prompts.items():
        if not structure_in_context:
            continue
        need_structure_proteins.extend(context)
    proteins_with_structure = fetch_structures(
        need_structure_proteins, cache_dir=args.AF2_cache_folder
    )
    # replace context proteins with proteins with structure as needed
    for (
        max_tokens,
        max_similarity,
        structure_in_context,
        inverse_folding_query,
        seed_offset,
    ), (context, _) in prompts.items():
        if not structure_in_context:
            continue
        n_replaced = 0
        for i in range(len(context)):
            name = context[i].name
            assert name is not None
            if name in proteins_with_structure:
                n_replaced += 1
            context[i] = proteins_with_structure.get(name, context[i])
        if VERBOSE:
            name = f"{max_tokens}_{max_similarity}_{structure_in_context}_{inverse_folding_query}"
            name = f"{name}_seed={args.seed + seed_offset}"
            context_hash = hash_of_list([c.sequence for c in context])
            print(
                f"Sampled prompt {name} ({n_replaced}/{len(context)}): {context_hash}"
            )
    # enforce max tokens again, as number of tokens may have changed after the above
    print("Reenforcing max tokens...")
    context_hashes: dict[tuple[int, float, bool, bool, int], str] = {}
    for key, (context, query) in prompts.items():
        (
            max_tokens,
            max_similarity,
            structure_in_context,
            inverse_folding_query,
            seed_offset,
        ) = key
        if structure_in_context:
            new_context: list[Protein] = []
            total_tokens = 0
            for protein in context:
                this_n_tokens = len(protein) + 2
                if this_n_tokens + total_tokens > max_tokens:
                    break
                new_context.append(protein)
                total_tokens += this_n_tokens
            n_old_context, n_new_context = len(context), len(new_context)
            context = new_context
            del new_context
            prompts[key] = (context, query)
        else:
            n_old_context, n_new_context = len(context), len(context)
        # save context hash
        context_hash = hash_of_list([c.sequence for c in context])
        context_hashes[key] = context_hash
        if VERBOSE:
            name = f"{max_tokens}_{max_similarity}_{structure_in_context}_{inverse_folding_query}"
            name = f"{name}_seed={args.seed + seed_offset}"
            print(
                f"Sampled prompt {name} ({n_new_context}/{n_old_context}): {context_hash}"
            )
    print("Trimming coordinates to backbone only to save memory...")
    for context, query in prompts.values():
        if query is not None:
            query.coordinates = query.coordinates[:, :3]
        for protein in context:
            protein.coordinates = protein.coordinates[:, :3]
    if args.sample_prompts_only:
        return

    ###########################################################################
    # Load PoET‑2 model
    ###########################################################################
    device = torch.device("cuda")
    model = helpers.load_model(path=args.checkpoint, device=device)

    #######################################################################
    # Score variants for each prompt
    #######################################################################
    variant_scores = np.empty((len(prompts), len(variants)), dtype=np.float32)
    for idx, (key, (context, query)) in enumerate(prompts.items()):
        (
            max_tokens,
            max_similarity,
            structure_in_context,
            inverse_folding_query,
            seed_offset,
        ) = key
        if VERBOSE:
            name = f"{max_tokens}_{max_similarity}_{structure_in_context}_{inverse_folding_query}"
            name = f"{name}_seed={args.seed + seed_offset}"
            tqdm.write(f"Processing prompt ({idx + 1}/{len(prompts)}): {name}")
        # Build memory for this prompt
        inputs: list[helpers.NamedInput] = []
        if query is not None:
            inputs.append(
                helpers.NamedInput(
                    sequence=query.sequence,
                    plddt=query.plddt.copy(),
                    atomx=query.coordinates[:, :3].copy(),
                ).enforce()
            )
        for p in context:
            inputs.append(
                helpers.NamedInput(
                    sequence=p.sequence,
                    plddt=p.plddt.copy(),
                    atomx=p.coordinates[:, :3].copy(),
                ).enforce()
            )
        memory, ys_ref = helpers.compute_memory(model=model, prompts=[inputs])
        # Score all variants
        logps = helpers.score_sequences(
            model=model,
            memory=memory,
            sequences=variants,
            ys_ref_values=ys_ref if query is not None else None,
            self_prompt=query.sequence if query is not None else None,
            batch_size=args.batch_size,
            verbose=True,
        )
        variant_scores[idx] = logps.cpu().float().numpy()

    #######################################################################
    # Aggregate & WT normalization
    #######################################################################
    # Mean over prompt ensemble
    final_scores = variant_scores.mean(axis=0)  # shape (n_variants,)
    # If WT supplied, convert to log‑likelihood difference
    if args.wt_sequence is not None:
        final_scores = final_scores[:-1] - final_scores[-1]

    #######################################################################
    # Save
    #######################################################################
    args.output_npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy_path, final_scores.astype(np.float32))
    print(f"Wrote {final_scores.shape[0]} scores → {args.output_npy_path}")


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        _main(Path(tmpdir))


if __name__ == "__main__":
    main()
