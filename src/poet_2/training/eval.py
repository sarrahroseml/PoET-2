"""M5: zero-shot eval hook to track regression during training.

Scores variants against a homolog context prompt. When a WT structure is provided,
runs multiple scoring modes and reports all:

1. **seq_only**: baseline, sequence-only context.
2. **struct**: WT added to context with real sequence + structure.
3. **ifq**: WT structure as inverse-folding query (masked sequence).
4. **af2**: homolog context with AlphaFold2 structures fetched + WT struct in context.

Model-free parts (a3m parsing, context selection, Spearman) are unit-tested locally.
The scoring reuses ``poet_2_helpers.compute_memory`` + ``score_sequences`` via a lazy
import (needs flash_attn / the GPU box).
"""

from __future__ import annotations

import csv
import gzip
import os
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from poet_2.fasta import parse_stream

ALPHA = 1.96


# ----------------------------------------------------------- a3m / fasta (model-free)


def _read(path: str, upper: bool) -> tuple[list[bytes], list[bytes]]:
    with open(path, "rb") as f:
        names, seqs = [], []
        for name, seq in parse_stream(f, upper=upper):
            names.append(name)
            seqs.append(seq)
    return names, seqs


def ungapped(seq: bytes) -> bytes:
    return seq.replace(b"-", b"").replace(b".", b"").upper()


def match_columns(seq: bytes) -> bytes:
    return bytes(c for c in seq if not (97 <= c <= 122))


def identity_to_query(row: bytes, query_match: bytes) -> float:
    q = np.frombuffer(query_match, dtype=np.uint8)
    r = np.frombuffer(match_columns(row), dtype=np.uint8)
    n = min(q.shape[0], r.shape[0])
    if n == 0:
        return 0.0
    q, r = q[:n], r[:n]
    nongap = q != ord("-")
    denom = int(nongap.sum())
    return float(((r == q) & nongap).sum() / denom) if denom else 0.0


def _uniprot_id(name: bytes) -> str:
    """Extract UniProt accession from MSA header (e.g. UniRef90_A0A1B2C3D4 -> A0A1B2C3D4).
    Strips range suffixes like /252-543."""
    uid = name.split(b"\t", 1)[0].split(b"_")[-1]
    uid = uid.split(b"/")[0]
    return uid.decode()


def select_context(
    names: list[bytes],
    rows: list[bytes],
    max_similarity: float = 1.0,
    max_tokens: int = 6144,
    seed: int = 0,
) -> tuple[list[bytes], list[bytes]]:
    """Pick ungapped context homologs. Returns (selected_names, selected_seqs)."""
    query_match = match_columns(rows[0]) if rows else b""
    rng = np.random.default_rng(seed)
    seen: set[bytes] = set()
    uniq_names: list[bytes] = []
    uniq_seqs: list[bytes] = []
    for name, row in zip(names[1:], rows[1:]):
        if query_match and identity_to_query(row, query_match) > max_similarity:
            continue
        s = ungapped(row)
        if s and s not in seen:
            seen.add(s)
            uniq_names.append(name)
            uniq_seqs.append(s)
    order = rng.permutation(len(uniq_seqs))
    out_names, out_seqs, used = [], [], 0
    for i in order:
        s = uniq_seqs[int(i)]
        cost = len(s) + 2
        if out_seqs and used + cost > max_tokens:
            continue
        out_names.append(uniq_names[int(i)])
        out_seqs.append(s)
        used += cost
        if used >= max_tokens:
            break
    return out_names, out_seqs


def read_variant_seqs(path: str, label_col: str = "DMS_score") -> list[bytes]:
    """Read variant sequences from FASTA or CSV (with mutated_sequence column)."""
    if path.endswith(".csv"):
        return _read_variants_csv(path, label_col=label_col)
    return [ungapped(s) for s in _read(path, upper=True)[1]]


def _read_variants_csv(path: str, label_col: str = "DMS_score") -> list[bytes]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if label_col in rows[0]:
        rows = [r for r in rows if r[label_col].strip() != ""]
    return [r["mutated_sequence"].encode() for r in rows]


def read_labels(csv_path: str, col: str = "DMS_score") -> np.ndarray:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r[col].strip() != ""]
    return np.array([float(r[col]) for r in rows], dtype=float)


def length_adjusted(logps, lengths, alpha: float = ALPHA) -> np.ndarray:
    return np.asarray(logps, dtype=float) + alpha * np.asarray(lengths, dtype=float)


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape[0] < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


# ----------------------------------------------------------- AF2 structure fetching


def _fetch_one(uniprot_id: str, cache_dir: Path) -> str:
    """Download a single structure from AlphaFold DB; cache as .cif.gz.
    Only creates a 0-byte sentinel for genuine 404s (structure doesn't exist)."""
    import requests

    cache_path = cache_dir / f"{uniprot_id}.cif.gz"
    if cache_path.is_file():
        return uniprot_id
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.cif"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 404:
            cache_path.touch()
            return uniprot_id
        response.raise_for_status()
    except Exception:
        return uniprot_id
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=cache_dir) as tmp:
        with gzip.GzipFile(fileobj=tmp, mode="wb") as gz:
            gz.write(response.content)
        temp_path = Path(tmp.name)
    try:
        temp_path.rename(cache_path)
    except FileExistsError:
        temp_path.unlink()
    return uniprot_id


def fetch_af2_structures(
    names: list[bytes], seqs: list[bytes], cache_dir: str, max_workers: int = 8,
) -> list:
    """Fetch AF2 structures for context homologs. Returns list of NamedInput (with or
    without structure, depending on availability)."""
    from openprotein.protein import Protein
    from poet_2.models.poet_2_helpers import NamedInput

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    ids = [_uniprot_id(n) for n in names]
    unique_ids = list({uid for uid in ids if not uid.startswith("UPI")})

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_fetch_one, uid, cache_path): uid for uid in unique_ids}
        for f in as_completed(futs):
            f.result()

    parsed: dict[str, Protein] = {}
    for uid in unique_ids:
        cif = cache_path / f"{uid}.cif.gz"
        if not cif.is_file() or cif.stat().st_size == 0:
            continue
        try:
            p = Protein.from_filepath(cif, chain_id="A")
            p.coordinates = p.coordinates[:, :3]
            parsed[uid] = p
        except Exception:
            continue

    inputs = []
    for uid, seq in zip(ids, seqs):
        if uid in parsed:
            p = parsed[uid]
            inp = NamedInput(
                sequence=seq,
                plddt=p.plddt.copy(),
                atomx=p.coordinates.copy(),
            ).enforce()
        else:
            inp = seq
        inputs.append(inp)
    return inputs


# ----------------------------------------------------------- structure loading


def _load_wt_protein(wt_structure_path: str):
    """Load WT structure, trim to backbone. Returns (protein, NamedInput with real seq, NamedInput with masked seq)."""
    from openprotein.protein import Protein
    from poet_2.models.poet_2_helpers import NamedInput

    protein = Protein.from_filepath(wt_structure_path, chain_id="A")
    protein.coordinates = protein.coordinates[:, :3]

    struct_context = NamedInput(
        sequence=protein.sequence,
        plddt=protein.plddt.copy(),
        atomx=protein.coordinates.copy(),
    ).enforce()

    ifq = NamedInput(
        sequence=b"X" * len(protein),
        plddt=protein.plddt.copy(),
        atomx=protein.coordinates.copy(),
    ).enforce()

    return protein, struct_context, ifq


def _load_context_structures(
    seqs: list[bytes],
    struct_npz_dir: str,
) -> list:
    """Load pre-folded structures for context sequences. Returns list of NamedInput or bytes.

    Matches sequences to NPZ files by md5 hash (same naming convention as
    extract_eval_context_seqs.py).
    """
    import hashlib
    from poet_2.models.poet_2_helpers import NamedInput

    npz_index: dict[str, str] | None = None

    def _get_index():
        nonlocal npz_index
        if npz_index is None:
            npz_index = {}
            for fname in os.listdir(struct_npz_dir):
                if fname.endswith(".npz"):
                    key = fname.replace(".npz", "")
                    npz_index[key] = os.path.join(struct_npz_dir, fname)
        return npz_index

    inputs = []
    for seq in seqs:
        seq_hash = hashlib.md5(seq).hexdigest()[:12]
        idx = _get_index()
        matched = None
        for key, path in idx.items():
            if seq_hash in key:
                matched = path
                break
        if matched is not None:
            try:
                data = np.load(matched)
                inp = NamedInput(
                    sequence=seq,
                    plddt=data["plddt"].copy(),
                    atomx=data["atomx"].copy(),
                ).enforce()
                inputs.append(inp)
                continue
            except Exception:
                pass
        inputs.append(seq)
    return inputs


def _score_prompt(model, prompt, query_seqs, ys_ref=None, self_prompt=None, alpha=ALPHA):
    from poet_2.models.poet_2_helpers import compute_memory, score_sequences

    memory, ref = compute_memory(model, [prompt])
    logps = score_sequences(
        model, memory, query_seqs,
        ys_ref_values=ref if ys_ref else None,
        self_prompt=self_prompt,
    ).float().cpu().numpy()
    lengths = [len(s) for s in query_seqs]
    return length_adjusted(logps, lengths, alpha)


ENSEMBLE_CONTEXT_LENGTHS = [6144, 12288, 24576]
ENSEMBLE_MAX_SIMILARITIES = [1.0, 0.95, 0.90, 0.70, 0.50]


def _score_ensemble(
    model,
    names: list[bytes],
    rows: list[bytes],
    query_seqs: list[bytes],
    *,
    ifq_input=None,
    alpha: float = ALPHA,
    seed: int = 0,
) -> np.ndarray:
    """Score with a 15-prompt ensemble (3 context lengths × 5 similarity thresholds).

    Averages raw scores across prompts before length adjustment, matching
    the scoring protocol in scripts/score.py.
    """
    all_adj = []
    for ctx_len in ENSEMBLE_CONTEXT_LENGTHS:
        for max_sim in ENSEMBLE_MAX_SIMILARITIES:
            prompt_seed = seed + len(all_adj)
            _, ctx_seqs = select_context(names, rows, max_sim, ctx_len, prompt_seed)
            if ifq_input is not None:
                prompt = [ifq_input] + list(ctx_seqs)
                adj = _score_prompt(
                    model, prompt, query_seqs,
                    ys_ref=True, self_prompt=ifq_input.sequence,
                    alpha=alpha,
                )
            else:
                adj = _score_prompt(model, list(ctx_seqs), query_seqs, alpha=alpha)
            all_adj.append(adj)
    return np.mean(all_adj, axis=0)


def evaluate(
    model,
    a3m_path: str,
    variants_fasta: str,
    *,
    wt_sequence: str | bytes | None = None,
    wt_structure_path: str | None = None,
    af2_cache_folder: str = "data/gitignore/cache/AF2",
    context_struct_dir: str | None = None,
    labels_csv: str | None = None,
    label_col: str = "DMS_score",
    alpha: float = ALPHA,
    max_similarity: float = 1.0,
    context_tokens: int = 6144,
    seed: int = 0,
    skip_ensemble: bool = False,
) -> tuple[dict, np.ndarray]:
    """Zero-shot evaluation for the training hook. Returns ``(metrics, scores)``.

    When ``wt_structure_path`` is provided, runs scoring modes:
    - ``spearman``: sequence-only baseline
    - ``spearman_struct``: WT (real seq + structure) added to context
    - ``spearman_ifq``: WT structure as inverse-folding query (masked seq)
    - ``spearman_af2``: homolog AF2 structures + WT structure in context

    When ``context_struct_dir`` is provided (directory of NPZ files from folded
    context homologs), also runs:
    - ``spearman_ctx_struct``: all context homologs with pre-folded structures
    - ``spearman_ctx_struct_wt``: WT structure + all context with structures
    """
    import torch

    names, rows = _read(a3m_path, upper=False)
    ctx_names, ctx_seqs = select_context(names, rows, max_similarity, context_tokens, seed)
    variants = read_variant_seqs(variants_fasta, label_col=label_col)
    wt = wt_sequence.encode() if isinstance(wt_sequence, str) else wt_sequence
    query = ([wt] + variants) if wt is not None else variants

    labels = None
    if labels_csv is not None:
        labels = read_labels(labels_csv, label_col)

    def to_scores(adj):
        return (adj[1:] - adj[0]) if wt is not None else adj

    def add_spearman(metrics, scores, key="spearman"):
        if labels is not None:
            m = min(scores.shape[0], labels.shape[0])
            metrics[key] = spearman(scores[:m], labels[:m])

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            metrics: dict = {"n_variants": len(variants), "n_context": len(ctx_seqs)}

            # 1. Baseline: sequence-only
            adj = _score_prompt(model, list(ctx_seqs), query, alpha=alpha)
            scores = to_scores(adj)
            add_spearman(metrics, scores)

            if wt_structure_path is not None:
                protein, struct_input, ifq_input = _load_wt_protein(wt_structure_path)
                pdb_len = len(protein)
                query_len = len(wt) if wt is not None else 0
                if pdb_len != query_len:
                    print(f"  SKIP struct/ifq: PDB length ({pdb_len}) != query ({query_len})",
                          flush=True)
                    wt_structure_path = None  # disable struct modes for this DMS
                else:
                    # 2. Structure-in-context: WT with real sequence + structure
                    adj_struct = _score_prompt(
                        model, [struct_input] + list(ctx_seqs), query, alpha=alpha,
                    )
                    add_spearman(metrics, to_scores(adj_struct), "spearman_struct")

                    # 3. Inverse-folding query: WT with masked sequence + structure
                    adj_ifq = _score_prompt(
                        model, [ifq_input] + list(ctx_seqs), query,
                        ys_ref=True, self_prompt=ifq_input.sequence,
                        alpha=alpha,
                    )
                    add_spearman(metrics, to_scores(adj_ifq), "spearman_ifq")

                    # 4. AF2 structures on homologs + WT structure in context
                    af2_ctx = fetch_af2_structures(ctx_names, ctx_seqs, af2_cache_folder)
                    n_with_struct = sum(1 for x in af2_ctx if not isinstance(x, bytes))
                    metrics["n_af2_structures"] = n_with_struct
                    adj_af2 = _score_prompt(
                        model, [struct_input] + af2_ctx, query, alpha=alpha,
                    )
                    add_spearman(metrics, to_scores(adj_af2), "spearman_af2")

            # 5+6. Pre-folded context structures
            if context_struct_dir is not None and os.path.isdir(context_struct_dir):
                ctx_struct = _load_context_structures(ctx_seqs, context_struct_dir)
                n_ctx_struct = sum(1 for x in ctx_struct if not isinstance(x, bytes))
                metrics["n_ctx_structures"] = n_ctx_struct

                # 5. All context with folded structures (no WT struct)
                adj_ctx = _score_prompt(model, ctx_struct, query, alpha=alpha)
                add_spearman(metrics, to_scores(adj_ctx), "spearman_ctx_struct")

                # 6. WT struct + all context with folded structures
                if wt_structure_path is not None:
                    adj_ctx_wt = _score_prompt(
                        model, [struct_input] + ctx_struct, query, alpha=alpha,
                    )
                    add_spearman(metrics, to_scores(adj_ctx_wt), "spearman_ctx_struct_wt")

                    # 7. IFQ decoder scoring + context with folded structures
                    adj_ctx_ifq = _score_prompt(
                        model, [ifq_input] + ctx_struct, query,
                        ys_ref=True, self_prompt=ifq_input.sequence,
                        alpha=alpha,
                    )
                    add_spearman(metrics, to_scores(adj_ctx_ifq), "spearman_ctx_ifq")

            # 8+9. 15-prompt ensemble (3 context lengths × 5 similarity thresholds)
            if not skip_ensemble:
                adj_ens = _score_ensemble(
                    model, names, rows, query, alpha=alpha, seed=seed,
                )
                add_spearman(metrics, to_scores(adj_ens), "spearman_ens")

                if wt_structure_path is not None and ifq_input is not None:
                    adj_ens_ifq = _score_ensemble(
                        model, names, rows, query,
                        ifq_input=ifq_input, alpha=alpha, seed=seed,
                    )
                    add_spearman(metrics, to_scores(adj_ens_ifq), "spearman_ens_ifq")
    finally:
        if was_training:
            model.train()

    return metrics, scores


# ----------------------------------------------------------- multi-DMS suite


def discover_dms_suite(
    dms_dir: str,
) -> list[dict]:
    """Discover all DMS eval triplets (a2m, variants CSV, optional PDB) in dms_dir.

    Expects subdirectories:
        alignments/          *.a2m files
        viral_dms_substitutions/  *_dms.csv files
        viral_dms_structures/     *.pdb files (optional)
    """
    align_dir = os.path.join(dms_dir, "alignments")
    subs_dir = os.path.join(dms_dir, "viral_dms_substitutions")
    struct_dir = os.path.join(dms_dir, "viral_dms_structures")

    if not os.path.isdir(align_dir) or not os.path.isdir(subs_dir):
        return []

    a2m_map = {}
    for f in os.listdir(align_dir):
        if f.endswith(".a2m"):
            name = f[: -len(".a2m")]
            a2m_map[name] = os.path.join(align_dir, f)

    suite = []
    for f in sorted(os.listdir(subs_dir)):
        if not f.endswith("_dms.csv"):
            continue
        name = f[: -len("_dms.csv")]
        if name not in a2m_map:
            continue
        entry = {
            "name": name,
            "a2m": a2m_map[name],
            "variants_csv": os.path.join(subs_dir, f),
        }
        pdb = os.path.join(struct_dir, f"{name}.pdb")
        if os.path.isfile(pdb):
            entry["structure"] = pdb
        suite.append(entry)
    return suite


def evaluate_dms_suite(
    model,
    dms_dir: str,
    *,
    alpha: float = ALPHA,
    max_similarity: float = 1.0,
    context_tokens: int = 6144,
    seed: int = 0,
    seq_only: bool = True,
    skip_ensemble: bool = False,
    context_struct_dir: str | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> dict:
    """Run eval over all DMS datasets in dms_dir. Returns per-DMS + mean Spearman.

    When ``world_size > 1``, each rank evaluates a shard of DMSes and results
    are all-gathered so every rank gets the full dict.
    """
    import torch

    suite = discover_dms_suite(dms_dir)
    if not suite:
        return {"error": f"no DMS triplets found in {dms_dir}"}

    shard = [e for i, e in enumerate(suite) if i % world_size == rank]

    results: dict = {}
    spearmans = []
    for entry in shard:
        name = entry["name"]
        wt_struct = None if seq_only else entry.get("structure")
        wt_names, wt_rows = _read(entry["a2m"], upper=False)
        wt_seq = ungapped(wt_rows[0]) if wt_rows else None
        try:
            metrics, _ = evaluate(
                model,
                entry["a2m"],
                entry["variants_csv"],
                wt_sequence=wt_seq,
                wt_structure_path=wt_struct,
                context_struct_dir=context_struct_dir,
                labels_csv=entry["variants_csv"],
                label_col="DMS_score",
                alpha=alpha,
                max_similarity=max_similarity,
                context_tokens=context_tokens,
                seed=seed,
                skip_ensemble=skip_ensemble,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            rho = metrics.get("spearman", float("nan"))
            if seq_only:
                results[name] = rho
            else:
                results[name] = {k: v for k, v in metrics.items()
                                 if k.startswith("spearman") or k.startswith("n_")}
            if not np.isnan(rho):
                spearmans.append(rho)
        except Exception as e:
            results[name] = f"error: {e}"
            print(f"  WARN: {name} failed: {e}", flush=True)

    if world_size > 1:
        import torch.distributed as dist
        import pickle
        data = pickle.dumps(results)
        size = torch.tensor([len(data)], dtype=torch.long, device="cuda")
        all_sizes = [torch.zeros_like(size) for _ in range(world_size)]
        dist.all_gather(all_sizes, size)
        max_size = max(s.item() for s in all_sizes)
        buf = torch.zeros(max_size, dtype=torch.uint8, device="cuda")
        buf[:len(data)] = torch.tensor(list(data), dtype=torch.uint8, device="cuda")
        all_bufs = [torch.zeros_like(buf) for _ in range(world_size)]
        dist.all_gather(all_bufs, buf)
        for i, (s, b) in enumerate(zip(all_sizes, all_bufs)):
            if i != rank:
                other = pickle.loads(bytes(b[:s.item()].cpu().tolist()))
                results.update(other)
        spearmans = []
        for name, v in results.items():
            if isinstance(v, dict):
                rho = v.get("spearman", float("nan"))
            elif isinstance(v, float):
                rho = v
            else:
                continue
            if not np.isnan(rho):
                spearmans.append(rho)

    results["mean_spearman"] = float(np.mean(spearmans)) if spearmans else float("nan")
    results["n_evaluated"] = len(spearmans)
    results["n_total"] = len(suite)

    if not seq_only:
        for mode in ("spearman_struct", "spearman_ifq",
                     "spearman_ctx_struct", "spearman_ctx_struct_wt", "spearman_ctx_ifq",
                     "spearman_ens", "spearman_ens_ifq"):
            vals = [v[mode] for v in results.values()
                    if isinstance(v, dict) and mode in v
                    and isinstance(v[mode], (int, float)) and not np.isnan(v[mode])]
            results[f"mean_{mode}"] = float(np.mean(vals)) if vals else float("nan")

    return results
