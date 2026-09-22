"""Score pretrained PoET (v1) on all 45 viral DMS eval datasets from PoET-2.

This script adapts PoET's scoring pipeline (ensemble of 15 MSA sampling
configurations × bidirectional scoring) to run over every DMS dataset
discovered in data/evals/.

Usage:
    python scripts/eval_poet1_baseline.py                          # full ensemble
    python scripts/eval_poet1_baseline.py --debug                  # 1/15 params (fast)
    python scripts/eval_poet1_baseline.py --batch_size 4           # lower VRAM

Requires:
    - GPU with CUDA
    - PoET checkpoint at data/poet.ckpt (run: make -C ../PoET download_model)
    - PoET source tree at ../PoET/ (used via sys.path)
"""

import argparse
import csv
import itertools
import json
import os
import string
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm, trange

POET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "PoET"))
sys.path.insert(0, POET_ROOT)

from poet.alphabets import Uniprot21
from poet.fasta import parse_stream
from poet.models.poet import PoET

ASCII_LOWERCASE_BYTES = string.ascii_lowercase.encode()


def append_startstop(x, alphabet):
    x_ndim = x.ndim
    if x_ndim == 1:
        x = x[None, :]
    x_ = np.empty((x.shape[0], x.shape[1] + 2), dtype=x.dtype)
    x_[:, 0] = alphabet.start_token
    x_[:, -1] = alphabet.stop_token
    x_[:, 1:-1] = x
    if x_ndim == 1:
        x_ = x_.flatten()
    return x_


def get_seqs_from_fastalike(filepath):
    return [s for _, s in parse_stream(open(filepath, "rb"), upper=False)]


def match_columns(seq):
    return bytes(c for c in seq if not (97 <= c <= 122))


def ungapped(seq):
    return seq.replace(b"-", b"").replace(b".", b"").upper()


def identity_to_query(row, query_match):
    q = np.frombuffer(query_match, dtype=np.uint8)
    r = np.frombuffer(match_columns(row), dtype=np.uint8)
    n = min(q.shape[0], r.shape[0])
    if n == 0:
        return 0.0
    q, r = q[:n], r[:n]
    nongap = (q != ord("-")) & (q != ord("."))
    denom = int(nongap.sum())
    return float(((r == q) & nongap).sum() / denom) if denom else 0.0


def preprocess_msa(msa_sequences):
    """Precompute identities and ungapped sequences once for the entire MSA."""
    if not msa_sequences:
        return [], []
    query_match = match_columns(msa_sequences[0])
    seen = set()
    candidates = []
    for row in msa_sequences[1:]:
        s = ungapped(row)
        if not s or s in seen:
            continue
        seen.add(s)
        ident = identity_to_query(row, query_match) if query_match else 0.0
        candidates.append((ident, s))
    return candidates


def select_context(candidates, max_similarity, max_tokens, seed):
    """Select context from precomputed candidates, filtered by identity threshold."""
    rng = np.random.default_rng(seed)
    filtered = [(ident, s) for ident, s in candidates if ident <= max_similarity]
    if not filtered:
        return []
    order = rng.permutation(len(filtered))
    selected = []
    used = 0
    for idx in order:
        _, s = filtered[int(idx)]
        cost = len(s) + 2
        if selected and used + cost > max_tokens:
            continue
        selected.append(s)
        used += cost
        if used >= max_tokens:
            break
    return selected


def sample_msa_sequences(get_sequence_fn, sample_idxs, max_tokens, alphabet, shuffle_seed=None):
    seqs, total_tokens = [], 0
    for idx in sample_idxs:
        next_sequence = get_sequence_fn(idx)
        seqs.append(append_startstop(alphabet.encode(next_sequence), alphabet=alphabet))
        total_tokens += len(seqs[-1])
        if total_tokens > max_tokens:
            break
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else np.random
    final_permutation = rng.permutation(len(seqs))
    final_seqs, total_tokens = [], 0
    for seq in [seqs[i] for i in final_permutation]:
        total_tokens += len(seq)
        final_seqs.append(seq)
        if total_tokens >= max_tokens:
            break
    return final_seqs


def score_variants_batch(memory, variants, model, batch_size, alphabet):
    max_variant_length = max(len(v) for v in variants)
    memory = model.logits_allocate_memory(
        memory=memory, batch_size=batch_size, length=max_variant_length - 1,
    )
    criteria = nn.CrossEntropyLoss(ignore_index=alphabet.mask_token, reduction="none")
    logps = []
    for start_idx in range(0, len(variants), batch_size):
        this_variants = variants[start_idx : start_idx + batch_size]
        this_variants = pad_sequence(
            [torch.from_numpy(v).long() for v in this_variants],
            batch_first=True, padding_value=alphabet.mask_token,
        )
        if this_variants.size(1) < max_variant_length:
            this_variants = F.pad(
                this_variants, (0, max_variant_length - this_variants.size(1)),
                value=alphabet.mask_token,
            )
        this_variants = this_variants.cuda()
        logits = model.logits(this_variants[:, :-1], memory, preallocated_memory=True)
        targets = this_variants[:, 1:]
        score = -criteria.forward(logits.transpose(1, 2), targets).float().sum(dim=1)
        logps.append(score.cpu().numpy())
    return np.hstack(logps)


def embed_and_score(msa_seqs_encoded, variants, model, batch_size, alphabet):
    if len(msa_seqs_encoded) > 0:
        segment_sizes = torch.tensor([len(s) for s in msa_seqs_encoded]).cuda()
        msa_flat = torch.cat([torch.from_numpy(s).long() for s in msa_seqs_encoded]).cuda()
        memory = model.embed(msa_flat.unsqueeze(0), segment_sizes.unsqueeze(0))
    else:
        memory = None
    return score_variants_batch(memory, variants, model, batch_size, alphabet)


def read_variants_from_csv(csv_path, label_col="DMS_score"):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r[label_col].strip() != ""]
    seqs = [r["mutated_sequence"].encode() for r in rows]
    labels = np.array([float(r[label_col]) for r in rows])
    return seqs, labels


def discover_dms_suite(dms_dir):
    align_dir = os.path.join(dms_dir, "alignments")
    subs_dir = os.path.join(dms_dir, "viral_dms_substitutions")
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
            print(f"WARNING: no alignment for {name}, skipping")
            continue
        suite.append({
            "name": name,
            "a2m": a2m_map[name],
            "variants_csv": os.path.join(subs_dir, f),
        })
    return suite


@torch.inference_mode()
def score_one_dms(model, alphabet, a2m_path, variants_csv, batch_size, seed, args):
    variant_seqs_raw, labels = read_variants_from_csv(variants_csv)
    variants = [
        append_startstop(alphabet.encode(v), alphabet=alphabet) for v in variant_seqs_raw
    ]

    msa_sequences = get_seqs_from_fastalike(Path(a2m_path))
    print(f"  MSA: {len(msa_sequences)} sequences", flush=True)
    candidates = preprocess_msa(msa_sequences)
    print(f"  Unique context candidates: {len(candidates)}", flush=True)

    if args.ensemble:
        params = list(itertools.product([6144, 12288, 24576], [1.0, 0.95, 0.90, 0.70, 0.50]))
    else:
        params = [(6144, 1.0)]

    logps_all = []
    for max_tokens, max_similarity in params:
        context_seqs = select_context(candidates, max_similarity, max_tokens, seed)
        this_msa = [
            append_startstop(alphabet.encode(s), alphabet=alphabet) for s in context_seqs
        ]

        forward_logps = embed_and_score(this_msa, variants, model, batch_size, alphabet)
        if args.ensemble:
            backward_logps = embed_and_score(
                [np.ascontiguousarray(s[::-1]) for s in this_msa],
                [np.ascontiguousarray(s[::-1]) for s in variants],
                model, batch_size, alphabet,
            )
            logps_all.append((forward_logps + backward_logps) / 2)
        else:
            logps_all.append(forward_logps)

    logps = np.vstack(logps_all).mean(axis=0)
    rho, _ = spearmanr(logps, labels)
    return rho, len(labels), logps


def parse_args():
    parser = argparse.ArgumentParser(description="Score PoET v1 on PoET-2 eval DMSes")
    parser.add_argument("--ckpt_path", type=str, default="data/poet.ckpt",
                        help="Path to PoET v1 checkpoint")
    parser.add_argument("--dms_dir", type=str, default="data/evals",
                        help="Directory containing alignments/ and viral_dms_substitutions/")
    parser.add_argument("--output_dir", type=str, default="data/gitignore/outputs/poet1_baseline",
                        help="Directory to save per-DMS scores and summary")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=188257)
    parser.add_argument("--ensemble", action="store_true",
                        help="Full 15-config ensemble with bidirectional scoring")
    parser.add_argument("--debug", action="store_true",
                        help="Run only 1/15 ensemble params (fast)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading PoET checkpoint from {args.ckpt_path} ...")
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model = PoET(**ckpt["hyper_parameters"]["model_spec"]["init_args"])
    model.load_state_dict({k.split(".", 1)[1]: v for k, v in ckpt["state_dict"].items()})
    del ckpt
    model = model.cuda().half().eval()

    alphabet = Uniprot21(include_gap=True, include_startstop=True, distinct_startstop=True)

    # warmup
    x = alphabet.encode(b"$WAAAGH*$WAAGW*")
    seg = torch.tensor([8, 7]).long().cuda()
    _ = model.embed(torch.from_numpy(x).long().cuda().unsqueeze(0), seg.unsqueeze(0))
    print("Model loaded and warmed up.", flush=True)

    suite = discover_dms_suite(args.dms_dir)
    print(f"Found {len(suite)} DMS datasets.\n", flush=True)

    all_results = {}
    all_spearmans = []

    for entry in suite:
        name = entry["name"]
        scores_path = os.path.join(args.output_dir, f"{name}_scores.npy")
        if os.path.exists(scores_path):
            logps = np.load(scores_path)
            _, labels = read_variants_from_csv(entry["variants_csv"])
            rho, _ = spearmanr(logps, labels)
            all_results[name] = {"spearman": rho, "n_variants": len(labels)}
            all_spearmans.append(rho)
            print(f"--- {name} --- SKIP (cached), spearman={rho:.4f}", flush=True)
            continue
        print(f"--- {name} ---", flush=True)
        try:
            rho, n_variants, logps = score_one_dms(
                model, alphabet,
                entry["a2m"], entry["variants_csv"],
                args.batch_size, args.seed, args,
            )
            all_results[name] = {"spearman": rho, "n_variants": n_variants}
            all_spearmans.append(rho)
            np.save(os.path.join(args.output_dir, f"{name}_scores.npy"), logps)
            print(f"  n_variants={n_variants}, spearman={rho:.4f}", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results[name] = {"error": str(e)}
            print(f"  ERROR: {e}", flush=True)

    print("\n\n=== SUMMARY (PoET v1 pretrained baseline) ===", flush=True)
    header = f"{'DMS':<45} {'Spearman':>10} {'N':>8}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for name, metrics in sorted(all_results.items()):
        if "error" not in metrics:
            print(f"{name:<45} {metrics['spearman']:>10.4f} {metrics['n_variants']:>8}", flush=True)
        else:
            print(f"{name:<45} {'ERROR':>10} {metrics['error']}", flush=True)
    print("-" * len(header), flush=True)
    if all_spearmans:
        print(f"{'MEAN':<45} {np.mean(all_spearmans):>10.4f} ({len(all_spearmans)} datasets)", flush=True)

    summary_path = os.path.join(args.output_dir, "poet1_baseline_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
