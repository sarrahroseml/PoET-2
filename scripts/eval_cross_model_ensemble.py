#!/usr/bin/env python3
"""Cross-model score-level ensemble: blend pretrained IFQ + fine-tuned seq/ens scores.

Loads both checkpoints, runs each on all 45 DMSes, averages per-variant scores
at different blend ratios, and reports Spearman correlations.

Blends tested:
  1. Pre:ens_ifq alone (baseline)
  2. FT:ens alone
  3. avg(Pre:ens_ifq, FT:ens)  — equal weight
  4. avg(Pre:ens_ifq, FT:seq_only)
  5. 0.7 * Pre:ens_ifq + 0.3 * FT:ens
  6. 0.3 * Pre:ens_ifq + 0.7 * FT:ens
  7. Oracle per-DMS: max(Pre:ens_ifq, FT:ens, blend)
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "src")

from poet_2.training.eval import (
    _read, ungapped, select_context, read_labels, _score_prompt,
    _load_wt_protein, discover_dms_suite, spearman, length_adjusted,
    ENSEMBLE_CONTEXT_LENGTHS, ENSEMBLE_MAX_SIMILARITIES, ALPHA,
    _load_context_structures,
)

CHECKPOINTS = {
    "Pretrained": "data/gitignore/models/poet-2.ckpt",
    "FT55": "data/gitignore/checkpoints/d1_div-lr5e-5-mask0.00-sdrop0.0/best.ckpt",
}

IID = {
    "IAV_H1_HA_Doud", "IAV_H1_HA_Wu", "IAV_H3_HA_Lee", "IAV_H5_HA_Dadonaite",
    "SARS2_BA1_SPIKE_Dadonaite", "SARS2_DELTA_SPIKE_Dadonaite",
    "SARS2_RBD_Starr_binding", "SARS2_RBD_Starr_expression",
    "SARS2_XBB15_RBD_Taylor", "SARS2_PRD0038_RBD_Starr",
    "RmYN02_RBD_Starr", "RsYN04_RBD_Starr",
    "NIPAH_F_Larsen",
    "HIV1_BF520_ENV_Haddox", "HIV1_HV1B9_ENV_DuenasDecamp", "HIV1_BG505_ENV_Haddox",
    "LASSA_GP_Carr",
}

DMS_DIR = "data/evals"
CTX_STRUCT_DIR = "data/eval_context_structures"


def score_ensemble_raw(model, names, rows, query_seqs, ifq_input=None,
                       alpha=ALPHA, seed=0):
    """15-prompt ensemble returning raw per-variant adjusted scores (not Spearman)."""
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


def run_dms(model, entry, mode="ens", seed=0):
    """Score one DMS dataset. Returns (variant_scores, labels) or (None, None) on error.

    mode: "ens" (seq+struct ensemble, no IFQ), "ens_ifq", "seq_only"
    """
    import torch

    names, rows = _read(entry["a2m"], upper=False)
    wt_seq = ungapped(rows[0]) if rows else None
    labels = read_labels(entry["variants_csv"], "DMS_score")

    from poet_2.training.eval import read_variant_seqs
    variants = read_variant_seqs(entry["variants_csv"], label_col="DMS_score")
    query = ([wt_seq] + variants) if wt_seq is not None else variants

    wt_struct_path = entry.get("structure")

    with torch.inference_mode():
        if mode == "seq_only":
            _, ctx_seqs = select_context(names, rows, 1.0, 6144, seed)
            adj = _score_prompt(model, list(ctx_seqs), query, alpha=ALPHA)
        elif mode == "ens":
            adj = score_ensemble_raw(model, names, rows, query, alpha=ALPHA, seed=seed)
        elif mode == "ens_ifq":
            if wt_struct_path is None or not os.path.isfile(wt_struct_path):
                return None, labels
            protein, struct_input, ifq_input = _load_wt_protein(wt_struct_path)
            if len(protein) != (len(wt_seq) if wt_seq else 0):
                return None, labels
            adj = score_ensemble_raw(
                model, names, rows, query, ifq_input=ifq_input,
                alpha=ALPHA, seed=seed,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    scores = (adj[1:] - adj[0]) if wt_seq is not None else adj
    return scores, labels


def main():
    import torch
    from poet_2.models.poet_2_helpers import load_model as _load

    device = torch.device("cuda")
    dtype = torch.bfloat16

    suite = discover_dms_suite(DMS_DIR)
    print(f"Found {len(suite)} DMS datasets", flush=True)

    # Storage for raw scores
    all_scores = {}  # {model_name: {dms_name: {mode: scores}}}

    for model_name, ckpt in CHECKPOINTS.items():
        print(f"\n{'='*70}", flush=True)
        print(f"Loading {model_name}: {ckpt}", flush=True)
        t0 = time.time()
        model = _load(ckpt, device=device, dtype=dtype)
        model.eval()
        print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

        all_scores[model_name] = {}

        modes = ["ens_ifq", "ens"] if model_name == "Pretrained" else ["ens", "seq_only"]

        for entry in suite:
            name = entry["name"]
            all_scores[model_name][name] = {}

            for mode in modes:
                t1 = time.time()
                scores, labels = run_dms(model, entry, mode=mode)
                elapsed = time.time() - t1

                if scores is not None:
                    all_scores[model_name][name][mode] = scores
                    rho = spearman(scores[:len(labels)], labels)
                    print(f"  {model_name}:{mode} {name}: rho={rho:.4f} ({elapsed:.1f}s)",
                          flush=True)
                else:
                    print(f"  {model_name}:{mode} {name}: SKIP (no struct)", flush=True)

                all_scores[model_name][name]["labels"] = labels

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Compute blends
    print(f"\n\n{'='*70}", flush=True)
    print("CROSS-MODEL ENSEMBLE RESULTS", flush=True)
    print(f"{'='*70}\n", flush=True)

    BLENDS = [
        ("Pre:ens_ifq", lambda p, f: p.get("ens_ifq")),
        ("Pre:ens", lambda p, f: p.get("ens")),
        ("FT:ens", lambda p, f: f.get("ens")),
        ("FT:seq_only", lambda p, f: f.get("seq_only")),
        ("50/50 Pre:eifq+FT:ens", lambda p, f: 0.5*p["ens_ifq"]+0.5*f["ens"] if "ens_ifq" in p and "ens" in f else None),
        ("70/30 Pre:eifq+FT:ens", lambda p, f: 0.7*p["ens_ifq"]+0.3*f["ens"] if "ens_ifq" in p and "ens" in f else None),
        ("30/70 Pre:eifq+FT:ens", lambda p, f: 0.3*p["ens_ifq"]+0.7*f["ens"] if "ens_ifq" in p and "ens" in f else None),
        ("50/50 Pre:ens+FT:ens", lambda p, f: 0.5*p["ens"]+0.5*f["ens"] if "ens" in p and "ens" in f else None),
        ("50/50 Pre:eifq+FT:seq", lambda p, f: 0.5*p["ens_ifq"]+0.5*f["seq_only"] if "ens_ifq" in p and "seq_only" in f else None),
    ]

    # Per-DMS results
    header = f"{'DMS':<40} {'Cat':>4}"
    for blend_name, _ in BLENDS:
        header += f"  {blend_name:>22}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    blend_results = {bn: {"iid": [], "ood": [], "all": []} for bn, _ in BLENDS}

    for entry in suite:
        name = entry["name"]
        tag = "IID" if name in IID else "OOD"
        pre_scores = all_scores.get("Pretrained", {}).get(name, {})
        ft_scores = all_scores.get("FT55", {}).get(name, {})
        labels = pre_scores.get("labels", ft_scores.get("labels"))
        if labels is None:
            continue

        row = f"{name:<40} {tag:>4}"
        for blend_name, blend_fn in BLENDS:
            blended = blend_fn(pre_scores, ft_scores)
            if blended is not None:
                m = min(len(blended), len(labels))
                rho = spearman(blended[:m], labels[:m])
                row += f"  {rho:>22.4f}"
                if not np.isnan(rho):
                    blend_results[blend_name]["all"].append(rho)
                    if name in IID:
                        blend_results[blend_name]["iid"].append(rho)
                    else:
                        blend_results[blend_name]["ood"].append(rho)
            else:
                row += f"  {'n/a':>22}"
        print(row, flush=True)

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Blend':<28} {'All':>7} {'IID':>7} {'OOD':>7} {'n_all':>6} {'n_iid':>6} {'n_ood':>6}", flush=True)
    print("-" * 75, flush=True)

    summary = {}
    for blend_name, _ in BLENDS:
        r = blend_results[blend_name]
        a = np.mean(r["all"]) if r["all"] else float("nan")
        i = np.mean(r["iid"]) if r["iid"] else float("nan")
        o = np.mean(r["ood"]) if r["ood"] else float("nan")
        print(f"{blend_name:<28} {a:>7.4f} {i:>7.4f} {o:>7.4f} {len(r['all']):>6} {len(r['iid']):>6} {len(r['ood']):>6}",
              flush=True)
        summary[blend_name] = {"all": a, "iid": i, "ood": o}

    # Save full results
    save_path = "data/gitignore/cross_model_ensemble_results.json"
    save_data = {}
    for entry in suite:
        name = entry["name"]
        pre_s = all_scores.get("Pretrained", {}).get(name, {})
        ft_s = all_scores.get("FT55", {}).get(name, {})
        labels = pre_s.get("labels", ft_s.get("labels"))
        if labels is None:
            continue
        save_data[name] = {}
        for blend_name, blend_fn in BLENDS:
            blended = blend_fn(pre_s, ft_s)
            if blended is not None:
                m = min(len(blended), len(labels))
                save_data[name][blend_name] = float(spearman(blended[:m], labels[:m]))

    save_data["_summary"] = summary
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {save_path}", flush=True)


if __name__ == "__main__":
    main()
