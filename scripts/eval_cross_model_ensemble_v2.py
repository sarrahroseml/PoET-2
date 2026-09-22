#!/usr/bin/env python3
"""Cross-model score-level ensemble v2: saves per-DMS variant scores incrementally.

Loads both models, scores all DMSes, saves raw variant scores to NPZ files
so we never lose progress. Then blends and reports.
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
    read_variant_seqs,
    ENSEMBLE_CONTEXT_LENGTHS, ENSEMBLE_MAX_SIMILARITIES, ALPHA,
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
SAVE_DIR = "data/gitignore/cross_ensemble_scores"


def score_ensemble_raw(model, names, rows, query_seqs, ifq_input=None,
                       alpha=ALPHA, seed=0):
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


def run_and_save(model, model_name, entry, save_dir, seed=0):
    """Score one DMS with ens and ens_ifq, save raw scores to NPZ."""
    import torch

    name = entry["name"]
    save_path = os.path.join(save_dir, f"{model_name}_{name}.npz")
    if os.path.exists(save_path):
        print(f"  SKIP (cached): {model_name} {name}", flush=True)
        return

    names, rows = _read(entry["a2m"], upper=False)
    wt_seq = ungapped(rows[0]) if rows else None
    labels = read_labels(entry["variants_csv"], "DMS_score")
    variants = read_variant_seqs(entry["variants_csv"], label_col="DMS_score")
    query = ([wt_seq] + variants) if wt_seq is not None else variants

    results = {"labels": labels}

    with torch.inference_mode():
        # ens (seq+struct, no IFQ)
        t0 = time.time()
        adj_ens = score_ensemble_raw(model, names, rows, query, alpha=ALPHA, seed=seed)
        scores_ens = (adj_ens[1:] - adj_ens[0]) if wt_seq is not None else adj_ens
        results["ens"] = scores_ens
        rho_ens = spearman(scores_ens[:len(labels)], labels)
        print(f"  {model_name}:ens {name}: rho={rho_ens:.4f} ({time.time()-t0:.1f}s)", flush=True)

        # seq_only (single prompt, no ensemble)
        t0 = time.time()
        _, ctx_seqs = select_context(names, rows, 1.0, 6144, seed)
        adj_seq = _score_prompt(model, list(ctx_seqs), query, alpha=ALPHA)
        scores_seq = (adj_seq[1:] - adj_seq[0]) if wt_seq is not None else adj_seq
        results["seq_only"] = scores_seq
        rho_seq = spearman(scores_seq[:len(labels)], labels)
        print(f"  {model_name}:seq {name}: rho={rho_seq:.4f} ({time.time()-t0:.1f}s)", flush=True)

        # ens_ifq
        wt_struct_path = entry.get("structure")
        if wt_struct_path and os.path.isfile(wt_struct_path):
            protein, struct_input, ifq_input = _load_wt_protein(wt_struct_path)
            if len(protein) == (len(wt_seq) if wt_seq else 0):
                t0 = time.time()
                adj_ifq = score_ensemble_raw(
                    model, names, rows, query, ifq_input=ifq_input,
                    alpha=ALPHA, seed=seed,
                )
                scores_ifq = (adj_ifq[1:] - adj_ifq[0]) if wt_seq is not None else adj_ifq
                results["ens_ifq"] = scores_ifq
                rho_ifq = spearman(scores_ifq[:len(labels)], labels)
                print(f"  {model_name}:eifq {name}: rho={rho_ifq:.4f} ({time.time()-t0:.1f}s)", flush=True)

    np.savez_compressed(save_path, **results)


def main():
    import torch
    from poet_2.models.poet_2_helpers import load_model as _load

    device = torch.device("cuda")
    dtype = torch.bfloat16

    os.makedirs(SAVE_DIR, exist_ok=True)
    suite = discover_dms_suite(DMS_DIR)
    print(f"Found {len(suite)} DMS datasets", flush=True)

    for model_name, ckpt in CHECKPOINTS.items():
        print(f"\n{'='*70}", flush=True)
        print(f"Loading {model_name}: {ckpt}", flush=True)
        t0 = time.time()
        model = _load(ckpt, device=device, dtype=dtype)
        model.eval()
        print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

        for entry in suite:
            try:
                run_and_save(model, model_name, entry, SAVE_DIR)
            except Exception as e:
                print(f"  ERROR {model_name} {entry['name']}: {e}", flush=True)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Compute blends
    print(f"\n\n{'='*70}", flush=True)
    print("CROSS-MODEL ENSEMBLE RESULTS", flush=True)
    print(f"{'='*70}\n", flush=True)

    BLENDS = [
        ("Pre:ens_ifq", "Pretrained", "ens_ifq", None, None, None),
        ("Pre:ens", "Pretrained", "ens", None, None, None),
        ("FT:ens", None, None, "FT55", "ens", None),
        ("FT:seq_only", None, None, "FT55", "seq_only", None),
        ("50/50 eifq+ens", "Pretrained", "ens_ifq", "FT55", "ens", 0.5),
        ("70/30 eifq+ens", "Pretrained", "ens_ifq", "FT55", "ens", 0.7),
        ("30/70 eifq+ens", "Pretrained", "ens_ifq", "FT55", "ens", 0.3),
        ("50/50 Pre+FT ens", "Pretrained", "ens", "FT55", "ens", 0.5),
        ("50/50 eifq+seq", "Pretrained", "ens_ifq", "FT55", "seq_only", 0.5),
    ]

    blend_results = {bn: {"iid": [], "ood": [], "all": []} for bn, *_ in BLENDS}
    per_dms = {}

    for entry in suite:
        name = entry["name"]
        tag = "IID" if name in IID else "OOD"
        per_dms[name] = {"cat": tag}

        # Load saved scores
        saved = {}
        for model_name in CHECKPOINTS:
            path = os.path.join(SAVE_DIR, f"{model_name}_{name}.npz")
            if os.path.exists(path):
                data = np.load(path)
                saved[model_name] = {k: data[k] for k in data.files}

        labels = None
        for mn in saved:
            if "labels" in saved[mn]:
                labels = saved[mn]["labels"]
                break
        if labels is None:
            continue

        row_vals = {}
        for blend_name, m1, mode1, m2, mode2, w in BLENDS:
            s1 = saved.get(m1, {}).get(mode1) if m1 else None
            s2 = saved.get(m2, {}).get(mode2) if m2 else None

            if m1 and m2:
                if s1 is None or s2 is None:
                    continue
                blended = w * s1 + (1 - w) * s2
            elif m1:
                if s1 is None:
                    continue
                blended = s1
            else:
                if s2 is None:
                    continue
                blended = s2

            m = min(len(blended), len(labels))
            rho = spearman(blended[:m], labels[:m])
            row_vals[blend_name] = rho
            if not np.isnan(rho):
                blend_results[blend_name]["all"].append(rho)
                if name in IID:
                    blend_results[blend_name]["iid"].append(rho)
                else:
                    blend_results[blend_name]["ood"].append(rho)

        per_dms[name].update(row_vals)

    # Print per-DMS table
    blend_names = [bn for bn, *_ in BLENDS]
    header = f"{'DMS':<40} {'Cat':>4}"
    for bn in blend_names:
        header += f"  {bn:>18}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for entry in suite:
        name = entry["name"]
        if name not in per_dms:
            continue
        d = per_dms[name]
        row = f"{name:<40} {d['cat']:>4}"
        for bn in blend_names:
            v = d.get(bn)
            if v is not None:
                row += f"  {v:>18.4f}"
            else:
                row += f"  {'n/a':>18}"
        print(row, flush=True)

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Blend':<22} {'All':>7} {'IID':>7} {'OOD':>7} {'n':>4}", flush=True)
    print("-" * 55, flush=True)

    summary = {}
    for bn, *_ in BLENDS:
        r = blend_results[bn]
        a = np.mean(r["all"]) if r["all"] else float("nan")
        i = np.mean(r["iid"]) if r["iid"] else float("nan")
        o = np.mean(r["ood"]) if r["ood"] else float("nan")
        print(f"{bn:<22} {a:>7.4f} {i:>7.4f} {o:>7.4f} {len(r['all']):>4}", flush=True)
        summary[bn] = {"all": float(a), "iid": float(i), "ood": float(o), "n": len(r["all"])}

    save_path = "data/gitignore/cross_model_ensemble_results.json"
    with open(save_path, "w") as f:
        json.dump({"per_dms": per_dms, "summary": summary}, f, indent=2)
    print(f"\nSaved to {save_path}", flush=True)


if __name__ == "__main__":
    main()
