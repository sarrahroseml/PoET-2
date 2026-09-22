#!/usr/bin/env python3
"""Multi-checkpoint score-level ensemble: blend 3-4 models trained differently.

Saves raw variant scores per (model, DMS) to NPZ files incrementally.
Skips already-computed (model, DMS) pairs, so it resumes cleanly.

After scoring, computes all pairwise and multi-model blends.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from itertools import combinations

import numpy as np

sys.path.insert(0, "src")

from poet_2.training.eval import (
    _read, ungapped, select_context, read_labels, _score_prompt,
    _load_wt_protein, discover_dms_suite, spearman,
    read_variant_seqs,
    ENSEMBLE_CONTEXT_LENGTHS, ENSEMBLE_MAX_SIMILARITIES, ALPHA,
)

CHECKPOINTS = {
    "Pre": "data/gitignore/models/poet-2.ckpt",
    "FT55": "data/gitignore/checkpoints/d1_div-lr5e-5-mask0.00-sdrop0.0/best.ckpt",
    "FT14": "data/gitignore/checkpoints/d1_div-lr1e-4-mask0.15-sdrop0.5/best.ckpt",
    "Mid15": "data/gitignore/checkpoints/d1_mid-lr1e-5-sdrop0.0/best.ckpt",
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
SAVE_DIR = "data/gitignore/multi_ckpt_scores"


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

        # seq_only (single prompt)
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


def load_scores(save_dir, model_name, dms_name):
    path = os.path.join(save_dir, f"{model_name}_{dms_name}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


def main():
    import torch
    from poet_2.models.poet_2_helpers import load_model as _load

    device = torch.device("cuda")
    dtype = torch.bfloat16

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Also reuse scores from the v2 run if available
    v2_dir = "data/gitignore/cross_ensemble_scores"

    suite = discover_dms_suite(DMS_DIR)
    print(f"Found {len(suite)} DMS datasets", flush=True)

    # Copy over any existing scores from v2 run
    # v2 used "Pretrained" as model name; this script uses "Pre"
    v2_to_local = {"Pretrained": "Pre"}
    if os.path.isdir(v2_dir):
        for fname in os.listdir(v2_dir):
            if fname.endswith(".npz"):
                src = os.path.join(v2_dir, fname)
                dst_name = fname
                for v2_name, local_name in v2_to_local.items():
                    if fname.startswith(v2_name + "_"):
                        dst_name = local_name + fname[len(v2_name):]
                dst = os.path.join(SAVE_DIR, dst_name)
                if not os.path.exists(dst):
                    import shutil
                    shutil.copy2(src, dst)
                    print(f"  Copied from v2: {fname} -> {dst_name}", flush=True)

    for model_name, ckpt in CHECKPOINTS.items():
        if not os.path.isfile(ckpt):
            print(f"\nSKIP {model_name}: checkpoint not found: {ckpt}", flush=True)
            continue

        # Check if all DMSes already cached
        all_cached = all(
            os.path.exists(os.path.join(SAVE_DIR, f"{model_name}_{e['name']}.npz"))
            for e in suite
        )
        if all_cached:
            print(f"\n{model_name}: all DMSes cached, skipping model load", flush=True)
            continue

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

    # ===== COMPUTE ALL BLENDS =====
    print(f"\n\n{'='*70}", flush=True)
    print("MULTI-CHECKPOINT ENSEMBLE RESULTS", flush=True)
    print(f"{'='*70}\n", flush=True)

    model_names = list(CHECKPOINTS.keys())

    # Define blend configs: (name, {model: (mode, weight)} dict)
    # Single models
    blends = []
    for mn in model_names:
        for mode in ["ens_ifq", "ens"]:
            blends.append((f"{mn}:{mode}", {mn: (mode, 1.0)}))

    # All pairwise equal-weight blends
    for m1, m2 in combinations(model_names, 2):
        # Best modes for each: Pre uses ens_ifq, FT models use ens
        mode1 = "ens_ifq" if m1 == "Pre" else "ens"
        mode2 = "ens_ifq" if m2 == "Pre" else "ens"
        blends.append((
            f"{m1}:{mode1}+{m2}:{mode2}",
            {m1: (mode1, 0.5), m2: (mode2, 0.5)}
        ))

    # 3-model blends
    for m1, m2, m3 in combinations(model_names, 3):
        modes = {mn: "ens_ifq" if mn == "Pre" else "ens" for mn in [m1, m2, m3]}
        w = 1.0 / 3.0
        blends.append((
            f"{m1}+{m2}+{m3}",
            {m1: (modes[m1], w), m2: (modes[m2], w), m3: (modes[m3], w)}
        ))

    # 4-model blend (all)
    if len(model_names) == 4:
        modes = {mn: "ens_ifq" if mn == "Pre" else "ens" for mn in model_names}
        w = 0.25
        blends.append((
            "All4",
            {mn: (modes[mn], w) for mn in model_names}
        ))

    # Pre-weighted blends (give Pre more weight)
    for w_pre in [0.4, 0.5, 0.6]:
        w_ft = (1.0 - w_pre) / (len(model_names) - 1)
        name = f"Pre({w_pre:.0%})+FTs({w_ft:.0%}ea)"
        blend_dict = {"Pre": ("ens_ifq", w_pre)}
        for mn in model_names:
            if mn != "Pre":
                blend_dict[mn] = ("ens", w_ft)
        blends.append((name, blend_dict))

    # Compute blends
    blend_results = {bn: {"iid": [], "ood": [], "all": []} for bn, _ in blends}
    per_dms = {}

    for entry in suite:
        dms_name = entry["name"]
        tag = "IID" if dms_name in IID else "OOD"
        per_dms[dms_name] = {"cat": tag}

        # Load all model scores for this DMS
        all_model_scores = {}
        labels = None
        for mn in model_names:
            data = load_scores(SAVE_DIR, mn, dms_name)
            if data is not None:
                all_model_scores[mn] = data
                if "labels" in data and labels is None:
                    labels = data["labels"]

        if labels is None:
            continue

        for blend_name, blend_spec in blends:
            parts = []
            valid = True
            for mn, (mode, weight) in blend_spec.items():
                scores = all_model_scores.get(mn, {}).get(mode)
                if scores is None:
                    valid = False
                    break
                parts.append((scores, weight))

            if not valid:
                continue

            blended = sum(s * w for s, w in parts)
            m = min(len(blended), len(labels))
            rho = spearman(blended[:m], labels[:m])

            per_dms[dms_name][blend_name] = float(rho)
            if not np.isnan(rho):
                blend_results[blend_name]["all"].append(rho)
                if dms_name in IID:
                    blend_results[blend_name]["iid"].append(rho)
                else:
                    blend_results[blend_name]["ood"].append(rho)

    # Summary table
    print(f"{'Blend':<40} {'All':>7} {'IID':>7} {'OOD':>7} {'n':>4}", flush=True)
    print("-" * 65, flush=True)

    summary = {}
    for bn, _ in blends:
        r = blend_results[bn]
        if not r["all"]:
            continue
        a = np.mean(r["all"])
        i = np.mean(r["iid"]) if r["iid"] else float("nan")
        o = np.mean(r["ood"]) if r["ood"] else float("nan")
        marker = " ***" if a > 0.5134 else ""
        print(f"{bn:<40} {a:>7.4f} {i:>7.4f} {o:>7.4f} {len(r['all']):>4}{marker}", flush=True)
        summary[bn] = {"all": float(a), "iid": float(i), "ood": float(o), "n": len(r["all"])}

    # Save
    save_path = "data/gitignore/multi_ckpt_ensemble_results.json"
    with open(save_path, "w") as f:
        json.dump({"per_dms": per_dms, "summary": summary}, f, indent=2)
    print(f"\nSaved to {save_path}", flush=True)
    print(f"\n*** marks Pre:ens_ifq baseline (0.5134) ***", flush=True)


if __name__ == "__main__":
    main()
