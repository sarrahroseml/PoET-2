#!/usr/bin/env python3
"""Model merging experiment: θ_merged = θ_pre + α * (θ_ft - θ_pre).

Tests whether small interpolation toward fine-tuned weights can improve IFQ
(or at least not hurt it) while gaining some seq_only improvement.

Evaluates single-prompt seq_only + IFQ on all 45 DMSes for each (FT, α) combo.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch

sys.path.insert(0, "src")

from poet_2.training.eval import (
    _read, ungapped, select_context, read_labels, _score_prompt,
    _load_wt_protein, discover_dms_suite, spearman,
    read_variant_seqs, ALPHA,
)

PRETRAINED = "data/gitignore/models/poet-2.ckpt"

FT_CHECKPOINTS = OrderedDict([
    ("FT55", "data/gitignore/checkpoints/d1_div-lr5e-5-mask0.00-sdrop0.0/best.ckpt"),
    ("Mid15", "data/gitignore/checkpoints/d1_mid-lr1e-5-sdrop0.0/best.ckpt"),
    ("LoRA8", "data/gitignore/checkpoints/d1_lora-r8-lr5e-4/best.ckpt"),
    ("FT14", "data/gitignore/checkpoints/d1_div-lr1e-4-mask0.15-sdrop0.5/best.ckpt"),
])

ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.5]

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


def extract_state_dict(model):
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def merge_state_dicts(pre_sd, ft_sd, alpha):
    merged = {}
    for k in pre_sd:
        if k in ft_sd and pre_sd[k].shape == ft_sd[k].shape:
            merged[k] = pre_sd[k] + alpha * (ft_sd[k] - pre_sd[k])
        else:
            merged[k] = pre_sd[k]
    return merged


def eval_model_on_suite(model, suite, alpha_val=ALPHA, seed=0):
    results = {}
    for entry in suite:
        name = entry["name"]
        try:
            names, rows = _read(entry["a2m"], upper=False)
            wt_seq = ungapped(rows[0]) if rows else None
            labels = read_labels(entry["variants_csv"], "DMS_score")
            variants = read_variant_seqs(entry["variants_csv"], label_col="DMS_score")
            query = ([wt_seq] + variants) if wt_seq is not None else variants

            _, ctx_seqs = select_context(names, rows, 1.0, 6144, seed)

            with torch.inference_mode():
                # seq_only
                adj_seq = _score_prompt(model, list(ctx_seqs), query, alpha=alpha_val)
                scores_seq = (adj_seq[1:] - adj_seq[0]) if wt_seq is not None else adj_seq
                m = min(len(scores_seq), len(labels))
                rho_seq = spearman(scores_seq[:m], labels[:m])

                # IFQ
                rho_ifq = float("nan")
                wt_struct = entry.get("structure")
                if wt_struct and os.path.isfile(wt_struct):
                    protein, struct_input, ifq_input = _load_wt_protein(wt_struct)
                    if len(protein) == (len(wt_seq) if wt_seq else 0):
                        adj_ifq = _score_prompt(
                            model, [ifq_input] + list(ctx_seqs), query,
                            ys_ref=True, self_prompt=ifq_input.sequence,
                            alpha=alpha_val,
                        )
                        scores_ifq = (adj_ifq[1:] - adj_ifq[0]) if wt_seq is not None else adj_ifq
                        rho_ifq = spearman(scores_ifq[:m], labels[:m])

            results[name] = {"seq": float(rho_seq), "ifq": float(rho_ifq)}
        except Exception as e:
            print(f"  ERROR {name}: {e}", flush=True)
            results[name] = {"seq": float("nan"), "ifq": float("nan")}
    return results


def summarize(results):
    seq_iid, seq_ood, ifq_iid, ifq_ood = [], [], [], []
    for name, r in results.items():
        s, i = r["seq"], r["ifq"]
        if name in IID:
            if not np.isnan(s): seq_iid.append(s)
            if not np.isnan(i): ifq_iid.append(i)
        else:
            if not np.isnan(s): seq_ood.append(s)
            if not np.isnan(i): ifq_ood.append(i)
    return {
        "seq_iid": float(np.mean(seq_iid)) if seq_iid else float("nan"),
        "seq_ood": float(np.mean(seq_ood)) if seq_ood else float("nan"),
        "seq_all": float(np.mean(seq_iid + seq_ood)) if (seq_iid or seq_ood) else float("nan"),
        "ifq_iid": float(np.mean(ifq_iid)) if ifq_iid else float("nan"),
        "ifq_ood": float(np.mean(ifq_ood)) if ifq_ood else float("nan"),
        "ifq_all": float(np.mean(ifq_iid + ifq_ood)) if (ifq_iid or ifq_ood) else float("nan"),
        "n_seq": len(seq_iid) + len(seq_ood),
        "n_ifq": len(ifq_iid) + len(ifq_ood),
    }


def main():
    from poet_2.models.poet_2_helpers import load_model as _load

    device = torch.device("cuda")
    dtype = torch.bfloat16

    suite = discover_dms_suite(DMS_DIR)
    print(f"Found {len(suite)} DMS datasets", flush=True)

    # Eval pretrained baseline and extract state dict for merging
    print("\n=== Pretrained baseline ===", flush=True)
    t0 = time.time()
    model = _load(PRETRAINED, device=device, dtype=dtype)
    model.eval()
    pre_sd = extract_state_dict(model)
    pre_results = eval_model_on_suite(model, suite)
    pre_summary = summarize(pre_results)
    print(f"  seq: IID={pre_summary['seq_iid']:.4f} OOD={pre_summary['seq_ood']:.4f} All={pre_summary['seq_all']:.4f}", flush=True)
    print(f"  ifq: IID={pre_summary['ifq_iid']:.4f} OOD={pre_summary['ifq_ood']:.4f} All={pre_summary['ifq_all']:.4f}", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    all_results = {"pretrained": {"per_dms": pre_results, "summary": pre_summary}}

    # For each FT checkpoint
    for ft_name, ft_path in FT_CHECKPOINTS.items():
        if not os.path.isfile(ft_path):
            print(f"\nSKIP {ft_name}: not found", flush=True)
            continue

        # Load FT model via load_model (handles all checkpoint formats) and extract state dict
        print(f"\nExtracting FT state dict: {ft_name} ({ft_path})", flush=True)
        ft_model = _load(ft_path, device=device, dtype=dtype)
        ft_sd = extract_state_dict(ft_model)
        del ft_model
        gc.collect()
        torch.cuda.empty_cache()

        for alpha in ALPHAS + [1.0]:
            label = f"{ft_name}_a{alpha:.2f}"
            print(f"\n=== {label} ===", flush=True)
            t0 = time.time()

            if alpha == 1.0:
                model = _load(ft_path, device=device, dtype=dtype)
            else:
                merged_sd = merge_state_dicts(pre_sd, ft_sd, alpha)
                model = _load(PRETRAINED, device=device, dtype=dtype)
                model.load_state_dict(merged_sd, strict=False)
                del merged_sd

            model.eval()
            results = eval_model_on_suite(model, suite)
            s = summarize(results)

            delta_seq = s["seq_all"] - pre_summary["seq_all"]
            delta_ifq = s["ifq_all"] - pre_summary["ifq_all"]
            marker = " ***" if delta_ifq > 0 else ""

            print(f"  seq: IID={s['seq_iid']:.4f} OOD={s['seq_ood']:.4f} All={s['seq_all']:.4f} (Δ={delta_seq:+.4f})", flush=True)
            print(f"  ifq: IID={s['ifq_iid']:.4f} OOD={s['ifq_ood']:.4f} All={s['ifq_all']:.4f} (Δ={delta_ifq:+.4f}){marker}", flush=True)
            print(f"  ({time.time()-t0:.0f}s)", flush=True)

            all_results[label] = {"per_dms": results, "summary": s}

            del model
            gc.collect()
            torch.cuda.empty_cache()

        del ft_sd
        gc.collect()

    # Summary table
    print(f"\n\n{'='*90}", flush=True)
    print("MODEL MERGING SUMMARY", flush=True)
    print(f"{'='*90}", flush=True)
    print(f"{'Config':<25} {'seq_IID':>8} {'seq_OOD':>8} {'seq_All':>8} {'ifq_IID':>8} {'ifq_OOD':>8} {'ifq_All':>8} {'Δifq':>7}", flush=True)
    print("-" * 90, flush=True)

    pre_ifq = pre_summary["ifq_all"]
    for label, data in all_results.items():
        s = data["summary"]
        delta = s["ifq_all"] - pre_ifq if not np.isnan(s["ifq_all"]) else float("nan")
        marker = " ***" if delta > 0 else ""
        print(f"{label:<25} {s['seq_iid']:>8.4f} {s['seq_ood']:>8.4f} {s['seq_all']:>8.4f} "
              f"{s['ifq_iid']:>8.4f} {s['ifq_ood']:>8.4f} {s['ifq_all']:>8.4f} {delta:>+7.4f}{marker}", flush=True)

    # Save
    save_path = "data/gitignore/model_merging_results.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {save_path}", flush=True)


if __name__ == "__main__":
    main()
