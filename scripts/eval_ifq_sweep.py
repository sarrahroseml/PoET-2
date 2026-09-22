#!/usr/bin/env python3
"""Inference-time IFQ parameter sweep on pretrained PoET-2.

Sweeps two parameters that affect IFQ scoring:
  1. ref_blend: ratio of encoder ref_values in the decoder blend (default 0.5)
  2. self_prompt mode: "default" (logaddexp) vs "consistency" (enforce consistency)

Uses the existing _score_prompt pipeline for correct tokenization.
Controls ref_blend via model._ref_blend and self_prompt mode via module variable.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")

from poet_2.training.eval import (
    _read, ungapped, select_context, read_labels, _score_prompt,
    _load_wt_protein, discover_dms_suite, spearman,
    read_variant_seqs, ALPHA,
)

PRETRAINED = "data/gitignore/models/poet-2.ckpt"
DMS_DIR = "data/evals"

REF_BLENDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SELF_PROMPT_MODES = ["default", "consistency"]

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


def main():
    import poet_2.models.poet_2_helpers as helpers
    from poet_2.models.poet_2_helpers import load_model as _load

    device = torch.device("cuda")
    dtype = torch.bfloat16

    suite = discover_dms_suite(DMS_DIR)
    ifq_suite = []
    for entry in suite:
        wt_struct = entry.get("structure")
        if wt_struct and os.path.isfile(wt_struct):
            ifq_suite.append(entry)
    print(f"Found {len(suite)} DMS datasets, {len(ifq_suite)} with structures for IFQ", flush=True)

    print(f"\nLoading pretrained: {PRETRAINED}", flush=True)
    model = _load(PRETRAINED, device=device, dtype=dtype)
    model.eval()

    all_results = {}

    for blend in REF_BLENDS:
        for sp_mode in SELF_PROMPT_MODES:
            label = f"blend{blend:.1f}_{sp_mode}"
            print(f"\n{'='*60}", flush=True)
            print(f"=== {label} ===", flush=True)
            t0 = time.time()

            model._ref_blend = blend
            helpers._SELF_PROMPT_MODE = sp_mode

            results = {}
            for entry in ifq_suite:
                name = entry["name"]
                try:
                    names, rows = _read(entry["a2m"], upper=False)
                    wt_seq = ungapped(rows[0]) if rows else None
                    labels = read_labels(entry["variants_csv"], "DMS_score")
                    variants = read_variant_seqs(entry["variants_csv"], label_col="DMS_score")
                    query = ([wt_seq] + variants) if wt_seq is not None else variants

                    _, ctx_seqs = select_context(names, rows, 1.0, 6144, 0)

                    protein, struct_input, ifq_input = _load_wt_protein(entry["structure"])
                    if len(protein) != (len(wt_seq) if wt_seq else 0):
                        print(f"  SKIP {name}: length mismatch", flush=True)
                        continue

                    with torch.inference_mode():
                        adj = _score_prompt(
                            model, [ifq_input] + list(ctx_seqs), query,
                            ys_ref=True, self_prompt=ifq_input.sequence,
                            alpha=ALPHA,
                        )
                        scores = (adj[1:] - adj[0]) if wt_seq is not None else adj
                        m = min(len(scores), len(labels))
                        rho = spearman(scores[:m], labels[:m])
                        results[name] = float(rho)
                        print(f"  {name}: rho={rho:.4f}", flush=True)

                except Exception as e:
                    print(f"  ERROR {name}: {e}", flush=True)
                    results[name] = float("nan")

            iid_rhos = [v for k, v in results.items() if k in IID and not np.isnan(v)]
            ood_rhos = [v for k, v in results.items() if k not in IID and not np.isnan(v)]
            all_rhos = [v for v in results.values() if not np.isnan(v)]

            s = {
                "iid": float(np.mean(iid_rhos)) if iid_rhos else float("nan"),
                "ood": float(np.mean(ood_rhos)) if ood_rhos else float("nan"),
                "all": float(np.mean(all_rhos)) if all_rhos else float("nan"),
                "n": len(all_rhos),
            }

            print(f"  IID={s['iid']:.4f} OOD={s['ood']:.4f} All={s['all']:.4f} (n={s['n']}, {time.time()-t0:.0f}s)", flush=True)
            all_results[label] = {"per_dms": results, "summary": s}

    # Reset to defaults
    model._ref_blend = 0.5
    helpers._SELF_PROMPT_MODE = "default"

    # Summary table
    print(f"\n\n{'='*80}", flush=True)
    print("IFQ PARAMETER SWEEP SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)

    baseline_all = all_results.get("blend0.5_default", {}).get("summary", {}).get("all", float("nan"))
    print(f"\nBaseline (blend=0.5, default self_prompt): ifq_All={baseline_all:.4f}\n", flush=True)

    print(f"{'Config':<30} {'IID':>7} {'OOD':>7} {'All':>7} {'Δ':>7} {'n':>3}", flush=True)
    print("-" * 65, flush=True)

    for label, data in all_results.items():
        s = data["summary"]
        delta = s["all"] - baseline_all if not np.isnan(s["all"]) and not np.isnan(baseline_all) else float("nan")
        marker = " ***" if delta > 0.001 else " *" if delta > 0 else ""
        print(f"{label:<30} {s['iid']:>7.4f} {s['ood']:>7.4f} {s['all']:>7.4f} {delta:>+7.4f}{marker}", flush=True)

    # Best per parameter
    print(f"\n--- Best ref_blend per self_prompt mode ---", flush=True)
    for sp_mode in SELF_PROMPT_MODES:
        mode_results = {k: v for k, v in all_results.items() if k.endswith(f"_{sp_mode}")}
        if mode_results:
            best_label = max(mode_results, key=lambda k: mode_results[k]["summary"]["all"])
            best_s = mode_results[best_label]["summary"]
            print(f"  {sp_mode}: best={best_label} All={best_s['all']:.4f}", flush=True)

    save_path = "data/gitignore/ifq_sweep_results.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {save_path}", flush=True)


if __name__ == "__main__":
    main()
