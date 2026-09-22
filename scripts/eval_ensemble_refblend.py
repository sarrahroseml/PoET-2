#!/usr/bin/env python3
"""15-prompt ensemble IFQ eval at ref_blend in {0.5, 0.6}.

Answers two questions:
  1. Does the single-prompt ref_blend=0.6 gain (+0.0027) hold for the full
     15-prompt ensemble (ens_ifq, the number that matters)?
  2. Does that free inference-time gain STACK on the untied LR 1e-5 checkpoint,
     i.e. best trained seq_only + best inference IFQ in one model?

Runs ens_ifq (15-prompt) for each (model, blend). Also records ens (no-IFQ) and
seq_only once per model as reference.  Results saved to JSON.
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
    _load_wt_protein, discover_dms_suite, spearman, read_variant_seqs,
    ENSEMBLE_CONTEXT_LENGTHS, ENSEMBLE_MAX_SIMILARITIES, ALPHA,
)

CHECKPOINTS = {
    "Untied15": "data/gitignore/checkpoints/d1_untied-lr1e-5-sdrop0.0/best.ckpt",
}
REF_BLENDS = [0.5]
DMS_DIR = "data/evals"
OUT_JSON = "data/gitignore/ensemble_refblend_untied15.json"

IID = {
    "IAV_H1_HA_Doud", "IAV_H1_HA_Wu", "IAV_H3_HA_Lee", "IAV_H5_HA_Dadonaite",
    "SARS2_BA1_SPIKE_Dadonaite", "SARS2_DELTA_SPIKE_Dadonaite",
    "SARS2_RBD_Starr_binding", "SARS2_RBD_Starr_expression",
    "SARS2_XBB15_RBD_Taylor", "SARS2_PRD0038_RBD_Starr",
    "RmYN02_RBD_Starr", "RsYN04_RBD_Starr", "NIPAH_F_Larsen",
    "HIV1_BF520_ENV_Haddox", "HIV1_HV1B9_ENV_DuenasDecamp", "HIV1_BG505_ENV_Haddox",
    "LASSA_GP_Carr",
}


def ens_ifq_score(model, names, rows, query_seqs, ifq_input, alpha=ALPHA, seed=0):
    """15-prompt ensemble IFQ (Cartesian product of ctx lengths x max sims)."""
    all_adj = []
    for ctx_len in ENSEMBLE_CONTEXT_LENGTHS:
        for max_sim in ENSEMBLE_MAX_SIMILARITIES:
            prompt_seed = seed + len(all_adj)
            _, ctx_seqs = select_context(names, rows, max_sim, ctx_len, prompt_seed)
            adj = _score_prompt(
                model, [ifq_input] + list(ctx_seqs), query_seqs,
                ys_ref=True, self_prompt=ifq_input.sequence, alpha=alpha,
            )
            all_adj.append(adj)
    return np.mean(all_adj, axis=0)


def summarize(results):
    iid = [v for k, v in results.items() if k in IID and not np.isnan(v)]
    ood = [v for k, v in results.items() if k not in IID and not np.isnan(v)]
    allv = [v for v in results.values() if not np.isnan(v)]
    return {
        "iid": float(np.mean(iid)) if iid else float("nan"),
        "ood": float(np.mean(ood)) if ood else float("nan"),
        "all": float(np.mean(allv)) if allv else float("nan"),
        "n": len(allv),
    }


def main():
    import torch
    from poet_2.models.poet_2_helpers import load_model as _load

    device = torch.device("cuda")
    dtype = torch.bfloat16

    suite = discover_dms_suite(DMS_DIR)
    ifq_suite = [e for e in suite if e.get("structure") and os.path.isfile(e["structure"])]
    print(f"Found {len(suite)} DMS, {len(ifq_suite)} with structures for IFQ", flush=True)

    out = {}
    for model_name, ckpt in CHECKPOINTS.items():
        if not os.path.isfile(ckpt):
            print(f"SKIP {model_name}: not found {ckpt}", flush=True)
            continue
        print(f"\n{'='*70}\nLoading {model_name}: {ckpt}", flush=True)
        t0 = time.time()
        model = _load(ckpt, device=device, dtype=dtype)
        model.eval()
        print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

        for blend in REF_BLENDS:
            model._ref_blend = blend
            label = f"{model_name}:ens_ifq:blend{blend:.1f}"
            print(f"\n--- {label} ---", flush=True)
            results = {}
            for entry in ifq_suite:
                name = entry["name"]
                try:
                    names, rows = _read(entry["a2m"], upper=False)
                    wt_seq = ungapped(rows[0]) if rows else None
                    labels = read_labels(entry["variants_csv"], "DMS_score")
                    variants = read_variant_seqs(entry["variants_csv"], label_col="DMS_score")
                    query = ([wt_seq] + variants) if wt_seq is not None else variants
                    protein, _, ifq_input = _load_wt_protein(entry["structure"])
                    if len(protein) != (len(wt_seq) if wt_seq else 0):
                        print(f"  SKIP {name}: length mismatch", flush=True)
                        continue
                    with torch.inference_mode():
                        adj = ens_ifq_score(model, names, rows, query, ifq_input)
                        scores = (adj[1:] - adj[0]) if wt_seq is not None else adj
                        m = min(len(scores), len(labels))
                        rho = spearman(scores[:m], labels[:m])
                    results[name] = float(rho)
                    print(f"  {name}: {rho:.4f}", flush=True)
                except Exception as e:
                    print(f"  ERROR {name}: {e}", flush=True)
                    results[name] = float("nan")
            s = summarize(results)
            out[label] = {"summary": s, "per_dms": results}
            print(f"  >> {label}: all={s['all']:.4f} iid={s['iid']:.4f} ood={s['ood']:.4f} (n={s['n']})", flush=True)
            with open(OUT_JSON, "w") as f:
                json.dump(out, f, indent=2)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n\n{'='*70}\nSUMMARY (ens_ifq)\n{'='*70}", flush=True)
    for label, d in out.items():
        s = d["summary"]
        print(f"  {label:32s} all={s['all']:.4f} iid={s['iid']:.4f} ood={s['ood']:.4f}", flush=True)
    print(f"\nSaved -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
