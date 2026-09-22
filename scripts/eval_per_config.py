#!/usr/bin/env python3
"""Evaluate each of the 15 ensemble configs individually to find the best one.

Runs Pretrained checkpoint with IFQ scoring for each (context_length, max_similarity)
combination across all 45 DMS datasets. Reports per-config average Spearman.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "src")

CHECKPOINT = "data/gitignore/models/poet-2.ckpt"
DMS_DIR = "data/evals"
CTX_STRUCT_DIR = "data/eval_context_structures"

CONTEXT_LENGTHS = [6144, 12288, 24576]
MAX_SIMILARITIES = [1.0, 0.95, 0.90, 0.70, 0.50]

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

ALPHA = 1.96


def main():
    import torch
    from poet_2.training.eval import (
        _read, ungapped, select_context, _score_prompt,
        _load_wt_protein, _load_context_structures,
        read_variant_seqs, read_labels, spearman, length_adjusted,
        discover_dms_suite,
    )
    from poet_2.models.poet_2_helpers import load_model

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"Loading model: {CHECKPOINT}", flush=True)
    model = load_model(CHECKPOINT, device=device, dtype=dtype)
    model.eval()

    suite = discover_dms_suite(DMS_DIR)
    print(f"Found {len(suite)} DMS datasets", flush=True)

    ctx_dir = CTX_STRUCT_DIR if os.path.isdir(CTX_STRUCT_DIR) else None

    # per_config[config_key][dms_name] = {"seq": spearman, "ifq": spearman}
    per_config: dict[str, dict[str, dict[str, float]]] = {}

    for ctx_len in CONTEXT_LENGTHS:
        for max_sim in MAX_SIMILARITIES:
            key = f"ctx{ctx_len}_sim{max_sim}"
            per_config[key] = {}

    t0 = time.time()
    with torch.inference_mode():
        for di, entry in enumerate(suite):
            name = entry["name"]
            wt_struct = entry.get("structure")
            names, rows = _read(entry["a2m"], upper=False)
            wt_seq = ungapped(rows[0]) if rows else None
            variants = read_variant_seqs(entry["variants_csv"], label_col="DMS_score")
            labels = read_labels(entry["variants_csv"], "DMS_score")
            query = ([wt_seq] + variants) if wt_seq is not None else variants

            protein, struct_input, ifq_input = None, None, None
            if wt_struct:
                try:
                    protein, struct_input, ifq_input = _load_wt_protein(wt_struct)
                    if len(protein) != len(wt_seq):
                        print(f"  SKIP struct for {name}: PDB len {len(protein)} != seq len {len(wt_seq)}", flush=True)
                        protein, struct_input, ifq_input = None, None, None
                except Exception as e:
                    print(f"  SKIP struct for {name}: {e}", flush=True)

            print(f"\n[{di+1}/{len(suite)}] {name} ({len(variants)} variants)", flush=True)

            for ctx_len in CONTEXT_LENGTHS:
                for max_sim in MAX_SIMILARITIES:
                    key = f"ctx{ctx_len}_sim{max_sim}"
                    seed = 42

                    _, ctx_seqs = select_context(names, rows, max_sim, ctx_len, seed)

                    # seq-only score
                    adj_seq = _score_prompt(model, list(ctx_seqs), query, alpha=ALPHA)
                    scores_seq = (adj_seq[1:] - adj_seq[0]) if wt_seq else adj_seq
                    m = min(scores_seq.shape[0], labels.shape[0])
                    rho_seq = spearman(scores_seq[:m], labels[:m])

                    result = {"seq": rho_seq}

                    # IFQ score
                    if ifq_input is not None:
                        adj_ifq = _score_prompt(
                            model, [ifq_input] + list(ctx_seqs), query,
                            ys_ref=True, self_prompt=ifq_input.sequence,
                            alpha=ALPHA,
                        )
                        scores_ifq = (adj_ifq[1:] - adj_ifq[0]) if wt_seq else adj_ifq
                        rho_ifq = spearman(scores_ifq[:m], labels[:m])
                        result["ifq"] = rho_ifq

                    per_config[key][name] = result

            elapsed = time.time() - t0
            print(f"  elapsed: {elapsed/60:.1f} min", flush=True)

    # Print results
    print(f"\n\n{'='*120}", flush=True)
    print("PER-CONFIG AVERAGE SPEARMAN (seq-only and IFQ)", flush=True)
    print(f"{'='*120}", flush=True)

    header = f"{'Config':<25} {'seq_all':>8} {'seq_IID':>8} {'seq_OOD':>8} {'ifq_all':>8} {'ifq_IID':>8} {'ifq_OOD':>8} {'n_ifq':>6}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    config_summary = {}
    for key in per_config:
        seq_iid, seq_ood = [], []
        ifq_iid, ifq_ood = [], []
        for dms, vals in per_config[key].items():
            rho_s = vals.get("seq", float("nan"))
            if not np.isnan(rho_s):
                (seq_iid if dms in IID else seq_ood).append(rho_s)
            rho_i = vals.get("ifq", float("nan"))
            if not np.isnan(rho_i):
                (ifq_iid if dms in IID else ifq_ood).append(rho_i)

        seq_all = seq_iid + seq_ood
        ifq_all = ifq_iid + ifq_ood
        row = {
            "seq_all": float(np.mean(seq_all)) if seq_all else float("nan"),
            "seq_IID": float(np.mean(seq_iid)) if seq_iid else float("nan"),
            "seq_OOD": float(np.mean(seq_ood)) if seq_ood else float("nan"),
            "ifq_all": float(np.mean(ifq_all)) if ifq_all else float("nan"),
            "ifq_IID": float(np.mean(ifq_iid)) if ifq_iid else float("nan"),
            "ifq_OOD": float(np.mean(ifq_ood)) if ifq_ood else float("nan"),
            "n_ifq": len(ifq_all),
        }
        config_summary[key] = row
        print(f"{key:<25} {row['seq_all']:>8.4f} {row['seq_IID']:>8.4f} {row['seq_OOD']:>8.4f} "
              f"{row['ifq_all']:>8.4f} {row['ifq_IID']:>8.4f} {row['ifq_OOD']:>8.4f} {row['n_ifq']:>6}",
              flush=True)

    # Sort by ifq_all
    print(f"\n\nRANKED BY IFQ (all):", flush=True)
    ranked = sorted(config_summary.items(), key=lambda x: -x[1]["ifq_all"])
    for i, (key, row) in enumerate(ranked):
        print(f"  {i+1}. {key:<25} ifq={row['ifq_all']:.4f}  seq={row['seq_all']:.4f}", flush=True)

    # Save full results
    out = "data/gitignore/per_config_results.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"per_config": per_config, "summary": config_summary}, f, indent=2)
    print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    main()
