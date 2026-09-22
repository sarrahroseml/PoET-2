#!/usr/bin/env python3
"""Evaluate top D1 sweep checkpoints with full struct/IFQ scoring modes.

Same logic as eval_struct_modes.py but targeting the best D1 checkpoints
to see whether continued training on D1 data hurts IFQ (like the original
sdiv training did).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, "src")

CHECKPOINTS = {
    "Pretrained": "data/gitignore/models/poet-2.ckpt",
    "D1_div_lr1e4_m15_sd5": "data/gitignore/checkpoints/d1_div-lr1e-4-mask0.15-sdrop0.5/best.ckpt",
    "D1_div_lr5e5_m0_sd0": "data/gitignore/checkpoints/d1_div-lr5e-5-mask0.00-sdrop0.0/best.ckpt",
    "D1_wgt_lr5e5_m0_sd0": "data/gitignore/checkpoints/d1_wgt-lr5e-5-mask0.00-sdrop0.0/best.ckpt",
}

DMS_DIR = "data/evals"

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

MODES = [
    "spearman", "spearman_struct", "spearman_ifq",
    "spearman_ctx_struct", "spearman_ctx_struct_wt", "spearman_ctx_ifq",
    "spearman_ens", "spearman_ens_ifq",
]

CTX_STRUCT_DIR = "data/eval_context_structures"


def _eval_one_checkpoint(run_name, ckpt, dms_dir, result_path):
    import gc
    import torch
    from poet_2.training.eval import evaluate_dms_suite
    from poet_2.models.poet_2_helpers import load_model as _load

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    ctx_dir = CTX_STRUCT_DIR if os.path.isdir(CTX_STRUCT_DIR) else None
    if ctx_dir:
        n_npz = len([f for f in os.listdir(ctx_dir) if f.endswith(".npz")])
        print(f"  Context struct dir: {ctx_dir} ({n_npz} NPZ files)", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"Evaluating: {run_name} ({ckpt})", flush=True)
    print(f"{'='*70}", flush=True)

    t0 = time.time()
    model = _load(ckpt, device=device, dtype=dtype)
    model.eval()
    print(f"  Model loaded in {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    results = evaluate_dms_suite(model, dms_dir, seq_only=False, context_struct_dir=ctx_dir)
    elapsed = time.time() - t1
    print(f"  Eval done in {elapsed:.1f}s", flush=True)
    print(f"  mean_spearman (seq_only) = {results.get('mean_spearman', 'N/A')}", flush=True)

    safe = {}
    for k, v in results.items():
        if isinstance(v, (int, float, str, dict)):
            safe[k] = v
        elif isinstance(v, np.floating):
            safe[k] = float(v)
        else:
            safe[k] = str(v)

    with open(result_path, "w") as f:
        json.dump(safe, f, indent=2)
    print(f"  Saved to {result_path}", flush=True)

    del model
    gc.collect()


def main():
    mp.set_start_method("spawn", force=True)
    print(f"Evaluating {len(CHECKPOINTS)} checkpoints (each in a subprocess)", flush=True)

    all_results = {}

    save_dir = "data/gitignore"
    os.makedirs(save_dir, exist_ok=True)
    for run_name, ckpt in CHECKPOINTS.items():
        if not os.path.isfile(ckpt):
            print(f"  SKIP: {run_name} — checkpoint not found: {ckpt}", flush=True)
            continue
        result_path = os.path.join(save_dir, f"eval_d1_{run_name}.json")
        p = mp.Process(target=_eval_one_checkpoint, args=(run_name, ckpt, DMS_DIR, result_path))
        p.start()
        p.join()

        if p.exitcode != 0:
            print(f"  ERROR: {run_name} subprocess exited with code {p.exitcode}", flush=True)

        try:
            with open(result_path) as f:
                all_results[run_name] = json.load(f)
        except FileNotFoundError:
            print(f"  ERROR: no results file for {run_name}", flush=True)
            all_results[run_name] = {}

    run_names = list(all_results.keys())

    dms_names = sorted(set().union(*(
        (k for k, v in all_results[rn].items() if isinstance(v, dict) and "spearman" in v)
        for rn in run_names if all_results[rn]
    )))

    print(f"\n\n{'='*140}", flush=True)
    print("PER-DMS SPEARMAN BY SCORING MODE", flush=True)
    print(f"{'='*140}", flush=True)

    header = f"{'DMS':<40} {'Cat':>4}"
    for rn in run_names:
        for mode in MODES:
            short_mode = mode.replace("spearman_", "").replace("spearman", "seq")
            header += f"  {rn[:8]}_{short_mode:>6}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for dms in dms_names:
        tag = "IID" if dms in IID else "OOD"
        row = f"{dms:<40} {tag:>4}"
        for rn in run_names:
            entry = all_results[rn].get(dms, {})
            if isinstance(entry, dict):
                for mode in MODES:
                    v = entry.get(mode, float("nan"))
                    if isinstance(v, str):
                        row += f"  {'error':>14}"
                    else:
                        row += f"  {v:>14.4f}"
            else:
                row += f"  {'error':>14}" * len(MODES)
        print(row, flush=True)

    print(f"\n{'='*140}", flush=True)
    print("AVERAGES BY MODE", flush=True)
    print(f"{'='*140}", flush=True)

    for rn in run_names:
        print(f"\n  {rn}:", flush=True)
        for mode in MODES:
            short = mode.replace("spearman_", "").replace("spearman", "seq_only")
            iid_vals = []
            ood_vals = []
            for dms in dms_names:
                entry = all_results[rn].get(dms, {})
                if not isinstance(entry, dict):
                    continue
                v = entry.get(mode, float("nan"))
                if isinstance(v, str) or np.isnan(v):
                    continue
                if dms in IID:
                    iid_vals.append(v)
                else:
                    ood_vals.append(v)
            all_vals = iid_vals + ood_vals
            iid_avg = np.mean(iid_vals) if iid_vals else float("nan")
            ood_avg = np.mean(ood_vals) if ood_vals else float("nan")
            all_avg = np.mean(all_vals) if all_vals else float("nan")
            print(f"    {short:<12} IID({len(iid_vals):>2})={iid_avg:.4f}  "
                  f"OOD({len(ood_vals):>2})={ood_avg:.4f}  "
                  f"All({len(all_vals):>2})={all_avg:.4f}", flush=True)

    out = "data/gitignore/d1_struct_eval_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    main()
