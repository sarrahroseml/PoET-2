#!/usr/bin/env python3
"""Evaluate multiple checkpoints on the full DMS suite and report per-DMS Spearman,
split into IID vs OOD categories.

Usage (GPU required):
    pixi run --frozen python scripts/eval_per_dms.py

Outputs a TSV table to stdout and saves results to data/gitignore/per_dms_results.tsv
"""

from __future__ import annotations

import gc
import json
import sys
import time

import torch

from poet_2.training.eval import evaluate_dms_suite

# ---------------------------------------------------------------------------
# Checkpoints to evaluate
# ---------------------------------------------------------------------------

CHECKPOINTS = {
    "Baseline (pretrained)": "data/gitignore/models/poet-2.ckpt",
    "LR=1e-4, mask=0.30": "data/gitignore/checkpoints/weighted-lr1e-4/best.ckpt",
    "LR=1e-4, mask=0.80": "data/gitignore/checkpoints/sweep-lr1e-4-mask0.80/best.ckpt",
    "LR=1e-5, mask=0.30": "data/gitignore/checkpoints/weighted-lr1e-5/best.ckpt",
}

DMS_DIR = "data/evals"

# ---------------------------------------------------------------------------
# IID / OOD split
# ---------------------------------------------------------------------------

IID_DATASETS = {
    # HA family
    "IAV_H1_HA_Doud",
    "IAV_H1_HA_Wu",
    "IAV_H3_HA_Lee",
    "IAV_H5_HA_Dadonaite",
    # Spike full
    "SARS2_BA1_SPIKE_Dadonaite",
    "SARS2_DELTA_SPIKE_Dadonaite",
    # Spike RBD
    "SARS2_RBD_Starr_binding",
    "SARS2_RBD_Starr_expression",
    "SARS2_XBB15_RBD_Taylor",
    "SARS2_PRD0038_RBD_Starr",
    "RmYN02_RBD_Starr",
    "RsYN04_RBD_Starr",
    # Paramyxo F
    "NIPAH_F_Larsen",
    # Retroviral Env
    "HIV1_BF520_ENV_Haddox",
    "HIV1_HV1B9_ENV_DuenasDecamp",
    "HIV1_BG505_ENV_Haddox",
    # Arenavirus GPC
    "LASSA_GP_Carr",
}

assert len(IID_DATASETS) == 17, f"Expected 17 IID datasets, got {len(IID_DATASETS)}"


def load_model(checkpoint_path: str, device: torch.device, dtype: torch.dtype):
    """Load a checkpoint for inference."""
    from poet_2.models.poet_2_helpers import load_model as _load

    model = _load(checkpoint_path, device=device, dtype=dtype)
    model.eval()
    return model


def evaluate_checkpoint(
    name: str, checkpoint_path: str, device: torch.device, dtype: torch.dtype
) -> dict[str, float]:
    """Load checkpoint, run full DMS suite, return per-DMS Spearman dict."""
    print(f"\n{'='*70}", flush=True)
    print(f"Evaluating: {name}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"{'='*70}", flush=True)

    t0 = time.time()
    model = load_model(checkpoint_path, device, dtype)
    print(f"  Model loaded in {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    results = evaluate_dms_suite(model, DMS_DIR, seq_only=True)
    elapsed = time.time() - t1
    print(f"  Eval done in {elapsed:.1f}s", flush=True)

    # Extract per-DMS Spearman values (exclude meta keys)
    per_dms = {}
    for k, v in results.items():
        if k in ("mean_spearman", "n_evaluated", "n_total", "error"):
            continue
        if isinstance(v, float):
            per_dms[k] = v
        else:
            print(f"  WARNING: {k} = {v}", flush=True)

    print(f"  mean_spearman = {results.get('mean_spearman', 'N/A')}", flush=True)
    print(f"  n_evaluated = {results.get('n_evaluated', 'N/A')}", flush=True)

    # Free GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return per_dms


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    print(f"Device: {device}, dtype: {dtype}", flush=True)

    # Collect results: {run_name: {dms_name: spearman}}
    all_results: dict[str, dict[str, float]] = {}

    for run_name, ckpt_path in CHECKPOINTS.items():
        per_dms = evaluate_checkpoint(run_name, ckpt_path, device, dtype)
        all_results[run_name] = per_dms

    # Get all DMS names (sorted)
    all_dms = sorted(
        set().union(*(r.keys() for r in all_results.values()))
    )

    # Classify IID / OOD
    iid_names = [d for d in all_dms if d in IID_DATASETS]
    ood_names = [d for d in all_dms if d not in IID_DATASETS]

    run_names = list(CHECKPOINTS.keys())

    # Compute averages
    def avg(names, results_dict):
        vals = [results_dict.get(n, float("nan")) for n in names]
        valid = [v for v in vals if v == v]  # exclude NaN
        return sum(valid) / len(valid) if valid else float("nan")

    # Print table
    print("\n\n" + "=" * 100, flush=True)
    print("PER-DMS SPEARMAN RESULTS", flush=True)
    print("=" * 100, flush=True)

    # Header
    col_w = 18
    header = f"{'DMS Dataset':<35} {'Cat':>4}"
    for rn in run_names:
        # Shorten run names for columns
        short = rn.replace("Baseline (pretrained)", "Baseline").replace(", mask=", "/m")
        header += f" {short:>{col_w}}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    # IID section
    print("--- IID (homologous to training) ---", flush=True)
    for dms in iid_names:
        row = f"{dms:<35} {'IID':>4}"
        for rn in run_names:
            v = all_results[rn].get(dms, float("nan"))
            row += f" {v:>{col_w}.5f}"
        print(row, flush=True)

    # OOD section
    print("\n--- OOD (no homology to training) ---", flush=True)
    for dms in ood_names:
        row = f"{dms:<35} {'OOD':>4}"
        for rn in run_names:
            v = all_results[rn].get(dms, float("nan"))
            row += f" {v:>{col_w}.5f}"
        print(row, flush=True)

    # Averages
    print("\n" + "-" * len(header), flush=True)
    for label, names in [("IID Average (17)", iid_names), ("OOD Average (28)", ood_names), ("Overall Average (45)", all_dms)]:
        row = f"{label:<35} {'':>4}"
        for rn in run_names:
            v = avg(names, all_results[rn])
            row += f" {v:>{col_w}.5f}"
        print(row, flush=True)

    # Also print deltas vs baseline
    print("\n\n" + "=" * 100, flush=True)
    print("DELTA vs BASELINE (positive = improvement)", flush=True)
    print("=" * 100, flush=True)

    baseline_results = all_results[run_names[0]]
    for label, names in [("IID Average", iid_names), ("OOD Average", ood_names), ("Overall Average", all_dms)]:
        row = f"{label:<35} {'':>4}"
        for rn in run_names:
            v = avg(names, all_results[rn]) - avg(names, baseline_results)
            sign = "+" if v >= 0 else ""
            row += f" {sign}{v:>{col_w-1}.5f}"
        print(row, flush=True)

    # Save full results as TSV
    out_path = "data/gitignore/per_dms_results.tsv"
    with open(out_path, "w") as f:
        # Header
        f.write("DMS\tCategory\t" + "\t".join(run_names) + "\n")
        for dms in all_dms:
            cat = "IID" if dms in IID_DATASETS else "OOD"
            vals = "\t".join(f"{all_results[rn].get(dms, float('nan')):.5f}" for rn in run_names)
            f.write(f"{dms}\t{cat}\t{vals}\n")
        # Averages
        for label, names in [("IID_Average", iid_names), ("OOD_Average", ood_names), ("Overall_Average", all_dms)]:
            vals = "\t".join(f"{avg(names, all_results[rn]):.5f}" for rn in run_names)
            f.write(f"{label}\t-\t{vals}\n")

    print(f"\nResults saved to {out_path}", flush=True)

    # Also save as JSON for programmatic access
    json_path = "data/gitignore/per_dms_results.json"
    json_data = {
        "checkpoints": CHECKPOINTS,
        "iid_datasets": sorted(IID_DATASETS),
        "results": {rn: {k: round(v, 5) for k, v in rd.items()} for rn, rd in all_results.items()},
        "averages": {
            rn: {
                "iid": round(avg(iid_names, all_results[rn]), 5),
                "ood": round(avg(ood_names, all_results[rn]), 5),
                "overall": round(avg(all_dms, all_results[rn]), 5),
            }
            for rn in run_names
        },
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON saved to {json_path}", flush=True)


if __name__ == "__main__":
    main()
