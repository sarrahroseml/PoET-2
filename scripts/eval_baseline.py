"""Run zero-shot eval of the base PoET-2 checkpoint on all viral DMSes.

Runs with seq_only=False so all scoring modes are reported:
  spearman       - sequence-only context
  spearman_struct - WT structure in context
  spearman_ifq   - inverse-folding query
  spearman_af2   - AF2 structures on homologs

Usage:
    pixi run python scripts/eval_baseline.py
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from poet_2.training.eval import discover_dms_suite, evaluate


def main():
    checkpoint = "data/gitignore/models/poet-2.ckpt"
    dms_dir = "data/evals"
    af2_cache = "data/gitignore/cache/AF2"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    from poet_2.models.poet_2_helpers import load_model
    model = load_model(checkpoint, device=device, dtype=torch.bfloat16)
    model.eval()
    print("Model loaded.", flush=True)

    suite = discover_dms_suite(dms_dir)
    print(f"Found {len(suite)} DMS datasets.", flush=True)

    from poet_2.training.eval import _read, ungapped

    all_results = {}
    for entry in suite:
        name = entry["name"]
        wt_names, wt_rows = _read(entry["a2m"], upper=False)
        wt_seq = ungapped(wt_rows[0]) if wt_rows else None
        wt_struct = entry.get("structure")

        print(f"\n--- {name} ---", flush=True)
        print(f"  a2m: {entry['a2m']}", flush=True)
        print(f"  variants: {entry['variants_csv']}", flush=True)
        print(f"  structure: {wt_struct}", flush=True)

        try:
            metrics, _ = evaluate(
                model,
                entry["a2m"],
                entry["variants_csv"],
                wt_sequence=wt_seq,
                wt_structure_path=wt_struct,
                af2_cache_folder=af2_cache,
                labels_csv=entry["variants_csv"],
                label_col="DMS_score",
                alpha=1.96,
                max_similarity=1.0,
                context_tokens=6144,
                seed=0,
            )
            all_results[name] = metrics
            print(f"  n_variants={metrics.get('n_variants')}, n_context={metrics.get('n_context')}", flush=True)
            for k in ("spearman", "spearman_struct", "spearman_ifq", "spearman_af2"):
                if k in metrics:
                    print(f"  {k}={metrics[k]:.4f}", flush=True)
        except Exception as e:
            all_results[name] = {"error": str(e)}
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {e}", flush=True)

    print("\n\n=== SUMMARY ===", flush=True)
    modes = ["spearman", "spearman_struct", "spearman_ifq", "spearman_af2"]
    header = f"{'DMS':<45}" + "".join(f"{m:>18}" for m in modes)
    print(header, flush=True)
    print("-" * len(header), flush=True)

    mode_values = {m: [] for m in modes}
    for name, metrics in sorted(all_results.items()):
        if isinstance(metrics, dict) and "error" not in metrics:
            row = f"{name:<45}"
            for m in modes:
                v = metrics.get(m)
                if v is not None and not (isinstance(v, float) and v != v):
                    row += f"{v:>18.4f}"
                    mode_values[m].append(v)
                else:
                    row += f"{'n/a':>18}"
            print(row, flush=True)
        else:
            err = metrics.get("error", "unknown") if isinstance(metrics, dict) else str(metrics)
            print(f"{name:<45}  ERROR: {err}", flush=True)

    print("-" * len(header), flush=True)
    import numpy as np
    means = f"{'MEAN':<45}"
    for m in modes:
        vals = mode_values[m]
        if vals:
            means += f"{np.mean(vals):>18.4f}"
        else:
            means += f"{'n/a':>18}"
    print(means, flush=True)
    print(f"\nCounts: " + ", ".join(f"{m}={len(mode_values[m])}" for m in modes), flush=True)

    os.makedirs("data/gitignore/outputs", exist_ok=True)
    with open("data/gitignore/outputs/baseline_eval.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved to data/gitignore/outputs/baseline_eval.json", flush=True)


if __name__ == "__main__":
    main()
