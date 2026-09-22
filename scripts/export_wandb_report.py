#!/usr/bin/env python3
"""Create a wandb report with all eval panels for every run in poet2-viral.

Usage:
    python scripts/export_wandb_report.py

Produces a shareable report URL with:
  - Training loss curves (total, enc, dec, clm)
  - Per-DMS eval Spearman panels
  - Mean Spearman comparison across runs
  - Run comparison table
"""

import wandb

PROJECT = "sarrah-sdpo/poet2-viral"
ENTITY = "sarrah-sdpo"
PROJECT_NAME = "poet2-viral"

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
    api = wandb.Api()
    runs = api.runs(PROJECT)

    # Filter to finished/running runs, skip crashed duplicates
    good_runs = [r for r in runs if r.state in ("finished", "running")]
    print(f"Including {len(good_runs)} runs in report")

    # Get all eval DMS keys from any run
    eval_keys = set()
    for run in good_runs:
        for k in run.summary._json_dict:
            if k.startswith("eval/") and k != "eval/mean_spearman" and k != "eval/n_evaluated":
                eval_keys.add(k)
    eval_keys = sorted(eval_keys)
    iid_keys = [k for k in eval_keys if k.replace("eval/", "") in IID]
    ood_keys = [k for k in eval_keys if k.replace("eval/", "") not in IID]

    print(f"  {len(iid_keys)} IID DMSes, {len(ood_keys)} OOD DMSes")

    import wandb.apis.reports as wr

    report = wr.Report(
        project=PROJECT_NAME,
        entity=ENTITY,
        title="PoET-2 Viral Training: All Runs Comparison",
        description="Comprehensive comparison of all training runs with per-DMS eval Spearman correlations.",
    )

    blocks = []

    # Section 1: Training loss curves
    blocks.append(wr.H1("Training Loss"))
    blocks.append(wr.PanelGrid(
        panels=[
            wr.LinePlot(x="Step", y=["train/loss"], title="Total Loss", groupby="run"),
            wr.LinePlot(x="Step", y=["train/L_mlm_enc"], title="Encoder MLM Loss", groupby="run"),
            wr.LinePlot(x="Step", y=["train/L_mlm_dec"], title="Decoder MLM Loss", groupby="run"),
            wr.LinePlot(x="Step", y=["train/L_clm_dec"], title="Decoder CLM Loss", groupby="run"),
            wr.LinePlot(x="Step", y=["train/lr"], title="Learning Rate", groupby="run"),
        ],
        runsets=[wr.Runset(project=PROJECT_NAME, entity=ENTITY)],
    ))

    # Section 2: Mean Spearman
    blocks.append(wr.H1("Mean Spearman"))
    blocks.append(wr.PanelGrid(
        panels=[
            wr.LinePlot(x="Step", y=["eval/mean_spearman"], title="Mean Spearman (all 45 DMSes)", groupby="run"),
        ],
        runsets=[wr.Runset(project=PROJECT_NAME, entity=ENTITY)],
    ))

    # Section 3: IID DMSes
    blocks.append(wr.H1("IID DMS Evals"))
    blocks.append(wr.PanelGrid(
        panels=[
            wr.LinePlot(x="Step", y=[k], title=k.replace("eval/", ""), groupby="run")
            for k in iid_keys
        ],
        runsets=[wr.Runset(project=PROJECT_NAME, entity=ENTITY)],
    ))

    # Section 4: OOD DMSes
    blocks.append(wr.H1("OOD DMS Evals"))
    blocks.append(wr.PanelGrid(
        panels=[
            wr.LinePlot(x="Step", y=[k], title=k.replace("eval/", ""), groupby="run")
            for k in ood_keys
        ],
        runsets=[wr.Runset(project=PROJECT_NAME, entity=ENTITY)],
    ))

    # Section 5: Run comparison table
    blocks.append(wr.H1("Run Summary Table"))
    blocks.append(wr.PanelGrid(
        panels=[
            wr.RunComparer(diff_only="split"),
        ],
        runsets=[wr.Runset(project=PROJECT_NAME, entity=ENTITY)],
    ))

    report.blocks = blocks
    report.save()
    print(f"\nReport URL: {report.url}")


if __name__ == "__main__":
    main()
