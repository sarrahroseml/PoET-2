"""Train-time configuration. Pure dataclass — no torch/model imports."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TrainConfig:
    # --- data (produced by your separate prep; see poet_2.training.data) ---
    data_dir: str = ""  # MaterializedDataset directory

    # --- model / checkpoint ---
    checkpoint: str = "data/gitignore/models/poet-2.ckpt"
    dtype: str = "bf16"  # bf16 | fp16 | fp32  (spec §8.4: bf16 for training)

    # --- optimizer / schedule (continual-training values; spec §8.4) ---
    optimizer: str = "adamw"  # adamw | adafactor
    peak_lr: float = 3e-4  # lowered continual peak (spec §8.4: ~1e-4..1e-3)
    weight_decay: float = 0.0
    schedule: str = "cosine"  # cosine | inverse_sqrt
    warmup_steps: int = 200  # short warmup for a continual run (spec §8.4)
    total_steps: int = 0  # 0 => derive from one pass over the materialized shard
    min_lr_frac: float = 0.0  # cosine floor as a fraction of peak_lr
    grad_clip: float = 1.0
    grad_accum: int = 1

    # --- batching ---
    tokens_per_gpu: int = 45056  # spec §8.4 (45,056 tokens/GPU)

    # --- train-time masking (NOT prep settings) ---
    seq_mask_max: float = 0.30
    rate_cap: float = 0.30
    reversal_p: float = 0.5

    # --- distributed ---
    # static_graph is cheaper but requires a constant param set each step (feed all-masked
    # structure so the bias params are always in-graph); else find_unused_parameters.
    ddp_mode: str = "find_unused"  # find_unused | static_graph

    # --- io / logging ---
    out_dir: str = "data/gitignore/checkpoints"
    save_every: int = 1000
    log_every: int = 10
    seed: int = 0

    # --- eval hook (spec §9 zero-shot LLR; 0 disables). Needs an a3m + variants (+labels). ---
    eval_every: int = 0
    eval_a3m: str = ""  # homolog MSA for context selection (e.g. data/BLAT_ECOLX_ColabFold_2202.a3m)
    eval_variants: str = ""  # variants FASTA to score
    eval_wt: str = ""  # WT sequence -> report WT-relative LLRs (else raw adjusted LLs)
    eval_labels: str = ""  # CSV of experimental fitness, for Spearman
    eval_label_col: str = "DMS_score"
    eval_alpha: float = 1.96  # length adjustment (spec §9)
    eval_max_similarity: float = 1.0  # keep context homologs with identity-to-WT <= this
    eval_context_tokens: int = 6144

    # --- debug ---
    overfit_one_batch: bool = False  # repeat the first batch (set --total-steps too)

    def collator_config(self):
        """Build the data-module CollatorConfig from the train-time masking knobs."""
        from poet_2.training.data import CollatorConfig

        return CollatorConfig(
            seq_mask_max=self.seq_mask_max,
            rate_cap=self.rate_cap,
            reversal_p=self.reversal_p,
        )

    def to_dict(self) -> dict:
        return asdict(self)
