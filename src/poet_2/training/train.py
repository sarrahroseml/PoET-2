"""M3 continued-training loop. Runs on the GPU box (pixi env). The model import is lazy
(inside :func:`load_trainable_model`) so this module imports without flash_attn for
local inspection of the non-model pieces.

Pipeline per step:
    PoET2Dataset -> DataLoader(workers) -> collate -> GPU -> training_forward -> total_loss
    -> backward (grad-accum) -> clip -> optimizer.step -> scheduler.step

Single-GPU:   pixi run --no-lockfile-update python -m poet_2.training.train --data-dir DIR
Multi-GPU:    torchrun --nproc_per_node=8 -m poet_2.training.train --data-dir DIR
Overfit test: ... --overfit-one-batch --total-steps 100   (loss should fall toward ~0)
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os

import torch
from torch.utils.data import DataLoader, Sampler

from poet_2.training.config import TrainConfig
from poet_2.training.data import PoET2Dataset, collate_token_budget
from poet_2.training.forward import training_forward
from poet_2.training.losses import total_loss
from poet_2.training.optim import build_optimizer
from poet_2.training.schedule import build_scheduler

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def resolve_dtype(name: str) -> torch.dtype:
    return _DTYPES[name]


def load_trainable_model(checkpoint: str, device: torch.device, dtype: torch.dtype):
    """Load the released checkpoint for training; returns (model, hyper_parameters).

    bf16 + requires_grad on; assert the flash-bias env flag is unset so the structure bias
    trains. The original ``hyper_parameters`` are returned verbatim for the save path.
    """
    assert os.environ.get("POET_2_ATOM3_BIAS_FORCE_FLASH_ATTN", "0") == "0", (
        "Unset POET_2_ATOM3_BIAS_FORCE_FLASH_ATTN (=0) so structure-bias gradients flow."
    )
    from poet_2.models.poet_2_helpers import load_model  # lazy: pulls flash_attn

    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hyper_parameters = raw["hyper_parameters"]
    del raw
    model = load_model(checkpoint, device=device, dtype=dtype)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    return model, hyper_parameters


def group_indices_by_budget(token_counts, budget: int) -> list[list[int]]:
    """Greedy token-budget batching over precomputed footprints (same logic as
    data.batch_by_token_budget, but on indices so no samples are loaded)."""
    groups: list[list[int]] = []
    cur: list[int] = []
    used = 0
    for i, c in enumerate(token_counts):
        c = int(c)
        if cur and used + c > budget:
            groups.append(cur)
            cur, used = [], 0
        cur.append(i)
        used += c
    if cur:
        groups.append(cur)
    return groups


class BudgetBatchSampler(Sampler):
    """Yields pre-computed token-budget batch groups, looping forever."""

    def __init__(self, groups: list[list[int]]):
        self.groups = groups

    def __iter__(self):
        while True:
            yield from self.groups

    def __len__(self):
        return len(self.groups)


class _TrainingWrapper(torch.nn.Module):
    """Thin wrapper so DDP hooks into forward() for gradient sync."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return training_forward(self.model, batch)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def save_checkpoint(model, hyper_parameters, optimizer, scheduler, step, path,
                    *, merge_lora: bool = False) -> None:
    """Save in a format reloadable by helpers.load_model: 'model.'-prefixed state_dict +
    the original hyper_parameters verbatim (so the architecture re-ties on load)."""
    m = model.module if hasattr(model, "module") else model
    if merge_lora:
        from poet_2.training.lora import merge_lora as _merge
        import copy
        m_save = copy.deepcopy(m)
        _merge(m_save)
    else:
        m_save = m
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "state_dict": {"model." + k: v for k, v in m_save.state_dict().items()},
            "hyper_parameters": hyper_parameters,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
        },
        path,
    )


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        import torch.distributed as dist
        from datetime import timedelta

        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            timeout=timedelta(hours=2),
        )
        return rank, world, local, True
    return 0, 1, 0, False


def train(cfg: TrainConfig) -> None:
    rank, world, local, ddp = setup_distributed()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(cfg.dtype)
    torch.manual_seed(cfg.seed + rank)

    model, hyper_parameters = load_trainable_model(cfg.checkpoint, device, dtype)

    if cfg.freeze_encoder:
        from poet_2.models.poet_2 import untie_decoders

        untie_decoders(model)
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        for p in model.norm.parameters():
            p.requires_grad_(False)
        if cfg.freeze_clm_xattn:
            for p in model.clm_decoder.layers[0].multihead_attn.k_proj.parameters():
                p.requires_grad_(False)
            for p in model.clm_decoder.layers[0].multihead_attn.v_proj.parameters():
                p.requires_grad_(False)
        hyper_parameters = {**hyper_parameters, "_untied_decoders": True}
        n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if rank == 0:
            print(f"Untied decoders: {n_trainable:,} trainable / {n_frozen:,} frozen params", flush=True)

    use_lora = cfg.lora_rank > 0
    if use_lora:
        from poet_2.training.lora import apply_lora, lora_parameters

        lora_alpha = cfg.lora_alpha if cfg.lora_alpha > 0 else None
        n_lora = apply_lora(model, rank=cfg.lora_rank, alpha=lora_alpha)
        if rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"LoRA rank={cfg.lora_rank}: {n_lora:,} trainable / {total_params:,} total params", flush=True)

    all_reduce = None
    ddp_model = None
    if ddp:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP

        all_reduce = lambda t: dist.all_reduce(t)  # in-place SUM for token-count sync
        ddp_kwargs = {}
        if cfg.ddp_mode == "static_graph":
            ddp_kwargs["static_graph"] = True
        else:
            ddp_kwargs["find_unused_parameters"] = True
        ddp_model = DDP(_TrainingWrapper(model), device_ids=[local], **ddp_kwargs)

    ds = PoET2Dataset(
        cfg.data_dir, cfg=cfg.collator_config(), seed=cfg.seed, rank=rank, world_size=world
    )
    groups = group_indices_by_budget(ds.token_counts(), cfg.tokens_per_gpu)
    if not groups:
        raise RuntimeError("no batches — check data_dir / tokens_per_gpu")
    total_steps = cfg.total_steps or math.ceil(len(groups) / cfg.grad_accum)
    if ddp:
        ts = torch.tensor(total_steps, device=device)
        dist.all_reduce(ts, op=dist.ReduceOp.MIN)
        total_steps = int(ts.item())

    if use_lora:
        train_params = list(lora_parameters(model))
    else:
        train_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(train_params, cfg)
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=cfg.warmup_steps,
        total_steps=total_steps,
        kind=cfg.schedule,
        min_lr_frac=cfg.min_lr_frac,
    )

    num_workers = min(4, os.cpu_count() or 1)
    if cfg.overfit_one_batch:
        def micro_batches():
            fixed = move_batch(collate_token_budget([ds[i] for i in groups[0]]), device)
            while True:
                yield fixed
        loader_iter = micro_batches()
    else:
        loader = DataLoader(
            ds,
            batch_sampler=BudgetBatchSampler(groups),
            collate_fn=collate_token_budget,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0),
        )
        loader_iter = iter(loader)

    autocast = (
        (lambda: torch.autocast(device_type="cuda", dtype=dtype))
        if device.type == "cuda"
        else contextlib.nullcontext
    )
    wb = None
    if rank == 0 and cfg.wandb_project:
        import wandb
        wb = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run or None,
            config=cfg.to_dict(),
        )
    if rank == 0:
        print(f"steps={total_steps} batches/pass={len(groups)} world={world} "
              f"device={device} dtype={cfg.dtype} opt={cfg.optimizer} "
              f"workers={num_workers}", flush=True)

    def run_eval(step_num):
        nonlocal best_spearman
        if cfg.eval_dms_dir:
            from poet_2.training.eval import evaluate_dms_suite

            eval_model = model
            eval_model.eval()
            suite_metrics = evaluate_dms_suite(
                eval_model, cfg.eval_dms_dir,
                alpha=cfg.eval_alpha, max_similarity=cfg.eval_max_similarity,
                context_tokens=cfg.eval_context_tokens, seed=cfg.seed,
                seq_only=cfg.eval_seq_only,
                skip_ensemble=cfg.eval_skip_ensemble,
                rank=rank, world_size=world,
            )
            eval_model.train()
            mean_rho = suite_metrics.get("mean_spearman", float("nan"))
            n_eval = suite_metrics.get("n_evaluated", 0)
            if rank == 0:
                extra = ""
                for mkey in ("mean_spearman_struct", "mean_spearman_ifq"):
                    mv = suite_metrics.get(mkey)
                    if mv is not None and mv == mv:
                        extra += f" {mkey.replace('mean_', '')}={mv:.4f}"
                print(f"[eval] step {step_num} mean_spearman={mean_rho:.4f}{extra} ({n_eval} DMSes)", flush=True)
                if wb:
                    wb.log({"eval/mean_spearman": mean_rho, "eval/n_evaluated": n_eval}, step=step_num)
                    for k, v in suite_metrics.items():
                        if isinstance(v, (int, float)) and k not in ("mean_spearman",):
                            wb.log({f"eval/{k}": v}, step=step_num)
                        elif isinstance(v, dict):
                            for sk, sv in v.items():
                                if isinstance(sv, (int, float)):
                                    wb.log({f"eval/{k}_{sk}": sv}, step=step_num)
                if mean_rho > best_spearman:
                    best_spearman = mean_rho
                    save_checkpoint(model, hyper_parameters, optimizer, scheduler, step_num,
                                    os.path.join(cfg.out_dir, "best.ckpt"),
                                    merge_lora=use_lora)
                    print(f"[eval] new best checkpoint at step {step_num} (spearman={mean_rho:.4f})", flush=True)
        elif cfg.eval_a3m:
            if ddp:
                dist.barrier()
            if rank == 0:
                from poet_2.training.eval import evaluate

                model.eval()
                metrics, _ = evaluate(
                    model, cfg.eval_a3m, cfg.eval_variants,
                    wt_sequence=cfg.eval_wt or None,
                    wt_structure_path=cfg.eval_wt_structure or None,
                    af2_cache_folder=cfg.eval_af2_cache,
                    labels_csv=cfg.eval_labels or None,
                    label_col=cfg.eval_label_col,
                    alpha=cfg.eval_alpha, max_similarity=cfg.eval_max_similarity,
                    context_tokens=cfg.eval_context_tokens, seed=cfg.seed,
                )
                model.train()
                print(f"[eval] step {step_num} {metrics}", flush=True)
                if wb:
                    wb.log({f"eval/{k}": v for k, v in metrics.items()
                            if isinstance(v, (int, float))}, step=step_num)
            if ddp:
                dist.barrier()

    optimizer.zero_grad(set_to_none=True)
    step = 0
    accum = 0
    best_spearman = float("-inf")

    if cfg.eval_every:
        run_eval(0)

    no_sync = ddp_model.no_sync if ddp_model is not None else contextlib.nullcontext

    while step < total_steps:
        batch = move_batch(next(loader_iter), device)
        is_last_accum = (accum + 1 >= cfg.grad_accum)
        sync_ctx = contextlib.nullcontext if is_last_accum else no_sync
        with sync_ctx():
            with autocast():
                if ddp_model is not None:
                    xs_l, mlm_l, clm_l = ddp_model(batch)
                else:
                    xs_l, mlm_l, clm_l = training_forward(model, batch)
            loss, logs = total_loss(
                xs_l, mlm_l, clm_l, batch, world_size=world, all_reduce=all_reduce,
            )
            (loss / cfg.grad_accum).backward()
        accum += 1
        if accum < cfg.grad_accum:
            continue
        torch.nn.utils.clip_grad_norm_(train_params, cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        step += 1
        if rank == 0 and (step % cfg.log_every == 0 or step == 1):
            lr = scheduler.get_last_lr()[0]
            print(
                f"step {step}/{total_steps} lr {lr:.2e} L {logs['L_total']:.4f} "
                f"(enc {logs['L_mlm_enc']:.3f} dec {logs['L_mlm_dec']:.3f} clm {logs['L_clm_dec']:.3f})",
                flush=True,
            )
            if wb:
                wb.log({"train/loss": logs["L_total"], "train/L_mlm_enc": logs["L_mlm_enc"],
                        "train/L_mlm_dec": logs["L_mlm_dec"], "train/L_clm_dec": logs["L_clm_dec"],
                        "train/lr": lr}, step=step)
        if rank == 0 and cfg.save_every and step % cfg.save_every == 0:
            save_checkpoint(model, hyper_parameters, optimizer, scheduler, step,
                            os.path.join(cfg.out_dir, f"step{step}.ckpt"),
                            merge_lora=use_lora)
        if cfg.eval_every and step % cfg.eval_every == 0:
            run_eval(step)

    if rank == 0:
        save_checkpoint(model, hyper_parameters, optimizer, scheduler, step,
                        os.path.join(cfg.out_dir, "final.ckpt"),
                        merge_lora=use_lora)
    if ddp:
        dist.barrier()
    if wb:
        wb.finish()
    if ddp:
        dist.destroy_process_group()


def _build_argparser() -> argparse.ArgumentParser:
    d = TrainConfig()
    p = argparse.ArgumentParser(description="PoET-2 continued training (M3)")
    p.add_argument("--data-dir", default=d.data_dir, required=True)
    p.add_argument("--checkpoint", default=d.checkpoint)
    p.add_argument("--out-dir", default=d.out_dir)
    p.add_argument("--dtype", default=d.dtype, choices=list(_DTYPES))
    p.add_argument("--optimizer", default=d.optimizer, choices=["adamw", "adafactor"])
    p.add_argument("--schedule", default=d.schedule, choices=["cosine", "inverse_sqrt"])
    p.add_argument("--peak-lr", type=float, default=d.peak_lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--warmup-steps", type=int, default=d.warmup_steps)
    p.add_argument("--total-steps", type=int, default=d.total_steps)
    p.add_argument("--min-lr-frac", type=float, default=d.min_lr_frac)
    p.add_argument("--grad-clip", type=float, default=d.grad_clip)
    p.add_argument("--grad-accum", type=int, default=d.grad_accum)
    p.add_argument("--tokens-per-gpu", type=int, default=d.tokens_per_gpu)
    p.add_argument("--seq-mask-max", type=float, default=d.seq_mask_max)
    p.add_argument("--rate-cap", type=float, default=d.rate_cap)
    p.add_argument("--reversal-p", type=float, default=d.reversal_p)
    p.add_argument("--struct-dropout", type=float, default=d.struct_dropout)
    p.add_argument("--ifq-p", type=float, default=d.ifq_p)
    p.add_argument("--lora-rank", type=int, default=d.lora_rank)
    p.add_argument("--lora-alpha", type=float, default=d.lora_alpha)
    p.add_argument("--freeze-encoder", action="store_true", default=d.freeze_encoder)
    p.add_argument("--freeze-clm-xattn", action="store_true", default=d.freeze_clm_xattn)
    p.add_argument("--ddp-mode", default=d.ddp_mode, choices=["find_unused", "static_graph"])
    p.add_argument("--save-every", type=int, default=d.save_every)
    p.add_argument("--log-every", type=int, default=d.log_every)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--overfit-one-batch", action="store_true", default=d.overfit_one_batch)
    # eval hook
    p.add_argument("--eval-every", type=int, default=d.eval_every)
    p.add_argument("--eval-dms-dir", default=d.eval_dms_dir)
    p.add_argument("--eval-a3m", default=d.eval_a3m)
    p.add_argument("--eval-variants", default=d.eval_variants)
    p.add_argument("--eval-wt", default=d.eval_wt)
    p.add_argument("--eval-labels", default=d.eval_labels)
    p.add_argument("--eval-label-col", default=d.eval_label_col)
    p.add_argument("--eval-alpha", type=float, default=d.eval_alpha)
    p.add_argument("--eval-max-similarity", type=float, default=d.eval_max_similarity)
    p.add_argument("--eval-context-tokens", type=int, default=d.eval_context_tokens)
    p.add_argument("--eval-wt-structure", default=d.eval_wt_structure)
    p.add_argument("--eval-seq-only", action="store_true", default=d.eval_seq_only)
    p.add_argument("--eval-all-modes", dest="eval_seq_only", action="store_false")
    p.add_argument("--eval-skip-ensemble", action="store_true", default=d.eval_skip_ensemble)
    p.add_argument("--eval-with-ensemble", dest="eval_skip_ensemble", action="store_false")
    p.add_argument("--eval-af2-cache", default=d.eval_af2_cache)
    # wandb
    p.add_argument("--wandb-project", default=d.wandb_project)
    p.add_argument("--wandb-run", default=d.wandb_run)
    return p


def main(argv=None) -> None:
    args = _build_argparser().parse_args(argv)
    cfg = TrainConfig(**{k.replace("-", "_"): v for k, v in vars(args).items()})
    train(cfg)


if __name__ == "__main__":
    main()
