"""M3 continued-training loop. Runs on the GPU box (pixi env). The model import is lazy
(inside :func:`load_trainable_model`) so this module imports without flash_attn for
local inspection of the non-model pieces.

Pipeline per step:
    MaterializedDataset -> token-budget batch -> collate -> training_forward -> total_loss
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

from poet_2.training.config import TrainConfig
from poet_2.training.data import MaterializedDataset, collate_token_budget
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


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def save_checkpoint(model, hyper_parameters, optimizer, scheduler, step, path) -> None:
    """Save in a format reloadable by helpers.load_model: 'model.'-prefixed state_dict +
    the original hyper_parameters verbatim (so the architecture re-ties on load)."""
    m = model.module if hasattr(model, "module") else model
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "state_dict": {"model." + k: v for k, v in m.state_dict().items()},
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

        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        return rank, world, local, True
    return 0, 1, 0, False


def train(cfg: TrainConfig) -> None:
    rank, world, local, ddp = setup_distributed()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(cfg.dtype)
    torch.manual_seed(cfg.seed + rank)

    model, hyper_parameters = load_trainable_model(cfg.checkpoint, device, dtype)
    all_reduce = None
    if ddp:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP

        ddp_kw = (
            {"static_graph": True}
            if cfg.ddp_mode == "static_graph"
            else {"find_unused_parameters": True}
        )
        device_ids = [local] if torch.cuda.is_available() else None
        model = DDP(model, device_ids=device_ids, **ddp_kw)
        all_reduce = lambda t: dist.all_reduce(t)  # in-place SUM

    ds = MaterializedDataset(
        cfg.data_dir, cfg=cfg.collator_config(), seed=cfg.seed, rank=rank, world_size=world
    )
    groups = group_indices_by_budget(ds.token_counts(), cfg.tokens_per_gpu)
    if not groups:
        raise RuntimeError("no batches — check data_dir / tokens_per_gpu")
    total_steps = cfg.total_steps or math.ceil(len(groups) / cfg.grad_accum)

    optimizer = build_optimizer((model.module if ddp else model).parameters(), cfg)
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=cfg.warmup_steps,
        total_steps=total_steps,
        kind=cfg.schedule,
        min_lr_frac=cfg.min_lr_frac,
    )
    raw_model = model.module if ddp else model

    def micro_batches():
        if cfg.overfit_one_batch:
            fixed = move_batch(collate_token_budget([ds[i] for i in groups[0]]), device)
            while True:
                yield fixed
        else:
            while True:  # loop the shard if total_steps exceeds one pass
                for g in groups:
                    yield move_batch(collate_token_budget([ds[i] for i in g]), device)

    autocast = (
        (lambda: torch.autocast(device_type="cuda", dtype=dtype))
        if device.type == "cuda"
        else contextlib.nullcontext
    )
    if rank == 0:
        print(f"steps={total_steps} batches/pass={len(groups)} world={world} "
              f"device={device} dtype={cfg.dtype} opt={cfg.optimizer}", flush=True)

    optimizer.zero_grad(set_to_none=True)
    step = 0
    accum = 0
    gen = micro_batches()
    while step < total_steps:
        batch = next(gen)
        with autocast():
            xs_l, mlm_l, clm_l = training_forward(model, batch)
        loss, logs = total_loss(
            xs_l, mlm_l, clm_l, batch, world_size=world, all_reduce=all_reduce
        )
        (loss / cfg.grad_accum).backward()
        accum += 1
        if accum < cfg.grad_accum:
            continue
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)
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
        if rank == 0 and cfg.save_every and step % cfg.save_every == 0:
            save_checkpoint(model, hyper_parameters, optimizer, scheduler, step,
                            os.path.join(cfg.out_dir, f"step{step}.ckpt"))
        if rank == 0 and cfg.eval_every and cfg.eval_a3m and step % cfg.eval_every == 0:
            from poet_2.training.eval import evaluate  # lazy

            metrics, _ = evaluate(
                raw_model,
                cfg.eval_a3m,
                cfg.eval_variants,
                wt_sequence=cfg.eval_wt or None,
                labels_csv=cfg.eval_labels or None,
                label_col=cfg.eval_label_col,
                alpha=cfg.eval_alpha,
                max_similarity=cfg.eval_max_similarity,
                context_tokens=cfg.eval_context_tokens,
                seed=cfg.seed,
            )
            print(f"[eval] step {step} {metrics}", flush=True)

    if rank == 0:
        save_checkpoint(model, hyper_parameters, optimizer, scheduler, step,
                        os.path.join(cfg.out_dir, "final.ckpt"))
    if ddp:
        import torch.distributed as dist

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
    p.add_argument("--ddp-mode", default=d.ddp_mode, choices=["find_unused", "static_graph"])
    p.add_argument("--save-every", type=int, default=d.save_every)
    p.add_argument("--log-every", type=int, default=d.log_every)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--overfit-one-batch", action="store_true", default=d.overfit_one_batch)
    # eval hook
    p.add_argument("--eval-every", type=int, default=d.eval_every)
    p.add_argument("--eval-a3m", default=d.eval_a3m)
    p.add_argument("--eval-variants", default=d.eval_variants)
    p.add_argument("--eval-wt", default=d.eval_wt)
    p.add_argument("--eval-labels", default=d.eval_labels)
    p.add_argument("--eval-label-col", default=d.eval_label_col)
    p.add_argument("--eval-alpha", type=float, default=d.eval_alpha)
    p.add_argument("--eval-max-similarity", type=float, default=d.eval_max_similarity)
    p.add_argument("--eval-context-tokens", type=int, default=d.eval_context_tokens)
    return p


def main(argv=None) -> None:
    args = _build_argparser().parse_args(argv)
    cfg = TrainConfig(**{k.replace("-", "_"): v for k, v in vars(args).items()})
    train(cfg)


if __name__ == "__main__":
    main()
