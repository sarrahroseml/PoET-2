#!/usr/bin/env python3
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional, TYPE_CHECKING


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from poet_2.models.poet_2 import PoET2
from poet_2.models import poet_2_helpers as helpers

from scripts.dataloader import (
    AugmentationConfig,
    PoET2Dataset,
    PoETBatch,
    build_collate_fn,
)

if TYPE_CHECKING:
    import wandb



@dataclass
class TrainConfig:
    """Configuration required to launch fine-tuning without relying on CLI flags."""

    data_root: Path
    checkpoint: Path
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1
    steps: int = 12000
    lr: float = 1e-6
    seed: int = 0
    save_dir: Optional[Path] = None 
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


def load_config_from_env() -> TrainConfig:
    """
    Build a TrainConfig from environment variables.

    Required:
        POET2_DATA_ROOT: directory containing *.npz prompts
        POET2_CHECKPOINT: path to a PoET-2 checkpoint file
    Optional overrides:
        POET2_DEVICE, POET2_BATCH_SIZE, POET2_STEPS, POET2_LR,
        POET2_SEED, POET2_WANDB_PROJECT, POET2_WANDB_RUN_NAME
    """

    data_root = os.environ.get("POET2_DATA_ROOT")
    checkpoint = os.environ.get("POET2_CHECKPOINT")
    if not data_root or not checkpoint:
        raise RuntimeError(
            "Set POET2_DATA_ROOT and POET2_CHECKPOINT before running train_poet2.py"
        )

    config_kwargs: dict[str, object] = {}
    if device := os.environ.get("POET2_DEVICE"):
        config_kwargs["device"] = device
    if batch_size := os.environ.get("POET2_BATCH_SIZE"):
        config_kwargs["batch_size"] = int(batch_size)
    if steps := os.environ.get("POET2_STEPS"):
        config_kwargs["steps"] = int(steps)
    if lr := os.environ.get("POET2_LR"):
        config_kwargs["lr"] = float(lr)
    if seed := os.environ.get("POET2_SEED"):
        config_kwargs["seed"] = int(seed)
    if wandb_project := os.environ.get("POET2_WANDB_PROJECT"):
        config_kwargs["wandb_project"] = wandb_project
    if wandb_run_name := os.environ.get("POET2_WANDB_RUN_NAME"):
        config_kwargs["wandb_run_name"] = wandb_run_name
    if save_dir := os.environ.get("SAVE_DIR"):
        config_kwargs["save_dir"] = Path(save_dir).expanduser()
    
    

    return TrainConfig(
        data_root=Path(data_root).expanduser(),
        checkpoint=Path(checkpoint).expanduser(),
        **config_kwargs,
    )


def move_to_device(batch: PoETBatch, device: torch.device) -> PoETBatch:
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=True)
    return batch

alphabet = helpers.Alphabet()
IGNORE = {
    alphabet.mask_token,   # 24
    alphabet.start_token,  # BOS
    alphabet.stop_token,   # EOS
}

def make_loss_mask(targets: torch.Tensor, base_mask: torch.Tensor | None):
    # base_mask is mlm_token_mask, clm_token_mask, or xs_mask
    if base_mask is None:
        base_mask = torch.ones_like(targets, dtype=torch.bool)
    ignore_mask = torch.zeros_like(targets, dtype=torch.bool)
    for idx in IGNORE:
        ignore_mask |= (targets == idx)
    # keep = base_mask AND not ignored
    return base_mask & (~ignore_mask)

def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor],
    alphabet: helpers.Alphabet,
    debug_name: str = "",
    debug: bool = False,
) -> torch.Tensor:
    # logits: [B, T, V], targets: [B, T]


    ce = F.cross_entropy(
        logits.transpose(1, 2),  # -> [B, V, T]
        targets,
        reduction="none",        # -> [B, T]
        ignore_index=alphabet.mask_token
    )

    ce = torch.clamp(ce, max=20.0)


    if mask is None:
        loss = ce.mean()
    else:
        mask = mask.to(dtype=ce.dtype)
        denom = mask.sum().clamp_min(1.0)
        loss = (ce * mask).sum() / denom

    if debug:
        # Look only at supervised positions
        if mask is not None:
            ce_eff = ce * (mask > 0)
        else:
            ce_eff = ce

        # 1) Check for non-finite tokens
        bad = ~torch.isfinite(ce_eff)
        if bad.any():
            print(f"[DEBUG:{debug_name}] Found non-finite CE tokens")
            idxs = bad.nonzero(as_tuple=False)
            # Print a few of them
            for (b, t) in idxs[:10]:
                b = b.item()
                t = t.item()
                tgt = targets[b, t].item()
                val = ce_eff[b, t].item()
                print(f"  (b={b}, t={t}) ce={val}, target={tgt}")
            # raise RuntimeError(f"Non-finite CE in {debug_name}")

        # 2) Print the worst finite tokens
        finite = torch.isfinite(ce_eff) & (ce_eff > 0)
        if finite.any():
            flat = ce_eff[finite].view(-1)
            topk_vals, topk_idx = torch.topk(flat, k=min(10, flat.numel()))
            # map flat indices back to (b, t)
            b_all, t_all = torch.nonzero(finite, as_tuple=True)
            # flatten them the same way
            flat_indices = torch.arange(finite.numel(), device=finite.device)[finite.view(-1)]
            for val, flat_i in zip(topk_vals, topk_idx):
                global_i = flat_indices[flat_i]
                b = (global_i // ce_eff.shape[1]).item()
                t = (global_i %  ce_eff.shape[1]).item()
                tgt = targets[b, t].item()
                print(f"[DEBUG:{debug_name}] top CE token: b={b}, t={t}, ce={val.item():.4f}, target={tgt}")

    return loss



def compute_losses(
    model: PoET2,
    batch: PoETBatch,
    alphabet: helpers.Alphabet,
) -> dict[str, torch.Tensor]:
    xs_logits, mlm_logits, clm_logits = model.forward(
        xs=batch["xs"],
        xs_plddts=batch.get("xs_plddts"),
        xs_s3dis=batch.get("xs_s3dis"),
        xs_atomxs=batch.get("xs_atomxs"),
        xs_atombs=batch.get("xs_atombs"),
        xs_segment_sizes=batch["xs_segment_sizes"],
        mlm_ys=batch["mlm_ys"],
        mlm_ys_seqids=None,
        mlm_ys_plddts=batch.get("mlm_ys_plddts"),
        mlm_ys_s3dis=batch.get("mlm_ys_s3dis"),
        mlm_ys_atomxs=batch.get("mlm_ys_atomxs"),
        mlm_ys_atombs=batch.get("mlm_ys_atombs"),
        mlm_ys_refs=batch["mlm_ys_refs"],
        mlm_ys_segment_sizes=batch["mlm_ys_segment_sizes"],
        clm_ys=batch["clm_ys"],
        clm_ys_seqids=None,
        clm_ys_plddts=batch.get("clm_ys_plddts"),
        clm_ys_s3dis=batch.get("clm_ys_s3dis"),
        clm_ys_atomxs=batch.get("clm_ys_atomxs"),
        clm_ys_atombs=batch.get("clm_ys_atombs"),
        clm_ys_refs=batch["clm_ys_refs"],
        clm_ys_segment_sizes=batch["clm_ys_segment_sizes"],
    )

    for name, t in [("xs_logits", xs_logits),
                    ("mlm_logits", mlm_logits),
                    ("clm_logits", clm_logits)]:
        print(
            f"[debug] {name} finite:",
            torch.isfinite(t).all().item(),
            "nan:", torch.isnan(t).any().item(),
            "inf:", torch.isinf(t).any().item(),
        )
    for name, t in [("xs_logits", xs_logits),
                ("mlm_logits", mlm_logits),
                ("clm_logits", clm_logits)]:
        print(
            f"[debug] {name} finite:",
            torch.isfinite(t).all().item(),
            "nan:", torch.isnan(t).any().item(),
            "inf:", torch.isinf(t).any().item(),
            "max:", t.max().item(),
            "min:", t.min().item(),
        )


    if not torch.isfinite(xs_logits).all():
        print("[fatal] xs_logits not finite at this step")
        print("max:", xs_logits.max().item(), "min:", xs_logits.min().item())
        raise RuntimeError("xs_logits not finite")


    target = batch["target_seqs"]
    #mlm_mask = batch["mlm_token_mask"]

    # 2) Check MLM targets are in range
    vocab_size = mlm_logits.shape[-1]
    bad = (target != alphabet.mask_token) & ((target < 0) | (target >= vocab_size))
    if bad.any():
        print("[debug] invalid target indices at MLM positions")
        print("min target:", target.min().item(), "max target:", target.max().item())
        raise RuntimeError("Bad indices in target")

    mlm_mask = make_loss_mask(target, batch["mlm_token_mask"])
    mlm_loss = masked_cross_entropy(mlm_logits, target, mlm_mask, alphabet,  debug=True)

    clm_logits = clm_logits[:, :-1]
    clm_target = target[:, 1:]
    #clm_mask = batch["clm_token_mask"][:, 1:]
    clm_mask   = make_loss_mask(clm_target, batch["clm_token_mask"][:, 1:])
    clm_loss   = masked_cross_entropy(clm_logits, clm_target, clm_mask, alphabet,  debug=True)

    # ----- encoder loss: MLM view only -----
    # xs_logits has shape [2B, L, V] because xs = cat([mlm, clm], dim=0)
    B = target.shape[0]
    xs_logits_mlm = xs_logits[:B]          # keep only MLM half

    target_xs = target                     # [B, L]
    #xs_mask = batch["mlm_token_mask"]      # [B, L]

    # Same range check for encoder targets, but only on MLM half
    bad_enc = (target_xs != alphabet.mask_token) & (
        (target_xs < 0) | (target_xs >= vocab_size)
    )
    if bad_enc.any():
        print("[debug] invalid target indices for encoder loss")
        print(
            "min target_xs:", target_xs.min().item(),
            "max target_xs:", target_xs.max().item(),
        )
        raise RuntimeError("Bad indices in target_xs")
    enc_mask   = make_loss_mask(target, batch["mlm_token_mask"])
    enc_loss   = masked_cross_entropy(xs_logits_mlm, target, enc_mask, alphabet,  debug=True)
    #enc_loss = masked_cross_entropy(xs_logits_mlm, target_xs, xs_mask, alphabet, debug=True)

    print("[debug] enc_loss", enc_loss.item())
    print("[debug] mlm_loss", mlm_loss.item())
    print("[debug] clm_loss", clm_loss.item())
    return {
        "encoder": enc_loss,
        "mlm": mlm_loss,
        "clm": clm_loss,
        "total": enc_loss + mlm_loss + clm_loss,
    }


def check_params(model: PoET2, tag: str):
    bad = False
    for name, p in model.named_parameters():
        if p.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            continue
        if not torch.isfinite(p).all():
            print(f"[PARAM {tag}] non-finite in {name}: "
                  f"min={p.detach().float().min().item()}, "
                  f"max={p.detach().float().max().item()}")
            bad = True
            break
    if not bad:
        print(f"[PARAM {tag}] all finite")

def train_loop( model: PoET2, dataloader: Iterable[PoETBatch], optimizer: torch.optim.Optimizer, steps: int, device: torch.device, wandb_run: Optional["wandb.sdk.wandb_run.Run"] = None,  
    save_dir: Path | None = None, cfg: TrainConfig | None = None) -> None: 
    alphabet = helpers.Alphabet() 
    print(f"alphabet mask token is {alphabet.mask_token}")
    iterator = iter(dataloader) 
    accum_steps = 4
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)

        batch = move_to_device(batch, device)
        losses = compute_losses(model, batch, alphabet)
        loss = losses["total"] / accum_steps
        loss.backward()

        if step % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses = compute_losses(model, batch, alphabet) 
            losses["total"].backward() 

        # Check for NaNs/infs in grads
        total_norm_sq = 0.0
        bad = False
        for p in model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                print("[DEBUG] non-finite grad in param with shape", p.shape)
                bad = True
                break
            total_norm_sq += p.grad.detach().float().pow(2).sum().item()
        grad_norm = total_norm_sq ** 0.5
        print(f"[DEBUG] step {step} grad_norm={grad_norm:.4f}, bad={bad}")

        if bad:
            print("[FATAL] Non-finite gradients, aborting before optimizer.step()")
            break
        check_params(model, "before_step")

        if bad:
            print("[FATAL] non-finite grads, skipping step")
            break

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        check_params(model, "after_step") 

         # Save checkpoint every 100 steps and on the final step
        if save_dir is not None and cfg is not None:
            if (step % 100 == 0) or (step == steps):
                save_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = save_dir / f"poet2_finetune_step{step}.pt"
                save_checkpoint(ckpt_path, model, optimizer, step, cfg)

        if step % 10 == 0: 
            print( f"step {step:05d} | " 
                  f"enc {losses['encoder'].item():.4f} " 
                  f"mlm {losses['mlm'].item():.4f} " 
                  f"clm {losses['clm'].item():.4f}" ) 
            if wandb_run is not None: # DEBUG print so we know this is actually executed 
                print( f"[wandb] logging step={step} " 
                      f"enc={losses['encoder'].item():.4f} " 
                      f"mlm={losses['mlm'].item():.4f} " 
                      f"clm={losses['clm'].item():.4f}" ) 
                wandb_run.log( { "step": step, 
                                "loss/encoder": losses["encoder"].item(), 
                                "loss/mlm": losses["mlm"].item(), 
                                "loss/clm": losses["clm"].item(), 
                                "loss/total": losses["total"].item(), } )



def save_checkpoint(
    path: Path,
    model: PoET2,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: TrainConfig,
):

    base_ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)

    if "hyper_parameters" not in base_ckpt:
        raise ValueError(
            f"Base checkpoint at {cfg.checkpoint} has no 'hyper_parameters' key; "
            "cannot create a load_model-compatible finetuned checkpoint."
        )


    hyper_parameters = base_ckpt["hyper_parameters"]


    state_dict = {f"model.{k}": v for k, v in model.state_dict().items()}

    ckpt = {
        "step": step,
        "hyper_parameters": hyper_parameters,
        "state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(), 
        "train_config": asdict(cfg),                 
    }

    torch.save(ckpt, path)
    print(f"[checkpoint] saved to {path}")
def main(config: Optional[TrainConfig] = None) -> None:
    cfg = config or load_config_from_env()
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    model = helpers.load_model(cfg.checkpoint, device=device)
    model.train()


    for name, p in model.named_parameters():
        print("[DTYPE param]", name, p.dtype)
    for name, b in model.named_buffers():
        if torch.is_floating_point(b):
            print("[DTYPE buffer]", name, b.dtype)

    # Force all params to float32
    for p in model.parameters():
        if torch.is_floating_point(p):
            p.data = p.data.float()

    # Force all buffers to float32
    for b in model.buffers():
        if torch.is_floating_point(b):
            b.data = b.data.float()

    dataset = PoET2Dataset(cfg.data_root)
    collate = build_collate_fn(AugmentationConfig(), seed=cfg.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate,
    )

   

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.lr,
        
    )
    save_dir = cfg.save_dir or (cfg.checkpoint.parent / "finetune_ckpts")
    wandb_run = None
    if cfg.wandb_project:
        try:
            import wandb
        except ImportError:
            raise RuntimeError("wandb is not installed but logging was requested.")
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config={
                "batch_size": cfg.batch_size,
                "steps": cfg.steps,
                "lr": cfg.lr,
                "device": cfg.device,
                "checkpoint": str(cfg.checkpoint),
            },
        )
        # Use our explicit "step" metric as x-axis for all loss/*
        wandb_run.define_metric("step")
        wandb_run.define_metric("loss/*", step_metric="step")
        wandb_run.log({"step": 0, "loss/total": 0.0})

    try:
        train_loop(model, dataloader, optimizer, cfg.steps, device, wandb_run,
            save_dir=save_dir, cfg=cfg)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
