"""Standalone M2 check (NO pytest needed): is the custom `training_forward` equivalent to
the released `PoET2.forward`?

Run on a GPU node, in the pixi env (so flash_attn imports):

    pixi run --no-lockfile-update python scripts/check_m2.py

Exit codes: 0 = passed, 1 = a check failed, 2 = flash_attn unavailable (skipped).
Uses a tiny model on CPU/fp32 (the GPU is only needed so flash_attn imports).
"""

import os
import sys

# make `poet_2` importable even without the editable install
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

try:
    import flash_attn  # noqa: F401
except Exception as e:  # pragma: no cover
    print(f"SKIP: flash_attn not importable ({e}). Run on a GPU node in the pixi env.")
    sys.exit(2)

import torch

from poet_2.models.poet_2 import PoET2
from poet_2.models.poet_2_helpers import tokenize_seq_of_seqs, tokenize_seqs
from poet_2.training.forward import training_forward


def tiny_model() -> PoET2:
    torch.manual_seed(0)
    # sequence-only tiny config; structure embeds default to zeros when their tensors are None
    return (
        PoET2(n_vocab=29, n_out=58, hidden_dim=64, ff_dim=128, n_layers=2, nhead=4)
        .to(torch.float32)
        .eval()
    )


def seqonly_batch(device="cpu") -> dict:
    ctx = [[b"MKTAYIAKQR", b"MRTAYIAKQS"], [b"AAAGGGCCCW", b"AACGGGCCDW"]]
    mlm = [b"MKTAYIAKQR", b"AAAGGGCCCW"]
    clm = [b"MKVAYIAKQR", b"AAAGGTCCCW"]
    dev = torch.device(device)
    tkc = tokenize_seq_of_seqs(ctx, device=dev)
    tkm = tokenize_seqs(mlm, device=dev)
    tkl = tokenize_seqs(clm, device=dev)
    return {
        "xs": tkc["seqs"].long(),
        "xs_segment_sizes": tkc["segment_sizes"],
        "mlm_ys": tkm["seqs"].long(),
        "mlm_ys_segment_sizes": tkm["segment_sizes"],
        "clm_ys": tkl["seqs"].long(),
        "clm_ys_segment_sizes": tkl["segment_sizes"],
    }


def main() -> None:
    print(f"torch {torch.__version__} | flash_attn {getattr(flash_attn, '__version__', '?')}")
    failures = []
    model = tiny_model()
    batch = seqonly_batch()
    B = batch["xs"].size(0)

    # [1/3] shapes
    with torch.no_grad():
        xs_l, mlm_l, clm_l = training_forward(model, batch)
    ok = (
        xs_l.shape[0] == B
        and xs_l.shape[-1] == 58
        and tuple(mlm_l.shape[:2]) == (B, batch["mlm_ys"].size(1))
        and tuple(clm_l.shape[:2]) == (B, batch["clm_ys"].size(1))
    )
    print(f"[1/3] shapes: {'PASS' if ok else 'FAIL'}  "
          f"xs={tuple(xs_l.shape)} mlm={tuple(mlm_l.shape)} clm={tuple(clm_l.shape)}")
    failures += [] if ok else ["shapes"]

    # [2/3] equivalence to stock forward (context duplicated into both [MLM|CLM] halves)
    with torch.no_grad():
        xs2 = torch.cat([batch["xs"], batch["xs"]], dim=0)
        seg2 = torch.cat([batch["xs_segment_sizes"], batch["xs_segment_sizes"]], dim=0)
        xs_s, mlm_s, clm_s = model.forward(
            xs=xs2, xs_plddts=None, xs_s3dis=None, xs_atomxs=None, xs_atombs=None,
            xs_segment_sizes=seg2,
            mlm_ys=batch["mlm_ys"], mlm_ys_seqids=None, mlm_ys_plddts=None,
            mlm_ys_s3dis=None, mlm_ys_atomxs=None, mlm_ys_atombs=None,
            mlm_ys_refs=torch.full_like(batch["mlm_ys"], -100),
            mlm_ys_segment_sizes=batch["mlm_ys_segment_sizes"],
            clm_ys=batch["clm_ys"], clm_ys_seqids=None, clm_ys_plddts=None,
            clm_ys_s3dis=None, clm_ys_atomxs=None, clm_ys_atombs=None,
            clm_ys_refs=torch.full_like(batch["clm_ys"], -100),
            clm_ys_segment_sizes=batch["clm_ys_segment_sizes"],
        )
    e_xs = torch.allclose(xs_l, xs_s[:B], atol=1e-4, rtol=1e-4)
    e_mlm = torch.allclose(mlm_l, mlm_s, atol=1e-4, rtol=1e-4)
    e_clm = torch.allclose(clm_l, clm_s, atol=1e-4, rtol=1e-4)
    ok = e_xs and e_mlm and e_clm
    print(f"[2/3] equivalence vs stock forward: {'PASS' if ok else 'FAIL'}  "
          f"(xs={e_xs} mlm={e_mlm} clm={e_clm})")
    if not ok:
        failures.append("equivalence")
        print(f"      max|Δ|  xs={(xs_l - xs_s[:B]).abs().max():.2e}  "
              f"mlm={(mlm_l - mlm_s).abs().max():.2e}  clm={(clm_l - clm_s).abs().max():.2e}")

    # [3/3] backward reaches tied params + structure bias
    m2 = tiny_model().train()
    xs_l, mlm_l, clm_l = training_forward(m2, batch)
    loss = (
        xs_l[..., :29].float().pow(2).mean()
        + mlm_l[..., :29].float().pow(2).mean()
        + clm_l[..., :29].float().pow(2).mean()
    )
    loss.backward()
    bias = m2.encoder.layers[0].bias_weights
    ok = (
        m2.token_embed.weight.grad is not None
        and bias.grad is not None
        and torch.isfinite(bias.grad).all()
    )
    print(f"[3/3] backward reaches tied params + structure bias: {'PASS' if ok else 'FAIL'}")
    failures += [] if ok else ["backward"]

    if failures:
        print(f"\nM2 FAILED: {failures}")
        sys.exit(1)
    print("\nM2 PASSED ✅  training_forward == stock forward; safe to build M3 on it.")


if __name__ == "__main__":
    main()
