"""M2 make-or-break check (runs where flash_attn is importable, e.g. the GPU box).

Validates the custom "shared context, two targets" ``training_forward`` against the stock
``PoET2.forward`` with the context duplicated into both halves. The two must agree, which
proves we can drive both decoders off one shared encoder memory without the stock
``chunk(2)`` ``[MLM | CLM]`` convention.

Skipped automatically on machines without ``flash_attn`` (e.g. macOS), since the model
module hard-imports it.
"""

import pytest

pytest.importorskip("flash_attn", reason="PoET2/tokenizers require flash_attn (Linux+CUDA)")

import torch

from poet_2.models.poet_2 import PoET2
from poet_2.models.poet_2_helpers import tokenize_seq_of_seqs, tokenize_seqs
from poet_2.training.forward import training_forward


def _tiny_model() -> PoET2:
    torch.manual_seed(0)
    # Sequence-only tiny config: structure embeds default to zeros when their tensors are
    # None (in_plddt/in_seqid handled inside the model), so no structure inputs are needed.
    model = PoET2(
        n_vocab=29,
        n_out=58,  # AA(29) ⊕ 3Di(29)
        hidden_dim=64,
        ff_dim=128,
        n_layers=2,
        nhead=4,
    )
    return model.to(torch.float32).eval()


def _seqonly_batch(device="cpu"):
    # B = 2 families, each context a sequence-of-sequences (2 homologs), one MLM + one CLM
    # target per family.
    ctx = [
        [b"MKTAYIAKQR", b"MRTAYIAKQS"],
        [b"AAAGGGCCCW", b"AACGGGCCDW"],
    ]
    mlm_targets = [b"MKTAYIAKQR", b"AAAGGGCCCW"]
    clm_targets = [b"MKVAYIAKQR", b"AAAGGTCCCW"]
    dev = torch.device(device)
    tk_ctx = tokenize_seq_of_seqs(ctx, device=dev)
    tk_mlm = tokenize_seqs(mlm_targets, device=dev)
    tk_clm = tokenize_seqs(clm_targets, device=dev)
    batch = {
        "xs": tk_ctx["seqs"].long(),
        "xs_segment_sizes": tk_ctx["segment_sizes"],
        "mlm_ys": tk_mlm["seqs"].long(),
        "mlm_ys_segment_sizes": tk_mlm["segment_sizes"],
        "clm_ys": tk_clm["seqs"].long(),
        "clm_ys_segment_sizes": tk_clm["segment_sizes"],
    }
    return batch


def test_training_forward_shapes():
    model = _tiny_model()
    batch = _seqonly_batch()
    with torch.no_grad():
        xs_logits, mlm_logits, clm_logits = training_forward(model, batch)
    B = batch["xs"].size(0)
    assert xs_logits.shape[0] == B and xs_logits.shape[-1] == 58
    assert mlm_logits.shape[0] == B and mlm_logits.shape[1] == batch["mlm_ys"].size(1)
    assert clm_logits.shape[0] == B and clm_logits.shape[1] == batch["clm_ys"].size(1)
    assert mlm_logits.shape[-1] == 58 and clm_logits.shape[-1] == 58


def test_training_forward_matches_stock_forward():
    model = _tiny_model()
    batch = _seqonly_batch()

    with torch.no_grad():
        xs_logits, mlm_logits, clm_logits = training_forward(model, batch)

        # Stock forward expects the encoder batch as [MLM half | CLM half]; duplicating the
        # same prompts into both halves makes each decoder see equivalent memory, so its
        # per-half decoder logits must match the shared-memory path. refs all -100 => no
        # ref-value blending (our no-query scope).
        xs2 = torch.cat([batch["xs"], batch["xs"]], dim=0)
        seg2 = torch.cat([batch["xs_segment_sizes"], batch["xs_segment_sizes"]], dim=0)
        xs_logits_stock, mlm_logits_stock, clm_logits_stock = model.forward(
            xs=xs2,
            xs_plddts=None,
            xs_s3dis=None,
            xs_atomxs=None,
            xs_atombs=None,
            xs_segment_sizes=seg2,
            mlm_ys=batch["mlm_ys"],
            mlm_ys_seqids=None,
            mlm_ys_plddts=None,
            mlm_ys_s3dis=None,
            mlm_ys_atomxs=None,
            mlm_ys_atombs=None,
            mlm_ys_refs=torch.full_like(batch["mlm_ys"], -100),
            mlm_ys_segment_sizes=batch["mlm_ys_segment_sizes"],
            clm_ys=batch["clm_ys"],
            clm_ys_seqids=None,
            clm_ys_plddts=None,
            clm_ys_s3dis=None,
            clm_ys_atomxs=None,
            clm_ys_atombs=None,
            clm_ys_refs=torch.full_like(batch["clm_ys"], -100),
            clm_ys_segment_sizes=batch["clm_ys_segment_sizes"],
        )

    B = batch["xs"].size(0)
    # encoder MLM logits: stock runs over the duplicated 2B prompts; first B == ours
    assert torch.allclose(xs_logits, xs_logits_stock[:B], atol=1e-4, rtol=1e-4)
    assert torch.allclose(mlm_logits, mlm_logits_stock, atol=1e-4, rtol=1e-4)
    assert torch.allclose(clm_logits, clm_logits_stock, atol=1e-4, rtol=1e-4)


def test_training_forward_backward_flows_to_shared_params():
    """One backward through the summed AA-logit means reaches encoder + both decoders
    (tied) and the structure bias weights."""
    model = _tiny_model().train()
    batch = _seqonly_batch()
    xs_logits, mlm_logits, clm_logits = training_forward(model, batch)
    loss = (
        xs_logits[..., :29].float().pow(2).mean()
        + mlm_logits[..., :29].float().pow(2).mean()
        + clm_logits[..., :29].float().pow(2).mean()
    )
    loss.backward()
    # token embedding (shared) and a structure-bias param must receive gradient
    assert model.token_embed.weight.grad is not None
    bias = model.encoder.layers[0].bias_weights
    assert bias.grad is not None and torch.isfinite(bias.grad).all()
