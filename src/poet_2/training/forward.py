"""Custom training forward pass: "shared context, two targets" (plan decision #4).

Encode each homolog-set context **once** and route the shared decoder memory to **both**
the MLM and CLM decoders, so every family yields all three training logits per step at
~1x encoder cost. This deliberately avoids the stock ``PoET2.forward``'s hardcoded
``xs_segment_sizes.chunk(2)`` ``[MLM half | CLM half]`` convention while reusing the exact
same model internals (``encoder_outputs`` / ``get_decoder_memory`` / ``outputs_from_memory``
/ ``head``). Nothing in the model package is modified.

Correctness rests on facts verified in ``poet_2/models/poet_2.py``:
* ``outputs_from_memory`` supports ``B(targets) == memory_B(prompts)`` 1:1 (:671-680), so
  the B-prompt memory pairs directly with the B MLM targets and the B CLM targets.
* Cross-attention K/V are tied across the two decoders, so a single ``get_decoder_memory``
  (built via ``mlm_decoder``) is valid for both — exactly what stock ``forward`` relies on.
* With ``ys_refs=None`` the query/self-conditioning ref-value blending (:697-704) is skipped
  (our no-query scope).

The equivalence of this path to stock ``forward`` (with the context duplicated into both
halves) is checked in ``tests/test_training_forward.py``.
"""

from __future__ import annotations

from typing import Mapping

import torch


def _extract_ifq_ref_values(xs_h, xs_segment_sizes, clm_ys_segment_sizes, ifq_active):
    """Extract encoder ref values for IFQ-active samples.

    For IFQ samples, the first encoder segment is the masked-X + structure input
    whose length matches the CLM target. Returns (ys_refs, ys_ref_values) for
    outputs_from_memory, or (None, None) if no IFQ samples.
    """
    B = xs_segment_sizes.size(0)
    if ifq_active is None or not ifq_active.any():
        return None, None

    all_ifq = ifq_active.all()
    clm_lengths = clm_ys_segment_sizes[:, 0]
    first_seg_lengths = xs_segment_sizes[:, 0]
    total_clm = int(clm_lengths.sum())

    ref_parts = []
    refs_parts = []
    offset = 0
    for i in range(B):
        L_clm = int(clm_lengths[i])
        L_seg = int(first_seg_lengths[i])
        enc_start = int(xs_h.cu_seqlens[i])
        if ifq_active[i] and L_seg >= L_clm:
            ref_parts.append(xs_h.x[enc_start:enc_start + L_clm])
            refs_parts.append(torch.arange(offset, offset + L_clm, device=xs_h.x.device))
        else:
            refs_parts.append(torch.full((L_clm,), -100, device=xs_h.x.device, dtype=torch.long))
        offset += L_clm

    if all_ifq:
        return None, torch.cat(ref_parts, dim=0)

    ys_refs = torch.cat(refs_parts, dim=0)
    if not ref_parts:
        return None, None
    ys_ref_values = torch.cat(ref_parts, dim=0)
    return ys_refs, ys_ref_values


def training_forward(
    model, batch: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the encoder once and both decoders off the shared memory.

    ``batch`` keys consumed (structure keys may be absent/``None`` for sequence-only):
        xs, xs_segment_sizes, [xs_plddts, xs_s3dis, xs_atomxs, xs_atombs]
        mlm_ys, mlm_ys_segment_sizes, [mlm_ys_plddts, mlm_ys_s3dis, mlm_ys_atomxs, mlm_ys_atombs]
        clm_ys, clm_ys_segment_sizes, [clm_ys_plddts, clm_ys_s3dis, clm_ys_atomxs, clm_ys_atombs]
        ifq_active (optional): (B,) bool tensor — IFQ ref-value blending for CLM decoder

    Returns ``(xs_logits, mlm_logits, clm_logits)``, each padded ``(B, L, 58)``. The loss
    (``poet_2.training.losses``) slices the AA half.
    """
    enc = model.encoder_outputs(
        xs=batch["xs"],
        segment_sizes=batch["xs_segment_sizes"],
        xs_plddts=batch.get("xs_plddts"),
        xs_s3dis=batch.get("xs_s3dis"),
        xs_atomxs=batch.get("xs_atomxs"),
        xs_atombs=batch.get("xs_atombs"),
        repr_layers=(-1,),
        return_logits=True,
    )
    xs_logits = enc.logits  # padded (B, max_len, 58) — encoder MLM logits
    xs_h = enc.reprs[-1]  # PackedTensorSequences (sequence-of-sequences form)

    # Build cross-attention memory. When weights are untied, each decoder needs its own.
    weights_tied = (
        model.encoder.layers[0].self_attn.q_proj.weight
        is model.mlm_decoder.layers[0].self_attn.q_proj.weight
    )
    mlm_memory = model.get_decoder_memory(decoder=model.mlm_decoder, xs_h=xs_h)
    clm_memory = mlm_memory if weights_tied else model.get_decoder_memory(decoder=model.clm_decoder, xs_h=xs_h)

    mlm_logits = model.outputs_from_memory(
        decoder=model.mlm_decoder,
        memory=mlm_memory,
        ys=batch["mlm_ys"],
        ys_segment_sizes=batch["mlm_ys_segment_sizes"],
        ys_plddts=batch.get("mlm_ys_plddts"),
        ys_s3dis=batch.get("mlm_ys_s3dis"),
        ys_atomxs=batch.get("mlm_ys_atomxs"),
        ys_atombs=batch.get("mlm_ys_atombs"),
        ys_seqids=None,
        ys_refs=None,
        ys_ref_values=None,
        return_logits=True,
    ).logits

    clm_ys_refs, clm_ys_ref_values = _extract_ifq_ref_values(
        xs_h, batch["xs_segment_sizes"], batch["clm_ys_segment_sizes"],
        batch.get("ifq_active"),
    )

    clm_logits = model.outputs_from_memory(
        decoder=model.clm_decoder,
        memory=clm_memory,
        ys=batch["clm_ys"],
        ys_segment_sizes=batch["clm_ys_segment_sizes"],
        ys_plddts=batch.get("clm_ys_plddts"),
        ys_s3dis=batch.get("clm_ys_s3dis"),
        ys_atomxs=batch.get("clm_ys_atomxs"),
        ys_atombs=batch.get("clm_ys_atombs"),
        ys_seqids=None,
        ys_refs=clm_ys_refs,
        ys_ref_values=clm_ys_ref_values,
        return_logits=True,
    ).logits

    return xs_logits, mlm_logits, clm_logits
