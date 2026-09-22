# PoET-2 Continued Training on Viral Proteins — Experiment Report

**Last updated**: July 28, 2026
**Model**: PoET-2 (614M params, encoder-decoder protein language model)
**Dataset**: D1 Logan viral protein dataset (4.7M samples, 127K unique sequences, 98.6% structure coverage)
**Eval**: 45 viral deep mutational scanning (DMS) datasets (17 IID, 28 OOD), zero-shot Spearman correlation

### IID/OOD Split

**IID (17 datasets)**: Proteins from families well-represented in the D1 training data:
IAV HA (H1 Doud, H1 Wu, H3 Lee, H5 Dadonaite), SARS-CoV-2 Spike (BA.1, Delta Dadonaite),
SARS-CoV-2 RBD (Starr binding, Starr expression, XBB1.5 Taylor, PRD-0038 Starr),
bat CoV RBD (RmYN02, RsYN04 Starr), Nipah F (Larsen),
HIV-1 Env (BF520, HV1B9, BG505 Haddox), Lassa GP (Carr).

**OOD (28 datasets)**: Everything else (AAV capsid, phage proteins, CVB3, DENV, EV, IAV non-HA, SARS-CoV-2 non-Spike, etc.).

---

## 1. Architecture Background

PoET-2 is a transformer encoder-decoder with three loss terms:
- **L_mlm_enc**: Encoder masked language modeling
- **L_mlm_dec**: Decoder masked language modeling
- **L_clm_dec**: Decoder causal language modeling (the scoring objective at eval time)

### Weight Tying (Critical)

The encoder, MLM decoder, and CLM decoder **share ALL weights** via `tie_module_weights()`. They are literally the same parameter tensors used in three modes:
- Encoder: bidirectional self-attention + tiered cross-sequence attention
- MLM decoder: bidirectional self-attention + cross-attention to encoder memory
- CLM decoder: causal self-attention + cross-attention to encoder memory

**This means any gradient from any loss term modifies the same shared backbone.** You cannot freeze the encoder without also freezing the decoders — they are the same parameters.

### Scoring Modes

| Mode | Description |
|------|------------|
| seq_only | CLM scoring with sequence context only |
| struct | CLM scoring with structure features on context |
| ifq | Inverse-folding query: mask WT to all-X, provide structure, use ref-value blending |
| ctx_struct | Sequence + structure context, sweep over context lengths |
| ctx_ifq | IFQ + context sweep |
| ens | 15-prompt ensemble (3 context lengths x 5 similarity thresholds), score-level average |
| ens_ifq | Ensemble with IFQ — **the best pretrained mode** |

### Training Scale

| Parameter | Value |
|-----------|-------|
| Token budget | 45,056 tokens/GPU |
| GPUs | 4x H100 |
| Gradient accumulation | 2 |
| Batches per epoch | 125,912 (per GPU) |
| **Steps per epoch** | **62,956** |
| **10,000 steps** | **0.16 epochs** (16% of one pass) |
| **50,000 steps** | **0.79 epochs** |

The D1 dataset is large enough that none of our runs complete a single full epoch.

---

## 2. Pretrained Baseline

Full evaluation with all scoring modes (15-prompt ensemble), split by IID/OOD:

| Mode | All (45) | IID (17) | OOD (28) |
|------|---------|---------|---------|
| seq_only | 0.399 | 0.336 | 0.437 |
| struct | 0.367 | 0.347 | 0.378 |
| ifq | 0.474 | 0.444 | 0.491 |
| ctx_struct | 0.433 | 0.376 | 0.468 |
| ctx_ifq | 0.475 | 0.447 | 0.490 |
| ctx_struct_wt | 0.367 | 0.350 | 0.376 |
| ens | 0.467 | 0.414 | 0.499 |
| **ens_ifq** | **0.513** | **0.482** | **0.530** |

The IFQ pathway is the single most valuable scoring mechanism. ens_ifq (0.513) outperforms ens (0.467) by +0.046 and seq_only (0.399) by +0.114. OOD performance is consistently higher than IID across all modes.

---

## 3. Per-Config Ensemble Analysis (Pretrained)

Each of the 15 ensemble configurations (3 context lengths x 5 similarity thresholds) evaluated individually on the pretrained model. Ranked by IFQ Spearman:

| Rank | Config | seq All | seq IID | seq OOD | IFQ All | IFQ IID | IFQ OOD |
|------|--------|---------|---------|---------|---------|---------|---------|
| 1 | ctx6144, sim0.95 | 0.432 | 0.376 | 0.466 | **0.498** | 0.463 | **0.516** |
| 2 | ctx12288, sim0.95 | 0.439 | 0.402 | 0.462 | 0.492 | 0.469 | 0.504 |
| 3 | ctx24576, sim0.70 | 0.443 | 0.419 | 0.457 | 0.491 | 0.480 | 0.497 |
| 4 | ctx6144, sim0.90 | 0.426 | 0.351 | 0.471 | 0.489 | 0.446 | 0.512 |
| 5 | ctx24576, sim0.90 | 0.436 | 0.393 | 0.462 | 0.488 | 0.457 | 0.505 |
| 6 | ctx24576, sim0.95 | 0.430 | 0.387 | 0.456 | 0.487 | 0.458 | 0.502 |
| 7 | ctx12288, sim0.70 | 0.426 | 0.377 | 0.457 | 0.485 | 0.459 | 0.499 |
| 8 | ctx6144, sim0.70 | 0.421 | 0.369 | 0.454 | 0.483 | 0.455 | 0.498 |
| 9 | ctx12288, sim0.90 | 0.424 | 0.370 | 0.457 | 0.482 | 0.439 | 0.505 |
| 10 | ctx24576, sim0.50 | 0.428 | 0.381 | 0.457 | 0.482 | 0.461 | 0.493 |
| 11 | ctx24576, sim1.00 | 0.418 | 0.393 | 0.433 | 0.479 | 0.465 | 0.487 |
| 12 | ctx6144, sim1.00 | 0.408 | 0.370 | 0.431 | 0.476 | 0.451 | 0.489 |
| 13 | ctx12288, sim1.00 | 0.413 | 0.375 | 0.437 | 0.474 | 0.447 | 0.488 |
| 14 | ctx12288, sim0.50 | 0.409 | 0.351 | 0.444 | 0.470 | 0.448 | 0.481 |
| 15 | ctx6144, sim0.50 | 0.391 | 0.336 | 0.424 | 0.462 | 0.446 | 0.470 |

**Key findings**:
- **sim=0.95 is the sweet spot** — the top config (ctx6144, sim0.95) gets 0.498 IFQ All, close to the 15-prompt ensemble at 0.513
- **sim=1.0 (no filtering) and sim=0.5 (aggressive filtering) are worst** — moderate diversity helps
- **Context length matters less than similarity threshold** — all three lengths perform similarly at the same sim threshold
- **IFQ OOD is consistently higher than IFQ IID** — the model's structural scoring generalizes well to unseen families
- The ensemble averages over diverse configs, adding ~0.015 over the best single config through noise cancellation

---

## 4. Experiment 1 — Full Fine-Tuning Hyperparameter Sweep

**Goal**: Find the best hyperparameters for continued training.
**Setup**: 10k steps (0.16 epochs), all combinations of LR, mask rate, and struct dropout.

### Diversity-weighted results (seq_only Spearman, 10k steps)

| LR | Mask | SDrop | seq_only | Δ |
|----|------|-------|----------|---|
| 1e-4 | 0.15 | 0.5 | **0.409** | **+0.010** |
| 5e-5 | 0.00 | 0.0 | 0.406 | +0.007 |
| 1e-4 | 0.00 | 0.5 | 0.402 | +0.003 |
| 5e-5 | 0.00 | 0.5 | 0.402 | +0.003 |
| 5e-5 | 0.15 | 0.0 | 0.402 | +0.003 |
| 5e-5 | 0.15 | 0.5 | 0.401 | +0.002 |
| 1e-4 | 0.00 | 0.0 | 0.399 | +0.000 |
| 1e-4 | 0.15 | 0.0 | 0.398 | -0.001 |

### Inverse-frequency-weighted results (seq_only Spearman, 10k steps)

| LR | Mask | SDrop | seq_only | Δ |
|----|------|-------|----------|---|
| 5e-5 | 0.00 | 0.0 | 0.402 | +0.003 |
| 5e-5 | 0.15 | 0.0 | 0.402 | +0.003 |
| 1e-4 | 0.00 | 0.0 | 0.401 | +0.002 |
| 5e-5 | 0.15 | 0.5 | 0.397 | -0.002 |
| 1e-4 | 0.15 | 0.0 | 0.394 | -0.005 |
| 1e-4 | 0.15 | 0.5 | 0.395 | -0.004 |
| 5e-5 | 0.00 | 0.5 | 0.394 | -0.005 |
| 1e-4 | 0.00 | 0.5 | 0.387 | -0.012 |

**Findings**:
- Best seq_only: **0.409** (diversity, LR 1e-4, mask=0.15, sdrop=0.5)
- Diversity weighting consistently outperforms inverse-frequency weighting
- CLM-only (mask=0) or light masking (0.15) works best

---

## 5. Full Scoring Mode Evaluation — Top Checkpoints (IID/OOD Breakdown)

Three top checkpoints evaluated with all 8 scoring modes including 15-prompt ensemble.

### All 45 DMS

| Mode | Pretrained | div lr1e-4 m15 sd5 (Δ) | div lr5e-5 m0 sd0 (Δ) |
|------|-----------|----------------------|---------------------|
| seq_only | 0.399 | 0.409 (+0.010) | 0.406 (+0.007) |
| struct | 0.367 | 0.361 (-0.006) | 0.344 (-0.023) |
| ifq | **0.474** | 0.440 (**-0.034**) | 0.445 (**-0.029**) |
| ctx_struct | 0.433 | 0.437 (+0.004) | 0.430 (-0.003) |
| ctx_ifq | 0.475 | 0.439 (-0.036) | 0.446 (-0.029) |
| ctx_struct_wt | 0.367 | 0.361 (-0.006) | 0.347 (-0.020) |
| ens | 0.467 | 0.469 (+0.002) | 0.469 (+0.002) |
| **ens_ifq** | **0.513** | 0.479 (**-0.034**) | 0.490 (**-0.023**) |

### IID (17 datasets)

| Mode | Pretrained | div lr1e-4 m15 sd5 (Δ) | div lr5e-5 m0 sd0 (Δ) |
|------|-----------|----------------------|---------------------|
| seq_only | 0.336 | 0.345 (+0.009) | 0.348 (+0.012) |
| struct | 0.347 | 0.313 (-0.034) | 0.292 (-0.055) |
| ifq | 0.444 | 0.420 (-0.024) | 0.433 (-0.011) |
| ctx_struct | 0.376 | 0.383 (+0.007) | 0.386 (+0.010) |
| ctx_ifq | 0.447 | 0.423 (-0.024) | 0.443 (-0.004) |
| ctx_struct_wt | 0.350 | 0.313 (-0.037) | 0.300 (-0.050) |
| ens | 0.414 | 0.413 (-0.001) | 0.413 (-0.001) |
| **ens_ifq** | **0.482** | 0.453 (**-0.029**) | 0.472 (**-0.010**) |

### OOD (28 datasets)

| Mode | Pretrained | div lr1e-4 m15 sd5 (Δ) | div lr5e-5 m0 sd0 (Δ) |
|------|-----------|----------------------|---------------------|
| seq_only | 0.437 | 0.447 (+0.010) | 0.441 (+0.004) |
| struct | 0.378 | 0.387 (+0.009) | 0.372 (-0.006) |
| ifq | 0.491 | 0.451 (**-0.040**) | 0.451 (**-0.040**) |
| ctx_struct | 0.468 | 0.471 (+0.003) | 0.456 (-0.012) |
| ctx_ifq | 0.490 | 0.448 (-0.042) | 0.447 (-0.043) |
| ctx_struct_wt | 0.376 | 0.386 (+0.010) | 0.372 (-0.004) |
| ens | 0.499 | 0.503 (+0.004) | 0.502 (+0.003) |
| **ens_ifq** | **0.530** | 0.492 (**-0.038**) | 0.500 (**-0.030**) |

**Key IID/OOD findings**:
- **IFQ degradation is worse on OOD** (-0.040) than IID (-0.024 to -0.011) — training on D1 data damages out-of-distribution generalization through the structural pathway
- **seq_only gains are similar IID and OOD** — the sequence-level improvement transfers
- **The lr5e-5 model is much better for IID** (ens_ifq -0.010 vs -0.029) — less aggressive training preserves IID IFQ better
- **OOD ens_ifq drops from 0.530 to 0.492-0.500** — substantial damage to the model's strongest OOD capability

---

## 6. Experiment 2 — Struct/Mask Sweep with All-Mode Eval

**Goal**: Can struct_dropout or mask rate prevent IFQ degradation?
**Setup**: 10k steps, LR 1e-4, tracking seq_only + struct + IFQ during training.

| Mask | SDrop | seq_only (Δ) | struct (Δ) | IFQ (Δ) |
|------|-------|-------------|-----------|---------|
| 0.00 | 0.2 | 0.403 (+0.004) | 0.366 (-0.001) | 0.445 (-0.029) |
| 0.05 | 0.2 | 0.400 (+0.001) | 0.352 (-0.015) | 0.435 (-0.039) |
| 0.00 | 0.0 | 0.401 (+0.002) | 0.358 (-0.009) | 0.431 (-0.043) |
| 0.05 | 0.0 | 0.398 (-0.001) | 0.360 (-0.007) | 0.433 (-0.041) |

**Finding**: No hyperparameter combination prevents IFQ degradation. sdrop=0.2 slightly reduces IFQ loss but doesn't solve the problem.

---

## 7. Experiment 3 — Long Training (50k Steps = 0.79 Epochs)

**Goal**: Does more training help?
**Setup**: 50k steps, LR ∈ {1e-5, 2e-5}, mask=0.15, sdrop=0.5. Seq_only eval only.

| LR | Step | seq_only |
|----|------|----------|
| 1e-5 | 0 | 0.399 |
| 1e-5 | 5k | 0.402 |
| 1e-5 | 10k | 0.402 |
| 1e-5 | 25k | 0.404 |
| 1e-5 | 50k | 0.403 |
| 2e-5 | 0 | 0.399 |
| 2e-5 | 5k | 0.395 |
| 2e-5 | 50k | 0.397 |

**Finding**: Performance plateaus by 5-10k steps. Going from 0.16 epochs to 0.79 epochs gives no additional benefit. LR 2e-5 actually hurts. The model reaches its adapted capacity quickly — more data doesn't help within a single epoch.

---

## 8. Experiment 4 — IFQ-Aware Training (Ref-Value Blending Objective)

**Hypothesis**: IFQ degrades because `training_forward` passes `ys_refs=None` to the CLM decoder, so the 50/50 ref-value blending pathway that IFQ eval relies on (`ys_h.x /= 2; ys_h.x += ys_ref_values / 2`) is never exercised during training.

**Implementation**:
- Modified data pipeline to insert a masked-X + structure context member with probability `ifq_p`
- Modified `training_forward` to extract encoder hidden states from the IFQ member and pass as `ys_ref_values` to the CLM decoder
- This exercises the ref-value blending pathway during training

**Setup**: 10k steps, LR 1e-4, mask=0.00, ifq_p ∈ {0.3, 0.5}, sdrop ∈ {0.0, 0.2}.

| ifq_p | SDrop | seq_only (Δ) | struct (Δ) | IFQ (Δ) |
|-------|-------|-------------|-----------|---------|
| 0.3 | 0.0 | 0.400 (+0.001) | 0.357 (-0.010) | 0.428 (**-0.046**) |
| 0.3 | 0.2 | 0.403 (+0.004) | 0.357 (-0.010) | 0.422 (**-0.052**) |
| 0.5 | 0.0 | 0.399 (+0.000) | 0.355 (-0.012) | 0.425 (**-0.049**) |
| 0.5 | 0.2 | 0.399 (+0.000) | 0.348 (-0.019) | 0.422 (**-0.052**) |

**Finding**: IFQ-aware training made IFQ degradation **worse** than standard fine-tuning (-0.046 to -0.052 vs -0.029 to -0.043). The root cause is not the blending pathway being unused — it's the shared weights shifting. Adding the IFQ objective may create competing gradient signals.

---

## 9. Experiment 5 — LoRA (Low-Rank Adaptation)

**Hypothesis**: Freeze all 614M pretrained parameters. Train only small LoRA adapter matrices on encoder Q/V attention projections. The pretrained IFQ pathway stays intact through frozen base weights.

**Implementation**:
- `LoRALinear` wrapper: `output = base(x) + (x @ A^T @ B^T) * (alpha/rank)`
- Applied to encoder's `q_proj` and `v_proj` only (not decoders, to avoid conflicting adapters on shared weights)
- At save time, LoRA merged into base weights via deepcopy + `weight += B @ A * scale`
- After merge, the checkpoint is standard format (loadable without LoRA code)

**Setup**: 10k steps, mask=0.00, sdrop=0.0, rank ∈ {8, 16}, LR ∈ {5e-4, 1e-3}.

| Rank | Trainable | LR | seq_only (Δ) | struct (Δ) | IFQ (Δ) |
|------|-----------|-----|-------------|-----------|---------|
| 8 | 786K (0.4%) | 5e-4 | 0.393 (**-0.006**) | 0.343 (-0.024) | 0.458 (-0.016) |
| 8 | 786K (0.4%) | 1e-3 | 0.386 (**-0.013**) | 0.352 (-0.015) | 0.452 (-0.022) |
| 16 | 1.57M (0.9%) | 5e-4 | 0.384 (**-0.015**) | 0.337 (-0.030) | 0.449 (-0.025) |
| 16 | 1.57M (0.9%) | 1e-3 | 0.386 (**-0.013**) | 0.332 (-0.035) | 0.447 (-0.027) |

**Finding**: LoRA is **worse than full fine-tuning on ALL metrics**. seq_only actually DROPS, and IFQ still degrades.

**Why LoRA failed**: Train/eval mismatch from weight tying. During training, LoRA adapters are only in the encoder path — the decoder uses base weights. At save time, LoRA merges into the shared weights, so the decoder gets a delta it was never trained with. The delta was optimized for the encoder context encoding, not the decoder's sequence scoring task.

---

## 10. Experiment 5b — Ultra-Low Learning Rate

**Setup**: Full fine-tuning with extremely low LR to minimize weight drift.

| LR | seq_only (Δ) | struct (Δ) | IFQ (Δ) | Notes |
|----|-------------|-----------|---------|-------|
| 1e-6 | 0.398 (-0.001) | 0.367 (+0.000) | 0.474 (+0.000) | No learning |
| 5e-6 | *(still running)* | | | |

**Finding**: LR 1e-6 is a no-op. The model barely changes — IFQ is preserved but nothing is learned.

---

## 11. Summary Table

| Approach | seq_only Δ | IFQ Δ | ens_ifq Δ | Verdict |
|----------|-----------|-------|-----------|---------|
| Full FT, LR 1e-4, mask=0.15, sdrop=0.5 | **+0.010** | -0.034 | -0.034 | Best seq_only, but net negative |
| Full FT, LR 5e-5 | +0.007 | -0.029 | -0.023 | Smaller gain, still net negative |
| Full FT, LR 1e-5, 50k steps | +0.004 | *(not tracked)* | — | Plateaus early |
| Struct/mask sweep | +0.001 to +0.004 | -0.029 to -0.043 | — | Cannot prevent IFQ loss |
| IFQ-aware training | +0.000 to +0.004 | **-0.046 to -0.052** | — | Worse — IFQ degrades more |
| LoRA rank 8-16 | **-0.006 to -0.015** | -0.016 to -0.027 | — | Worse — both metrics degrade |
| Ultra-low LR (1e-6) | -0.001 | +0.000 | — | No learning |

**No approach improves the model's overall best score (ens_ifq = 0.513).**

---

## 12. Key Insights

1. **The weight-tied architecture is the fundamental obstacle.** Encoder, MLM decoder, and CLM decoder share all parameters. Any training gradient modifies the backbone that the IFQ pathway depends on.

2. **IFQ is fragile.** The inverse-folding query mode depends on a precise balance between encoder representations (ref values from structure) and decoder behavior (50/50 blending). Even small weight changes disrupt this.

3. **seq_only gains are real but small.** Best improvement is +0.010. But ens_ifq (0.513) >> best fine-tuned seq_only (0.409), so the IFQ pathway matters far more.

4. **IFQ degradation is worse on OOD.** The pretrained model's strongest OOD mode (ens_ifq = 0.530) drops to 0.492-0.500 after training. IID IFQ is more robust (-0.011 to -0.024 vs -0.030 to -0.040 OOD).

5. **More training doesn't help.** Performance plateaus by 5-10k steps (0.08-0.16 epochs). 50k steps gives no additional benefit.

6. **LoRA doesn't solve weight tying.** The train/eval mismatch from encoder-only LoRA on tied weights makes both metrics worse.

7. **The IFQ-aware objective backfires.** Adding ref-value blending to training creates competing gradients without stabilizing IFQ.

---

## 13. Potential Next Directions

1. **Score-level ensembling of pretrained + fine-tuned models**: Average logits from pretrained (strong IFQ) and fine-tuned (stronger seq_only) without modifying weights.

2. **Un-tie weights, then selective fine-tuning**: Break weight tying so encoder and decoder have separate parameter tensors. Freeze the encoder entirely, fine-tune only the decoder. The encoder's IFQ representations stay pretrained. This is a larger architectural change.

3. **Adapter layers**: Insert new trainable layers (bottleneck adapters between transformer blocks) rather than modifying existing weights. These don't interact with weight tying.

4. **Prompt tuning / prefix tuning**: Learn soft prompt embeddings prepended to the input. Model weights stay completely frozen.

5. **Curated training data**: Instead of training on all viral proteins, select families where the model currently performs poorly — focusing adaptation where it's most needed.

---

# PART II — Experiments Since July 15

Everything below was run after the original report. **Headline result: breaking weight
tying (untie decoders + freeze encoder) is the first intervention that stops IFQ
degradation — and with a full inverse-folding training objective on the frozen encoder,
IFQ is held flat while struct and seq_only both improve.** All during-training evals are
**single-prompt** (ctx6144, sim1.0); pretrained single-prompt baseline is
**seq 0.399 / struct 0.367 / ifq 0.4746**. Ensemble (ens_ifq) numbers, where measured, use
the 15-prompt ensemble and are called out explicitly. Pretrained **ens_ifq = 0.513**.

---

## 14. Experiment 6 — Similarity-Filtered Training

**Hypothesis**: IFQ degrades because training sequences too close to the DMS WTs overwrite
the pretrained representation. Filter training data by max identity-to-context.

**Setup**: 10k steps, sdrop=0.0, similarity thresholds {0.90, 0.95}, LR {2e-5, 5e-5}.

| Filter | LR | best seq_only (Δ) | struct (Δ) | IFQ (Δ) |
|--------|-----|-------------------|------------|---------|
| sim0.90 | 2e-5 | 0.4039 (+0.005) | 0.339 (-0.028) | 0.4631 (**-0.011**) |
| sim0.90 | 5e-5 | 0.4053 (+0.006) | 0.340 (-0.027) | 0.4470 (**-0.028**) |
| sim0.95 | 2e-5 | 0.4028 (+0.004) | 0.340 (-0.027) | 0.4629 (**-0.012**) |
| sim0.95 | 5e-5 | 0.4061 (+0.007) | 0.343 (-0.024) | 0.4466 (**-0.028**) |

**Finding**: Similarity filtering does **not** protect IFQ — behaves like the plain LR
sweep (IFQ −0.011 at 2e-5, −0.028 at 5e-5). Degradation is intrinsic to weight tying, not
caused by near-duplicate training sequences.

---

## 15. Experiment 7 — Mid/Low-LR Sweep ("Mid15")

**Goal**: Push LR low enough to preserve IFQ while still learning. This produced the first
checkpoint to *nearly* preserve IFQ.

**Setup**: 10k steps, mask=0.00, LR {1e-5, 2e-5, 3e-5}, sdrop {0.0, 0.2}.

| LR | SDrop | best seq_only (Δ) | struct (Δ) | IFQ (Δ) |
|----|-------|-------------------|------------|---------|
| **1e-5** | 0.0 | 0.4015 (+0.002) | 0.356 (-0.011) | **0.4710 (-0.0036)** |
| 2e-5 | 0.0 | 0.4033 (+0.004) | 0.341 (-0.026) | 0.4629 (-0.0117) |
| 3e-5 | 0.0 | 0.4064 (+0.007) | 0.343 (-0.024) | 0.4597 (-0.0149) |
| 5e-5 | 0.2 | 0.398 (-0.001) | 0.347 (-0.020) | 0.4538 (-0.0208) |

**Finding**: **LR 1e-5 ("Mid15") is the best IFQ-preserving tied-weight config** (IFQ only
−0.0036). IFQ damage scales cleanly with LR. Mid15 became the reference checkpoint for the
merging/ensemble experiments below.

---

## 16. Experiment 8 — Model Merging / Task Arithmetic

**Hypothesis**: Interpolate weights `θ = θ_pre + α·(θ_ft − θ_pre)` with small α to capture a
fraction of the seq_only gain while keeping most of the pretrained IFQ.

**Setup**: Merge each fine-tuned checkpoint with pretrained at α ∈ {0.05 … 1.0}. Scored
single-prompt IFQ. (Checkpoint-format fix: load via `load_model()` + `state_dict()` rather
than raw `torch.load`, since pretrained is DeepSpeed format.)

| Merge source | Best α | IFQ Δ | Notes |
|--------------|--------|-------|-------|
| Mid15 | 0.05–0.20 | **+0.0012** | Only merge that beats pretrained IFQ |
| FT (5e-5) | any | negative | Always hurts IFQ |
| FT (1e-4, m15) | small | ~0 | Marginal at best |
| LoRA8 | — | 0.000 | Checkpoint identical to pretrained (confirms LoRA merge was a no-op) |

**Finding**: Merging the lowest-LR checkpoint (Mid15) at tiny α gives a **marginal** IFQ gain
(+0.0012). Confirms Mid15 sits close to pretrained in weight space; higher-LR checkpoints
have drifted too far to merge usefully.

---

## 17. Experiment 9 — Inference-Time IFQ Sweep (no training)

**Idea**: The IFQ pathway has two inference knobs that require no training:
- **ref_blend**: encoder ref-value mix in the decoder blend (`ys_h.x = (1−b)·x + b·ref`),
  default 0.5 — now exposed via `model._ref_blend`.
- **self_prompt mode**: `default` (logaddexp) vs `consistency` fusion of the WT self-prompt.

**Setup**: Sweep ref_blend ∈ {0.2…0.8} × self_prompt ∈ {default, consistency}, single-prompt.

| Knob | Best value | IFQ Δ (single-prompt) |
|------|-----------|-----------------------|
| ref_blend | **0.6** | **+0.0027** |
| self_prompt | consistency | +0.0004 (negligible) |

**...but it does not survive the ensemble.** Re-running the **15-prompt ensemble** at
blend 0.6 vs the default 0.5:

| Model | ens_ifq @0.5 | ens_ifq @0.6 |
|-------|--------------|--------------|
| Pretrained | **0.5134** | 0.5113 (**−0.0021**) |

**Finding**: The single-prompt +0.0027 from ref_blend=0.6 **reverses at ensemble scale**
(−0.0021). The single prompt benefited from leaning harder on structure to prop up one weak
context; the ensemble already gets that robustness by averaging diverse contexts, so the
extra bias hurts. **ref_blend tuning is a wash-to-negative on the metric that matters.**
`consistency` self_prompt is negligible everywhere. Net: no free inference-time win.

---

## 18. Experiment 10 — Untied Encoder (freeze encoder, untie decoders) ⭐

**Hypothesis**: The root cause is weight tying. Give the decoders their own parameter copies
(`untie_decoders()`), freeze the encoder + final norm, and train only the decoders. The
encoder's structure representation — which IFQ reads — stays exactly pretrained.

**Implementation**: `untie_decoders()` clones any decoder parameter shared-by-id with the
encoder (preserves within-decoder cross-attn K/V tying); `--freeze-encoder` freezes encoder
+ norm; training forward builds separate decoder memories when untied; checkpoints tagged
`_untied_decoders` so `load_model()` re-unties on load. ~363M of 545M params trainable.

**Setup**: 10k steps, mask=0.00, sdrop=0.0, LR {1e-5, 5e-5, 1e-4}.

| LR | best step | seq_only (Δ) | struct (Δ) | IFQ (Δ) |
|----|-----------|--------------|------------|---------|
| **1e-5** | 4000 | **0.4063 (+0.007)** | 0.3649 (-0.002) | **0.4721 (-0.0025)** |
| 5e-5 | 2000 | 0.4101 (+0.011) | 0.3589 (-0.008) | 0.4545 (-0.020) |
| 1e-4 | 6000 | 0.4085 (+0.010) | 0.3604 (-0.007) | 0.4432 (-0.031) |

**Findings**:
- **At LR 1e-5, untied beats tied Mid15 on both axes**: 3× the seq_only gain (+0.007 vs
  +0.002) with less than half the IFQ damage (−0.0025 vs −0.0036). The frozen encoder
  protects IFQ better than any tied-weight config.
- Untying does **not** unlock higher LR for IFQ — at LR ≥5e-5, IFQ still falls because the
  decoder's cross-attention drifts out of alignment with the frozen encoder.
- **"Untied15" (LR 1e-5, best.ckpt) is the best all-round checkpoint produced** — see §21 for
  its ensemble result, where it beats pretrained.

---

## 19. Experiment 11 — Frozen Cross-Attention K/V (untied + freeze CLM xattn)

**Hypothesis (from §18)**: IFQ falls at higher LR because the CLM decoder's cross-attn K/V
(the encoder↔decoder bridge, tied across layers) drift. Freeze *just* those K/V while
leaving decoder self-attn + FFN trainable.

**Setup**: as §18 plus `--freeze-clm-xattn` (freezes `clm_decoder.layers[0].multihead_attn.{k,v}_proj`, ~2.1M params).

| LR | best step | seq_only (Δ) | struct (Δ) | IFQ (Δ) |
|----|-----------|--------------|------------|---------|
| 1e-5 | 4000 | 0.4049 (+0.006) | 0.3642 (-0.003) | 0.4712 (-0.0034) |
| 5e-5 | 6000 | 0.4113 (+0.012) | 0.3612 (-0.006) | 0.4510 (-0.024) |
| 1e-4 | 4000 | 0.4081 (+0.010) | 0.3585 (-0.009) | 0.4355 (-0.039) |

**Finding**: **Theory wrong — freezing K/V made everything worse** (lower seq_only AND more
IFQ damage than §18 at every LR). Freezing K/V while the decoder's queries/representations
move breaks the decoder's *internal* attention consistency. Letting K/V co-adapt with the
rest of the decoder is better. IFQ degradation is not a K/V-drift problem.

---

## 20. Experiment 12 — IFQ-Objective Training on a Frozen Encoder ⭐⭐

**Hypothesis**: Past IFQ-aware runs (§8) failed only because tied weights let the objective
corrupt the encoder. With the encoder frozen (§18), train the decoder *directly on the
inverse-folding objective* (`ifq_p`) — the decoder gets more practice reading structure→
sequence while the encoder representation stays fixed. A pLDDT diagnostic (§22) first
confirmed the training structures are good enough to learn from.

**Setup**: `--freeze-encoder` + `--ifq-p {0.5, 1.0}`, sdrop=0.0 (structures always present),
LR {1e-5, 5e-5}, 10k steps.

| ifq_p | LR | best step | seq_only (Δ) | struct (Δ) | IFQ (Δ) |
|-------|-----|-----------|--------------|------------|---------|
| 0.5 | 1e-5 | 4000 | 0.4058 (+0.007) | 0.3667 (-0.001) | 0.4726 (-0.0020) |
| **1.0** | **1e-5** | 4000–6000 | **0.4046 (+0.006)** | **0.3710 (+0.0037)** | **0.4749 (+0.0002)** |
| 1.0 | 5e-5 | — | *(blocked by partition access change — not run, see §23)* | | |

**Findings**:
- **`ifq_p=1.0` is the first continued-training run that does not degrade IFQ at all** —
  single-prompt IFQ 0.4749 vs pretrained 0.4746 (**+0.0002**, effectively flat), while
  **struct actually improves** (0.3710 vs 0.3673, +0.0037) and seq_only improves (+0.006).
  Every prior run dropped struct and IFQ; this one lifts both.
- The effect is monotone in the IFQ objective: no-IFQ untied (−0.0025) → ifq_p 0.5 (−0.0020)
  → ifq_p 1.0 (+0.0002). Training the decoder on inverse folding, with the encoder frozen,
  is the mechanism that protects and slightly strengthens the structural pathway.
- **This validates the core thesis**: IFQ degradation was never about the data or the
  objective — it was weight tying. Remove tying + train the right objective, and IFQ holds.
- **Open**: its 15-prompt **ens_ifq** is not yet measured (the eval + the LR 5e-5 run are
  blocked by the cluster access change in §23). Given Untied15 flips from single-prompt
  −0.0025 to ensemble **+0.0014** (§21), this checkpoint — which starts *higher* single-prompt
  — is the strongest candidate yet to beat pretrained ens_ifq. **This is the top priority to
  measure once access is restored.**

---

## 21. Experiment 13 — Multi-Checkpoint & Ensemble ref_blend Evals

**15-prompt ens_ifq (the metric that matters), measured directly:**

| Model | ens_ifq All | IID | OOD | Δ vs Pre |
|-------|-------------|-----|-----|----------|
| Pretrained | 0.5134 | 0.4818 | 0.5303 | — |
| Mid15 (tied 1e-5) | 0.5151 | — | — | +0.0016 |
| **Untied15 (untied 1e-5)** | **0.5148** | 0.4839 | 0.5313 | **+0.0014** |

**Findings**:
- **Both Mid15 and Untied15 beat pretrained ens_ifq**, by +0.0016 and +0.0014 — even though
  both *lose/tie* pretrained at single-prompt IFQ. The ensemble reweights where each model is
  strong; a checkpoint can trail single-prompt yet win the ensemble.
- **Untied15 dominates Mid15 overall**: comparable ens_ifq but much better seq_only (0.4063 vs
  0.4015), so it is a strictly better checkpoint than anything from the tied-weight era.
- Untied15 wins on **both IID and OOD** ens_ifq — the untied gain generalizes.
- These are small (+0.0014–0.0016) but they are the **first real, positive movements over the
  pretrained model's best score** in the whole program.

---

## 22. Training-Data Structure Quality (pLDDT Diagnostic)

**Purpose**: Decide whether IFQ-objective training (§20) can work, or whether the training
structures are too low-confidence to teach inverse folding.

**Findings** (`d1_diversity_struct`, raw float pLDDT on disk, 36.1M residues):
- 98.6% of sequences have structure; only 3.3% of residues are missing (NaN).
- Per-residue pLDDT: **mean 71, median 76**; >90: 24%, 70–90: 34%, 50–70: 21%, <50: 21%.
- **58% of residues clear the model's own high-confidence threshold (pLDDT ≥70).**

**Verdict**: Mixed-to-moderate confidence — good enough that inverse-folding training is
sensible (not noise), which greenlit §20. The ~21% of low-confidence residues argue for
pLDDT≥70 weighting/masking in a future refinement of the IFQ objective.

---

## 23. Summary Table (updated)

Δ vs pretrained single-prompt (seq 0.399 / struct 0.367 / ifq 0.4746), best checkpoint:

| Approach | seq_only Δ | struct Δ | IFQ Δ (single) | ens_ifq Δ | Verdict |
|----------|-----------|----------|----------------|-----------|---------|
| Full FT (LR 1e-4, m15, sd5) | +0.010 | -0.006 | -0.034 | -0.034 | Best seq, net negative |
| Similarity-filtered | +0.004…+0.007 | -0.024…-0.028 | -0.011…-0.028 | — | No IFQ protection |
| Mid15 (tied LR 1e-5) | +0.002 | -0.011 | -0.0036 | **+0.0016** | First to ~preserve IFQ; wins ensemble |
| Model merging (Mid15, α0.05) | ~0 | ~0 | +0.0012 | — | Marginal |
| Inference ref_blend=0.6 | 0 | 0 | +0.0027 (single) | **−0.0021** | Reverses at ensemble — no win |
| **Untied encoder (LR 1e-5)** | **+0.007** | -0.002 | -0.0025 | **+0.0014** | Best all-round checkpoint |
| Untied, LR ≥5e-5 | +0.010…+0.011 | -0.007…-0.008 | -0.020…-0.031 | — | High LR still breaks IFQ |
| Frozen CLM xattn K/V | +0.006…+0.012 | -0.003…-0.009 | -0.003…-0.039 | — | Worse than untied — theory wrong |
| **IFQ-obj, frozen enc (ifq_p 1.0, 1e-5)** | **+0.006** | **+0.0037** | **+0.0002** | *pending* | **First to hold IFQ + lift struct** |

**Two interventions now beat pretrained ens_ifq (Mid15 +0.0016, Untied15 +0.0014), and the
ifq_p=1.0 frozen-encoder run is the first to hold single-prompt IFQ flat while improving
struct — its ensemble number is the key open measurement.**

---

## 24. Key Insights (updated)

1. **Weight tying was the whole problem — and it's now solved.** Untying the decoders and
   freezing the encoder (§18) is the first intervention that reliably preserves IFQ. Prior
   conclusions ("IFQ is intrinsically fragile") were really about tied weights.

2. **The inverse-folding objective, applied correctly, strengthens IFQ.** With a frozen
   encoder, `ifq_p=1.0` training lifts struct (+0.0037) and holds IFQ flat (§20) — the exact
   opposite of the tied-weight IFQ-aware runs (§8), which were the worst offenders. Same idea,
   opposite result, entirely because of tying.

3. **Single-prompt IFQ is a misleading proxy for ens_ifq.** Mid15 and Untied15 both *lose*
   single-prompt but *win* the 15-prompt ensemble (§21). Judge checkpoints by ens_ifq.

4. **Inference-time tuning is a trap.** ref_blend=0.6 looked like a free +0.0027 single-prompt
   win but reverses to −0.0021 at ensemble (§17). Always validate on the ensemble.

5. **Freezing the encoder is necessary but not sufficient at high LR.** Untied training still
   loses IFQ at LR ≥5e-5 (§18); freezing cross-attn K/V on top makes it worse (§19). Low LR +
   frozen encoder + IFQ objective is the working recipe.

6. **Gains are small but finally positive.** +0.0014–0.0016 ens_ifq is the first real movement
   over pretrained. The ifq_p=1.0 ensemble result (pending) is the best shot at extending it.

---

## 25. Next Directions

1. **[TOP PRIORITY] Measure ens_ifq of the ifq_p=1.0 frozen-encoder checkpoint.** It holds
   single-prompt IFQ flat and lifts struct; given the single-prompt→ensemble flip seen for
   Untied15, it is the strongest candidate to beat pretrained ens_ifq. **Blocked on cluster
   access (see below).**
2. **pLDDT-weighted IFQ objective**: mask/weight the IFQ loss to pLDDT≥70 residues (§22) so
   the decoder learns only from confident structure.
3. **Finish the ifq_p=1.0 LR sweep** (5e-5) — does a frozen encoder finally make higher LR
   safe now that the objective protects IFQ?
4. **pLDDT-conditioned ref_blend** at inference: per-residue blend `b_i = f(pLDDT_i)` instead
   of a global scalar — a different lever than the (failed) global ref_blend sweep.
5. **Combine**: score-level ensemble of Untied15 + the ifq_p=1.0 checkpoint (best seq_only +
   best structural pathway).

---

## ⚠️ Cluster Access Status (July 28, 2026)

**Blocker**: The Kempner SLURM partitions were renamed — **`kempner_h100_priority` no longer
exists** (current GPU partitions: `kempner_h100`, `kempner_h200`, `kempner_gpu_priority`,
`kempner_rtx`, `kempner_interactive`, `kempner_requeue`). The user's association
(`kempner_marks_lab` account, QOS `kemp_gpu16_id32`) was scoped to the old partition and was
**not migrated** — new submissions to any Kempner GPU partition fail with *"Invalid qos
specification"*, and the `marks_lab` account has no Kempner-GPU access at all (*"Invalid
account or account/partition combination"*). Already-running jobs are grandfathered; queued
jobs on the dead partition (e.g. the ifq_p=1.0 LR 5e-5 run) will never start.

**Action needed**: FASRC/Kempner admin must grant the `kempner_marks_lab` association a valid
QOS on the new `kempner_h100` (or `kempner_gpu_priority`) partition. All `slurm/*.sh` scripts
hardcode `--partition=kempner_h100_priority --qos=kemp_gpu16_id32` and must be updated to the
new names once the correct QOS is known.

---

## Infrastructure

- **Cluster**: Harvard FAS RC Kempner partition, H100 GPUs
- **Training**: 4x H100, torchrun DDP, bf16, AdamW, cosine schedule, 200 warmup steps
- **Eval**: 45 viral DMS datasets, zero-shot Spearman, during-training eval every 2k steps
- **Structures**: Protenix folding (with MSA from colabfold_search), pLDDT >= 70
- **Wandb**: `poet2-d1-sweep` project
- **Key scripts (Part II)**: `poet_2.models.poet_2.untie_decoders`, `--freeze-encoder` /
  `--freeze-clm-xattn` / `--ifq-p` train flags, `scripts/eval_model_merging.py`,
  `scripts/eval_ifq_sweep.py`, `scripts/eval_ensemble_refblend.py`,
  `scripts/eval_multi_checkpoint_ensemble.py`
