# Evaluation summary (Bovine-Enformer project)

All evaluations use leave-chromosome-out cross-validation (chr1-3 held out) unless noted.

## Task-level capability map (established by this project)

| Task | Result | Evidence |
|---|---|---|
| Region-level CRE accessibility (59 ctx) | **AUROC 0.9117** | v1.0 head, this repo |
| Quantitative accessibility (log-TMM) | Spearman 0.642 | quant head |
| Variant prioritization (enrichment) | 1.5-7.5x | atlas/motif/E4 shared-causal |
| Matched eQTL discrimination (tab. features) | 0.61-0.69 (power-confounded) | E1-E3, mt5v2 |
| Credible-set fine-mapping (within-CS ranking) | functional features cannot beat \|Z\| | CS E1/E2 |
| Zero-shot variant scoring (Enformer/Borzoi/DNABERT-2) | AUROC ≈ 0.51 (random) | H3a, ISM/grad |
| Effect direction prediction | sign concordance ≈ 0.50 | E5 |

## Ablations

| Variant of the head | test/val macro AUROC |
|---|---|
| Frozen trunk + linear head (478k, center) | 0.9003 (test) |
| Frozen trunk + MLP head (478k, center) | 0.9073 (test) |
| End-to-end fine-tune (last 3 layers), 60k subset | 0.8936 (val) |
| MLP + jitter pooling + background, 60k subset | 0.8878 (test) |
| **MLP + background (v1.0, 478k + 100k bg)** | **0.9117 (test)** |

## Numerical/precision checks
- bf16 vs fp32 trunk inference for variant deltas: corr 0.9986 — bf16 is not the cause of
  small deltas; head-logit saturation is (see MODEL_CARD limitations).
- ISM (saturation mutagenesis) and gradient saliency scoring: AUROC 0.508/0.510 — scoring
  function is not the bottleneck for variant-level tasks.
