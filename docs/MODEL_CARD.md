# Model Card: Bovine-Enformer v1.0

## Model description
Bovine-Enformer adapts the human/mouse-pretrained Enformer backbone (251M parameters, frozen)
to bovine regulatory genomics via a task head trained on the BovineSCellRegAtlas single-cell
multi-ome atlas. The model predicts chromatin accessibility across **59 bovine cell-type
contexts** (12 tissues) for any input DNA sequence, and scores variant perturbation by
reference/alternate sequence comparison, complemented by atlas open-peak and JASPAR TF-motif
annotation.

- **Backbone**: Enformer (EleutherAI/enformer-official-rough, PyTorch port), frozen
- **Head**: MLP 3072→512(GELU)→59, sigmoid outputs
- **Training data**: 478,422 unified CREs × 59 context open labels (from 221 sample-level
  pseudobulk open calls) + 100,000 background windows (all-zero labels for calibration)
- **Training split**: chromosomes 5-29 train / chr4 validation / chr1-3 test
- **Reference genome**: ARS-UCD1.2

## Performance (held-out chromosomes 1-3)
| Metric | Value |
|---|---|
| CRE context macro AUROC (46 evaluable contexts) | **0.9117** |
| Background P(open) mean / p95 | 0.042 / 0.229 |
| CRE vs background separation AUROC | 0.871 |
| Quantitative head (log-TMM) per-context Spearman (v0.9 quant head) | 0.642 |

## Honest limitations (measured, not speculative)
1. **Variant-level effect discrimination is weak**: zero-shot and fine-tuned single-base
   perturbation deltas do not discriminate conditional eQTL from matched background variants
   (AUROC ≈ 0.51 across Enformer, Borzoi, DNABERT-2, ISM and gradient scoring). Use the
   perturbation score as one annotation among several, not as a causal verdict.
2. **Effect direction cannot be predicted** (sign concordance ≈ 0.50).
3. Training supervision derives from ~2 individuals (2 breeds); context labels aggregate
   few samples per context. Probabilities are calibrated against genomic background but
   not against held-out individuals.
4. No mammary context is present in the atlas v1 label set.
5. The model is not validated for non-SNV variants.

## Intended use
- Research use: regulatory annotation of bovine sequences/variants, variant prioritization,
  input feature generation for downstream models.
- Not intended for: clinical/breeding decisions on individual animals without further validation.

## Citation
See README.md (manuscript in preparation).
