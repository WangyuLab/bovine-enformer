# Bovine-Enformer

> https://github.com/WangyuLab/bovine-enformer

**Bovine cell-context accessibility & variant perturbation prediction from DNA sequence**

牛细胞类型上下文染色质可及性与变异扰动预测工具 —— 人源 Enformer 经 BovineSCellRegAtlas 单细胞图谱监督的领域适配模型。

---

## Overview | 概述

Bovine-Enformer adapts the human/mouse-pretrained [Enformer](https://www.nature.com/articles/s41592-021-01252-x) trunk to cattle (*Bos taurus*, ARS-UCD1.2) by supervising a bovine output head with **BovineSCellRegAtlas** — a single-cell multi-omics regulatory atlas (478,422 unified CREs × 153 cell-type contexts across 12 tissues).

Given any bovine DNA sequence, region, or variant, the model predicts:

- **Accessibility profile**: probability of open chromatin in **59 bovine cell-type contexts** (test macro AUROC **0.9117**, chromosome-held-out)
- **Variant perturbation score**: ref/alt difference of the predicted profile (per-context Δ, max|Δ|, L2, most-sensitive context)

Why not zero-shot? We show that zero-shot scoring with human-pretrained models fails on bovine variant tasks (AUROC ≈ 0.51 for Enformer, Borzoi, and DNABERT-2 alike), while atlas-supervised adaptation recovers strong region-level performance (0.90) — the core finding of the companion paper.

## Key results | 关键结果

| Task | Metric | Value |
|---|---|---|
| Region-level CRE accessibility (59 ctx) | test macro AUROC (chr1-3 held out) | **0.9117** (v1.0) |
| Best contexts | lung alveolar epithelial / skin spinous / spleen plasma | 0.996 / 0.988 / 0.985 |
| Zero-shot Enformer/Borzoi/DNABERT-2 (variant discrimination) | AUROC | 0.50–0.51 (≈ random) |

## Installation | 安装

```bash
# python 3.11 + torch (CUDA build recommended; CPU works with fallback attention)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install enformer-pytorch==0.8.12 transformers==4.56.2 numpy pandas pyfaidx

# model weights
# 1) Enformer trunk (EleutherAI PyTorch port)
huggingface-cli download EleutherAI/enformer-official-rough --local-dir models/enformer-official-rough
# 2) Bovine head v1.0 (GitHub Release asset)
curl -L -o head_v10.pt https://github.com/WangyuLab/bovine-enformer/releases/download/v1.0.0/head_v10.pt
#    -> models/bovine_enformer/head_v10.pt  (test AUROC 0.9117, background-calibrated)
```

> ⚠️ transformers ≥5.0 breaks enformer-pytorch — pin 4.56.2. numpy ≥2.0 breaks scipy in older stacks — pin numpy 1.x if needed.

## Quickstart | 快速开始

```bash
# 1) any DNA sequences (FASTA) -> 59-context accessibility probabilities
python -m bovine_enformer.cli seq --fa seqs.fa --out preds.tsv

# 2) genomic regions (BED, ARS-UCD1.2) -> accessibility profiles
python -m bovine_enformer.cli region --bed regions.bed \
    --genome cattle_ARS-UCD1.2.fa --fai cattle_ARS-UCD1.2.fa.fai --out preds.tsv

# 3) variants (chr pos ref alt, 1-based) -> perturbation scores
python -m bovine_enformer.cli var --variants variants.txt \
    --genome cattle_ARS-UCD1.2.fa --fai cattle_ARS-UCD1.2.fa.fai --out scores.tsv
```

Variant output columns: `max_abs_delta`, `l2_delta`, `argmax_ctx` (most sensitive context), plus the full 59-dim Δ vector.

## Python API

```python
from bovine_enformer.model import BovineEnformer

be = BovineEnformer("models/enformer-official-rough", "models/bovine_enformer/head_mlp.pt")

# sequence -> (n, 59) accessibility probabilities
P = be.predict_accessibility(["ACGTACGT..." * 2000])

# variant perturbation
r = be.score_variant(ref_seq, alt_base="G", center=len(ref_seq) // 2)
print(r["max_abs_delta"], r["argmax_ctx"])
```

## Model card | 模型说明

- **Trunk**: Enformer (251.2M params, frozen in v0.1), input 196,608 bp one-hot
- **Head**: MLP 3072→512→59 (GELU, dropout 0.2), trained on center-17-bin mean embeddings
- **Supervision**: 478,422 atlas CREs × 153-context open/closed matrix (sample-level calls aggregated to context level; 59 contexts with sufficient coverage)
- **Split**: train chr5-29 / val chr4 / test chr1-3 (anti-leakage chromosome holdout)
- **Training**: BCE with per-context pos_weight (≤50), AdamW, early stopping on val macro AUROC
- **Stage B (optional)**: end-to-end fine-tune unfreezing the last 3 transformer layers

## Repository layout | 目录结构

```
bovine_enformer/
├── bovine_enformer/
│   ├── model.py      # BovineEnformer: trunk + head, accessibility & variant scoring
│   └── cli.py        # seq / region / var subcommands
├── models/
│   ├── enformer-official-rough/   # trunk weights (downloaded)
│   └── bovine_enformer/           # bovine head + per-context metrics
├── scripts/          # training & embedding extraction pipelines
└── docs/             # evaluation reports, method notes
```

## Citation | 引用

```text
Bovine-Enformer: cross-species adaptation of genomic foundation models
via single-cell atlas supervision. (manuscript in preparation)
```

## License

MIT (code). Model weights: Enformer weights follow the original DeepMind/EleutherAI terms; bovine head weights are released by the authors.
