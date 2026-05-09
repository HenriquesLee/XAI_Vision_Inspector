# 📊 Benchmark Results

Raw per-image faithfulness evaluation results for the Eigen-IG paper:
**"Eigen-IG: SVD-Weighted Path Integration for Sparser CNN Attribution Maps"**

---

## Files

| File | Architecture | Images | Methods |
|---|---|---|---|
| `benchmark_ResNet50.csv` | ResNet-50 | 500 | Eigen-IG, IG, Grad-CAM |
| `benchmark_VGG16.csv` | VGG-16 | 500 | Eigen-IG, IG, Grad-CAM |
| `benchmark_EfficientNet-B0.csv` | EfficientNet-B0 | 500 | Eigen-IG, IG, Grad-CAM |

---

## Experimental Setup

| Setting | Value |
|---|---|
| Dataset | ImageNet ILSVRC-2012 validation set |
| Images per architecture | 500 (randomly sampled) |
| Pretrained weights | torchvision ImageNet1K |
| Hardware | NVIDIA RTX 5050 Laptop GPU (8 GB) |
| PyTorch | 2.12.0 + CUDA 12.8 |
| Framework | Python 3.13 |

**Eigen-IG hyperparameters:**

| Parameter | Value |
|---|---|
| Integration steps (T) | 50 |
| SVD rank (k) | 10 |
| Softmax temperature (τ) | 1.0 |
| Baseline | Zero tensor |

**IG hyperparameters:**

| Parameter | Value |
|---|---|
| Integration steps (T) | 50 |
| Baseline | Zero tensor |

**Faithfulness metric settings:**

| Parameter | Value |
|---|---|
| Deletion/Insertion steps | 5 |
| Infidelity perturbations | 5 |
| Noise std (infidelity) | 0.1 |

---

## CSV Schema

Each CSV has one row per image per method:

| Column | Type | Description |
|---|---|---|
| `image` | string | Filename |
| `class_idx` | int | ImageNet class index (0–999) |
| `class_label` | string | Human-readable class name |
| `confidence` | float | Top-1 softmax confidence |
| `method` | string | `Eigen-IG`, `Integrated Gradients`, or `Grad-CAM` |
| `deletion_auc` | float | Deletion AUC — lower is better |
| `insertion_auc` | float | Insertion AUC — higher is better |
| `infidelity` | float | Infidelity — lower is better |

---

## Reproducing the Summary Statistics

```python
import pandas as pd
from scipy import stats
import numpy as np

df = pd.read_csv('benchmark_ResNet50.csv')

# Mean ± std per method
summary = df.groupby('method')[['deletion_auc','insertion_auc','infidelity']]\
            .agg(['mean','std']).round(4)
print(summary)

# Paired t-test: Eigen-IG vs IG on deletion
eig = df[df.method == 'Eigen-IG']['deletion_auc']
ig  = df[df.method == 'Integrated Gradients']['deletion_auc']
# Align on same images
common = set(df[df.method=='Eigen-IG']['image']) & set(df[df.method=='Integrated Gradients']['image'])
eig_v = df[(df.method=='Eigen-IG') & (df.image.isin(common))].sort_values('image')['deletion_auc']
ig_v  = df[(df.method=='Integrated Gradients') & (df.image.isin(common))].sort_values('image')['deletion_auc']
t, p  = stats.ttest_rel(eig_v, ig_v)
print(f"t={t:.3f}, p={p:.4f}")
```

---

## Summary (from paper)

### ResNet-50

| Method | Deletion AUC ↓ | Insertion AUC ↑ | Infidelity ↓ |
|---|---|---|---|
| Grad-CAM | 0.0629 ± 0.0852 | 0.1558 ± 0.1283 | 14.847 |
| IG | 0.0336 ± 0.0429 | **0.0879** ± 0.1008 | **0.651** |
| **Eigen-IG** | **0.0278** ± 0.0280 | 0.0560 ± 0.0695 | 1.315 |

Eigen-IG vs IG deletion: −17.4%, p < 0.0001 ✅

### VGG-16

| Method | Deletion AUC ↓ | Insertion AUC ↑ | Infidelity ↓ |
|---|---|---|---|
| Grad-CAM | 0.0805 ± 0.0986 | 0.1799 ± 0.1733 | 6.891 |
| IG | **0.0482** ± 0.0430 | **0.1216** ± 0.1516 | **0.814** |
| **Eigen-IG** | 0.0455 ± 0.0435 | 0.0762 ± 0.1114 | 1.438 |

Eigen-IG vs IG deletion: −5.6%, p = 0.12 ❌ (not significant)

### EfficientNet-B0

| Method | Deletion AUC ↓ | Insertion AUC ↑ | Infidelity ↓ |
|---|---|---|---|
| Grad-CAM | 0.0766 ± 0.0955 | 0.3186 ± 0.2218 | 30.195 |
| IG | **0.0643** ± 0.0692 | 0.1764 ± 0.1815 | **0.825** |
| **Eigen-IG** | 0.0625 ± 0.0719 | **0.2385** ± 0.2031 | 1.398 |

Eigen-IG vs IG insertion: +35.2%, p < 0.0001 ✅

---

## Notes

- All pretrained models loaded via `torchvision.models` with `ImageNet1K` weights — no fine-tuning.
- Infidelity is consistently ~2× higher for Eigen-IG than IG across all architectures, reflecting the sparsity–completeness tradeoff discussed in the paper.
- VGG-16 results show no significant deletion gain, consistent with the gradient vanishing / rank collapse analysis in Section 5 of the paper.
- Deletion/insertion steps set to 5 (vs 10–15 in the paper's single-image analysis) for computational efficiency across 500 images. This reduces resolution of the curves but does not materially affect mean AUC rankings.