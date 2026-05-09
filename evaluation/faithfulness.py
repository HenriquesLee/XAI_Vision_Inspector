"""
Faithfulness Evaluation Suite
==============================
Three standard metrics for evaluating XAI attribution maps:

  1. Deletion AUC  — progressively remove most-important pixels; lower = better
  2. Insertion AUC — progressively reveal most-important pixels; higher = better
  3. Infidelity    — how well attributions predict score changes under noise; lower = better

Bug fixes applied (v2):
  - BUG 1 (NaN root cause): infidelity dot-product shape mismatch.
      saliency_flat was (H*W,) but noise.flatten() was (B*C*H*W,) = 3x larger.
      RuntimeError was silently caught -> NaN.
      Fix: average noise across channels -> (H*W,) matches saliency.

  - BUG 2: perturbed.clamp(0, 1) corrupts ImageNet-normalized inputs.
      Normalized pixel values range ~ [-2.1, 2.6], not [0, 1].
      Fix: clamp removed entirely.

  - BUG 3: np.trapz was renamed np.trapezoid in NumPy 2.0.
      Fix: try/except fallback covers both versions.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def _trapz(y, dx):
    """Compatibility shim: NumPy 2.0 renamed np.trapz to np.trapezoid."""
    try:
        return float(np.trapezoid(y, dx=dx))
    except AttributeError:
        return float(np.trapz(y, dx=dx))


def safe_normalize(saliency_map: np.ndarray) -> np.ndarray:
    sal = np.asarray(saliency_map, dtype=np.float32)
    sal = np.clip(sal, 0.0, None)
    lo, hi = sal.min(), sal.max()
    if hi - lo < 1e-8:
        return np.ones_like(sal) * 0.5
    return (sal - lo) / (hi - lo)


def compute_deletion_curve(
    model, input_tensor, saliency_map, class_idx, steps=15, baseline_value=0.0,
):
    sal_map = safe_normalize(saliency_map)
    H, W = sal_map.shape
    total_pixels = H * W
    sorted_idx = np.argsort(sal_map.flatten())[::-1]
    masked = input_tensor.clone().detach()
    confidences = []

    for step in range(steps + 1):
        if step > 0:
            n_mask = max(1, int(total_pixels * step / steps))
            idx = sorted_idx[:n_mask]
            ys, xs = idx // W, idx % W
            masked[0, :, ys, xs] = baseline_value
        with torch.no_grad():
            prob = F.softmax(model(masked), dim=1)[0, class_idx].item()
        confidences.append(prob)

    auc = _trapz(confidences, dx=1.0 / steps)
    return np.array(confidences), auc


def compute_insertion_curve(
    model, input_tensor, saliency_map, class_idx, steps=15,
):
    sal_map = safe_normalize(saliency_map)
    H, W = sal_map.shape
    total_pixels = H * W
    sorted_idx = np.argsort(sal_map.flatten())[::-1]
    revealed = torch.zeros_like(input_tensor)
    confidences = []

    for step in range(steps + 1):
        if step > 0:
            n_add = max(1, int(total_pixels * step / steps))
            idx = sorted_idx[:n_add]
            ys, xs = idx // W, idx % W
            revealed[0, :, ys, xs] = input_tensor[0, :, ys, xs]
        with torch.no_grad():
            prob = F.softmax(model(revealed), dim=1)[0, class_idx].item()
        confidences.append(prob)

    auc = _trapz(confidences, dx=1.0 / steps)
    return np.array(confidences), auc


def compute_infidelity(
    model, input_tensor, saliency_map, class_idx, n_perturb=20, noise_std=0.1,
):
    """
    BUG FIX: saliency_flat is (H*W,). noise.flatten() was (B*C*H*W,) - 3x mismatch.
    Fix: reduce noise to (H*W,) by averaging across batch+channels before dot product.
    BUG FIX: removed .clamp(0,1) - ImageNet-normalized inputs are NOT in [0,1].
    """
    sal_map = safe_normalize(saliency_map)
    device = input_tensor.device
    saliency_flat = torch.from_numpy(sal_map.flatten()).float().to(device)

    with torch.no_grad():
        orig_score = F.softmax(model(input_tensor), dim=1)[0, class_idx].item()

    diffs = []
    for _ in range(n_perturb):
        noise = torch.randn_like(input_tensor) * noise_std
        noise_spatial = noise[0].mean(dim=0).flatten()
        perturbed = input_tensor + noise

        with torch.no_grad():
            pert_score = F.softmax(model(perturbed), dim=1)[0, class_idx].item()

        model_delta = orig_score - pert_score
        attr_delta  = (saliency_flat * noise_spatial).sum().item()
        diffs.append((model_delta - attr_delta) ** 2)

    return float(np.mean(diffs)) if diffs else 0.0


def run_full_faithfulness_eval(
    model, input_tensor, saliency_maps: Dict[str, np.ndarray], class_idx, steps=15,
) -> Dict[str, Dict[str, float]]:
    """
    Run deletion, insertion, and infidelity for every method.
    
    Interpretation:
        Deletion_AUC        lower is better
        Insertion_AUC       higher is better  
        Infidelity          lower is better
        Deletion_final_drop (%) higher is better
    """
    model.eval()
    results = {}

    for name, sal_map in saliency_maps.items():
        try:
            del_conf, del_auc = compute_deletion_curve(model, input_tensor, sal_map, class_idx, steps)
            _, ins_auc        = compute_insertion_curve(model, input_tensor, sal_map, class_idx, steps)
            infid             = compute_infidelity(model, input_tensor, sal_map, class_idx)
            final_drop        = float(del_conf[0] - del_conf[-1])

            results[name] = {
                "Deletion_AUC":        round(del_auc,        4),
                "Insertion_AUC":       round(ins_auc,        4),
                "Infidelity":          round(infid,          6),
                "Deletion_final_drop": round(final_drop * 100, 1),
            }
            print(f"✓ {name:25s}  Del={del_auc:.4f}  Ins={ins_auc:.4f}  Infid={infid:.6f}  Drop={final_drop*100:.1f}%")

        except Exception as e:
            print(f"✗ {name}: {type(e).__name__} — {e}")
            results[name] = {"Deletion_AUC": float("nan"), "Insertion_AUC": float("nan"),
                             "Infidelity": float("nan"), "Deletion_final_drop": float("nan")}

    return results
