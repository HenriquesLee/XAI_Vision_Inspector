"""
Integrated Gradients
Paper: https://arxiv.org/abs/1703.01365  (Sundararajan et al., 2017)

Implemented from scratch using PyTorch autograd.

The key insight: attribution should satisfy two axioms —
  1. Sensitivity: if input differs from baseline at feature i and predictions differ, i gets non-zero attribution.
  2. Implementation Invariance: attributions are identical for functionally equivalent networks.

IG satisfies both by integrating gradients along the straight-line path
from a baseline (typically all-zeros or all-gray image) to the actual input.

Formula:
  IG_i(x) = (x_i - x'_i) × ∫[α=0→1] ∂F(x' + α(x - x')) / ∂x_i  dα

Approximated via Riemann summation with `n_steps` interpolation points.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class IntegratedGradients:
    """
    Integrated Gradients attribution method.

    Attributes:
        model: PyTorch model (must support gradient computation)
        n_steps: Number of interpolation steps for Riemann approximation (50–300 typical)
        baseline_type: 'zeros' | 'uniform_noise' | 'gaussian_noise' | 'blur'
    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_steps: int = 100,
        baseline_type: str = "zeros",
    ):
        self.model = model
        self.n_steps = n_steps
        self.baseline_type = baseline_type
        self.model.eval()

    def _get_baseline(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Create the baseline input (what we integrate from)."""
        if self.baseline_type == "zeros":
            return torch.zeros_like(input_tensor)
        elif self.baseline_type == "uniform_noise":
            return torch.rand_like(input_tensor)
        elif self.baseline_type == "gaussian_noise":
            return torch.randn_like(input_tensor) * 0.1
        else:
            return torch.zeros_like(input_tensor)

    def _interpolate_inputs(
        self,
        baseline: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate n_steps interpolated inputs along the path baseline → input.
        Returns shape: (n_steps, C, H, W)
        """
        alphas = torch.linspace(0.0, 1.0, self.n_steps, device=input_tensor.device)
        alphas = alphas.view(-1, 1, 1, 1)
        delta = input_tensor - baseline
        interpolated = baseline + alphas * delta
        return interpolated

    def _compute_gradients(
        self,
        interpolated: torch.Tensor,
        class_idx: int,
    ) -> torch.Tensor:
        """
        Compute gradients of class_idx score w.r.t. each interpolated input.
        Processes in a single batched forward+backward pass for efficiency.
        """
        interpolated = interpolated.requires_grad_(True)

        logits = self.model(interpolated)
        scores = logits[:, class_idx].sum()

        grads = torch.autograd.grad(scores, interpolated)[0].clone()
        return grads.detach()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Integrated Gradients attribution.

        Args:
            input_tensor: (1, C, H, W) preprocessed image tensor
            class_idx: Target class. If None, uses the argmax prediction.

        Returns:
            ig_attrs: (C, H, W) numpy array — raw channel-wise attribution
            ig_map:   (H, W) numpy array in [0, 1] — collapsed & normalized saliency map
        """
        with torch.no_grad():
            logits = self.model(input_tensor)
            if class_idx is None:
                class_idx = logits.argmax(dim=1).item()

        baseline = self._get_baseline(input_tensor)

        interpolated = self._interpolate_inputs(baseline, input_tensor)

        grads = self._compute_gradients(interpolated, class_idx)

        # Riemann sum (trapezoidal)
        avg_grads = (grads[:-1] + grads[1:]).mean(dim=0)

        delta = (input_tensor.squeeze(0) - baseline.squeeze(0)).cpu().detach()

        ig_attrs = avg_grads.cpu().detach() * delta

        # Collapse to saliency map
        ig_map = ig_attrs.norm(dim=0).numpy()
        ig_map = self._normalize(ig_map)

        return ig_attrs.numpy(), ig_map

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        min_val, max_val = arr.min(), arr.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    def convergence_delta(
        self,
        input_tensor: torch.Tensor,
        ig_attrs: np.ndarray,
        class_idx: Optional[int] = None,
    ) -> float:
        """
        Completeness check: sum of attributions should equal F(x) - F(baseline).
        A small delta means the approximation is accurate.
        """
        with torch.no_grad():
            baseline = self._get_baseline(input_tensor)
            f_input = self.model(input_tensor)[0, class_idx or 0].item()
            f_baseline = self.model(baseline)[0, class_idx or 0].item()

        sum_attrs = ig_attrs.sum()
        delta = abs(sum_attrs - (f_input - f_baseline))
        return delta
