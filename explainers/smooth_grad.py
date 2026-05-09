"""
SmoothGrad: Removing Noise from Gradients
Paper: https://arxiv.org/abs/1706.03825  (Smilkov et al., 2017)

Vanilla gradients of the output w.r.t. the input are noisy because the gradient
function is highly non-linear in practice. SmoothGrad averages gradients computed
on noisy copies of the input to denoise the attribution map.

Formula:
  SmoothGrad(x) = (1/N) x Σ_k ∂F(x + ε_k) / ∂x
  where ε_k ~ N(0, σ²)

σ is typically set as a fraction of the input range (0.1-0.2 x (max - min)).

Variants included:
  - SmoothGrad (vanilla)
  - SmoothGrad-Squared (SG-SQ): emphasizes strongest signals
  - SmoothGrad-VAR (SG-VAR): highlights where gradients are consistent
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class SmoothGrad:
    """
    SmoothGrad noise-averaged gradient attribution.

    Args:
        model:       PyTorch model
        n_samples:   Number of noisy samples to average (50 typical)
        noise_level: Noise std as fraction of input value range (default 0.15)
        variant:     'standard' | 'squared' | 'var'
    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_samples: int = 50,
        noise_level: float = 0.15,
        variant: str = "standard",
    ):
        self.model = model
        self.n_samples = n_samples
        self.noise_level = noise_level
        self.variant = variant
        self.model.eval()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute SmoothGrad attribution map.

        Args:
            input_tensor: (1, C, H, W) preprocessed image tensor
            class_idx: Target class. If None, uses the argmax prediction.

        Returns:
            smooth_attrs: (C, H, W) numpy array — channel-wise attributions
            smooth_map:   (H, W) numpy array in [0, 1] — collapsed saliency map
        """
        with torch.no_grad():
            logits = self.model(input_tensor)
            if class_idx is None:
                class_idx = logits.argmax(dim=1).item()

        # Noise standard deviation proportional to input value range
        val_range = input_tensor.max().item() - input_tensor.min().item()
        sigma = self.noise_level * val_range

        device = input_tensor.device
        all_grads = []

        for _ in range(self.n_samples):
            noise = torch.randn_like(input_tensor) * sigma
            noisy_input = (input_tensor.detach() + noise).requires_grad_(True)

            logits = self.model(noisy_input)
            score = logits[0, class_idx]

            self.model.zero_grad()
            score.backward(retain_graph=False)

            grad = noisy_input.grad.detach().cpu()
            all_grads.append(grad.squeeze(0))

        # Stack: (n_samples, C, H, W)
        grads_stack = torch.stack(all_grads, dim=0)

        if self.variant == "standard":
            # Average of gradients
            smooth_attrs = grads_stack.mean(dim=0)
        elif self.variant == "squared":
            # Average of squared gradients — amplifies confident attributions
            smooth_attrs = (grads_stack ** 2).mean(dim=0)
        elif self.variant == "var":
            # Variance of gradients — highlights regions where model is certain
            smooth_attrs = grads_stack.var(dim=0)
        else:
            smooth_attrs = grads_stack.mean(dim=0)

        smooth_attrs_np = smooth_attrs.numpy()

        # Collapse to single map: take absolute value, then mean across channels
        smooth_map = np.abs(smooth_attrs_np).mean(axis=0)
        smooth_map = self._normalize(smooth_map)

        return smooth_attrs_np, smooth_map

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        min_val, max_val = arr.min(), arr.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)


class VanillaGradients:
    """
    Vanilla gradient saliency map (baseline comparison).
    ∂F(x) / ∂x — how much does each pixel affect the prediction?
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        input_tensor = input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        grads = input_tensor.grad.detach().cpu().squeeze(0).numpy()
        saliency = np.abs(grads).mean(axis=0)
        saliency = self._normalize(saliency)

        return grads, saliency

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        min_val, max_val = arr.min(), arr.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
