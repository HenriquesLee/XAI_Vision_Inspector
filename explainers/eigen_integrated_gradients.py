"""
Eigen-Integrated Gradients (Eigen-IG) v3
==========================================
Options 1 + 2 implemented:

OPTION 2 — Per-channel trajectory SVD
---------------------------------------
Previous versions mixed all 3 RGB channels into one matrix (n_steps, C*H*W).
SVD then finds cross-channel correlations rather than pure spatial patterns.

Fix: decompose gradient trajectory independently per channel:
    G_c in R^(n_steps x H*W)   for c = 0, 1, 2

n_components is now meaningful per-channel.

OPTION 1 — Eigenvalue-weighted integration
-------------------------------------------
Standard IG uses uniform average of all gradient steps. Near alpha=0
(close to zero baseline) gradients are noisy — model barely activates.
Near alpha=1 (close to real image) gradients carry the most signal.

Fix: compute per-step alignment score with dominant spatial structure
(top-k singular vectors of the channel trajectory), use as integration weights.

    weight_t = || G_t . Vh_k^T ||   (how much step t aligns with dominant patterns)
    weights  = softmax(weight_t / temperature)
    avg_grad = sum(weights_t * G_t)

Steps consistent with the overall signal get upweighted.
Noisy incoherent steps near alpha=0 get downweighted automatically.

OPTION 3 — Gaussian blur for spatial coherence
-----------------------------------------------
After the weighted average, a small Gaussian blur is applied to eigen_attrs
before the ReLU and normalization step. This merges nearby high-attribution
pixels into coherent regions, reducing salt-and-pepper noise in the saliency map.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional
from typing import Optional, Tuple


class EigenIntegratedGradients:
    """
    Eigen-IG v3: per-channel trajectory SVD + eigenvalue-weighted integration
    + Gaussian blur for spatial coherence before saliency map generation.

    Args:
        model:         PyTorch model (eval mode)
        n_steps:       IG interpolation steps (50-150)
        baseline_type: 'zeros' | 'uniform_noise' | 'gaussian_noise'
        n_components:  Singular vectors used for step-weighting (1-20)
        weight_temp:   Softmax temperature for step weights.
                       Lower = more aggressive focus on best steps.
                       Higher = closer to uniform IG. Default 1.0.
        blur_kernel:   Kernel size for post-attribution Gaussian blur (odd int).
                       Set to 0 or 1 to disable. Default 11.
        blur_sigma:    Sigma for Gaussian blur. Default 2.0.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_steps: int = 100,
        baseline_type: str = "zeros",
        n_components: int = 10,
        weight_temp: float = 1.0,
        blur_kernel: int = 11,
        blur_sigma: float = 2.0,
    ):
        self.model = model
        self.n_steps = n_steps
        self.baseline_type = baseline_type
        self.n_components = n_components
        self.weight_temp = weight_temp
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.model.eval()

    def _get_baseline(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.baseline_type == "zeros":
            return torch.zeros_like(input_tensor)
        elif self.baseline_type == "uniform_noise":
            return torch.rand_like(input_tensor)
        elif self.baseline_type == "gaussian_noise":
            return torch.randn_like(input_tensor) * 0.1
        return torch.zeros_like(input_tensor)

    def _weighted_average(self, grads_c: torch.Tensor) -> torch.Tensor:
        """
        Options 1+2 combined on a single channel's trajectory.

        Args:
            grads_c: (n_steps, H*W) gradient trajectory for one channel

        Returns:
            (H*W,) weighted average gradient
        """
        n_steps, HW = grads_c.shape

        U, S, Vh = torch.linalg.svd(grads_c, full_matrices=False)

        k = min(self.n_components, len(S))

        proj        = grads_c @ Vh[:k].T
        step_scores = proj.norm(dim=1)

        # Softmax normalisation → weights sum to 1
        weights = F.softmax(step_scores / self.weight_temp, dim=0)

        # Weighted sum (replaces trapezoidal mean)
        return (weights.unsqueeze(1) * grads_c).sum(dim=0)

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            logits = self.model(input_tensor)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())

        _, C, H, W = input_tensor.shape
        baseline = self._get_baseline(input_tensor)

        # Collect full gradient trajectory
        alphas = torch.linspace(0.0, 1.0, self.n_steps, device=input_tensor.device)
        interp = (baseline + alphas.view(-1,1,1,1) * (input_tensor - baseline)).requires_grad_(True)
        grads  = torch.autograd.grad(
            self.model(interp)[:, class_idx].sum(), interp
        )[0].detach().clone()

        # Per-channel weighted average
        avg_grads = torch.zeros(C, H, W)
        for c in range(C):
            grads_c      = grads[:, c].reshape(self.n_steps, H * W).float()
            avg_grads[c] = self._weighted_average(grads_c).reshape(H, W)

        # Scale by (input - baseline) delta
        delta       = (input_tensor.squeeze(0) - baseline.squeeze(0)).detach().cpu()
        eigen_attrs = avg_grads.cpu() * delta
        if self.blur_kernel > 1:
            eigen_attrs = torchvision.transforms.functional.gaussian_blur(
                eigen_attrs,
                kernel_size=[self.blur_kernel, self.blur_kernel],
                sigma=[self.blur_sigma, self.blur_sigma],
            )

        # ReLU + L2 norm collapse → saliency map
        eigen_map = F.relu(eigen_attrs).norm(dim=0).numpy()
        eigen_map = self._normalize(eigen_map)

        return eigen_attrs.numpy(), eigen_map

    def convergence_delta(
        self,
        input_tensor: torch.Tensor,
        eigen_attrs: np.ndarray,
        class_idx: Optional[int] = None,
    ) -> float:
        with torch.no_grad():
            baseline = self._get_baseline(input_tensor)
            if class_idx is None:
                class_idx = int(self.model(input_tensor).argmax(dim=1).item())
            f_x    = self.model(input_tensor)[0, class_idx].item()
            f_base = self.model(baseline)[0, class_idx].item()
        return abs(float(eigen_attrs.sum()) - (f_x - f_base))

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-8:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)