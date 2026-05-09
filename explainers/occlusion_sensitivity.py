"""
Occlusion Sensitivity Analysis
Original: Zeiler & Fergus, 2014 (https://arxiv.org/abs/1311.2901)

A perturbation-based XAI method that systematically occludes (masks) patches
of the input image with a neutral value and measures how much the prediction
confidence drops — revealing which regions the model relies on.

Unlike gradient-based methods, this is model-agnostic (works on any classifier)
at the cost of being computationally expensive (O(HxW / stride²) forward passes).
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings


class OcclusionSensitivity:
    """
    Occlusion Sensitivity attribution map.

    For each spatial position (i, j), replaces a (patch_size x patch_size) window
    centered at that position with `occlusion_value`, runs the model forward,
    and records the drop in the target class probability.

    Attributes:
        model:           PyTorch model
        patch_size:      Size of the occlusion square patch (pixels)
        stride:          Step size for sliding the patch (smaller = higher res map)
        occlusion_value: Value to fill the patch with (default 0 = black)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        patch_size: int = 32,
        stride: int = 8,
        occlusion_value: float = 0.0,
    ):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
        self.occlusion_value = occlusion_value
        self.model.eval()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, int]:
        """
        Compute the occlusion sensitivity map.

        Args:
            input_tensor: (1, C, H, W) preprocessed image tensor
            class_idx:    Target class. If None, uses the argmax prediction.
            batch_size:   Number of occluded images to process at once.

        Returns:
            sensitivity_map: (H, W) numpy array in [0, 1]
            class_idx:       Resolved target class index
        """
        _, C, H, W = input_tensor.shape
        device = input_tensor.device

        with torch.no_grad():
            baseline_logits = self.model(input_tensor)
            if class_idx is None:
                class_idx = baseline_logits.argmax(dim=1).item()
            baseline_prob = F.softmax(baseline_logits, dim=1)[0, class_idx].item()

        # Accumulate sensitivity scores in a (H, W) map and count map
        sensitivity_map = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)

        # Generate all (top, left) patch positions
        positions = []
        y = 0
        while y < H:
            x = 0
            while x < W:
                positions.append((y, x))
                x += self.stride
            y += self.stride

        # Process positions in batches for efficiency
        for batch_start in range(0, len(positions), batch_size):
            batch_positions = positions[batch_start: batch_start + batch_size]

            batch_tensors = []
            for (top, left) in batch_positions:
                occluded = input_tensor.clone()
                bottom = min(top + self.patch_size, H)
                right = min(left + self.patch_size, W)
                occluded[0, :, top:bottom, left:right] = self.occlusion_value
                batch_tensors.append(occluded)

            batch = torch.cat(batch_tensors, dim=0).to(device)

            with torch.no_grad():
                logits = self.model(batch)
                probs = F.softmax(logits, dim=1)[:, class_idx].cpu().numpy()

            # The sensitivity at this patch = how much the probability dropped
            for i, (top, left) in enumerate(batch_positions):
                bottom = min(top + self.patch_size, H)
                right = min(left + self.patch_size, W)
                drop = baseline_prob - probs[i]  # positive = important region
                sensitivity_map[top:bottom, left:right] += drop
                count_map[top:bottom, left:right] += 1.0

        # Average overlapping regions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sensitivity_map = np.where(
                count_map > 0,
                sensitivity_map / count_map,
                0.0,
            )

        sensitivity_map = self._normalize(sensitivity_map)
        return sensitivity_map, class_idx

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        min_val, max_val = arr.min(), arr.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    def compute_mean_sensitivity(
        self,
        sensitivity_map: np.ndarray,
        top_k_percent: float = 0.1,
    ) -> float:
        """Return the mean sensitivity of the top-k% most important pixels."""
        flat = sensitivity_map.flatten()
        k = max(1, int(len(flat) * top_k_percent))
        return float(np.partition(flat, -k)[-k:].mean())
