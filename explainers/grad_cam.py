"""
Grad-CAM: Gradient-weighted Class Activation Mapping
Paper: https://arxiv.org/abs/1610.02391

Implemented from scratch using PyTorch tensor-level hooks.
Uses tensor hooks instead of module backward hooks to avoid
inplace operation conflicts in VGG, EfficientNet, MobileNet.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional


class GradCAM:
    """
    Grad-CAM implementation using forward hook + tensor-level gradient hook.

    The algorithm:
      1. Forward pass → capture feature maps at target conv layer
      2. Backward pass from class score → capture gradients via tensor hook
      3. Global average pool the gradients → per-channel importance weights
      4. Weighted combination of feature maps → class activation map
      5. ReLU (we only care about features with positive influence)
      6. Bilinear upsample to input size
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.model.eval()

        self._feature_maps: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Forward hook only — no backward hook
        self._forward_hook = target_layer.register_forward_hook(self._save_feature_maps)

    def _save_feature_maps(self, module, input, output):
        # No .detach() — tensor must stay in graph so register_hook fires
        self._feature_maps = output

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        self._gradients = None

        def save_grad(grad):
            self._gradients = grad.detach().clone()

        input_tensor = input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Tensor-level hook on the feature map
        hook_handle = self._feature_maps.register_hook(save_grad)

        self.model.zero_grad()
        logits[0, class_idx].backward()

        hook_handle.remove()

        feature_maps = self._feature_maps.detach()
        gradients    = self._gradients

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * feature_maps).sum(dim=1, keepdim=True))

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam  = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam  = cam.squeeze().cpu().numpy()
        return self._normalize(cam)

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        min_val, max_val = arr.min(), arr.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    def remove_hooks(self):
        self._forward_hook.remove()


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ — improved version with better localization for multiple instances.
    Paper: https://arxiv.org/abs/1710.11063

    Uses second-order gradient approximation for per-pixel importance weights.
    Inherits tensor-level hook approach from GradCAM for VGG compatibility.
    """

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        self._gradients = None

        def save_grad(grad):
            self._gradients = grad.detach().clone()

        input_tensor = input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        hook_handle = self._feature_maps.register_hook(save_grad)

        self.model.zero_grad()
        logits[0, class_idx].backward()

        hook_handle.remove()

        feature_maps = self._feature_maps.detach()
        gradients    = self._gradients

        # Grad-CAM++ second-order weight computation
        grads_power_2   = gradients ** 2
        grads_power_3   = gradients ** 3
        sum_activations = feature_maps.sum(dim=(2, 3), keepdim=True)

        denom  = 2 * grads_power_2 + sum_activations * grads_power_3
        denom  = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha  = grads_power_2 / denom

        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * feature_maps).sum(dim=1, keepdim=True))

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam  = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam  = cam.squeeze().cpu().numpy()
        return self._normalize(cam)
    