"""
Model Loader — supports multiple pretrained torchvision architectures
with helpers for extracting named convolutional layers for Grad-CAM targeting.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ModelConfig:
    """Configuration for a pretrained vision model."""
    name: str
    constructor: callable
    weights_class: object
    default_target_layer: str  # Attribute path string for the default Grad-CAM layer
    cam_layers: List[str]      # All available layers for Grad-CAM
    description: str


# Registry of supported models
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "ResNet-50": ModelConfig(
        name="ResNet-50",
        constructor=models.resnet50,
        weights_class=models.ResNet50_Weights.IMAGENET1K_V2,
        default_target_layer="layer4.2.conv3",
        cam_layers=["layer1", "layer2", "layer3", "layer4", "layer4.2.conv3"],
        description="50-layer residual network. Great balance of speed and accuracy.",
    ),
    "ResNet-101": ModelConfig(
        name="ResNet-101",
        constructor=models.resnet101,
        weights_class=models.ResNet101_Weights.IMAGENET1K_V2,
        default_target_layer="layer4.2.conv3",
        cam_layers=["layer1", "layer2", "layer3", "layer4", "layer4.2.conv3"],
        description="101-layer residual network. Higher accuracy, slower inference.",
    ),
    "VGG-16": ModelConfig(
        name="VGG-16",
        constructor=models.vgg16,
        weights_class=models.VGG16_Weights.IMAGENET1K_V1,
        default_target_layer="features.28",
        cam_layers=["features.14", "features.21", "features.28"],
        description="Classic VGG architecture. Very clear gradient signals.",
    ),
    "EfficientNet-B0": ModelConfig(
        name="EfficientNet-B0",
        constructor=models.efficientnet_b0,
        weights_class=models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        default_target_layer="features.8.0",
        cam_layers=["features.3", "features.5", "features.7", "features.8.0"],
        description="Efficient compound-scaled architecture. Small and fast.",
    ),
    "DenseNet-121": ModelConfig(
        name="DenseNet-121",
        constructor=models.densenet121,
        weights_class=models.DenseNet121_Weights.IMAGENET1K_V1,
        default_target_layer="features.denseblock4.denselayer16.conv2",
        cam_layers=["features.denseblock2", "features.denseblock3", "features.denseblock4"],
        description="Dense connections between all layers. Strong feature reuse.",
    ),
    "MobileNet-V3": ModelConfig(
        name="MobileNet-V3",
        constructor=models.mobilenet_v3_large,
        weights_class=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        default_target_layer="features.16.block.2.0",
        cam_layers=["features.7", "features.12", "features.16.block.2.0"],
        description="Lightweight mobile-optimized network.",
    ),
}


def load_model(model_name: str, device: Optional[str] = None) -> Tuple[nn.Module, ModelConfig]:
    """
    Load a pretrained model by name from the registry.

    Args:
        model_name: Key in MODEL_REGISTRY
        device: 'cpu', 'cuda', or None (auto-detect)

    Returns:
        (model, config) tuple
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")

    config = MODEL_REGISTRY[model_name]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = config.constructor(weights=config.weights_class)
    model = model.to(device)
    model.eval()
    # Disable inplace ops (VGG/EfficientNet/MobileNet use inplace ReLU
    # which causes autograd view errors with backward hooks)
    for module in model.modules():
        if hasattr(module, 'inplace'):
            module.inplace = False

    return model, config


def get_layer_by_name(model: nn.Module, layer_name: str) -> nn.Module:
    """
    Navigate the model hierarchy using a dot-separated attribute path.
    E.g., 'layer4.2.conv3' → model.layer4[2].conv3

    Handles both attribute access and integer indexing for Sequential/ModuleList.
    """
    parts = layer_name.split(".")
    current = model
    for part in parts:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def list_conv_layers(model: nn.Module, prefix: str = "") -> List[Tuple[str, nn.Module]]:
    """
    Recursively list all Conv2d layers in the model with their full attribute paths.
    Useful for letting users explore all possible Grad-CAM target layers.
    """
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))
    return conv_layers


def get_model_summary(model: nn.Module) -> Dict:
    """Return a summary dict with parameter counts and layer info."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    conv_count = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    bn_count = sum(1 for m in model.modules() if isinstance(m, (nn.BatchNorm2d,)))

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "conv_layers": conv_count,
        "batch_norm_layers": bn_count,
    }
