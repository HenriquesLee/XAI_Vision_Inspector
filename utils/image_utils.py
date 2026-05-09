"""
Image preprocessing utilities for XAI Vision Inspector.
Handles loading, preprocessing for model inference, and denormalization for display.
"""

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from typing import Tuple, Optional
import io
import urllib.request


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Standard input size for most torchvision models
DEFAULT_SIZE = 224


def get_transform(input_size: int = DEFAULT_SIZE) -> T.Compose:
    """Standard ImageNet preprocessing pipeline."""
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_image(
    pil_image: Image.Image,
    input_size: int = DEFAULT_SIZE,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocess a PIL Image for model inference.

    Args:
        pil_image: Input PIL image (any mode)
        input_size: Target spatial size
        device: Target device

    Returns:
        tensor:    (1, 3, H, W) normalized float32 tensor, on device
        display:   (H, W, 3) uint8 numpy array for visualization (no normalization)
    """
    # Ensure RGB
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Resize for display
    display_img = pil_image.resize((input_size, input_size), Image.LANCZOS)
    display_np = np.array(display_img, dtype=np.uint8)

    # Preprocess for model
    transform = get_transform(input_size)
    tensor = transform(pil_image).unsqueeze(0)  

    if device:
        tensor = tensor.to(device)

    return tensor, display_np


def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalization and convert tensor to displayable numpy array.

    Args:
        tensor: (1, 3, H, W) or (3, H, W) normalized tensor

    Returns:
        (H, W, 3) uint8 numpy array
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    img = tensor.cpu() * std + mean
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def load_image_from_bytes(data: bytes) -> Image.Image:
    """Load a PIL Image from raw bytes (e.g. from Streamlit uploader)."""
    return Image.open(io.BytesIO(data)).convert("RGB")


def load_image_from_url(url: str, timeout: int = 10) -> Image.Image:
    """Load a PIL Image from a URL."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = response.read()
    return load_image_from_bytes(data)


def pil_to_numpy(pil_image: Image.Image, size: int = DEFAULT_SIZE) -> np.ndarray:
    """Resize and convert PIL image to (H, W, 3) uint8 numpy array."""
    img = pil_image.resize((size, size), Image.LANCZOS).convert("RGB")
    return np.array(img, dtype=np.uint8)


# ─── Sample images ────────────────────────────────────────────────────────────

SAMPLE_IMAGES = {
    "Golden Retriever": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Golden_Retriever_Smiling.jpg/640px-Golden_Retriever_Smiling.jpg",
    "Tabby Cat": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/640px-Cat_November_2010-1a.jpg",
    "Red Fox": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/03/Red_Fox_%28Vulpes_vulpes%29_-_British_Wildlife_Centre-3.jpg/640px-Red_Fox_%28Vulpes_vulpes%29_-_British_Wildlife_Centre-3.jpg",
    "African Elephant": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/African_Bush_Elephant.jpg/640px-African_Bush_Elephant.jpg",
}
