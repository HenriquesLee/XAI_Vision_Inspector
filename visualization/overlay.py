"""
Visualization utilities for XAI heatmap overlays.

Provides:
  - Heatmap colorization with multiple colormaps
  - Alpha blending of heatmap over original image
  - Side-by-side comparison figures
  - Plotly-based interactive figures
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Optional, Tuple, Dict
import io


# ─── Colormaps ────────────────────────────────────────────────────────────────

COLORMAPS = {
    "jet":       cv2.COLORMAP_JET,
    "hot":       cv2.COLORMAP_HOT,
    "plasma":    cv2.COLORMAP_PLASMA,
    "inferno":   cv2.COLORMAP_INFERNO,
    "viridis":   cv2.COLORMAP_VIRIDIS,
    "turbo":     cv2.COLORMAP_TURBO,
    "cool":      cv2.COLORMAP_COOL,
    "spring":    cv2.COLORMAP_SPRING,
}


def normalize_map(saliency_map: np.ndarray, percentile_clip: float = 99.0) -> np.ndarray:
    """
    Normalize a saliency map to [0, 1], optionally clipping outliers.

    Args:
        saliency_map: 2D numpy array
        percentile_clip: Values above this percentile are clipped (reduces noise spikes)
    """
    if percentile_clip < 100.0:
        clip_val = np.percentile(saliency_map, percentile_clip)
        saliency_map = np.clip(saliency_map, None, clip_val)

    min_val, max_val = saliency_map.min(), saliency_map.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(saliency_map, dtype=np.float32)
    return ((saliency_map - min_val) / (max_val - min_val)).astype(np.float32)


def apply_colormap(
    saliency_map: np.ndarray,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Apply a colormap to a normalized [0, 1] saliency map.

    Returns:
        RGB image array (H, W, 3) uint8
    """
    saliency_uint8 = (saliency_map * 255).astype(np.uint8)
    cmap_code = COLORMAPS.get(colormap, cv2.COLORMAP_JET)
    colored = cv2.applyColorMap(saliency_uint8, cmap_code)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def overlay_heatmap(
    original_image: np.ndarray,
    saliency_map: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
    percentile_clip: float = 99.0,
) -> np.ndarray:
    """
    Blend a heatmap overlay onto the original image.

    Args:
        original_image: (H, W, 3) uint8 numpy array
        saliency_map:   (H, W) float saliency map
        alpha:          Heatmap opacity [0, 1]
        colormap:       Colormap name from COLORMAPS
        percentile_clip: Clip saliency values above this percentile

    Returns:
        Blended (H, W, 3) uint8 numpy array
    """
    H, W = original_image.shape[:2]

    # Resize map to match image
    if saliency_map.shape != (H, W):
        saliency_map = cv2.resize(saliency_map, (W, H), interpolation=cv2.INTER_LINEAR)

    norm_map = normalize_map(saliency_map, percentile_clip)
    colored = apply_colormap(norm_map, colormap)

    original_float = original_image.astype(np.float32)
    colored_float = colored.astype(np.float32)

    blended = (1 - alpha) * original_float + alpha * colored_float
    return np.clip(blended, 0, 255).astype(np.uint8)


def make_comparison_figure(
    original_image: np.ndarray,
    results: Dict[str, np.ndarray],
    colormap: str = "jet",
    alpha: float = 0.5,
    figsize_per_col: Tuple[float, float] = (4.0, 4.5),
) -> plt.Figure:
    """
    Create a matplotlib figure comparing multiple XAI methods side by side.

    Args:
        original_image: (H, W, 3) uint8 image
        results:        {method_name: saliency_map (H, W)}
        colormap:       Colormap for all heatmaps
        alpha:          Overlay opacity

    Returns:
        matplotlib Figure
    """
    n_methods = len(results)
    n_cols = n_methods + 1  # +1 for original
    fig_w = figsize_per_col[0] * n_cols
    fig_h = figsize_per_col[1]

    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#0e1117")

    titles = ["Original"] + list(results.keys())
    images = [original_image] + [
        overlay_heatmap(original_image, m, alpha=alpha, colormap=colormap)
        for m in results.values()
    ]

    for ax, title, img in zip(axes, titles, images):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=6)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout(pad=0.5)
    return fig


def fig_to_pil(fig: plt.Figure) -> Image.Image:
    """Convert a matplotlib Figure to a PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf).copy()


def make_plotly_heatmap(
    original_image: np.ndarray,
    saliency_map: np.ndarray,
    method_name: str,
    colormap: str = "Hot",
) -> go.Figure:
    """
    Create an interactive Plotly figure with zoomable heatmap overlay.
    """
    H, W = original_image.shape[:2]
    if saliency_map.shape != (H, W):
        saliency_map = cv2.resize(saliency_map, (W, H), interpolation=cv2.INTER_LINEAR)

    norm_map = normalize_map(saliency_map)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Original", f"{method_name} Overlay"),
        horizontal_spacing=0.05,
    )

    fig.add_trace(
        go.Image(z=original_image, name="Original"),
        row=1, col=1,
    )

    overlay = overlay_heatmap(original_image, saliency_map, alpha=0.55, colormap="jet")
    fig.add_trace(
        go.Image(z=overlay, name=method_name),
        row=1, col=2,
    )

    fig.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=380,
    )

    for ann in fig.layout.annotations:
        ann.font.color = "white"
        ann.font.size = 13

    return fig


def plot_top_predictions(
    probs: np.ndarray,
    labels: List[str],
    top_k: int = 5,
) -> go.Figure:
    """
    Create a horizontal bar chart for top-k class predictions.
    """
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_probs = probs[top_indices]
    top_labels = [labels[i] for i in top_indices]

    # Truncate long labels
    top_labels = [lbl[:35] + "…" if len(lbl) > 35 else lbl for lbl in top_labels]

    colors = [
        "#ff4b4b" if i == 0 else "#4b8bff"
        for i in range(len(top_probs))
    ]

    fig = go.Figure(go.Bar(
        x=top_probs[::-1],
        y=top_labels[::-1],
        orientation="h",
        marker_color=colors[::-1],
        text=[f"{p*100:.1f}%" for p in top_probs[::-1]],
        textposition="outside",
        textfont=dict(color="white", size=12),
    ))

    fig.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161c27",
        font=dict(color="#c8d0e0", size=12),
        xaxis=dict(
            range=[0, min(1.0, top_probs.max() * 1.3)],
            title="Confidence",
            gridcolor="#2a2f3e",
            tickformat=".0%",
        ),
        yaxis=dict(gridcolor="#2a2f3e"),
        margin=dict(l=10, r=60, t=10, b=30),
        height=220,
    )
    return fig


def compute_attribution_stats(saliency_map: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics for a saliency map."""
    flat = saliency_map.flatten()
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "max": float(flat.max()),
        "sparsity": float((flat < 0.05).mean()),  # fraction of near-zero pixels
        "top10_mean": float(np.partition(flat, -max(1, len(flat)//10))[-len(flat)//10:].mean()),
    }
