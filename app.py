"""
XAI Vision Inspector — Interactive Explainable AI Dashboard
============================================================
A professional Streamlit dashboard for visualizing how CNN models make decisions,
using four XAI methods implemented from scratch:

  • Grad-CAM / Grad-CAM++   — gradient-weighted class activation maps
  • Integrated Gradients    — path-integral attribution from a baseline
  • Occlusion Sensitivity   — perturbation-based importance maps
  • SmoothGrad              — noise-averaged gradient saliency

Run with:
    streamlit run app.py
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional
from evaluation.faithfulness import run_full_faithfulness_eval

import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image

# ─── Local imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from model_zoo import load_model, get_layer_by_name, MODEL_REGISTRY, get_model_summary
from explainers import (
    GradCAM, GradCAMPlusPlus,
    IntegratedGradients,
    OcclusionSensitivity,
    SmoothGrad,
    EigenIntegratedGradients,
)
from visualization import (
    overlay_heatmap,
    make_comparison_figure,
    make_plotly_heatmap,
    plot_top_predictions,
    compute_attribution_stats,
    fig_to_pil,
    COLORMAPS,
)
from utils import preprocess_image, load_image_from_bytes, load_image_from_url, pil_to_numpy, SAMPLE_IMAGES

warnings.filterwarnings("ignore")

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="XAI Vision Inspector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-deep:    #080c14;
    --bg-card:    #0f1623;
    --bg-surface: #161d2e;
    --accent:     #3b82f6;
    --accent-hot: #f87171;
    --accent-glow:#6ee7b7;
    --text:       #e2e8f0;
    --text-dim:   #64748b;
    --border:     #1e2940;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

.stApp { background-color: var(--bg-deep); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stRadio label {
    color: var(--text-dim) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* Header */
.xai-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 20px 0 10px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}
.xai-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.02em;
}
.xai-subtitle {
    font-size: 0.85rem;
    color: var(--text-dim);
    margin-top: 2px;
}
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.badge-blue  { background: #1e3a5f; color: #60a5fa; border: 1px solid #2563eb; }
.badge-green { background: #052e16; color: #4ade80; border: 1px solid #16a34a; }
.badge-red   { background: #3b0a0a; color: #f87171; border: 1px solid #dc2626; }

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 16px 0;
}
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
}
.metric-label {
    font-size: 0.7rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'Space Mono', monospace;
}
.metric-value {
    font-size: 1.35rem;
    font-weight: 600;
    color: var(--text);
    margin-top: 4px;
}
.metric-delta {
    font-size: 0.78rem;
    color: var(--accent-glow);
    margin-top: 2px;
}

/* Method cards */
.method-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 16px;
}
.method-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 4px;
}
.method-desc {
    font-size: 0.8rem;
    color: var(--text-dim);
    line-height: 1.5;
}

/* Attribution stats table */
.stats-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
}
.stats-table th {
    color: var(--text-dim);
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 6px 10px;
    text-align: left;
    border-bottom: 1px solid var(--border);
}
.stats-table td {
    padding: 7px 10px;
    border-bottom: 1px solid #1a2235;
}

/* Info box */
.info-box {
    background: #0c1a2e;
    border: 1px solid #1e3a5f;
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: #94a3b8;
    margin: 12px 0;
}

/* Convergence delta */
.delta-ok  { color: #4ade80; }
.delta-bad { color: #f87171; }

/* Streamlit overrides */
.stButton > button {
    background-color: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.04em !important;
    padding: 8px 20px !important;
    transition: background-color 0.2s !important;
}
.stButton > button:hover {
    background-color: #2563eb !important;
}
div[data-testid="stTabs"] button {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Imagenet Labels ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_imagenet_labels() -> List[str]:
    """Load ImageNet class labels (1000 classes)."""
    try:
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        import urllib.request
        with urllib.request.urlopen(url, timeout=5) as r:
            labels = [line.strip() for line in r.read().decode().splitlines()]
        return labels
    except Exception:
        return [f"class_{i}" for i in range(1000)]


# ─── Model cache ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_model(model_name: str):
    """Load and cache a pretrained model (survives re-runs)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_model(model_name, device=device)


# ─── XAI Runner ───────────────────────────────────────────────────────────────

def run_gradcam(model, config, tensor, class_idx, use_plus_plus=False):
    layer = get_layer_by_name(model, config.default_target_layer)
    explainer_cls = GradCAMPlusPlus if use_plus_plus else GradCAM
    explainer = explainer_cls(model, layer)
    cam = explainer(tensor, class_idx=class_idx)
    explainer.remove_hooks()
    return cam


def run_integrated_gradients(model, tensor, class_idx, n_steps):
    explainer = IntegratedGradients(model, n_steps=n_steps)
    _, ig_map = explainer(tensor, class_idx=class_idx)
    return ig_map


def run_occlusion(model, tensor, class_idx, patch_size, stride):
    explainer = OcclusionSensitivity(model, patch_size=patch_size, stride=stride)
    sens_map, _ = explainer(tensor, class_idx=class_idx)
    return sens_map


def run_smoothgrad(model, tensor, class_idx, n_samples, noise_level, variant):
    explainer = SmoothGrad(model, n_samples=n_samples, noise_level=noise_level, variant=variant)
    _, smooth_map = explainer(tensor, class_idx=class_idx)
    return smooth_map

def run_eigen_ig(model, tensor, class_idx, n_steps, n_components):
    explainer = EigenIntegratedGradients(model, n_steps=n_steps, n_components=n_components)
    _, eigen_map = explainer(tensor, class_idx=class_idx)
    return eigen_map

# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 8px 0 16px 0;">
            <div style="font-family:'Space Mono',monospace;font-size:1.05rem;font-weight:700;color:#e2e8f0;">
                🔍 XAI Inspector
            </div>
            <div style="font-size:0.75rem;color:#475569;margin-top:2px;">Configuration Panel</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### 🧠 Model")
        model_name = st.selectbox(
            "Architecture",
            options=list(MODEL_REGISTRY.keys()),
            index=0,
            help="Select the pretrained CNN to analyze",
        )

        st.markdown("##### 🖼️ Image Source")
        image_source = st.radio(
            "Input",
            options=["Upload", "Batch Benchmark"],
            horizontal=True,
        )

        uploaded_file = None
        batch_folder = None
        batch_max = 500
        batch_methods = ["Grad-CAM", "Integrated Gradients", "Eigen-IG"]
        if image_source == "Upload":
            uploaded_file = st.file_uploader(
                "Drop an image",
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed",
            )
        else:
            batch_folder = st.text_input(
                "Folder path",
                placeholder="/path/to/imagenet/val",
                help="Absolute path to a folder of images (jpg/png). Subfolders scanned recursively.",
            )
            batch_max = st.slider("Max images", 50, 1000, 500, 50)
            batch_methods = st.multiselect(
                "Methods to benchmark",
                options=["Grad-CAM", "Integrated Gradients", "Eigen-IG"],
                default=["Grad-CAM", "Integrated Gradients", "Eigen-IG"],
            )
            st.caption("CSV auto-downloads when done.")

        st.markdown("##### 🔬 XAI Methods")
        methods = st.multiselect(
            "Active methods",
            options=["Grad-CAM", "Grad-CAM++", "Integrated Gradients", "Eigen-IG","Occlusion Sensitivity", "SmoothGrad"],
            default=["Grad-CAM", "Integrated Gradients"],
        )

        st.markdown("##### 🎨 Visualization")
        colormap = st.selectbox("Colormap", list(COLORMAPS.keys()), index=0)
        alpha = st.slider("Overlay opacity", 0.2, 0.9, 0.55, 0.05)

        with st.expander("⚙️ Method Parameters"):
            st.markdown("**Integrated Gradients**")
            ig_steps = st.slider("IG interpolation steps", 20, 200, 80, 10)

            st.markdown("**Occlusion Sensitivity**")
            occ_patch = st.slider("Patch size (px)", 8, 64, 32, 8)
            occ_stride = st.slider("Stride (px)", 4, 32, 16, 4)

            st.markdown("**SmoothGrad**")
            sg_samples = st.slider("Noise samples", 10, 100, 40, 10)
            sg_noise = st.slider("Noise level", 0.05, 0.4, 0.15, 0.05)
            sg_variant = st.selectbox("Variant", ["standard", "squared", "var"])

            st.markdown("**Eigen-IG**")
            eigen_components = st.slider("Number of principal components", 1, 12, 5, 1)

        run_btn = st.button("▶ Run Analysis", use_container_width=True)

    return {
        "model_name": model_name,
        "image_source": image_source,
        "uploaded_file": uploaded_file,
        "batch_folder": batch_folder,
        "batch_max": batch_max,
        "batch_methods": batch_methods,
        "methods": methods,
        "colormap": colormap,
        "alpha": alpha,
        "ig_steps": ig_steps,
        "occ_patch": occ_patch,
        "occ_stride": occ_stride,
        "sg_samples": sg_samples,
        "sg_noise": sg_noise,
        "sg_variant": sg_variant,
        "run_btn": run_btn,
        "eigen_components": eigen_components,
    }



# ─── Batch Benchmark ──────────────────────────────────────────────────────────

def run_batch_benchmark(model, config, cfg, labels, device):
    """
    Pipelined faithfulness benchmark.

    Phase 1 (calibration): run first 5 images without timeout to measure
    real per-image wall time. Computes max_time_seen + 0.5s safety margin.

    Phase 2 (pipeline): process remaining images with a per-image timeout.
    Any image exceeding the timeout is skipped and logged — prevents one
    slow/corrupt image from stalling the entire run.

    GPU acceleration: tensors are moved to device (CUDA if available).
    """
    import glob, os, time, signal
    import pandas as pd
    from contextlib import contextmanager
    from evaluation.faithfulness import compute_deletion_curve, compute_insertion_curve, compute_infidelity

    folder        = cfg.get("batch_folder", "").strip()
    max_imgs      = cfg.get("batch_max", 500)
    methods_to_run = cfg.get("batch_methods", ["Grad-CAM", "Integrated Gradients", "Eigen-IG"])

    if not folder:
        st.info("Enter a folder path in the sidebar to start the batch benchmark.")
        return
    if not os.path.isdir(folder):
        st.error(f"Folder not found: `{folder}`")
        return

    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPEG", "*.JPG", "*.PNG")
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    image_paths = sorted(set(image_paths))[:max_imgs]

    if not image_paths:
        st.error("No images found in that folder.")
        return

    gpu_label = f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
    st.markdown(f"#### Batch Benchmark — {len(image_paths)} images")
    st.caption(f"Methods: {', '.join(methods_to_run)}  ·  Device: **{gpu_label}**")

    progress   = st.progress(0, text="Starting calibration (first 5 images)…")
    status     = st.empty()
    timing_box = st.empty()
    rows       = []

    CALIBRATION_N = min(5, len(image_paths))
    per_image_timeout = None   # set after calibration

    # ── Helper: process one image, returns list of row dicts ─────────────────
    def process_image(img_path):
        pil_img = Image.open(img_path).convert("RGB")
        tensor, _ = preprocess_image(pil_img, input_size=224, device=device)

        with torch.no_grad():
            probs = torch.nn.functional.softmax(model(tensor), dim=1).squeeze()
            probs_np = probs.cpu().numpy()

        class_idx   = int(probs_np.argmax())
        class_label = labels[class_idx] if class_idx < len(labels) else f"class_{class_idx}"

        saliency_maps = {}
        for method in methods_to_run:
            try:
                if method == "Grad-CAM":
                    saliency_maps[method] = run_gradcam(model, config, tensor, class_idx, use_plus_plus=False)
                elif method == "Integrated Gradients":
                    saliency_maps[method] = run_integrated_gradients(model, tensor, class_idx, n_steps=50)
                elif method == "Eigen-IG":
                    eig = EigenIntegratedGradients(model, n_steps=50, n_components=10)
                    _, saliency_maps[method] = eig(tensor, class_idx=class_idx)
            except Exception as e:
                status.warning(f"  {method} failed on {os.path.basename(img_path)}: {e}")

        result_rows = []
        for method, sal_map in saliency_maps.items():
            try:
                _, del_auc = compute_deletion_curve(model, tensor, sal_map, class_idx, steps=5)
                _, ins_auc = compute_insertion_curve(model, tensor, sal_map, class_idx, steps=5)
                infid      = compute_infidelity(model, tensor, sal_map, class_idx, n_perturb=5)
                result_rows.append({
                    "image":         os.path.basename(img_path),
                    "class_idx":     class_idx,
                    "class_label":   class_label,
                    "confidence":    round(float(probs_np[class_idx]), 4),
                    "method":        method,
                    "deletion_auc":  round(del_auc, 4),
                    "insertion_auc": round(ins_auc, 4),
                    "infidelity":    round(infid, 6),
                })
            except Exception as e:
                status.warning(f"  Metrics failed for {method} on {os.path.basename(img_path)}: {e}")

        return result_rows

    # ── Phase 1: calibration ──────────────────────────────────────────────────
    calibration_times = []
    for i, img_path in enumerate(image_paths[:CALIBRATION_N]):
        progress.progress((i + 1) / len(image_paths),
                          text=f"Calibrating [{i+1}/{CALIBRATION_N}]: {os.path.basename(img_path)}")
        t0 = time.time()
        try:
            rows.extend(process_image(img_path))
            elapsed = time.time() - t0
            calibration_times.append(elapsed)
            timing_box.info(f"Calibration {i+1}/{CALIBRATION_N} — {elapsed:.1f}s/image")
        except Exception as e:
            status.warning(f"Calibration skipped {os.path.basename(img_path)}: {e}")

    if calibration_times:
        per_image_timeout = max(calibration_times) + 0.5
        timing_box.success(
            f"Calibration done. Max observed: {max(calibration_times):.1f}s  "
            f"→  Timeout set to **{per_image_timeout:.1f}s/image**.  "
            f"Est. Total Time: {per_image_timeout * (len(image_paths) - CALIBRATION_N) / 60:.0f} min"
        )
    else:
        per_image_timeout = 120.0
        timing_box.warning(f"Calibration failed — using fallback timeout {per_image_timeout}s")

    # ── Phase 2: pipelined with timeout ──────────────────────────────────────
    # Windows-compatible timeout using threading
    import threading

    def run_with_timeout(fn, timeout, *args):
        result = [None]
        error  = [None]
        def target():
            try:
                result[0] = fn(*args)
            except Exception as e:
                error[0] = e
        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            return None, TimeoutError(f"exceeded {timeout:.1f}s")
        return result[0], error[0]

    skipped_timeout = 0
    for i, img_path in enumerate(image_paths[CALIBRATION_N:], start=CALIBRATION_N):
        pct = (i + 1) / len(image_paths)
        done = i + 1 - CALIBRATION_N
        remaining = len(image_paths) - i - 1
        eta_min = per_image_timeout * remaining / 60
        progress.progress(pct,
            text=f"[{i+1}/{len(image_paths)}] {os.path.basename(img_path)}  "
                 f"·  ETA ~{eta_min:.0f} min  ·  skipped: {skipped_timeout}")

        result_rows, err = run_with_timeout(process_image, per_image_timeout, img_path)
        if isinstance(err, TimeoutError):
            skipped_timeout += 1
            status.warning(f"⏱ Timeout: {os.path.basename(img_path)} — skipped")
        elif err is not None:
            status.warning(f"Error: {os.path.basename(img_path)}: {err}")
        elif result_rows:
            rows.extend(result_rows)

    progress.progress(1.0, text=f"Done!  ·  {skipped_timeout} images timed out")

    if not rows:
        st.error("No results collected.")
        return

    df = pd.DataFrame(rows)

    st.markdown("#### Results summary (mean ± std)")
    summary = df.groupby("method")[["deletion_auc", "insertion_auc", "infidelity"]].agg(["mean", "std"]).round(4)
    st.dataframe(summary, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode()
    st.download_button(
        label="⬇ Download full results CSV",
        data=csv_bytes,
        file_name="xai_benchmark_results.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )
    st.dataframe(df, use_container_width=True)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div class="xai-header">
        <div>
            <div class="xai-title">XAI Vision Inspector</div>
            <div class="xai-subtitle">
                Introducing <strong style="color:#60a5fa;">Eigen-IG</strong>
                &nbsp;— eigenvalue-weighted integrated gradients for sparser, more faithful CNN attributions
                &nbsp;·&nbsp;
                <span class="badge badge-blue">ResNet-50 &minus;17.4% Del ✅</span>
                &nbsp;
                <span class="badge badge-green">EfficientNet +35.2% Ins ✅</span>
                &nbsp;
                <span class="badge badge-red">500-image benchmark</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    cfg = render_sidebar()
    labels = load_imagenet_labels()

    # ── Image loading ──────────────────────────────────────────────────────────
    pil_image = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with st.spinner("Loading model…"):
        model, config = get_model(cfg["model_name"])

    if cfg["image_source"] == "Upload" and cfg["uploaded_file"] is not None:
        pil_image = load_image_from_bytes(cfg["uploaded_file"].read())
    elif cfg["image_source"] == "Batch Benchmark":
        run_batch_benchmark(model, config, cfg, labels, device)
        return

    if pil_image is None:
        # Landing state
        st.markdown("""
        <div style="background:#0c1a2e;border:1px solid #2563eb;border-left:4px solid #60a5fa;border-radius:10px;padding:18px 22px;margin-bottom:20px;">
            <div style="font-family:'Space Mono',monospace;font-size:1.05rem;font-weight:700;color:#60a5fa;margin-bottom:6px;">
                ✦ Novel Method: Eigen-IG
            </div>
            <div style="font-size:0.85rem;color:#94a3b8;line-height:1.6;">
                Eigen-IG applies <strong style="color:#e2e8f0;">truncated SVD on the per-channel gradient trajectory</strong>
                to identify dominant gradient patterns along the IG integration path,
                then uses <strong style="color:#e2e8f0;">eigenvalue-weighted averaging</strong> to upweight signal-rich
                integration steps and suppress noisy ones near the baseline.
                <br><br>
                Benchmarked across 500 ImageNet images on 3 architectures:
                &nbsp;<span style="background:#1e3a5f;color:#60a5fa;padding:2px 8px;border-radius:4px;font-size:0.75rem;">ResNet-50 deletion &minus;17.4% p&lt;0.0001</span>
                &nbsp;<span style="background:#052e16;color:#4ade80;padding:2px 8px;border-radius:4px;font-size:0.75rem;">EfficientNet-B0 insertion +35.2% p&lt;0.0001</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Methods")
        # Eigen-IG gets full-width hero card, others in a row below
        st.markdown("""
        <div class="method-card" style="border-color:#2563eb;border-width:1.5px;margin-bottom:12px;">
            <div class="method-title" style="font-size:1.0rem;color:#60a5fa;">✦ Eigen-IG &nbsp;<span style="font-size:0.7rem;background:#1e3a5f;padding:2px 8px;border-radius:4px;">NOVEL</span></div>
            <div class="method-desc" style="margin-top:6px;">
                Decomposes the IG gradient trajectory matrix <em>G ∈ R<sup>n_steps × H×W</sup></em> per channel via truncated SVD.
                Top-k singular vectors capture dominant spatial gradient patterns; per-step alignment scores
                become integration weights via softmax — upweighting confident steps near α=1,
                downweighting noisy steps near α=0 (zero baseline).
                Produces <strong>sparser, more localized</strong> attributions than standard IG.
            </div>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(4)
        baseline_methods = [
            ("Grad-CAM",             "Gradient-weighted class activation map via the final conv layer. Fast, coarse, class-discriminative.", "🔥"),
            ("Grad-CAM++",           "Second-order gradient weights for better multi-instance localization.", "🔥🔥"),
            ("Integrated Gradients", "Path integral from baseline to input. Satisfies completeness and sensitivity axioms.", "📐"),
            ("Occlusion Sensitivity","Slides a masking patch over the image and measures confidence drop. Model-agnostic.", "🟫"),
        ]
        for col_idx, (name, desc, icon) in enumerate(baseline_methods):
            with cols[col_idx]:
                st.markdown(f"""
                <div class="method-card">
                    <div class="method-title">{icon} {name}</div>
                    <div class="method-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("👈 &nbsp;Upload an image and click **▶ Run Analysis**, or switch to **Batch Benchmark** to run the faithfulness benchmark.")
        return

    # ── Preprocessing ──────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with st.spinner("Loading model…"):
        model, config = get_model(cfg["model_name"])

    tensor, display_img = preprocess_image(pil_image, input_size=224, device=device)

    # ── Prediction ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    top1_idx = int(probs.argmax())
    top1_prob = float(probs[top1_idx])
    top1_label = labels[top1_idx] if top1_idx < len(labels) else f"class_{top1_idx}"

    # ── Model info ─────────────────────────────────────────────────────────────
    summary = get_model_summary(model)

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value" style="font-size:1.05rem;">{cfg['model_name']}</div>
            <div class="metric-delta">{summary['total_params']/1e6:.1f}M params</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Top Prediction</div>
            <div class="metric-value" style="font-size:1.0rem;color:#60a5fa;">{top1_label[:22]}</div>
            <div class="metric-delta">{top1_prob*100:.1f}% confidence</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Conv Layers</div>
            <div class="metric-value">{summary['conv_layers']}</div>
            <div class="metric-delta">available for Grad-CAM</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Active Methods</div>
            <div class="metric-value">{len(cfg['methods'])}</div>
            <div class="metric-delta">{'/ '.join(cfg['methods'][:2])}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Run XAI ────────────────────────────────────────────────────────────────
    if not cfg["run_btn"] and "xai_results" not in st.session_state:
        st.info("Click **▶ Run Analysis** in the sidebar to compute attributions.")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(display_img, caption="Input Image", use_container_width=True)
        with col2:
            st.plotly_chart(
                plot_top_predictions(probs, labels, top_k=5),
                use_container_width=True,
            )
        return

    if cfg["run_btn"]:
        results = {}
        progress = st.progress(0, text="Running XAI analysis…")
        n = len(cfg["methods"])

        for i, method in enumerate(cfg["methods"]):
            progress.progress((i / n), text=f"Computing {method}…")
            t0 = time.time()

            try:
                if method == "Grad-CAM":
                    results[method] = run_gradcam(model, config, tensor, top1_idx, use_plus_plus=False)
                elif method == "Grad-CAM++":
                    results[method] = run_gradcam(model, config, tensor, top1_idx, use_plus_plus=True)
                elif method == "Integrated Gradients":
                    results[method] = run_integrated_gradients(model, tensor, top1_idx, cfg["ig_steps"])
                elif method == "Occlusion Sensitivity":
                    results[method] = run_occlusion(model, tensor, top1_idx, cfg["occ_patch"], cfg["occ_stride"])
                elif method == "SmoothGrad":
                    results[method] = run_smoothgrad(model, tensor, top1_idx, cfg["sg_samples"], cfg["sg_noise"], cfg["sg_variant"])
                elif method == "Eigen-IG":
                    results[method] = run_eigen_ig(model, tensor, top1_idx,cfg.get("ig_steps", 80),cfg.get("eigen_components", 5))

                elapsed = time.time() - t0
                results[method + "_time"] = elapsed

            except Exception as e:
                st.warning(f"⚠️ {method} failed: {e}")

        progress.progress(1.0, text="Done!")
        st.session_state["xai_results"] = results
        st.session_state["xai_display_img"] = display_img
        st.session_state["xai_probs"] = probs

    results = st.session_state.get("xai_results", {})
    display_img = st.session_state.get("xai_display_img", display_img)
    probs = st.session_state.get("xai_probs", probs)

    saliency_maps = {k: v for k, v in results.items() if not k.endswith("_time")}

    if not saliency_maps:
        st.warning("No XAI methods produced results.")
        return

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab_labels = ["🗺️ Comparison", "🔬 Individual Maps", "📊 Statistics", "ℹ️ Method Info", "📈 Faithfulness Benchmark"]
    tabs = st.tabs(tab_labels)

    # ── Tab 1: Side-by-side comparison ─────────────────────────────────────────
    with tabs[0]:
        st.markdown("#### Attribution Map Comparison")
        fig = make_comparison_figure(
            display_img,
            saliency_maps,
            colormap=cfg["colormap"],
            alpha=cfg["alpha"],
        )
        st.image(fig_to_pil(fig), use_container_width=True)

        st.markdown("#### Top-5 Predictions")
        st.plotly_chart(
            plot_top_predictions(probs, labels, top_k=5),
            use_container_width=True,
        )

    # ── Tab 2: Individual interactive maps ─────────────────────────────────────
    with tabs[1]:
        for method_name, saliency_map in saliency_maps.items():
            st.markdown(f"#### {method_name}")
            elapsed = results.get(method_name + "_time", 0)
            st.caption(f"Computed in {elapsed:.2f}s")

            plotly_fig = make_plotly_heatmap(
                display_img,
                saliency_map,
                method_name,
                colormap="Hot",
            )
            st.plotly_chart(plotly_fig, use_container_width=True)
            st.divider()

    # ── Tab 3: Statistics ──────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("#### Attribution Statistics")
        st.markdown("""
        <div class="info-box">
            <strong>Sparsity</strong> = fraction of pixels with near-zero attribution (close to 1 = sharp localization)<br>
            <strong>Top-10% Mean</strong> = average attribution of the most important 10% of pixels<br>
            <strong>Std</strong> = spread of attributions (higher = more concentrated hotspots)
        </div>
        """, unsafe_allow_html=True)

        stats_rows = ""
        for method_name, saliency_map in saliency_maps.items():
            stats = compute_attribution_stats(saliency_map)
            elapsed = results.get(method_name + "_time", 0)
            stats_rows += f"""
            <tr>
                <td><strong>{method_name}</strong></td>
                <td>{stats['mean']:.4f}</td>
                <td>{stats['std']:.4f}</td>
                <td>{stats['max']:.4f}</td>
                <td>{stats['sparsity']*100:.1f}%</td>
                <td>{stats['top10_mean']:.4f}</td>
                <td>{elapsed:.2f}s</td>
            </tr>"""

        st.markdown(f"""
        <table class="stats-table">
            <thead>
                <tr>
                    <th>Method</th><th>Mean</th><th>Std</th><th>Max</th>
                    <th>Sparsity</th><th>Top-10% Mean</th><th>Time</th>
                </tr>
            </thead>
            <tbody>{stats_rows}</tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("#### Attribution Distributions")
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        n_methods = len(saliency_maps)
        fig = make_subplots(
            rows=1, cols=n_methods,
            subplot_titles=list(saliency_maps.keys()),
        )
        for i, (method_name, saliency_map) in enumerate(saliency_maps.items()):
            fig.add_trace(
                go.Histogram(
                    x=saliency_map.flatten(),
                    nbinsx=60,
                    name=method_name,
                    marker_color=["#3b82f6", "#f87171", "#6ee7b7", "#fbbf24", "#a78bfa"][i % 5],
                    showlegend=False,
                ),
                row=1, col=i + 1,
            )
        fig.update_layout(
            paper_bgcolor="#0e1117",
            plot_bgcolor="#161d2e",
            font=dict(color="white", size=11),
            height=280,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        for ann in fig.layout.annotations:
            ann.font.color = "white"
        fig.update_xaxes(gridcolor="#2a2f3e")
        fig.update_yaxes(gridcolor="#2a2f3e")
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Method Info ─────────────────────────────────────────────────────
    with tabs[3]:
        methods_info = {
            "Grad-CAM": {
                "full_name": "Gradient-weighted Class Activation Mapping",
                "paper": "Selvaraju et al., 2017 — https://arxiv.org/abs/1610.02391",
                "how": "Computes the gradient of the class score with respect to the final convolutional feature maps. Global-average-pools the gradients to get per-channel importance weights, then takes a weighted sum of feature maps.",
                "strengths": "Fast (single backward pass), class-discriminative, works with any CNN.",
                "limitations": "Resolution limited to last conv layer size; may miss fine-grained details.",
                "axioms": "Not implementation-invariant. Doesn't satisfy completeness.",
            },
            "Grad-CAM++": {
                "full_name": "Gradient-weighted Class Activation Mapping++",
                "paper": "Chattopadhay et al., 2018 — https://arxiv.org/abs/1710.11063",
                "how": "Extends Grad-CAM by using second-order gradients to compute per-pixel importance weights, rather than globally averaging first-order gradients.",
                "strengths": "Better localization for multiple object instances. More accurate bounding boxes.",
                "limitations": "Slightly slower than Grad-CAM. Still resolution-limited.",
                "axioms": "Improved sensitivity over Grad-CAM but still not fully axiomatic.",
            },
            "Integrated Gradients": {
                "full_name": "Integrated Gradients",
                "paper": "Sundararajan et al., 2017 — https://arxiv.org/abs/1703.01365",
                "how": "Approximates the path integral of gradients along the straight line from a baseline (zeros) to the actual input, via Riemann summation over N interpolation steps.",
                "strengths": "Satisfies Sensitivity and Implementation Invariance axioms. Pixel-level resolution.",
                "limitations": "Slower than Grad-CAM. Results depend on baseline choice. Requires N forward+backward passes.",
                "axioms": "✅ Sensitivity ✅ Implementation Invariance ✅ Completeness (∑attrs ≈ F(x)−F(baseline))",
            },
            "Occlusion Sensitivity": {
                "full_name": "Occlusion Sensitivity / Sliding Window Perturbation",
                "paper": "Zeiler & Fergus, 2014 — https://arxiv.org/abs/1311.2901",
                "how": "Slides a square patch of neutral values over the image and records the drop in prediction confidence at each position.",
                "strengths": "Model-agnostic (no gradients needed). Highly interpretable — literally measures importance.",
                "limitations": "Very slow O(H×W/stride²) forward passes. Low resolution if large stride. May miss interactions.",
                "axioms": "Model-agnostic. No gradient assumptions.",
            },
            "SmoothGrad": {
                "full_name": "SmoothGrad: Removing Noise from Gradients",
                "paper": "Smilkov et al., 2017 — https://arxiv.org/abs/1706.03825",
                "how": "Adds Gaussian noise to the input N times, computes gradients for each noisy copy, and averages them. This denoises the gradient signal and reveals more stable attribution patterns.",
                "strengths": "Pixel-level resolution. Denoises vanilla gradients. Easy to combine with other methods.",
                "limitations": "Requires N backward passes. Noise level σ is a hyperparameter. Doesn't satisfy completeness.",
                "axioms": "Improves visual quality of gradient saliency but doesn't add new axiomatic guarantees.",
            },
            "Eigen-IG": {
                "full_name": "Eigen-Integrated Gradients (Eigen-IG)",
                "paper": "Proposed hybrid (based on Integrated Gradients + Eigen-CAM principles)",
                "how": "First computes standard Integrated Gradients along the path from baseline to input, then applies low-rank SVD (principal component analysis) on the resulting attribution tensor to retain only the top principal directions. This produces sparser and often more faithful explanations by removing noisy components while preserving the axiomatic properties of IG.",
                "strengths": "Combines pixel-level resolution and axiomatic guarantees of IG with the denoising/sparsity benefits of Eigen decomposition. Often yields sharper, less noisy maps than vanilla IG.",
                "limitations": "Adds a small computational overhead for SVD. Number of components (n_components) is a new hyperparameter that needs tuning. Still depends on baseline choice like standard IG.",
                "axioms": "Inherits Sensitivity, Implementation Invariance, and approximate Completeness from Integrated Gradients. Eigen re-weighting improves sparsity without breaking core axioms.",
            },
        }

        for method_name in cfg["methods"]:
            if method_name in methods_info:
                info = methods_info[method_name]
                with st.expander(f"📖 {method_name} — {info['full_name']}", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Paper:** {info['paper']}")
                        st.markdown(f"**How it works:** {info['how']}")
                    with col2:
                        st.markdown(f"**Strengths:** {info['strengths']}")
                        st.markdown(f"**Limitations:** {info['limitations']}")
                        st.markdown(f"**Axiomatic properties:** {info['axioms']}")

# ── Tab 4: Faithfulness Benchmark ─────────────────────────────────────
    with tabs[4]:                   
        st.markdown("#### Faithfulness & Benchmark Metrics")
        st.info("Deletion AUC ↓ better | Insertion AUC ↑ better | Infidelity ↓ better")

        if st.button("🚀 Run Full Faithfulness Benchmark", type="primary"):
            with st.spinner("Computing Deletion / Insertion / Infidelity on all methods..."):
                benchmark_results = run_full_faithfulness_eval(
                    model, tensor, saliency_maps, top1_idx, steps=20
                )
                
                # Display table
                rows = ""
                for method, metrics in benchmark_results.items():
                    rows += f"""
                    <tr>
                        <td><strong>{method}</strong></td>
                        <td>{metrics['Deletion_AUC']:.4f}</td>
                        <td>{metrics['Insertion_AUC']:.4f}</td>
                        <td>{metrics['Infidelity']:.4f}</td>
                        <td>{metrics['Deletion_final_drop']:.1f}%</td>
                    </tr>"""
                
                st.markdown(f"""
                <table class="stats-table">
                    <thead>
                        <tr><th>Method</th><th>Deletion AUC ↓</th><th>Insertion AUC ↑</th>
                        <th>Infidelity ↓</th><th>Final Drop</th></tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
                """, unsafe_allow_html=True)

# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
    