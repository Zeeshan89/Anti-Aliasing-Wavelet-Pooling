"""
Aliasing Analysis  –  Empirically show WHERE aliasing occurs in the backbone.

This script provides the scientific justification for applying WaveletPool
selectively rather than blindly everywhere.

For each down-sampling layer in the baseline YOLOv8-S backbone, it:
  1. Forward-passes a batch of images and captures feature maps BEFORE and
     AFTER the stride-2 convolution.
  2. Computes the 2-D FFT (power spectrum) of each feature map to measure
     high-frequency energy.
  3. Measures the "aliasing score" = fraction of spectral energy above the
     Nyquist frequency (i.e., energy that WILL alias upon down-sampling).
  4. Visualises the spatial feature maps and their frequency spectra.
  5. Outputs a per-layer aliasing severity table and recommends which layers
     to target with WaveletPool.

Run:
    python aliasing_analysis.py

Outputs (saved to  runs/aliasing_analysis/):
    - aliasing_scores.txt          per-layer aliasing severity table
    - <layer>_spatial.png          spatial feature-map visualisation
    - <layer>_spectrum.png         frequency-domain (FFT) visualisation
    - recommendation.txt           which layers to replace with WaveletPool
"""

import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")               # headless rendering
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2

import config as cfg
from model import YOLOv8S, ConvBnSiLU


# ─────────────────────────────────────────────────────────────────────
# 1.  Hook-based Feature Capture
# ─────────────────────────────────────────────────────────────────────

class FeatureCapture:
    """Register forward hooks on target modules to capture input/output."""

    def __init__(self):
        self.captured = {}      # name → {"input": tensor, "output": tensor}
        self._hooks = []

    def register(self, name: str, module: nn.Module):
        def hook_fn(mod, inp, out, name=name):
            self.captured[name] = {
                "input":  inp[0].detach().cpu(),
                "output": out.detach().cpu(),
            }
        self._hooks.append(module.register_forward_hook(hook_fn))

    def clear(self):
        self.captured.clear()

    def remove_all(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ─────────────────────────────────────────────────────────────────────
# 2.  Frequency-Domain Analysis
# ─────────────────────────────────────────────────────────────────────

def compute_power_spectrum(feat: torch.Tensor) -> np.ndarray:
    """
    Compute the average 2-D power spectrum across batch & channels.
    Input:  (B, C, H, W)  →  Output:  (H, W) power spectrum (log scale).
    """
    # average over batch and channels
    feat_np = feat.numpy().mean(axis=(0, 1))             # (H, W)
    fft = np.fft.fft2(feat_np)
    fft_shift = np.fft.fftshift(fft)
    power = np.log1p(np.abs(fft_shift) ** 2)
    return power


def aliasing_score(feat_before: torch.Tensor) -> float:
    """
    Measure the fraction of spectral energy above the Nyquist frequency.

    Nyquist criterion:  when down-sampling by 2×, frequencies above π/2
    (i.e., the outer 50% of the spectrum in each dimension) will alias.

    A higher score means MORE aliasing → stronger justification to apply
    WaveletPool at this layer.

    Returns:  score in [0, 1]  (0 = no aliasing risk, 1 = maximum risk)
    """
    feat_np = feat_before.numpy().mean(axis=(0, 1))      # (H, W)
    fft = np.fft.fft2(feat_np)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift) ** 2

    H, W = power.shape
    cy, cx = H // 2, W // 2
    # Nyquist region = outer annulus (beyond r = min(H,W)//4 from center)
    ry, rx = H // 4, W // 4
    Y, X = np.ogrid[:H, :W]
    nyquist_mask = ((Y - cy) / ry) ** 2 + ((X - cx) / rx) ** 2 > 1.0

    total_energy = power.sum()
    if total_energy < 1e-12:
        return 0.0
    alias_energy = power[nyquist_mask].sum()
    return float(alias_energy / total_energy)


# ─────────────────────────────────────────────────────────────────────
# 2b.  Adaptive Threshold  (replaces fixed ALIASING_THRESHOLD)
# ─────────────────────────────────────────────────────────────────────

def compute_adaptive_threshold(scores_dict, min_ratio=3.0):
    """
    Find the optimal split between HIGH and LOW aliasing layers using
    Otsu's method (maximize between-class variance on the 1-D score
    distribution).  This replaces the fixed threshold from config.py.

    An additional safety check ensures the high group's mean is at least
    `min_ratio`× the low group's mean; otherwise nothing is replaced
    (i.e., all layers are genuinely low-aliasing).

    Returns:  (threshold, low_names, high_names)
              threshold is None when no meaningful split exists.
    """
    names  = list(scores_dict.keys())
    values = np.array([scores_dict[n] for n in names])
    n = len(values)
    if n <= 1:
        return None, names, []

    idx = np.argsort(values)
    sv   = values[idx]                       # sorted values
    sn   = [names[i] for i in idx]           # sorted names

    best_var, best_k = -1, n                 # default: no split
    for k in range(1, n):
        lo, hi = sv[:k], sv[k:]
        w_lo, w_hi = k / n, (n - k) / n
        bv = w_lo * w_hi * (lo.mean() - hi.mean()) ** 2
        if bv > best_var:
            best_var, best_k = bv, k

    lo_mean = sv[:best_k].mean()
    hi_mean = sv[best_k:].mean() if best_k < n else 0

    # safety: reject split if high group isn't clearly separated
    if best_k >= n or hi_mean < lo_mean * min_ratio:
        return None, sn, []

    threshold = float((sv[best_k - 1] + sv[best_k]) / 2)
    return threshold, sn[:best_k], sn[best_k:]


# ─────────────────────────────────────────────────────────────────────
# 3.  Visualisation Helpers
# ─────────────────────────────────────────────────────────────────────

def _save_spatial_vis(feat: torch.Tensor, title: str, path: str):
    """Save a grid of the first 16 feature-map channels."""
    n_show = min(16, feat.shape[1])
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(title, fontsize=14)
    for i, ax in enumerate(axes.flat):
        if i < n_show:
            fm = feat[0, i].numpy()
            ax.imshow(fm, cmap="viridis")
            ax.set_title(f"ch {i}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def _save_spectrum_vis(feat: torch.Tensor, title: str, path: str):
    """Save the averaged 2-D power spectrum."""
    ps = compute_power_spectrum(feat)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(ps, cmap="inferno")
    ax.set_title(title, fontsize=12)
    # draw Nyquist boundary (ellipse at 50% of spectrum)
    H, W = ps.shape
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(W // 2 + W // 4 * np.cos(theta),
            H // 2 + H // 4 * np.sin(theta),
            "w--", linewidth=1.5, label="Nyquist boundary")
    ax.legend(loc="lower right", fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def _save_comparison_vis(before: torch.Tensor, after: torch.Tensor,
                         layer_name: str, score: float, path: str):
    """Side-by-side: input vs output spatial maps + spectra, with aliasing score."""
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 4, figure=fig)

    fig.suptitle(f"{layer_name}   (aliasing score = {score:.3f})", fontsize=14)

    # before: spatial (first 4 channels)
    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        ch = min(i, before.shape[1] - 1)
        ax.imshow(before[0, ch].numpy(), cmap="viridis")
        ax.set_title(f"BEFORE ch{ch}", fontsize=8)
        ax.axis("off")

    # after: spatial (first 4 channels)
    for i in range(4):
        ax = fig.add_subplot(gs[1, i])
        ch = min(i, after.shape[1] - 1)
        ax.imshow(after[0, ch].numpy(), cmap="viridis")
        ax.set_title(f"AFTER ch{ch}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# 4.  Main Analysis
# ─────────────────────────────────────────────────────────────────────

def _load_sample_images(n=4, dataset="moth"):
    """Load a few images from the requested dataset for analysis."""
    ds_cfg = cfg.load_dataset_cfg(dataset)
    img_dir = os.path.join(ds_cfg["split_dir"], "train", "images")
    if not os.path.isdir(img_dir):
        print(f"ERROR: No images found at {img_dir}. Run the dataset preprocess script first.")
        sys.exit(1)
    paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))[:n]
    if not paths:
        # try other extensions
        for ext in ("*.jpeg", "*.png", "*.bmp"):
            paths = sorted(glob.glob(os.path.join(img_dir, ext)))[:n]
            if paths:
                break
    if not paths:
        print(f"ERROR: No images found in {img_dir}")
        sys.exit(1)

    imgs = []
    for p in paths:
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (cfg.INPUT_SIZE, cfg.INPUT_SIZE))
        imgs.append(img.transpose(2, 0, 1).astype(np.float32) / 255.0)

    return torch.tensor(np.stack(imgs))


def run_analysis(model=None, dataset: str = "moth"):
    # Per-dataset output directory so each dataset gets its own analysis
    out_subdir = "aliasing_analysis" if dataset == "moth" else f"aliasing_analysis_{dataset}"
    out_dir = os.path.join(cfg.RUNS_DIR, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"  Aliasing Analysis — dataset: {dataset}")
    print("=" * 60)

    # ── Build or use provided model ───────────────────────────────
    ds_cfg = cfg.load_dataset_cfg(dataset)
    if model is None:
        model = YOLOv8S(num_classes=ds_cfg["num_classes"])
    model.eval()

    # ── Register hooks on every down-sampling layer ──────────────
    capture = FeatureCapture()
    ds_layers = model.get_downsample_layers()
    for name, layer in ds_layers.items():
        capture.register(name, layer)
    print(f"\n  Analysing {len(ds_layers)} down-sampling layers:")
    for n in ds_layers:
        print(f"    • {n}")

    # ── Forward pass ──────────────────────────────────────────────
    images = _load_sample_images(n=4, dataset=dataset)
    print(f"\n  Loaded {images.shape[0]} sample images  ({images.shape})")

    with torch.no_grad():
        _ = model(images)

    # ── Compute aliasing scores ──────────────────────────────────
    print(f"\n  {'Layer':<25s}  {'Resolution':<15s}  {'Aliasing Score':<15s}  Severity")
    print("  " + "─" * 72)

    scores = {}
    for name in ds_layers:
        data = capture.captured.get(name)
        if data is None:
            print(f"  {name:<25s}  (not captured – skipping)")
            continue

        inp = data["input"]
        out = data["output"]
        score = aliasing_score(inp)
        scores[name] = score

        # severity label
        if score > 0.15:
            sev = "HIGH  <-- apply WaveletPool"
        elif score > 0.08:
            sev = "MEDIUM (candidate for WaveletPool)"
        else:
            sev = "LOW"

        res_str = f"{inp.shape[2]}×{inp.shape[3]} → {out.shape[2]}×{out.shape[3]}"
        print(f"  {name:<25s}  {res_str:<15s}  {score:<15.4f}  {sev}")

        # save visualisations
        safe_name = name.replace(".", "_")
        _save_spectrum_vis(inp, f"{name} — INPUT spectrum", os.path.join(out_dir, f"{safe_name}_input_spectrum.png"))
        _save_spectrum_vis(out, f"{name} — OUTPUT spectrum", os.path.join(out_dir, f"{safe_name}_output_spectrum.png"))
        _save_comparison_vis(inp, out, name, score, os.path.join(out_dir, f"{safe_name}_comparison.png"))

    # ── Adaptive threshold (Otsu 1-D) ──────────────────────────────
    threshold, low_group, high_group = compute_adaptive_threshold(scores)
    replace_layers = high_group
    keep_layers = low_group

    if threshold is None:
        thr_str = "N/A (no meaningful split)"
        print(f"\n  Adaptive threshold: {thr_str}")
        print("  All layers have similar (low) aliasing — no replacement needed.")
    else:
        thr_str = f"{threshold:.4f}"
        print(f"\n  Adaptive threshold (Otsu): {thr_str}")

    # aliasing scores table
    with open(os.path.join(out_dir, "aliasing_scores.txt"), "w") as f:
        f.write(f"{'Layer':<25s}  {'Aliasing Score':<15s}  Recommendation\n")
        f.write("─" * 65 + "\n")
        for name, score in sorted(scores.items(), key=lambda x: -x[1]):
            rec = "REPLACE with WaveletPool" if name in replace_layers else "Keep standard"
            f.write(f"{name:<25s}  {score:<15.4f}  {rec}\n")
        f.write(f"\nAdaptive threshold (Otsu 1-D): {thr_str}\n")

    # recommendation
    with open(os.path.join(out_dir, "recommendation.txt"), "w") as f:
        f.write("Aliasing Analysis — Selective WaveletPool Recommendation\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Threshold method: Otsu 1-D (maximize between-class variance)\n")
        f.write(f"Computed threshold: {thr_str}\n\n")
        if replace_layers:
            f.write(f"Layers to REPLACE with WaveletPool ({len(replace_layers)}):\n")
            for n in replace_layers:
                f.write(f"  ✓ {n}  (score = {scores[n]:.4f})\n")
        else:
            f.write("No layers require replacement (all scores are uniformly low).\n")
        f.write(f"\nLayers to KEEP as standard Conv(stride=2) ({len(keep_layers)}):\n")
        for n in keep_layers:
            f.write(f"  ✗ {n}  (score = {scores[n]:.4f})\n")
        f.write(f"\nJustification:\n")
        f.write("  Instead of a fixed threshold, we use Otsu's 1-D method to find\n")
        f.write("  the split that maximally separates high-aliasing layers from\n")
        f.write("  low-aliasing ones.  A safety check ensures the high group's\n")
        f.write("  mean score is ≥3× the low group's mean, preventing spurious\n")
        f.write("  replacements when all layers are uniformly low-aliasing.\n")
        f.write("\n  Layers with high aliasing scores have significant spectral energy\n")
        f.write("  above the Nyquist frequency. Down-sampling these layers without\n")
        f.write("  anti-aliasing causes this energy to fold back into the low-frequency\n")
        f.write("  band, corrupting the very features needed to detect tiny objects.\n")
        f.write("  WaveletPool (DWT→LL→IDWT) cleanly removes this high-frequency content\n")
        f.write("  before down-sampling, preserving small-object detail.\n")

    capture.remove_all()

    print(f"\n  Results saved to: {out_dir}/")
    print(f"\n  RECOMMENDATION (adaptive Otsu threshold = {thr_str}):")
    if replace_layers:
        print(f"    Replace {len(replace_layers)} layer(s) with WaveletPool:")
        for n in replace_layers:
            print(f"      ✓ {n}  (score = {scores[n]:.4f})")
    else:
        print("    No layers need replacement (scores uniformly low).")
    if keep_layers:
        print(f"    Keep {len(keep_layers)} layer(s) as standard:")
        for n in keep_layers:
            print(f"      ✗ {n}  (score = {scores[n]:.4f})")

    print("\nDone.")
    return scores, replace_layers


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Per-layer aliasing analysis (multi-dataset)")
    p.add_argument("--dataset", type=str, default="moth",
                   choices=list(cfg.DATASET_CONFIGS.keys()))
    args = p.parse_args()
    run_analysis(dataset=args.dataset)
