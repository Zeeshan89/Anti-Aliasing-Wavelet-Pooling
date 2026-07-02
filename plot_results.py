"""
Plot all training results, comparisons, and aliasing analysis.

Generates publication-quality figures from the saved training histories,
test metrics, and aliasing analysis outputs.

Run:
    python plot_results.py

Outputs saved to:  runs/plots/
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

import config as cfg

# Active dataset (set by main() via --dataset flag).  Default "moth" preserves
# legacy behaviour so existing scripts/tests keep working.
DATASET = "moth"
PLOT_DIR = os.path.join(cfg.RUNS_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
})

COLORS = {
    "baseline": "#2196F3",   # blue
    "A":        "#4CAF50",   # green
    "B":        "#FF9800",   # orange
    "blurpool": "#E91E63",   # pink/magenta
    "gaussian": "#9C27B0",   # purple
}
LABELS = {
    "baseline": "Baseline (YOLOv8-S)",
    "A":        "Variant A — Wavelet (Coif3)",
    "B":        "Variant B (Bottom-Heavy + Wavelet)",
    "blurpool": "Variant C — BlurPool",
    "gaussian": "Variant D — Gaussian",
}


# ── Load data ────────────────────────────────────────────────────────

def load_history(variant_dir):
    path = os.path.join(cfg.RUNS_DIR, variant_dir, "history.json")
    with open(path) as f:
        return json.load(f)


def load_test_metrics(variant_dir):
    path = os.path.join(cfg.RUNS_DIR, variant_dir, "test_metrics.json")
    with open(path) as f:
        return json.load(f)


def load_test_metrics_tta(variant_dir):
    """Load TTA test metrics if available; returns None if file missing."""
    path = os.path.join(cfg.RUNS_DIR, variant_dir, "test_metrics_tta.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# These module-level globals are populated by init_dataset() — call it
# before invoking any plot functions.  Defaults to the moth dataset so
# legacy `python plot_results.py` (no args) still works.
variants = {}
histories = {}
test_metrics = {}
avail_variants = {}
CLASS_NAMES_FOR_PLOTS = []


def init_dataset(dataset: str = "moth"):
    """
    Configure all module-level state for plotting a specific dataset.
    Builds dataset-prefixed variant directory names (e.g. ip102_baseline),
    loads each variant's history.json / test_metrics.json if it exists,
    and points PLOT_DIR at runs/plots/{dataset}/ for non-moth datasets.
    """
    global DATASET, PLOT_DIR, variants, histories, test_metrics, avail_variants
    global CLASS_NAMES_FOR_PLOTS

    DATASET = dataset
    # Keep moth plots at runs/plots/ for backward compatibility; other
    # datasets get their own subdir.
    PLOT_DIR = (os.path.join(cfg.RUNS_DIR, "plots")
                if dataset == "moth"
                else os.path.join(cfg.RUNS_DIR, "plots", dataset))
    os.makedirs(PLOT_DIR, exist_ok=True)

    variants = {
        "baseline": (f"{dataset}_baseline", COLORS["baseline"], LABELS["baseline"]),
        "A":        (f"{dataset}_a",        COLORS["A"],        LABELS["A"]),
        "blurpool": (f"{dataset}_blurpool", COLORS["blurpool"], LABELS["blurpool"]),
        "gaussian": (f"{dataset}_gaussian", COLORS["gaussian"], LABELS["gaussian"]),
        # "B":        (f"{dataset}_b",        COLORS["B"],        LABELS["B"]),
    }

    histories.clear()
    test_metrics.clear()
    for key, (dirname, _, _) in variants.items():
        try:
            histories[key] = load_history(dirname)
            test_metrics[key] = load_test_metrics(dirname)
        except FileNotFoundError:
            pass  # variant not yet trained — plots will skip it automatically

    avail_variants.clear()
    avail_variants.update({k: v for k, v in variants.items() if k in histories})

    # Resolve dataset-specific class names for per-class plots
    try:
        ds_cfg = cfg.load_dataset_cfg(dataset)
        CLASS_NAMES_FOR_PLOTS[:] = ds_cfg["class_names"]
    except Exception:
        CLASS_NAMES_FOR_PLOTS[:] = cfg.CLASS_NAMES

    return avail_variants


# Initialize with the default dataset so all top-level references resolve.
init_dataset("moth")


# =====================================================================
# Plot 1:  Training Loss Curves (all variants overlaid)
# =====================================================================

def plot_training_loss():
    fig, ax = plt.subplots(figsize=(10, 6))

    for key, (dirname, color, label) in avail_variants.items():
        h = histories[key]
        epochs = [e["epoch"] for e in h]
        train_loss = [e["train_loss"] for e in h]
        ax.plot(epochs, train_loss, color=color, label=label, marker="o", markersize=4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss Comparison Across Variants")
    ax.legend(loc="upper right")
    ax.set_yscale("log")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "01_training_loss.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# =====================================================================
# Plot 2:  Validation Loss Curves (all variants overlaid)
# =====================================================================

def plot_validation_loss():
    fig, ax = plt.subplots(figsize=(10, 6))

    for key, (dirname, color, label) in avail_variants.items():
        h = histories[key]
        epochs = [e["epoch"] for e in h]
        val_loss = [e["val_loss"] for e in h]
        ax.plot(epochs, val_loss, color=color, label=label, marker="s", markersize=4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss Comparison Across Variants")
    ax.legend(loc="upper right")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "02_validation_loss.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# =====================================================================
# Plot 3:  Train vs Val Loss per Variant (overfitting check)
# =====================================================================

def plot_train_vs_val():
    avail = list(avail_variants.items())
    n = len(avail)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes_arr = plt.subplots(rows, cols, figsize=(9 * cols, 5 * rows), sharey=True)
    axes_arr = np.array(axes_arr).flatten()

    for i, (key, (dirname, color, label)) in enumerate(avail):
        ax = axes_arr[i]
        h = histories[key]
        epochs = [e["epoch"] for e in h]
        train_loss = [e["train_loss"] for e in h]
        val_loss = [e["val_loss"] for e in h]

        ax.plot(epochs, train_loss, color=color, label="Train", linestyle="-", marker="o", markersize=3)
        ax.plot(epochs, val_loss, color=color, label="Val", linestyle="--", marker="s", markersize=3, alpha=0.7)
        ax.fill_between(epochs, train_loss, val_loss, color=color, alpha=0.1)

        ax.set_xlabel("Epoch")
        if i % cols == 0:
            ax.set_ylabel("Loss")
        ax.set_title(label)
        ax.legend(loc="upper right")

    for j in range(n, len(axes_arr)):
        axes_arr[j].axis("off")

    fig.suptitle("Train vs Validation Loss (Overfitting Check)", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "03_train_vs_val.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# =====================================================================
# Plot 4:  Learning Rate Schedule
# =====================================================================

def plot_lr_schedule():
    fig, ax = plt.subplots(figsize=(10, 5))

    for key, (dirname, color, label) in avail_variants.items():
        h = histories[key]
        epochs = [e["epoch"] for e in h]
        lr = [e["lr"] for e in h]
        ax.plot(epochs, lr, color=color, label=label, marker=".", markersize=5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule (Cosine Annealing with Warm-up)")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "04_lr_schedule.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# =====================================================================
# Plot 5:  Test Metrics Bar Chart (final comparison)
# =====================================================================

def plot_test_metrics_bar():
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    metric_names = ["loss", "cls_loss", "reg_loss"]
    metric_titles = ["Total Test Loss", "Classification Loss", "Regression Loss"]

    for ax, mname, mtitle in zip(axes, metric_names, metric_titles):
        vals = []
        colors = []
        labels = []
        for key, (dirname, color, label) in avail_variants.items():
            vals.append(test_metrics[key][mname])
            colors.append(color)
            labels.append(key.upper() if key != "baseline" else "Baseline")

        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5, width=0.5)

        # add value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_ylabel("Loss")
        ax.set_title(mtitle)
        ax.set_ylim(0, max(vals) * 1.3)

    fig.suptitle("Test Set Results — Final Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "05_test_metrics_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# =====================================================================
# Plot 6:  Aliasing Scores Horizontal Bar Chart
# =====================================================================

def plot_aliasing_scores():
    # Look in the per-dataset aliasing analysis directory
    subdir = "aliasing_analysis" if DATASET == "moth" else f"aliasing_analysis_{DATASET}"
    scores_path = os.path.join(cfg.RUNS_DIR, subdir, "aliasing_scores.txt")
    if not os.path.exists(scores_path):
        print(f"  [SKIP] aliasing_scores.txt not found at {scores_path}")
        return

    layers = []
    scores = []
    with open(scores_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Layer") or line.startswith("─"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    layers.append(parts[0])
                    scores.append(float(parts[1]))
                except ValueError:
                    continue

    if not layers:
        print("  [SKIP] No aliasing scores parsed")
        return

    # sort by score ascending for horizontal bar
    paired = sorted(zip(scores, layers))
    scores_sorted = [p[0] for p in paired]
    layers_sorted = [p[1] for p in paired]

    fig, ax = plt.subplots(figsize=(10, 6))

    threshold = cfg.ALIASING_THRESHOLD
    colors_bar = ["#f44336" if s > threshold else "#90CAF9" for s in scores_sorted]

    bars = ax.barh(layers_sorted, scores_sorted, color=colors_bar, edgecolor="black", linewidth=0.5, height=0.6)

    # threshold line
    # ax.axvline(x=threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold}")

    # value labels
    for bar, score in zip(bars, scores_sorted):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", fontsize=10)

    ax.set_xlabel("Aliasing Score")
    ax.set_title("Per-Layer Aliasing Score (FFT Energy Above Nyquist)")
    # ax.legend(loc="lower right")

    # add annotation
    # ax.annotate("RED = replaced with WaveletPool", xy=(0.98, 0.02), xycoords="axes fraction",
    #             ha="right", fontsize=9, fontstyle="italic", color="#c62828")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "06_aliasing_scores.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# =====================================================================
# Plot 7:  Convergence Speed Comparison (epochs to reach threshold)
# =====================================================================

def plot_convergence_speed():
    fig, ax = plt.subplots(figsize=(10, 6))

    # plot cumulative min validation loss over epochs
    for key, (dirname, color, label) in avail_variants.items():
        h = histories[key]
        epochs = [e["epoch"] for e in h]
        val_loss = [e["val_loss"] for e in h]

        # cumulative minimum
        cum_min = []
        best = float("inf")
        for v in val_loss:
            best = min(best, v)
            cum_min.append(best)

        ax.plot(epochs, cum_min, color=color, label=label, marker="D", markersize=4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Best Validation Loss So Far")
    ax.set_title("Convergence Speed — Best Val Loss Over Training")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "07_convergence_speed.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# =====================================================================
# Plot 8:  Summary Dashboard (combined)
# =====================================================================

def plot_dashboard():
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ── Panel 1: Training Loss ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0:2])
    for key, (dirname, color, label) in avail_variants.items():
        h = histories[key]
        epochs = [e["epoch"] for e in h]
        ax1.plot(epochs, [e["train_loss"] for e in h], color=color, label=label, marker="o", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.set_yscale("log")
    ax1.legend(fontsize=8)

    # ── Panel 2: Validation Loss ──────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    for key, (dirname, color, label) in avail_variants.items():
        h = histories[key]
        epochs = [e["epoch"] for e in h]
        ax2.plot(epochs, [e["val_loss"] for e in h], color=color, label=label.split("—")[0].strip(), marker="s", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Validation Loss")
    ax2.legend(fontsize=8)

    # ── Panel 3: Test Total Loss Bar ──────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    vnames = []
    vvals = []
    vcols = []
    for key, (dirname, color, label) in avail_variants.items():
        vnames.append(key.upper() if key != "baseline" else "Base")
        vvals.append(test_metrics[key]["loss"])
        vcols.append(color)
    bars = ax3.bar(vnames, vvals, color=vcols, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, vvals):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax3.set_ylabel("Loss")
    ax3.set_title("Test Total Loss")
    ax3.set_ylim(0, max(vvals) * 1.35)

    # ── Panel 4: Test Cls Loss Bar ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    cls_vals = [test_metrics[key]["cls_loss"] for key in avail_variants]
    bars = ax4.bar(vnames, cls_vals, color=vcols, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, cls_vals):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax4.set_ylabel("Loss")
    ax4.set_title("Test Classification Loss")
    ax4.set_ylim(0, max(cls_vals) * 1.35)

    # ── Panel 5: Test Reg Loss Bar ────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    reg_vals = [test_metrics[key]["reg_loss"] for key in avail_variants]
    bars = ax5.bar(vnames, reg_vals, color=vcols, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, reg_vals):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax5.set_ylabel("Loss")
    ax5.set_title("Test Regression Loss")
    ax5.set_ylim(0, max(reg_vals) * 1.35)

    # ── Panel 6: LR Schedule ─────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    for key, (dirname, color, label) in avail_variants.items():
        h = histories[key]
        ax6.plot([e["epoch"] for e in h], [e["lr"] for e in h], color=color,
                label=key.upper() if key != "baseline" else "Base", marker=".", markersize=3)
    ax6.set_xlabel("Epoch")
    ax6.set_ylabel("LR")
    ax6.set_title("Learning Rate Schedule")
    ax6.legend(fontsize=8)

    # ── Panel 7: Convergence ──────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 1])
    for key, (dirname, color, label) in avail_variants.items():
        h = histories[key]
        val_loss = [e["val_loss"] for e in h]
        cum_min = []
        best = float("inf")
        for v in val_loss:
            best = min(best, v)
            cum_min.append(best)
        ax7.plot([e["epoch"] for e in h], cum_min, color=color,
                label=key.upper() if key != "baseline" else "Base", marker="D", markersize=3)
    ax7.set_xlabel("Epoch")
    ax7.set_ylabel("Best Val Loss")
    ax7.set_title("Convergence Speed")
    ax7.legend(fontsize=8)

    # ── Panel 8: Aliasing Scores ──────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 2])
    subdir = "aliasing_analysis" if DATASET == "moth" else f"aliasing_analysis_{DATASET}"
    scores_path = os.path.join(cfg.RUNS_DIR, subdir, "aliasing_scores.txt")
    if os.path.exists(scores_path):
        layers_a = []
        scores_a = []
        with open(scores_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("Layer") or line.startswith("─"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        layers_a.append(parts[0].replace("backbone.", "bb.").replace("neck.", "n."))
                        scores_a.append(float(parts[1]))
                    except ValueError:
                        continue
        if layers_a:
            paired = sorted(zip(scores_a, layers_a))
            s_sorted = [p[0] for p in paired]
            l_sorted = [p[1] for p in paired]
            bar_colors = ["#f44336" if s > cfg.ALIASING_THRESHOLD else "#90CAF9" for s in s_sorted]
            ax8.barh(l_sorted, s_sorted, color=bar_colors, edgecolor="black", linewidth=0.5, height=0.5)
            ax8.axvline(x=cfg.ALIASING_THRESHOLD, color="red", linestyle="--", linewidth=1)
            ax8.set_xlabel("Score")
            ax8.set_title("Aliasing Scores")
    else:
        ax8.text(0.5, 0.5, "No aliasing data", ha="center", va="center", transform=ax8.transAxes)
        ax8.set_title("Aliasing Scores")

    fig.suptitle("Moth Detection — Training Results Dashboard", fontsize=16, fontweight="bold", y=0.98)
    path = os.path.join(PLOT_DIR, "08_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# =====================================================================
# Plot 9:  Results Summary Table (as image)
# =====================================================================

def plot_results_table():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    col_labels = ["Variant", "Train Loss\n(final)", "Val Loss\n(final)",
                  "Test Loss", "mAP@0.5", "mAP@0.75", "Weighted\nF1",
                  "Avg\nPrec", "Avg\nRecall"]
    cell_data = []
    cell_colors = []

    for key, (dirname, color, label) in avail_variants.items():
        h = histories[key]
        tm = test_metrics[key]
        row = [
            label,
            f"{h[-1]['train_loss']:.4f}",
            f"{h[-1]['val_loss']:.4f}",
            f"{tm['loss']:.4f}",
            f"{tm.get('mAP@0.5', 0):.4f}",
            f"{tm.get('mAP@0.75', 0):.4f}",
            f"{tm.get('weighted_f1', 0):.4f}",
            f"{tm.get('avg_precision', 0):.4f}",
            f"{tm.get('avg_recall', 0):.4f}",
        ]
        cell_data.append(row)
        cell_colors.append([color + "30"] * len(col_labels))

    table = ax.table(cellText=cell_data, colLabels=col_labels, cellColours=cell_colors,
                     colColours=["#E0E0E0"] * len(col_labels),
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)

    # bold best for loss cols (lower=better) and metric cols (higher=better)
    for col_idx in [3]:
        vals = [float(cell_data[r][col_idx]) for r in range(len(cell_data))]
        best_row = vals.index(min(vals))
        table[best_row + 1, col_idx].set_text_props(fontweight="bold")
        table[best_row + 1, col_idx].set_facecolor("#C8E6C9")
    for col_idx in [4, 5, 6, 7, 8]:
        vals = [float(cell_data[r][col_idx]) for r in range(len(cell_data))]
        best_row = vals.index(max(vals))
        table[best_row + 1, col_idx].set_text_props(fontweight="bold")
        table[best_row + 1, col_idx].set_facecolor("#C8E6C9")

    ax.set_title("Results Summary Table", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "09_results_table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# =====================================================================
# Plot 10:  mAP Comparison Bar Chart
# =====================================================================

def plot_map_comparison():
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flat

    # mAP@0.5
    vnames = [k.upper() if k != "baseline" else "Baseline" for k in avail_variants]
    vcols = [avail_variants[k][1] for k in avail_variants]

    map50 = [test_metrics[k].get("mAP@0.5", 0) for k in avail_variants]
    bars = axes[0].bar(vnames, map50, color=vcols, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, map50):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
    axes[0].set_title("mAP @ IoU=0.5")
    axes[0].set_ylabel("mAP")
    axes[0].set_ylim(0, max(max(map50) * 1.4, 0.05))

    # mAP@0.75
    map75 = [test_metrics[k].get("mAP@0.75", 0) for k in avail_variants]
    bars = axes[1].bar(vnames, map75, color=vcols, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, map75):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
    axes[1].set_title("mAP @ IoU=0.75")
    axes[1].set_ylabel("mAP")
    axes[1].set_ylim(0, max(max(map75) * 1.4, 0.05))

    # weighted F1
    wf1 = [test_metrics[k].get("weighted_f1", 0) for k in avail_variants]
    bars = axes[2].bar(vnames, wf1, color=vcols, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, wf1):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
    axes[2].set_title("Weighted F1 Score")
    axes[2].set_ylabel("F1")
    axes[2].set_ylim(0, max(max(wf1) * 1.4, 0.05))

    # Average Precision
    avgp = [test_metrics[k].get("avg_precision", 0) for k in avail_variants]
    bars = axes[3].bar(vnames, avgp, color=vcols, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, avgp):
        axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
    axes[3].set_title("Average Precision")
    axes[3].set_ylabel("Precision")
    axes[3].set_ylim(0, max(max(avgp) * 1.4, 0.05))

    # Average Recall
    avgr = [test_metrics[k].get("avg_recall", 0) for k in avail_variants]
    bars = axes[4].bar(vnames, avgr, color=vcols, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, avgr):
        axes[4].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
    axes[4].set_title("Average Recall")
    axes[4].set_ylabel("Recall")
    axes[4].set_ylim(0, max(max(avgr) * 1.4, 0.05))

    # hide unused subplot
    axes[5].axis("off")

    fig.suptitle("Detection Metrics Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "10_map_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# =====================================================================
# Plot 11:  Per-Class AP / Precision / Recall / F1 Grouped Bar
# =====================================================================

def plot_per_class_metrics():
    class_names = list(CLASS_NAMES_FOR_PLOTS)
    nc = len(class_names)
    # For datasets with many classes (IP102 has 102), per-class grouped bars
    # are unreadable.  Show the 20 most-frequent GT classes only.
    MAX_CLASSES = 20
    if nc > MAX_CLASSES and avail_variants:
        # rank classes by total GT count across all available variants
        gt_counts = np.zeros(nc, dtype=int)
        for k in avail_variants:
            pc = test_metrics[k].get("per_class", {})
            for c in range(nc):
                gt_counts[c] += pc.get(str(c), pc.get(c, {})).get("n_gt", 0)
        top_idx = list(np.argsort(-gt_counts)[:MAX_CLASSES])
        keep_idx = sorted(top_idx)
        class_names = [class_names[c] for c in keep_idx]
        nc = len(class_names)
    else:
        keep_idx = list(range(nc))
    x = np.arange(nc)
    width = 0.25

    n_av = max(len(avail_variants), 1)
    width = 0.8 / n_av

    for metric_key, metric_label in [("ap50", "AP @ IoU=0.5"),
                                      ("precision", "Precision"),
                                      ("recall", "Recall"),
                                      ("f1", "F1 Score")]:
        fig, ax = plt.subplots(figsize=(max(10, nc * 0.5), 5))
        for i, (key, (dirname, color, label)) in enumerate(avail_variants.items()):
            pc = test_metrics[key].get("per_class", {})
            vals = [pc.get(str(c), pc.get(c, {})).get(metric_key, 0) for c in keep_idx]
            short = key.upper() if key != "baseline" else "Baseline"
            bars = ax.bar(x + i * width, vals, width, label=short, color=color,
                         edgecolor="black", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f"{val:.2f}", ha="center", fontsize=8)

        ax.set_xticks(x + width * (n_av - 1) / 2)
        rotation = 45 if nc > 8 else 15
        ax.set_xticklabels([n[:18] for n in class_names], rotation=rotation, ha="right" if rotation == 45 else "center")
        ax.set_ylabel(metric_label)
        ax.set_title(f"Per-Class {metric_label}")
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        safe = metric_key.replace("@", "")
        path = os.path.join(PLOT_DIR, f"11_perclass_{safe}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")


# =====================================================================
# Plot 12:  Equivariance Error Comparison
# =====================================================================

def plot_equivariance():
    fig, ax = plt.subplots(figsize=(12, 6))
    layer_names = None
    x = None
    n_av = max(len(avail_variants), 1)
    width = 0.8 / n_av

    for i, (key, (dirname, color, label)) in enumerate(avail_variants.items()):
        ee = test_metrics[key].get("equivariance_error", {})
        if not ee:
            continue
        if layer_names is None:
            layer_names = sorted(ee.keys())
            x = np.arange(len(layer_names))
        vals = [ee.get(n, 0) for n in layer_names]
        short = key.upper() if key != "baseline" else "Baseline"
        ax.bar(x + i * width, vals, width, label=short, color=color,
               edgecolor="black", linewidth=0.5)

    if layer_names is not None:
        short_names = [n.replace("backbone.", "bb.").replace("neck.", "n.") for n in layer_names]
        ax.set_xticks(x + width * (n_av - 1) / 2)
        ax.set_xticklabels(short_names, rotation=30)
        ax.set_ylabel("Equivariance Error (lower = better)")
        ax.set_title("Shift Equivariance Error — Per Layer Comparison")
        ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "12_equivariance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# =====================================================================
# Plot 14:  Train vs Val mAP / P / R Curves (diagnostic)
# =====================================================================

def plot_train_val_map_curves():
    """Plot train/val mAP@0.5, precision, recall over epochs for each variant."""
    metrics_to_plot = [
        ("mAP50",  "mAP @ IoU=0.5"),
        ("wF1",    "Weighted F1"),
        ("avgP",   "Avg Precision"),
        ("avgR",   "Avg Recall"),
    ]

    for key, (dirname, color, label) in avail_variants.items():
        h = histories[key]
        # filter epochs that have detection metrics
        eval_epochs = [e for e in h if "train_mAP50" in e]
        if not eval_epochs:
            print(f"  [SKIP] No train/val detection metrics for {key}")
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

        for ax, (mkey, mtitle) in zip(axes.flat, metrics_to_plot):
            train_key = f"train_{mkey}"
            val_key = f"val_{mkey}"
            eps = [e["epoch"] for e in eval_epochs]
            train_vals = [e.get(train_key, 0) for e in eval_epochs]
            val_vals = [e.get(val_key, 0) for e in eval_epochs]
            ax.plot(eps, train_vals, "o-", color=color, label="Train", markersize=5)
            ax.plot(eps, val_vals, "s--", color=color, label="Val", alpha=0.7, markersize=5)
            ax.fill_between(eps, train_vals, val_vals, color=color, alpha=0.1)
            ax.set_ylabel(mtitle)
            ax.set_title(mtitle)
            ax.legend()
            ax.set_ylim(0, 1.05)

        axes[1, 0].set_xlabel("Epoch")
        axes[1, 1].set_xlabel("Epoch")

        vname = key.upper() if key != "baseline" else "Baseline"
        fig.suptitle(f"Train vs Val Detection Metrics — {vname}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, f"14_train_val_metrics_{key}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")

    # Also plot mAP@0.5 for all variants on one chart
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, (dirname, color, label) in avail_variants.items():
        h = histories[key]
        eval_epochs = [e for e in h if "val_mAP50" in e]
        if not eval_epochs:
            continue
        eps = [e["epoch"] for e in eval_epochs]
        val_map = [e["val_mAP50"] for e in eval_epochs]
        vname = key.upper() if key != "baseline" else "Baseline"
        ax.plot(eps, val_map, "o-", color=color, label=vname, markersize=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val mAP @ IoU=0.5")
    ax.set_title("Validation mAP@0.5 Over Training — All Variants")
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "14_val_map_all_variants.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# =====================================================================
# Plot 13:  Sample Detection Outputs (GT vs Predictions per variant)
# =====================================================================

def plot_sample_detections():
    """Load best model for each variant, run on test images, draw GT + predictions."""
    import sys
    sys.path.insert(0, cfg.PROJECT_ROOT)
    import torch
    import cv2
    from model import YOLOv8S
    from train import (decode_predictions, nms,
                       apply_wavelet_selective, apply_bottom_heavy,
                       apply_blurpool_selective, apply_gaussian_selective)

    device = "cpu"
    ds_cfg = cfg.load_dataset_cfg(DATASET)
    class_names = ds_cfg["class_names"]
    num_classes = ds_cfg["num_classes"]
    # generate a palette covering all classes
    base_palette = [(46, 204, 113), (52, 152, 219), (231, 76, 60), (241, 196, 15),
                    (155, 89, 182), (52, 73, 94), (230, 126, 34), (149, 165, 166)]
    cls_colors = [base_palette[i % len(base_palette)] for i in range(num_classes)]

    # load up to 8 test images from the active dataset's split
    test_img_dir = os.path.join(ds_cfg["split_dir"], "test", "images")
    test_lbl_dir = os.path.join(ds_cfg["split_dir"], "test", "labels")
    IMG_EXTS = (".jpg", ".jpeg", ".png")
    img_files = sorted([f for f in os.listdir(test_img_dir)
                        if os.path.splitext(f)[1].lower() in IMG_EXTS])[:8]
    if not img_files:
        print("  [SKIP] No test images found")
        return

    # load images + GT
    images_np, images_tensor, gt_labels = [], [], []
    for fname in img_files:
        img = cv2.imread(os.path.join(test_img_dir, fname))
        img = cv2.resize(img, (cfg.INPUT_SIZE, cfg.INPUT_SIZE))
        images_np.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        t = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        images_tensor.append(torch.from_numpy(t.copy()))
        lbl_path = os.path.join(test_lbl_dir, f"{os.path.splitext(fname)[0]}.txt")
        boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        boxes.append([float(x) for x in parts])
        gt_labels.append(boxes)

    batch = torch.stack(images_tensor)

    for vkey, (dirname, color, vlabel) in avail_variants.items():
        weights_path = os.path.join(cfg.RUNS_DIR, dirname, "weights", "best.pt")
        if not os.path.exists(weights_path):
            print(f"  [SKIP] No weights for {vkey}")
            continue

        import json as _json

        def _load_target_layers(dirname):
            wl_path = os.path.join(cfg.RUNS_DIR, dirname, "wavelet_layers.json")
            if os.path.exists(wl_path):
                with open(wl_path) as _f:
                    return _json.load(_f)
            return None

        model = YOLOv8S(num_classes=num_classes)
        if vkey.upper() in ("A", "B"):
            tgt = _load_target_layers(dirname)
            apply_wavelet_selective(model, target_layers=tgt) if tgt else apply_wavelet_selective(model)
        elif vkey.lower() == "blurpool":
            tgt = _load_target_layers(dirname)
            apply_blurpool_selective(model, target_layers=tgt) if tgt else apply_blurpool_selective(model)
        elif vkey.lower() == "gaussian":
            tgt = _load_target_layers(dirname)
            apply_gaussian_selective(model, target_layers=tgt) if tgt else apply_gaussian_selective(model)
        if vkey.upper() == "B":
            apply_bottom_heavy(model)
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        model.eval()

        with torch.no_grad():
            preds = model(batch.to(device))
        batch_dets = decode_predictions(preds, conf_thresh=0.25, img_size=cfg.INPUT_SIZE)

        n_imgs = len(images_np)
        cols = min(4, n_imgs)
        rows = (n_imgs + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        if n_imgs == 1:
            axes = np.array([axes])
        axes = axes.flat

        for idx in range(n_imgs):
            ax = axes[idx]
            img_show = images_np[idx].copy()
            ax.imshow(img_show)
            S = cfg.INPUT_SIZE

            # draw GT boxes (dashed)
            for box in gt_labels[idx]:
                cid = int(box[0])
                cx, cy, w, h = box[1]*S, box[2]*S, box[3]*S, box[4]*S
                x1, y1 = cx - w/2, cy - h/2
                c = tuple(v/255 for v in cls_colors[cid % len(cls_colors)])
                rect = plt.Rectangle((x1, y1), w, h, linewidth=2,
                                     edgecolor=c, facecolor="none", linestyle="--")
                ax.add_patch(rect)
                ax.text(x1, y1 - 3, f"GT:{class_names[cid]}", fontsize=7,
                        color="white", bbox=dict(facecolor=c, alpha=0.7, pad=1))

            # draw predictions (solid)
            dets = nms(batch_dets[idx].cpu(), 0.45)
            for d in range(dets.shape[0]):
                x1, y1, x2, y2, conf, cid = dets[d].tolist()
                cid = int(cid)
                c = tuple(v/255 for v in cls_colors[cid % len(cls_colors)])
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor=c, facecolor="none")
                ax.add_patch(rect)
                ax.text(x1, y2 + 10, f"{class_names[cid]} {conf:.2f}", fontsize=7,
                        color="white", bbox=dict(facecolor=c, alpha=0.7, pad=1))

            ax.set_title(img_files[idx][:25], fontsize=8)
            ax.axis("off")

        # hide unused axes
        for idx in range(n_imgs, len(list(axes))):
            axes[idx].axis("off")

        vname = vkey.upper() if vkey != "baseline" else "Baseline"
        fig.suptitle(f"Sample Detections — {vname}\n(dashed=GT, solid=predicted)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, f"13_detections_{vkey}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


# =====================================================================
# Plot 15:  Anti-Aliasing Method Summary (3-panel bar chart)
# =====================================================================

def plot_antialiasing_summary():
    """
    Single figure comparing all available anti-aliasing variants on the
    three most informative metrics: mAP@0.5, mAP@0.75, Weighted F1.
    Each bar is annotated with its absolute value and delta vs baseline.
    """
    if not avail_variants:
        print("  [SKIP] No variants available for antialiasing summary")
        return

    metrics = [
        ("mAP@0.5",      "mAP @ IoU=0.5"),
        ("mAP@0.75",     "mAP @ IoU=0.75"),
        ("weighted_f1",  "Weighted F1"),
    ]

    baseline_tm = test_metrics.get("baseline", {})
    vnames = [k.upper() if k != "baseline" else "Baseline" for k in avail_variants]
    vcols  = [avail_variants[k][1] for k in avail_variants]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, (mkey, mtitle) in zip(axes, metrics):
        vals = [test_metrics[k].get(mkey, 0) for k in avail_variants]
        base_val = baseline_tm.get(mkey, 0)

        bars = ax.bar(vnames, vals, color=vcols, edgecolor="black", linewidth=0.6, width=0.55)
        for bar, val, k in zip(bars, vals, avail_variants):
            delta = val - base_val
            sign = "+" if delta >= 0 else ""
            label_txt = f"{val:.3f}\n({sign}{delta*100:.1f}%)" if k != "baseline" else f"{val:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    label_txt, ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_title(mtitle, fontsize=12)
        ax.set_ylabel(mtitle.split("@")[0].strip())
        ax.set_ylim(0, max(vals) * 1.35 if vals else 1)

    fig.suptitle("Anti-Aliasing Methods Comparison — All Metrics",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "15_antialiasing_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# =====================================================================
# Plot 16: TTA vs no-TTA comparison (3-panel grouped bar chart)
# =====================================================================

def plot_tta_comparison():
    """
    Compare Test-Time Augmentation vs standard inference for each variant.
    Reads {dataset}_{variant}/test_metrics_tta.json if present and pairs
    it with test_metrics.json.  Skips silently if no TTA results found.
    """
    pairs = {}
    for key, (dirname, color, label) in avail_variants.items():
        tta = load_test_metrics_tta(dirname)
        if tta is None:
            continue
        pairs[key] = (test_metrics[key], tta, color, label)

    if not pairs:
        print("  [SKIP] No TTA result files found (run with --eval-tta to generate).")
        return

    metrics = [
        ("mAP@0.5",     "mAP @ IoU=0.5"),
        ("mAP@0.75",    "mAP @ IoU=0.75"),
        ("weighted_f1", "Weighted F1"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    keys   = list(pairs.keys())
    x      = np.arange(len(keys))
    width  = 0.4
    colors = [pairs[k][2] for k in keys]
    names  = [k.upper() if k != "baseline" else "Baseline" for k in keys]

    for ax, (mkey, mlabel) in zip(axes, metrics):
        no_tta = [pairs[k][0].get(mkey, 0) for k in keys]
        tta    = [pairs[k][1].get(mkey, 0) for k in keys]

        b1 = ax.bar(x - width/2, no_tta, width, color=colors, edgecolor="black",
                    linewidth=0.5, label="no-TTA")
        b2 = ax.bar(x + width/2, tta,    width, color=colors, edgecolor="black",
                    linewidth=0.5, hatch="//", label="TTA", alpha=0.85)

        ymax = max(max(no_tta), max(tta)) if (no_tta or tta) else 1
        for bar, val in zip(b1, no_tta):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + ymax * 0.015,
                    f"{val:.3f}", ha="center", fontsize=8)
        for bar, val, a in zip(b2, tta, no_tta):
            d = val - a
            sign = "+" if d >= 0 else ""
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + ymax * 0.015,
                    f"{val:.3f}\n({sign}{d:.3f})",
                    ha="center", fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_title(mlabel, fontsize=12)
        ax.set_ylim(0, ymax * 1.35)
        ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(f"Test-Time Augmentation vs Standard Inference — {DATASET}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "16_tta_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # Console table (so you don't need to open the PNG to read the numbers)
    print()
    print(f"  TTA vs no-TTA — {DATASET}")
    print(f"  {'Variant':<10s} {'Metric':<15s} {'no-TTA':>10s} {'TTA':>10s} {'Δ':>10s}")
    print("  " + "-" * 58)
    all_metric_keys = ("mAP@0.5", "mAP@0.75", "weighted_f1", "avg_precision", "avg_recall")
    for key in keys:
        orig, tta, _, _ = pairs[key]
        short = key.upper() if key != "baseline" else "Baseline"
        for mkey in all_metric_keys:
            a, b = orig.get(mkey, 0), tta.get(mkey, 0)
            print(f"  {short:<10s} {mkey:<15s} {a:>10.4f} {b:>10.4f} {b-a:>+10.4f}")
        print("  " + "-" * 58)


# =====================================================================
# Plot set: per-technique separate comparisons (baseline vs one method)
# =====================================================================

def plot_per_technique_comparisons():
    """
    For each anti-aliasing technique (wavelet, blurpool, gaussian), generate
    a dedicated set of comparison plots against baseline saved to a subdirectory.
    This makes it easy to compare each technique independently.
    """
    technique_keys = {
        "wavelet":  "A",
        "blurpool": "blurpool",
        "gaussian": "gaussian",
    }

    for tech_name, variant_key in technique_keys.items():
        if variant_key not in avail_variants or "baseline" not in avail_variants:
            continue

        subdir = os.path.join(PLOT_DIR, tech_name)
        os.makedirs(subdir, exist_ok=True)

        pair = {
            "baseline": avail_variants["baseline"],
            variant_key: avail_variants[variant_key],
        }
        pair_tm = {k: test_metrics[k] for k in pair}
        pair_h  = {k: histories[k]    for k in pair}

        # ── mAP / F1 bar comparison ───────────────────────────────
        metric_specs = [
            ("mAP@0.5",     "mAP @ IoU=0.5"),
            ("mAP@0.75",    "mAP @ IoU=0.75"),
            ("weighted_f1", "Weighted F1"),
            ("avg_precision","Avg Precision"),
            ("avg_recall",  "Avg Recall"),
        ]
        fig, axes = plt.subplots(1, len(metric_specs), figsize=(18, 5))
        vnames_p = [k.upper() if k != "baseline" else "Baseline" for k in pair]
        vcols_p  = [pair[k][1] for k in pair]
        base_tm  = pair_tm["baseline"]

        for ax, (mkey, mlabel) in zip(axes, metric_specs):
            vals = [pair_tm[k].get(mkey, 0) for k in pair]
            bars = ax.bar(vnames_p, vals, color=vcols_p,
                          edgecolor="black", linewidth=0.5, width=0.45)
            for bar, val, k in zip(bars, vals, pair):
                delta = val - base_tm.get(mkey, 0)
                sign = "+" if delta >= 0 else ""
                ann = f"{val:.3f}\n({sign}{delta*100:.1f}%)" if k != "baseline" else f"{val:.3f}"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.03,
                        ann, ha="center", fontsize=9, fontweight="bold")
            ax.set_title(mlabel, fontsize=10)
            ax.set_ylim(0, max(vals) * 1.4 if vals else 1)

        tech_label = LABELS.get(variant_key, variant_key)
        fig.suptitle(f"Baseline vs {tech_label}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(subdir, "metrics_comparison.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # ── Training & validation loss curves ────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for k, (_, color, label) in pair.items():
            h = pair_h[k]
            eps = [e["epoch"] for e in h]
            axes[0].plot(eps, [e["train_loss"] for e in h], color=color, label=label, marker="o", markersize=3)
            axes[1].plot(eps, [e["val_loss"]   for e in h], color=color, label=label, marker="s", markersize=3)
        axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[1].set_title("Validation Loss"); axes[1].set_xlabel("Epoch")
        for ax in axes:
            ax.legend(fontsize=9)
        fig.suptitle(f"Loss Curves — Baseline vs {tech_label}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(subdir, "loss_curves.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # ── Per-class AP@0.5 ─────────────────────────────────────
        class_names = list(CLASS_NAMES_FOR_PLOTS)
        nc = len(class_names)
        # For many-class datasets, show the 20 most-frequent classes
        MAX_CLASSES = 20
        if nc > MAX_CLASSES:
            gt_counts = np.zeros(nc, dtype=int)
            for k in pair:
                pc = pair_tm[k].get("per_class", {})
                for c in range(nc):
                    gt_counts[c] += pc.get(str(c), pc.get(c, {})).get("n_gt", 0)
            top_idx = sorted(np.argsort(-gt_counts)[:MAX_CLASSES].tolist())
            class_names = [class_names[c] for c in top_idx]
            nc = len(class_names)
        else:
            top_idx = list(range(nc))
        x = np.arange(nc)
        w = 0.35
        fig, ax = plt.subplots(figsize=(max(9, nc * 0.5), 5))
        for i, (k, (_, color, label)) in enumerate(pair.items()):
            pc = pair_tm[k].get("per_class", {})
            vals = [pc.get(str(c), pc.get(c, {})).get("ap50", 0) for c in top_idx]
            short = k.upper() if k != "baseline" else "Baseline"
            bars = ax.bar(x + i * w, vals, w, label=short, color=color,
                          edgecolor="black", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01, f"{val:.2f}",
                            ha="center", fontsize=8)
        ax.set_xticks(x + w / 2)
        rotation = 45 if nc > 8 else 15
        ax.set_xticklabels([n[:18] for n in class_names], rotation=rotation,
                           ha="right" if rotation == 45 else "center")
        ax.set_ylabel("AP @ IoU=0.5")
        ax.set_title(f"Per-Class AP@0.5 — Baseline vs {tech_label}")
        ax.legend()
        ax.set_ylim(0, 1.15)
        plt.tight_layout()
        plt.savefig(os.path.join(subdir, "perclass_ap50.png"), dpi=150)
        plt.close()

        print(f"  Saved per-technique plots → {subdir}/")


# =====================================================================
# Main
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate plots for a trained dataset")
    parser.add_argument("--dataset", type=str, default="moth",
                        choices=list(cfg.DATASET_CONFIGS.keys()))
    args = parser.parse_args()
    init_dataset(args.dataset)

    print("=" * 60)
    print(f"  Generating Result Plots  (dataset: {args.dataset})")
    print(f"  Output dir: {PLOT_DIR}")
    print(f"  Available variants: {list(avail_variants.keys())}")
    print("=" * 60)

    if not avail_variants:
        print(f"  No trained variants found for '{args.dataset}'. Train at least one variant first.")
        return

    plot_training_loss()
    plot_validation_loss()
    plot_train_vs_val()
    plot_lr_schedule()
    plot_test_metrics_bar()
    plot_aliasing_scores()
    plot_convergence_speed()
    plot_dashboard()
    plot_results_table()
    plot_map_comparison()
    plot_per_class_metrics()
    plot_equivariance()
    plot_train_val_map_curves()
    plot_sample_detections()
    plot_antialiasing_summary()
    plot_per_technique_comparisons()
    plot_tta_comparison()

    print(f"\n  All plots saved to: {PLOT_DIR}/")
    print(f"  Per-technique plots: {PLOT_DIR}/wavelet/  blurpool/  gaussian/  (only those that have trained data)")
    print(f"  Dataset: {DATASET}")
    print("  Done.")


if __name__ == "__main__":
    main()
