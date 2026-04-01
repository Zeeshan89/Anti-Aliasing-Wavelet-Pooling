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
}
LABELS = {
    "baseline": "Baseline (YOLOv8-S)",
    "A":        "Variant A (+ WaveletPool)",
    "B":        "Variant B (Bottom-Heavy + WaveletPool)",
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


variants = {
    "baseline": ("moth_baseline", COLORS["baseline"], LABELS["baseline"]),
    "A":        ("moth_a",        COLORS["A"],        LABELS["A"]),
    "B":        ("moth_b",        COLORS["B"],        LABELS["B"]),
}

histories = {}
test_metrics = {}
for key, (dirname, _, _) in variants.items():
    histories[key] = load_history(dirname)
    test_metrics[key] = load_test_metrics(dirname)


# =====================================================================
# Plot 1:  Training Loss Curves (all variants overlaid)
# =====================================================================

def plot_training_loss():
    fig, ax = plt.subplots(figsize=(10, 6))

    for key, (dirname, color, label) in variants.items():
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

    for key, (dirname, color, label) in variants.items():
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
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for i, (key, (dirname, color, label)) in enumerate(variants.items()):
        ax = axes[i]
        h = histories[key]
        epochs = [e["epoch"] for e in h]
        train_loss = [e["train_loss"] for e in h]
        val_loss = [e["val_loss"] for e in h]

        ax.plot(epochs, train_loss, color=color, label="Train", linestyle="-", marker="o", markersize=3)
        ax.plot(epochs, val_loss, color=color, label="Val", linestyle="--", marker="s", markersize=3, alpha=0.7)
        ax.fill_between(epochs, train_loss, val_loss, color=color, alpha=0.1)

        ax.set_xlabel("Epoch")
        if i == 0:
            ax.set_ylabel("Loss")
        ax.set_title(label)
        ax.legend(loc="upper right")

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

    for key, (dirname, color, label) in variants.items():
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
        for key, (dirname, color, label) in variants.items():
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
    scores_path = os.path.join(cfg.RUNS_DIR, "aliasing_analysis", "aliasing_scores.txt")
    if not os.path.exists(scores_path):
        print("  [SKIP] aliasing_scores.txt not found")
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
    ax.axvline(x=threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold}")

    # value labels
    for bar, score in zip(bars, scores_sorted):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", fontsize=10)

    ax.set_xlabel("Aliasing Score")
    ax.set_title("Per-Layer Aliasing Score (FFT Energy Above Nyquist)")
    ax.legend(loc="lower right")

    # add annotation
    ax.annotate("RED = replaced with WaveletPool", xy=(0.98, 0.02), xycoords="axes fraction",
                ha="right", fontsize=9, fontstyle="italic", color="#c62828")

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
    for key, (dirname, color, label) in variants.items():
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
    for key, (dirname, color, label) in variants.items():
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
    for key, (dirname, color, label) in variants.items():
        h = histories[key]
        epochs = [e["epoch"] for e in h]
        ax2.plot(epochs, [e["val_loss"] for e in h], color=color, label=label.split("(")[0].strip(), marker="s", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Validation Loss")
    ax2.legend(fontsize=8)

    # ── Panel 3: Test Total Loss Bar ──────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    vnames = []
    vvals = []
    vcols = []
    for key, (dirname, color, label) in variants.items():
        vnames.append(key.upper() if key != "baseline" else "Baseline")
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
    cls_vals = [test_metrics[key]["cls_loss"] for key in variants]
    bars = ax4.bar(vnames, cls_vals, color=vcols, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, cls_vals):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax4.set_ylabel("Loss")
    ax4.set_title("Test Classification Loss")
    ax4.set_ylim(0, max(cls_vals) * 1.35)

    # ── Panel 5: Test Reg Loss Bar ────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    reg_vals = [test_metrics[key]["reg_loss"] for key in variants]
    bars = ax5.bar(vnames, reg_vals, color=vcols, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, reg_vals):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax5.set_ylabel("Loss")
    ax5.set_title("Test Regression Loss")
    ax5.set_ylim(0, max(reg_vals) * 1.35)

    # ── Panel 6: LR Schedule ─────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    for key, (dirname, color, label) in variants.items():
        h = histories[key]
        ax6.plot([e["epoch"] for e in h], [e["lr"] for e in h], color=color,
                label=key.upper() if key != "baseline" else "Baseline", marker=".", markersize=3)
    ax6.set_xlabel("Epoch")
    ax6.set_ylabel("LR")
    ax6.set_title("Learning Rate Schedule")
    ax6.legend(fontsize=8)

    # ── Panel 7: Convergence ──────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 1])
    for key, (dirname, color, label) in variants.items():
        h = histories[key]
        val_loss = [e["val_loss"] for e in h]
        cum_min = []
        best = float("inf")
        for v in val_loss:
            best = min(best, v)
            cum_min.append(best)
        ax7.plot([e["epoch"] for e in h], cum_min, color=color,
                label=key.upper() if key != "baseline" else "Baseline", marker="D", markersize=3)
    ax7.set_xlabel("Epoch")
    ax7.set_ylabel("Best Val Loss")
    ax7.set_title("Convergence Speed")
    ax7.legend(fontsize=8)

    # ── Panel 8: Aliasing Scores ──────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 2])
    scores_path = os.path.join(cfg.RUNS_DIR, "aliasing_analysis", "aliasing_scores.txt")
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
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    # build table data
    col_labels = ["Variant", "Train Loss\n(final)", "Val Loss\n(final)", "Test Loss", "Test Cls Loss", "Test Reg Loss"]
    cell_data = []
    cell_colors = []

    for key, (dirname, color, label) in variants.items():
        h = histories[key]
        tm = test_metrics[key]
        row = [
            label,
            f"{h[-1]['train_loss']:.4f}",
            f"{h[-1]['val_loss']:.4f}",
            f"{tm['loss']:.4f}",
            f"{tm['cls_loss']:.4f}",
            f"{tm['reg_loss']:.4f}",
        ]
        cell_data.append(row)
        cell_colors.append([color + "30"] * len(col_labels))  # light tint

    table = ax.table(cellText=cell_data, colLabels=col_labels, cellColours=cell_colors,
                     colColours=["#E0E0E0"] * len(col_labels),
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)

    # bold the best values in each metric column
    for col_idx in range(3, 6):
        vals = [float(cell_data[r][col_idx]) for r in range(len(cell_data))]
        best_row = vals.index(min(vals))
        cell = table[best_row + 1, col_idx]
        cell.set_text_props(fontweight="bold")
        cell.set_facecolor("#C8E6C9")  # highlight green

    ax.set_title("Results Summary Table", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "09_results_table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 60)
    print("  Generating Result Plots")
    print("=" * 60)

    plot_training_loss()
    plot_validation_loss()
    plot_train_vs_val()
    plot_lr_schedule()
    plot_test_metrics_bar()
    plot_aliasing_scores()
    plot_convergence_speed()
    plot_dashboard()
    plot_results_table()

    print(f"\n  All plots saved to: {PLOT_DIR}/")
    print("  Done.")


if __name__ == "__main__":
    main()
