"""
system_model.py .standalone generator for the paper's system-model figure
(Figure 6 / fig_06_pipeline).

Kept separate from paper_figures.py so the system model can be edited and
re-rendered on its own, without regenerating all 12 figures:

    python system_model.py

Writes runs/plots/paper/fig_06_pipeline.png (300 DPI) and .pdf (vector),
overwriting the versions used by the LaTeX build.  Figure size is kept near
the printed \\textwidth so the ~12 pt labels stay readable in the paper.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from matplotlib.lines import Line2D


# ─────────────────────────────────────────────────────────────────────
#  Style + output (self-contained; mirrors paper_figures.py)
# ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "runs" / "plots" / "paper"
# real dataset sample used as the input thumbnail (DIOR aerial: dense small objects)
INPUT_IMAGE = PROJECT_ROOT / "dataset" / "dior" / "split" / "test" / "images" / "11726.jpg"

STYLE = {"dpi_png": 300}


def apply_style():
    plt.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size":       12,
        "figure.constrained_layout.use": True,
    })


def save_both(fig, stem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=STYLE["dpi_png"], bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {png} + {pdf.name}")


def shade(color, amt):
    """Lighten (amt>0, toward white) or darken (amt<0, toward black) a color."""
    r, g, b = mcolors.to_rgb(color)
    if amt >= 0:
        return (r + (1 - r) * amt, g + (1 - g) * amt, b + (1 - b) * amt)
    a = -amt
    return (r * (1 - a), g * (1 - a), b * (1 - a))


# ═════════════════════════════════════════════════════════════════════
#  System model  (Figure 6 / fig_06_pipeline)
# ═════════════════════════════════════════════════════════════════════
def fig_06_pipeline():
    """Professional system-model diagram of the AA-aware detection pipeline."""
    apply_style()

    # width ~ printed \textwidth so 12 pt labels stay legible after scaling
    fig, ax = plt.subplots(figsize=(9.0, 4.886))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── palette ──────────────────────────────────────────────────
    C = {
        "panel_fc": "#f5f7f9", "panel_ec": "#c3ccd6",
        "ribbon_fc": "#eef3f2", "ribbon_ec": "#bcd0cc",
        "c2f_fc":  "#cdeae5", "c2f_ec":  "#2a9d8f",
        "down_ec": "#7b52a0",
        "aa_fc":   "#ffd2b3", "aa_ec":   "#d84315", "aa_glow": "#ff7043",
        "loss_fc": "#eceff1", "loss_ec": "#8d9aa6",
        "fft_fc":  "#d4e6f7", "fft_ec":  "#3d78b0",
        "sel_fc":  "#ffe0cc", "sel_ec":  "#d84315",
        "det_fc":  "#d8dee4", "det_ec":  "#54606b",
        "ink": "#1f2a33", "arrow": "#3a4750", "muted": "#5e6b75",
        "green": "#2e7d32", "red": "#c62828",
    }
    # light color families graded bottom->top (backbone) and P5->P3 (head)
    BB_FACES = ["#ecf3fb", "#d9e8f6", "#c4d9ef", "#accbe8", "#93bbdf"]   # P1..P5
    BB_MID = "#c4d9ef"
    HEAD_FACES = {"P5": "#c8bce4", "P4": "#d6cdec", "P3": "#e4def4"}
    HEAD_MID = "#d6cdec"

    def panel(x, y, w, h, title, subtitle=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.14",
                     facecolor=C["panel_fc"], edgecolor=C["panel_ec"], linewidth=1.0, zorder=1))
        ax.text(x + 0.22, y + h - 0.22, title, ha="left", va="top",
                fontsize=13, weight="bold", color=C["ink"], zorder=6)
        if subtitle:
            ax.text(x + 0.22, y + h - 0.62, subtitle, ha="left", va="top",
                    fontsize=9.5, color=C["muted"], zorder=6)

    def rbox(x, y, w, h, text, fc, ec, tc=None, fs=12, weight="bold", z=5, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.03,rounding_size=0.09",
                     facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=z, linestyle=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc or C["ink"], weight=weight, zorder=z + 1)

    def cuboid(x, y, w, h, face, dx=0.16, dy=0.24, lw=0.9, z=4):
        top, side, edge = shade(face, 0.45), shade(face, -0.14), shade(face, -0.5)
        ax.add_patch(Polygon([(x + w, y), (x + w + dx, y + dy),
                              (x + w + dx, y + h + dy), (x + w, y + h)],
                             closed=True, facecolor=side, edgecolor=edge, linewidth=lw, zorder=z))
        ax.add_patch(Polygon([(x, y + h), (x + dx, y + h + dy),
                              (x + w + dx, y + h + dy), (x + w, y + h)],
                             closed=True, facecolor=top, edgecolor=edge, linewidth=lw, zorder=z))
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=lw, zorder=z + 1))

    def arrow(x1, y1, x2, y2, color=None, lw=1.7, z=6, ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.32,head_length=0.55",
                                    color=color or C["arrow"], lw=lw, linestyle=ls,
                                    shrinkA=0, shrinkB=0), zorder=z)

    def aa_tag(cx, cy):
        ax.add_patch(mpatches.Circle((cx, cy), 0.30, facecolor=C["aa_glow"],
                                     alpha=0.28, edgecolor="none", zorder=7))
        ax.add_patch(mpatches.Circle((cx, cy), 0.21, facecolor="#fff3ec",
                                     edgecolor=C["aa_ec"], linewidth=1.1, zorder=8))
        ax.text(cx, cy, "AA", ha="center", va="center", fontsize=9.5,
                weight="bold", color=C["aa_ec"], zorder=9)

    # ═══ TOP RIBBON  ..analysis / layer-selection phase ═══════════
    rx, ry, rw, rh = 0.25, 6.05, 13.5, 1.35
    ax.add_patch(FancyBboxPatch((rx, ry), rw, rh,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 facecolor=C["ribbon_fc"], edgecolor=C["ribbon_ec"], linewidth=1.0, zorder=1))
    ax.text(rx + 0.22, ry + rh - 0.16, "Analysis phase   (once per dataset)",
            ha="left", va="top", fontsize=12, weight="bold", color=C["ink"], zorder=6)
    yc = ry + 0.50
    rbox(0.75, yc - 0.36, 1.75, 0.72, "Train\nimages", C["det_fc"], C["det_ec"], fs=11)
    rbox(3.05, yc - 0.36, 1.95, 0.72, "Per-layer\n2-D FFT", C["fft_fc"], C["fft_ec"], fs=11)

    # mini aliasing-score bar chart with Otsu threshold
    bx0, bw, bgap = 5.75, 0.26, 0.12
    bar_h = [0.18, 0.24, 0.30, 0.62, 0.40, 0.72, 0.55]
    thr = 0.45
    base_y = yc - 0.44
    ax.text(bx0 + 3 * (bw + bgap), yc + 0.58, "aliasing score", ha="center",
            va="bottom", fontsize=9.5, color=C["muted"], zorder=6)
    for i, bh in enumerate(bar_h):
        bx = bx0 + i * (bw + bgap)
        above = bh >= thr
        ax.add_patch(Rectangle((bx, base_y), bw, bh,
                     facecolor=C["aa_fc"] if above else "#c7d2db",
                     edgecolor=C["aa_ec"] if above else "#8d9aa6", linewidth=0.7, zorder=5))
    ax.plot([bx0 - 0.06, bx0 + 7 * (bw + bgap) - bgap + 0.02],
            [base_y + thr, base_y + thr], ls="--", color=C["down_ec"], lw=1.3, zorder=6)
    ax.text(bx0 + 7 * (bw + bgap) + 0.04, base_y + thr + 0.07, "Otsu", ha="left", va="bottom",
            fontsize=9.5, color=C["down_ec"], weight="bold", zorder=6)

    rbox(9.15, yc - 0.36, 2.05, 0.72, "Layers above\nthreshold", C["sel_fc"], C["sel_ec"],
         tc=C["aa_ec"], fs=11)
    rbox(11.55, yc - 0.36, 1.95, 0.72, "Insert AA\npre-filter", C["aa_fc"], C["aa_ec"],
         tc=C["aa_ec"], fs=11)
    arrow(2.50, yc, 3.05, yc)
    arrow(5.00, yc, 5.55, yc)
    arrow(8.30, yc, 9.15, yc)
    arrow(11.20, yc, 11.55, yc)

    # ═══ THREE MAIN PANELS ════════════════════════════════════════
    py, ph = 0.95, 4.75
    panel(0.25, py, 4.05, ph, "Backbone", "CSPDarknet  (C2f)")
    panel(4.55, py, 5.05, ph, "Neck", "PANet  (top-down + bottom-up)")
    panel(9.85, py, 3.90, ph, "Head", "decoupled detect")

    # ---- Backbone: real input image + isometric feature pyramid P1..P5 ----
    img_x0, img_x1, img_y0, img_y1 = 0.48, 1.60, 1.05, 2.17
    try:
        img = plt.imread(str(INPUT_IMAGE))
        ax.imshow(img, extent=[img_x0, img_x1, img_y0, img_y1], aspect="auto",
                  zorder=3, interpolation="antialiased")
        ax.add_patch(Rectangle((img_x0, img_y0), img_x1 - img_x0, img_y1 - img_y0,
                     facecolor="none", edgecolor="#5c80a6", linewidth=1.0, zorder=4))
        ax.set_xlim(0, 14); ax.set_ylim(0, 7.6)
    except (FileNotFoundError, OSError):
        rbox(img_x0, img_y0, img_x1 - img_x0, img_y1 - img_y0, "Input\n640$\\times$640",
             "#e8edf1", "#9aa7b2", tc=C["ink"], fs=10)
    # label placed just above the image box (inside the panel, left of P1)
    ax.text((img_x0 + img_x1) / 2, img_y1 + 0.08, "Input image", ha="center", va="bottom",
            fontsize=9, color=C["muted"], weight="bold", zorder=6)

    # pyramid blocks (P1 widest at bottom, narrowing up to P5 = deepest)
    levels = [("P1", 1.45, 2.35, 0.42),
              ("P2", 1.18, 2.82, 0.42),
              ("P3", 0.95, 3.29, 0.46),
              ("P4", 0.74, 3.77, 0.46),
              ("P5", 0.56, 4.25, 0.48)]
    px_center = 2.55
    tap_pts = {}
    for (name, w, by, bh), face in zip(levels, BB_FACES):
        bx = px_center - w / 2
        cuboid(bx, by, w, bh, face)
        ax.text(px_center, by + bh / 2, name, ha="center", va="center",
                fontsize=11, weight="bold", color=C["ink"], zorder=7)
        tap_pts[name] = (bx + w, by + bh / 2)
    arrow(1.60, 1.75, 1.90, 2.45)   # input image -> P1
    # backbone.down3 is the dataset-dependent AA site (e.g. IP102)
    aa_tag(3.15, 3.64)
    ax.text(3.50, 3.64, "down3", ha="left", va="center",
            fontsize=9, color=C["aa_ec"], weight="bold", zorder=9)

    # ---- Neck: PANet with two AA-modified downsampling layers ----
    lane = {"P5": 4.35, "P4": 3.35, "P3": 2.35}
    ncx, ncw, nch = 4.95, 0.95, 0.62
    c2f = {}
    for lvl in ("P5", "P4", "P3"):
        rbox(ncx, lane[lvl] - nch / 2, ncw, nch, "C2f", C["c2f_fc"], C["c2f_ec"], fs=12)
        c2f[lvl] = ncx + ncw
        x0, y0 = tap_pts[lvl]
        arrow(x0, y0, ncx, lane[lvl], lw=1.3)

    # top-down (upsample) path: arrows BETWEEN the stacked C2f blocks
    cxc = ncx + ncw / 2
    for a, b in (("P5", "P4"), ("P4", "P3")):
        arrow(cxc, lane[a] - nch / 2, cxc, lane[b] + nch / 2, color="#8d9aa6", lw=1.4)

    # bottom-up path: two stride-2 downsampling layers with AA pre-filter
    d1x, d1y = 6.65, lane["P4"]     # down p3 -> p4
    d2x, d2y = 7.95, lane["P5"]     # down p4 -> p5
    for (dx, dy, lab) in ((d1x, d1y, "down\n$p3\\!\\to\\!p4$"),
                          (d2x, d2y, "down\n$p4\\!\\to\\!p5$")):
        rbox(dx, dy - 0.31, 0.98, 0.62, lab, C["aa_fc"], C["aa_ec"],
             tc=C["aa_ec"], fs=10.5, ls="--")
        aa_tag(dx + 0.49, dy + 0.50)
    arrow(c2f["P3"], lane["P3"], d1x, d1y - 0.14, color=C["aa_ec"], lw=1.6)
    arrow(d1x + 0.98, d1y + 0.12, d2x, d2y - 0.14, color=C["aa_ec"], lw=1.6)

    # AA-feedback wire from the ribbon into the neck (orthogonal, in the gap band)
    dash = (0, (5, 3))
    ax.add_line(Line2D([12.5, 12.5], [ry, 5.92], color=C["aa_ec"], lw=1.4, ls=dash, zorder=6))
    ax.add_line(Line2D([12.5, 7.45], [5.92, 5.92], color=C["aa_ec"], lw=1.4, ls=dash, zorder=6))
    arrow(7.45, 5.92, 7.45, 5.68, color=C["aa_ec"], lw=1.4, ls="--")
    ax.text(7.60, 5.60, "insert AA on\nselected layers", ha="left", va="top",
            fontsize=9, color=C["aa_ec"], zorder=7)

    # ---- Head: three detect heads (graded family) + loss ----
    for lvl in ("P5", "P4", "P3"):
        rbox(10.05, lane[lvl] - 0.30, 1.75, 0.60, f"Detect {lvl}",
             HEAD_FACES[lvl], shade(HEAD_FACES[lvl], -0.5), tc=C["ink"], fs=12)
    rbox(12.15, 2.85, 1.45, 1.10, "Loss\nCIoU+DFL\n/ BCE", C["loss_fc"], C["loss_ec"],
         tc=C["ink"], fs=10.5)
    for lvl in ("P5", "P4", "P3"):
        arrow(11.80, lane[lvl], 12.15, 3.40, lw=1.0)

    # neck outputs -> head (clean horizontals, each scale to its detect head)
    arrow(d2x + 0.98, lane["P5"], 10.05, lane["P5"], lw=1.3)
    arrow(d1x + 0.98, lane["P4"], 10.05, lane["P4"], lw=1.3)
    arrow(c2f["P3"], lane["P3"], 10.05, lane["P3"], lw=1.3)

    # ═══ BOTTOM STRIP  ..legend + predictive rule ═════════════════
    leg = [("C2f", C["c2f_fc"], C["c2f_ec"]),
           ("Feature map", BB_MID, shade(BB_MID, -0.5)),
           ("Detect head", HEAD_MID, shade(HEAD_MID, -0.5)),
           ("AA pre-filter", C["aa_fc"], C["aa_ec"])]
    lx, ly = 0.35, 0.46
    for txt, fc, ec in leg:
        ax.add_patch(Rectangle((lx, ly - 0.13), 0.34, 0.26, facecolor=fc,
                     edgecolor=ec, linewidth=1.0, zorder=5))
        ax.text(lx + 0.46, ly, txt, ha="left", va="center",
                fontsize=11, color=C["ink"], zorder=5)
        lx += 0.46 + 0.115 * len(txt) + 0.55

    # predictive-rule mini panel (bottom-right)
    prx, prw = 9.85, 3.90
    ax.add_patch(FancyBboxPatch((prx, 0.12), prw, 0.74,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 facecolor="#fbfcfd", edgecolor=C["panel_ec"], linewidth=1.0, zorder=4))
    ax.text(prx + 0.15, 0.63, "Rule:", ha="left", va="center",
            fontsize=11, weight="bold", color=C["ink"], zorder=5)
    ax.text(prx + 0.90, 0.63, "$F = k\\,s$  (filter footprint)", ha="left", va="center",
            fontsize=11, color=C["ink"], zorder=5)
    ax.text(prx + 0.15, 0.31, "$F < q_{25}$:", ha="left", va="center",
            fontsize=10.5, color=C["ink"], zorder=5)
    ax.text(prx + 1.15, 0.31, "AA helps", ha="left", va="center",
            fontsize=10.5, weight="bold", color=C["green"], zorder=5)
    ax.text(prx + 2.30, 0.31, "$F > q_{25}$:", ha="left", va="center",
            fontsize=10.5, color=C["ink"], zorder=5)
    ax.text(prx + 3.30, 0.31, "hurts", ha="left", va="center",
            fontsize=10.5, weight="bold", color=C["red"], zorder=5)

    save_both(fig, "fig_06_pipeline")


if __name__ == "__main__":
    print("Generating system model -> fig_06_pipeline")
    fig_06_pipeline()
    print("Done.")
