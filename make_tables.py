"""
Consolidated absolute-results tables from the existing test_metrics.json files.

The paper (Sec 4.5) promises mAP@0.5, mAP@0.75, weighted-F1, average precision and
average recall, but reports numbers only inline.  This emits complete LaTeX tables
(clean and augmented) plus a plain-text preview.  Read-only over runs/*/; writes
only LaTeX snippets under runs/plots/paper/.

Usage:  python make_tables.py
"""
import os, json
from pathlib import Path
import config as cfg

RUNS = Path(cfg.RUNS_DIR)
OUT = RUNS / "plots" / "paper"

DATASETS = ["dior", "dota", "ip102"]
VARIANTS = ["baseline", "a", "blurpool", "gaussian", "blurpool_small", "gaussian_small"]
DLAB = {"dior": "DIOR", "dota": "DOTA v1.5", "ip102": "IP102"}
VLAB = {"baseline": "Baseline", "a": "Wavelet (Coif3)", "blurpool": "BlurPool (3-tap)",
        "gaussian": r"Gaussian (7$\times$7)", "blurpool_small": "BlurPool (2-tap)",
        "gaussian_small": r"Gaussian (3$\times$3)"}
# (json key, latex header, plain header)
METRICS = [("mAP@0.5", r"mAP$_{.5}$", "mAP.5"), ("mAP@0.75", r"mAP$_{.75}$", "mAP.75"),
           ("weighted_f1", "wF1", "wF1"), ("avg_precision", "P", "P"), ("avg_recall", "R", "R")]


def load(dataset, variant, aug=False):
    name = f"{dataset}_aug_{variant}" if aug else f"{dataset}_{variant}"
    p = RUNS / name / "test_metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def collect(aug):
    """dataset -> variant -> {metric_key: value}."""
    out = {}
    for ds in DATASETS:
        out[ds] = {}
        for v in VARIANTS:
            m = load(ds, v, aug)
            if m is None:
                continue
            out[ds][v] = {k: m.get(k) for k, _, _ in METRICS}
    return out


def best_per_metric(rows):
    """Return {metric_key: best_value} across variants in one dataset block."""
    best = {}
    for k, _, _ in METRICS:
        vals = [r[k] for r in rows.values() if r.get(k) is not None]
        best[k] = max(vals) if vals else None
    return best


def latex_table(data, aug):
    cap = ("Augmented" if aug else "Clean") + " test results (custom evaluator, stretched-640)."
    lab = "tab:absresults_aug" if aug else "tab:absresults"
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{ll" + "c" * len(METRICS) + "}",
             r"\toprule",
             "Dataset & Variant & " + " & ".join(h for _, h, _ in METRICS) + r" \\",
             r"\midrule"]
    for di, ds in enumerate(DATASETS):
        rows = data[ds]
        if not rows:
            continue
        best = best_per_metric(rows)
        present = [v for v in VARIANTS if v in rows]
        for i, v in enumerate(present):
            dcell = r"\multirow{%d}{*}{%s}" % (len(present), DLAB[ds]) if i == 0 else ""
            cells = []
            for k, _, _ in METRICS:
                val = rows[v].get(k)
                if val is None:
                    cells.append("--"); continue
                s = f"{val:.3f}"
                if best[k] is not None and abs(val - best[k]) < 1e-9:
                    s = r"\textbf{%s}" % s
                cells.append(s)
            lines.append(f"{dcell} & {VLAB[v]} & " + " & ".join(cells) + r" \\")
        if di != len(DATASETS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{%s Best per column per dataset in \textbf{bold}.}" % cap,
              r"\label{%s}" % lab, r"\end{table}"]
    return "\n".join(lines)


def text_preview(data, title):
    w = 22
    print(f"\n===== {title} =====")
    print(f"{'dataset/variant':<28s}" + "".join(f"{h:>9s}" for _, _, h in METRICS))
    for ds in DATASETS:
        rows = data[ds]
        if not rows:
            continue
        base = rows.get("baseline", {})
        for v in VARIANTS:
            if v not in rows:
                continue
            cells = ""
            for k, _, _ in METRICS:
                val = rows[v].get(k)
                cells += f"{val:>9.3f}" if val is not None else f"{'--':>9s}"
            # delta on mAP@0.5 vs baseline
            d = ""
            if v != "baseline" and base.get("mAP@0.5") is not None and rows[v].get("mAP@0.5") is not None:
                d = f"   (Δ.5 {rows[v]['mAP@0.5'] - base['mAP@0.5']:+.3f})"
            print(f"{DLAB[ds][:12]:<12s} {VLAB[v][:15]:<15s}" + cells + d)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clean = collect(aug=False)
    aug = collect(aug=True)

    text_preview(clean, "CLEAN (main experiments)")
    text_preview(aug, "AUGMENTED (robustness)")

    (OUT / "tab_absolute_results.tex").write_text(latex_table(clean, aug=False))
    (OUT / "tab_absolute_results_aug.tex").write_text(latex_table(aug, aug=True))
    print(f"\nsaved LaTeX: {OUT/'tab_absolute_results.tex'}")
    print(f"saved LaTeX: {OUT/'tab_absolute_results_aug.tex'}")


if __name__ == "__main__":
    main()
