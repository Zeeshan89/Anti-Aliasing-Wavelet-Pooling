"""
Scale-stratified AP table (AP_small / AP_medium / AP_large at IoU 0.5) from the
runs/scale_ap/*.json produced by scale_ap.py.  Emits LaTeX + a text preview.

Read-only over runs/scale_ap/; writes only a LaTeX snippet under runs/plots/paper/.
Usage:  python make_scale_table.py
"""
import json
from pathlib import Path
import config as cfg

RUNS = Path(cfg.RUNS_DIR)
OUT = RUNS / "plots" / "paper"
DATASETS = ["dior", "dota", "ip102"]
DLAB = {"dior": "DIOR", "dota": "DOTA v1.5", "ip102": "IP102"}
VARIANTS = ["baseline", "a", "blurpool", "gaussian", "blurpool_small", "gaussian_small"]
VLAB = {"baseline": "Baseline", "a": "Wavelet (Coif3)", "blurpool": "BlurPool (3-tap)",
        "gaussian": r"Gaussian (7$\times$7)", "blurpool_small": "BlurPool (2-tap)",
        "gaussian_small": r"Gaussian (3$\times$3)"}
BUCKETS = ["small", "medium", "large"]


def load(ds):
    p = RUNS / "scale_ap" / f"{ds}_scale_ap.json"
    return json.load(open(p)) if p.exists() else None


def cell(v):
    return "--" if v is None else f"{v:.3f}"


def latex():
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
             r"\begin{tabular}{ll ccc}", r"\toprule",
             r"Dataset & Variant & AP$_{\mathrm{S}}$ & AP$_{\mathrm{M}}$ & AP$_{\mathrm{L}}$ \\",
             r"\midrule"]
    for di, ds in enumerate(DATASETS):
        d = load(ds)
        if d is None:
            continue
        census = d.get("gt_counts_by_bucket", {})
        dv = d["variants"]
        present = [v for v in VARIANTS if v in dv]
        for i, v in enumerate(present):
            ap = dv[v]["AP@0.5"]
            dcell = r"\multirow{%d}{*}{%s}" % (len(present), DLAB[ds]) if i == 0 else ""
            lines.append(f"{dcell} & {VLAB[v]} & {cell(ap['small'])} & {cell(ap['medium'])} "
                         f"& {cell(ap['large'])} \\\\")
        # census footnote row
        lines.append(r"\cmidrule(l){2-5}")
        lines.append(r"\multicolumn{2}{l}{\footnotesize GT objects} & "
                     f"\\footnotesize {census.get('small',0):,} & "
                     f"\\footnotesize {census.get('medium',0):,} & "
                     f"\\footnotesize {census.get('large',0):,} \\\\")
        if di != len(DATASETS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Scale-stratified AP@0.5 (COCO small/medium/large: side "
              r"$<32$, $32$--$96$, $>96$ px at the 640 input). Recomputed from the "
              r"released checkpoints; the sample-weighted mean over buckets reproduces "
              r"each model's overall mAP@0.5 to within 0.002. IP102 contains no small "
              r"objects. Within DIOR and DOTA the standard filters lower AP$_{\mathrm{S}}$ "
              r"and raise AP$_{\mathrm{L}}$ (see Fig.~\ref{fig:scaleheat}).}",
              r"\label{tab:scaleap}", r"\end{table}"]
    return "\n".join(lines)


def preview():
    print(f"{'dataset':10s} {'variant':16s} {'AP_S':>7s} {'AP_M':>7s} {'AP_L':>7s}")
    for ds in DATASETS:
        d = load(ds)
        if d is None:
            print(f"{ds}: (no json)"); continue
        for v in VARIANTS:
            if v not in d["variants"]:
                continue
            ap = d["variants"][v]["AP@0.5"]
            def f(x):
                return f"{x:7.3f}" if x is not None else f"{'n/a':>7s}"
            print(f"{DLAB[ds][:10]:10s} {VLAB[v][:16]:16s} {f(ap['small'])} {f(ap['medium'])} {f(ap['large'])}")
        c = d.get("gt_counts_by_bucket", {})
        print(f"{'':10s} {'(GT counts)':16s} {c.get('small',0):7,d} {c.get('medium',0):7,d} {c.get('large',0):7,d}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    preview()
    (OUT / "tab_scale_ap.tex").write_text(latex())
    print(f"\nsaved LaTeX: {OUT/'tab_scale_ap.tex'}")


if __name__ == "__main__":
    main()
