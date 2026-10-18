"""
Lightweight generalization check on the SECOND detector (YOLOv10-L, Ultralytics),
reusing the already-stored per-class metrics (no re-inference).

The YOLOv8-S result is that anti-aliasing's benefit rises with object size (it hurts
small objects, helps large ones).  Here we test whether the SAME size-dependence
appears on YOLOv10-L DOTA at the per-class level: correlate each class's AA-induced
change in AP with that class's median object size.  A positive correlation means the
size-dependence generalizes across architectures, even if the size-composition of DOTA
makes the class-averaged aggregate small or slightly positive (the apparent
contradiction the paper must address).

Read-only over runs/ and dataset labels.  Usage:  python yolov10_generalization.py
"""
import os, json
from pathlib import Path
import numpy as np
import config as cfg

RUNS = Path(cfg.RUNS_DIR)
VARIANTS = ["a", "blurpool", "gaussian", "blurpool_small", "gaussian_small"]
VLAB = {"a": "Wavelet", "blurpool": "BlurPool", "gaussian": "Gaussian",
        "blurpool_small": "BlurPool-sm", "gaussian_small": "Gaussian-sm"}


def load_perclass(variant):
    p = RUNS / f"yolov10l_dota_{variant}" / "test_metrics.json"
    if not p.exists():
        return None, None
    d = json.load(open(p))
    pc = {int(k): v["ap50_95"] for k, v in d["per_class"].items()}
    return d["mAP@0.5"], pc


def class_median_sizes():
    """class id -> median sqrt(w*h)*640 over DOTA test labels."""
    lbl = Path(cfg.load_dataset_cfg("dota")["split_dir"]) / "test" / "labels"
    per = {}
    for f in lbl.glob("*.txt"):
        try:
            for line in open(f):
                s = line.split()
                if len(s) == 5:
                    c = int(s[0]); w, h = float(s[3]), float(s[4])
                    per.setdefault(c, []).append(np.sqrt(w * h) * cfg.INPUT_SIZE)
        except Exception:
            continue
    return {c: float(np.median(v)) for c, v in per.items()}


def pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    base_map, base_pc = load_perclass("baseline")
    if base_pc is None:
        print("no yolov10 baseline metrics"); return
    sizes = class_median_sizes()
    classes = sorted(base_pc.keys())

    print(f"YOLOv10-L DOTA baseline mAP@0.5 = {base_map:.4f}\n")
    print(f"{'variant':13s} {'ΔmAP@0.5':>9s}   {'r(Δap, class size)':>20s}   {'n_classes':>9s}")
    rows = []
    for v in VARIANTS:
        vmap, vpc = load_perclass(v)
        if vpc is None:
            continue
        cs = [c for c in classes if c in vpc and c in sizes]
        d_ap = [vpc[c] - base_pc[c] for c in cs]
        sz = [sizes[c] for c in cs]
        r = pearson(sz, d_ap)
        rows.append((v, vmap - base_map, r, len(cs)))
        print(f"{VLAB[v]:13s} {vmap - base_map:+9.4f}   {r:>20.3f}   {len(cs):>9d}")

    # pooled across the 3 standard filters (a, blurpool, gaussian)
    main3 = ["a", "blurpool", "gaussian"]
    sz_all, d_all = [], []
    for v in main3:
        _, vpc = load_perclass(v)
        if vpc is None:
            continue
        for c in classes:
            if c in vpc and c in sizes:
                sz_all.append(sizes[c]); d_all.append(vpc[c] - base_pc[c])
    print(f"\npooled over standard filters (Wavelet+BlurPool+Gaussian): "
          f"r(Δap, class size) = {pearson(sz_all, d_all):+.3f}  (n={len(sz_all)})")
    print("Interpretation: r>0 means the AA benefit rises with object size on YOLOv10-L too,\n"
          "generalizing the YOLOv8-S per-scale law across architectures.")


if __name__ == "__main__":
    main()
