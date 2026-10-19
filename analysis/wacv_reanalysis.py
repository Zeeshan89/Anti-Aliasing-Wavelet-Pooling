"""
Re-analysis of the WACV results that needs no training (plan Phase 1.1-1.3).

Inputs (copied from the Mac): runs/runs/{ds}_{variant}/test_metrics.json,
runs/runs/bootstrap_ci/*.json, runs/runs/scale_ap/cache/*.npz, dataset labels.

Outputs (analysis/out/):
  size_stats.json           q05..q95, mean of sqrt(area) at 640 px, train and test splits
  cases.csv                 the 13 dataset x filter cases (F, observed delta, bootstrap CI)
  percentile_sensitivity.csv  rule accuracy for each candidate statistic (Mn63 #1, wJpy)
  uncertainty_band.csv      coverage / accuracy of a help-hurt-uncertain rule vs band width
  bucket_bootstrap.csv      paired image-bootstrap CI of delta AP@0.5 per size bucket
  bucket_regression.json    delta AP vs log2(F / bucket median size): fit + zero crossing

Usage:  python analysis/wacv_reanalysis.py [--B 1000] [--workers 32]
"""

import argparse
import contextlib
import csv
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs", "runs")
OUT = os.path.join(ROOT, "analysis", "out")
IMGSZ = 640

# Paper Table 1: footprint F = k * s at the finest selected layer (s = 8).
FOOTPRINT = {"a": 144, "gaussian": 56, "blurpool": 24, "gaussian_small": 24, "blurpool_small": 16}
CASES = {"ip102": ["a", "blurpool", "gaussian"],
         "dior": list(FOOTPRINT), "dota": list(FOOTPRINT)}
BUCKETS = ["small", "medium", "large"]            # COCO areas at 640: <32^2, <96^2, rest
BUCKET_EDGES = [0, 32, 96, 1e9]
PCTS = list(range(5, 100, 5))


# ──────────────────────────────────────────────────────────────────────
# Object sizes
# ──────────────────────────────────────────────────────────────────────

def object_sizes(ds, split):
    """sqrt(box area) in letterboxed 640 px, from the Ultralytics label cache."""
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset
    data = check_det_dataset(os.path.join(ROOT, "dataset", ds, "split", "data.yaml"))
    with contextlib.redirect_stdout(io.StringIO()):
        d = YOLODataset(img_path=data[split], imgsz=IMGSZ, augment=False, data=data, task="detect")
    out = []
    for lab in d.labels:
        h, w = lab["shape"]
        r = IMGSZ / max(h, w)
        b = lab["bboxes"]
        if len(b):
            out.append(np.sqrt(b[:, 2] * w * b[:, 3] * h) * r)
    return np.concatenate(out)


def size_stats():
    path = os.path.join(OUT, "size_stats.json")
    if os.path.exists(path):
        return json.load(open(path))
    stats = {}
    for ds in CASES:
        stats[ds] = {}
        for split in ("train", "test"):
            s = object_sizes(ds, split)
            st = {f"q{p:02d}": float(np.percentile(s, p)) for p in PCTS}
            st["mean"] = float(s.mean())
            st["n"] = int(s.size)
            st["bucket_median"] = [float(np.median(s[(s >= lo) & (s < hi)])) if
                                   ((s >= lo) & (s < hi)).any() else None
                                   for lo, hi in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:])]
            st["frac_below"] = {str(F): float((s < F).mean()) for F in sorted(set(FOOTPRINT.values()))}
            stats[ds][split] = st
    json.dump(stats, open(path, "w"), indent=1)
    return stats


# ──────────────────────────────────────────────────────────────────────
# Per-bucket AP from the cached detections (all-point interpolation;
# reproduces runs/runs/scale_ap/*.json exactly)
# ──────────────────────────────────────────────────────────────────────

def load_bucket_dets(ds, v):
    """Per (bucket, class): image index and TP flag of detections, sorted by conf."""
    z = np.load(os.path.join(RUNS, "scale_ap", "cache", f"{ds}_{v}.npz"))
    cls, conf, img, ob, mb = z["cls"], z["conf"], z["img"], z["ob"], z["mb50"]
    order = np.argsort(-conf, kind="stable")
    cls, img, ob, mb = cls[order], img[order], ob[order], mb[order]
    out = {}
    for b in range(3):
        imgs, tps, grp = [], [], []
        sel = (mb == b) | ((mb == -1) & (ob == b))
        for c in np.unique(cls[sel]):
            m = np.where(sel & (cls == c))[0]
            tp = (mb[m] == b)
            # Detections after the last TP never change all-point AP (recall is flat
            # there), for any bootstrap weights, so drop them.
            last = np.flatnonzero(tp)
            k = last[-1] + 1 if last.size else 0
            if k:
                imgs.append(img[m[:k]])
                tps.append(tp[:k].astype(np.float64))
                grp.append(np.full(k, c, np.int64))
        if imgs:
            g = np.concatenate(grp)
            starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
            gi = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(g)]))  # group index
            # rank offset: earlier groups exceed later ones, so a reversed running max
            # never carries across a class boundary (prec is in [0, 1])
            rank = (len(starts) - 1 - gi) * 2.0
            n = len(g)
            bufs = {k: np.empty(n) for k in ("ww", "a", "ctp", "call", "tmp")}
            out[b] = (np.concatenate(imgs), np.concatenate(tps), g, starts, gi, rank, bufs)
    return out, z["ngt"]


def bucket_ap(dets, ngt, w):
    """Class-averaged all-point AP@0.5 per bucket with image weights w (bootstrap counts).

    Vectorized over classes: AP_c = sum_j (w_j tp_j / n_c) * max_{k>=j, same class} prec_k.
    """
    aps = np.full(3, np.nan)
    n_cls = ngt.shape[1]
    for b in range(3):
        n_c = (ngt[:, :, b] * w[:, None]).sum(0)
        valid = n_c > 0
        if not valid.any():
            continue
        ap_c = np.zeros(n_cls)
        if b in dets:
            # Preallocated buffers: fresh multi-MB allocations per resample were the
            # bottleneck (page-fault cost), not the arithmetic.
            im, tp, g, starts, gi, rank, B_ = dets[b]
            ww, a, ctp, call, tmp = B_["ww"], B_["a"], B_["ctp"], B_["call"], B_["tmp"]
            np.take(w, im, out=ww)
            np.multiply(ww, tp, out=a)
            np.cumsum(a, out=ctp)
            np.cumsum(ww, out=call)
            np.take(np.r_[0.0, ctp[starts[1:] - 1]], gi, out=tmp)
            ctp -= tmp
            np.take(np.r_[0.0, call[starts[1:] - 1]], gi, out=tmp)
            call -= tmp
            np.maximum(call, 1e-12, out=call)
            np.divide(ctp, call, out=tmp)                       # precision
            tmp += rank
            rev = tmp[::-1]
            np.maximum.accumulate(rev, out=rev)                 # reversed running max, in place
            tmp -= rank                                          # envelope
            tmp *= a
            ap_c = np.bincount(g, weights=tmp, minlength=n_cls)
        aps[b] = float(np.mean(ap_c[valid] / n_c[valid]))
    return aps


def _boot_dataset(args):
    ds, B, seed = args
    base, ngt = load_bucket_dets(ds, "baseline")
    variants = {v: load_bucket_dets(ds, v)[0] for v in CASES[ds]}
    n_img = ngt.shape[0]
    ones = np.ones(n_img)
    point = {"baseline": bucket_ap(base, ngt, ones)}
    point.update({v: bucket_ap(d, ngt, ones) for v, d in variants.items()})
    rng = np.random.default_rng(seed)
    deltas = {v: [] for v in variants}
    for _ in range(B):
        w = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img).astype(np.float64)
        ab = bucket_ap(base, ngt, w)
        for v, d in variants.items():
            deltas[v].append(bucket_ap(d, ngt, w) - ab)
    rows = []
    for v in variants:
        dv = np.array(deltas[v])
        for b, name in enumerate(BUCKETS):
            if np.isnan(point["baseline"][b]):
                continue
            lo, hi = np.nanpercentile(dv[:, b], [2.5, 97.5])
            rows.append({"dataset": ds, "variant": v, "bucket": name,
                         "ap_base": round(point["baseline"][b], 4),
                         "ap_var": round(point[v][b], 4),
                         "delta": round(point[v][b] - point["baseline"][b], 4),
                         "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                         "excludes_0": bool(lo > 0 or hi < 0),
                         "n_gt": int(ngt[:, :, b].sum())})
    return rows


# ──────────────────────────────────────────────────────────────────────
# Rule analyses
# ──────────────────────────────────────────────────────────────────────

def cases_table():
    rows = []
    for ds, vs in CASES.items():
        base = json.load(open(os.path.join(RUNS, f"{ds}_baseline", "test_metrics.json")))["mAP@0.5"]
        ci = json.load(open(os.path.join(RUNS, "bootstrap_ci", f"{ds}_ci.json")))["variants"]
        for v in vs:
            m = json.load(open(os.path.join(RUNS, f"{ds}_{v}", "test_metrics.json")))["mAP@0.5"]
            rows.append({"dataset": ds, "variant": v, "F": FOOTPRINT[v],
                         "delta": round(m - base, 4),
                         "boot_delta": ci[v]["delta_mAP"],
                         "boot_ci_lo": ci[v]["delta_ci95"][0], "boot_ci_hi": ci[v]["delta_ci95"][1]})
    return rows


def rule_accuracy(cases, stats, key, split):
    correct = 0
    for c in cases:
        pred = 1 if c["F"] <= stats[c["dataset"]][split][key] else -1
        correct += pred == np.sign(c["delta"])
    return correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--skip-bootstrap", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    stats = size_stats()
    print("q25 (train / test) at 640 px:",
          {ds: (round(stats[ds]["train"]["q25"], 1), round(stats[ds]["test"]["q25"], 1)) for ds in CASES},
          " paper: dior 13, dota 26, ip102 288")

    cases = cases_table()
    with open(os.path.join(OUT, "cases.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cases[0]))
        w.writeheader()
        w.writerows(cases)

    # 1.1 percentile sensitivity
    rows = []
    for split in ("train", "test"):
        for key in [f"q{p:02d}" for p in PCTS] + ["mean"]:
            rows.append({"split": split, "statistic": key,
                         **{ds: round(stats[ds][split][key], 1) for ds in CASES},
                         "correct_of_13": rule_accuracy(cases, stats, key, split)})
    with open(os.path.join(OUT, "percentile_sensitivity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nRule accuracy by statistic (train split):")
    print("  " + "  ".join(f"{r['statistic']}:{r['correct_of_13']}" for r in rows if r["split"] == "train"))

    # Continuous predictor: fraction of objects smaller than F
    from scipy.stats import spearmanr
    frac = [stats[c["dataset"]]["train"]["frac_below"][str(c["F"])] for c in cases]
    rho, p = spearmanr(frac, [c["delta"] for c in cases])
    print(f"Spearman(frac objects < F, delta mAP) over 13 cases: rho={rho:.2f} p={p:.3f}")

    # 1.3 uncertainty band on log2(F / q25)
    band = []
    for delta in np.round(np.arange(0, 1.6, 0.1), 2):
        conf_correct = conf_n = 0
        for c in cases:
            r = np.log2(c["F"] / stats[c["dataset"]]["train"]["q25"])
            if abs(r) < delta:
                continue
            conf_n += 1
            conf_correct += (1 if r <= 0 else -1) == np.sign(c["delta"])
        band.append({"band_log2": delta, "band_factor": round(2 ** delta, 2), "confident": conf_n,
                     "correct": conf_correct, "accuracy": round(conf_correct / conf_n, 3) if conf_n else None})
    with open(os.path.join(OUT, "uncertainty_band.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(band[0]))
        w.writeheader()
        w.writerows(band)
    print("\nUncertainty band |log2(F/q25)| < d  ->  confident calls / correct:")
    print("  " + "  ".join(f"d={b['band_log2']}:{b['correct']}/{b['confident']}" for b in band))

    if args.skip_bootstrap:
        return

    # 1.2 per-bucket paired bootstrap + coarse regression
    with ProcessPoolExecutor(args.workers) as ex:
        rows = [r for rs in ex.map(_boot_dataset, [(ds, args.B, i) for i, ds in enumerate(CASES)]) for r in rs]
    with open(os.path.join(OUT, "bucket_bootstrap.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nPer-bucket delta AP@0.5 with paired image-bootstrap 95% CI:")
    for r in rows:
        print(f"  {r['dataset']:6}{r['variant']:16}{r['bucket']:7} {r['delta']:+.4f} "
              f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]{' *' if r['excludes_0'] else ''}")

    x, y, wts = [], [], []
    for r in rows:
        med = stats[r["dataset"]]["test"]["bucket_median"][BUCKETS.index(r["bucket"])]
        x.append(np.log2(FOOTPRINT[r["variant"]] / med))
        y.append(r["delta"])
        wts.append(1 / max((r["ci_hi"] - r["ci_lo"]) / 3.92, 1e-4) ** 2)
    x, y, wts = map(np.array, (x, y, wts))
    slope, icpt = np.polyfit(x, y, 1, w=np.sqrt(wts))
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(x), len(x))
        s, c = np.polyfit(x[i], y[i], 1, w=np.sqrt(wts[i]))
        bs.append((s, -c / s if s != 0 else np.nan))
    bs = np.array(bs)
    res = {"n_points": int(len(x)), "slope_per_log2": float(slope), "intercept": float(icpt),
           "slope_ci95": np.nanpercentile(bs[:, 0], [2.5, 97.5]).tolist(),
           "zero_crossing_log2": float(-icpt / slope),
           "zero_crossing_ratio": float(2 ** (-icpt / slope)),
           "zero_crossing_ratio_ci95": (2 ** np.nanpercentile(bs[:, 1], [2.5, 97.5])).tolist(),
           "note": "x = log2(F / median object size of the bucket); coarse preview of Phase 1.2"}
    json.dump(res, open(os.path.join(OUT, "bucket_regression.json"), "w"), indent=1)
    print(f"\nRegression delta AP ~ log2(F/size): slope {slope:+.4f} per doubling "
          f"(95% CI {res['slope_ci95'][0]:+.4f}..{res['slope_ci95'][1]:+.4f}); "
          f"zero crossing at F/size = {res['zero_crossing_ratio']:.2f} "
          f"(CI {res['zero_crossing_ratio_ci95'][0]:.2f}..{res['zero_crossing_ratio_ci95'][1]:.2f})")


if __name__ == "__main__":
    main()
