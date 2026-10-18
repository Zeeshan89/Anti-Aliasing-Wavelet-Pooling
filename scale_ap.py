"""
Scale-stratified AP (small / medium / large) on already-trained checkpoints.

Inference-only, no retraining.  Clones the MPS-safe capture harness of
bootstrap_ci.py (reusing its build_model / pick_device) and extends the capture
loop to record, for every detection: class, conf, image_id, own-area bucket, and
(at IoU 0.5 and 0.75) whether it matched a GT and the AREA BUCKET of the matched
GT.  Per image it records GT counts per (class, bucket).

From this we compute COCO-style area-range AP following COCOeval `areaRng`
semantics:
  - a detection matched to a GT counts as a TP in that GT's bucket and is IGNORED
    for the other buckets;
  - an unmatched detection (false positive) counts only in its own area bucket and
    is ignored for the others.

Area thresholds are in 640-input-space (the dataset is stretched to 640 via
cv2.resize, so area = (norm_w * 640) * (norm_h * 640) exactly, matching the paper's
q25 convention):
    small  : area <  32^2 = 1024 px^2
    medium : 1024 <= area < 96^2 = 9216 px^2
    large  : area >= 9216 px^2

Two averaging conventions are reported, both stated explicitly:
  * overall mAP  -> mean over ALL classes (absent class = AP 0).  This matches the
    paper's evaluate_detection_metrics and is used only as a SANITY GATE: the
    recomputed overall mAP must reproduce the stored test_metrics.json value,
    proving the inference + matching pipeline is faithful before we trust buckets.
  * AP_bucket    -> mean over classes that HAVE GT in that bucket (standard COCO
    areaRng).  This is the reported AP_small / AP_medium / AP_large.

Outputs (separate dir, nothing else touched):
    runs/scale_ap/cache/{dataset}_{variant}.npz     per-variant capture
    runs/scale_ap/{dataset}_scale_ap.json           per-scale AP + deltas (+CIs)
Usage:
    python scale_ap.py --dataset dota                     # all present variants
    python scale_ap.py --dataset dota --limit 200 --B 0   # smoke test, no bootstrap
    python scale_ap.py --dataset dior --B 1000            # with paired bootstrap CIs
"""
import os, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
import train as T
import bootstrap_ci as B  # reuse pick_device, build_model, VARIANTS (MPS-proven)

OUT = os.path.join(cfg.RUNS_DIR, "scale_ap")
CACHE = os.path.join(OUT, "cache")
BUCKETS = ("small", "medium", "large")

# COCO area thresholds in 640-space (px^2)
_SMALL_MAX = 32 * 32      # 1024
_MEDIUM_MAX = 96 * 96     # 9216


def area_bucket(area):
    """0 = small, 1 = medium, 2 = large (half-open COCO ranges)."""
    if area < _SMALL_MAX:
        return 0
    if area < _MEDIUM_MAX:
        return 1
    return 2


@torch.no_grad()
def capture_scale(model, dl, device, num_classes, img_size, limit=None):
    """One inference pass -> per-detection rows (with area buckets) + per-image
    per-(class, bucket) GT counts.  Mirrors bootstrap_ci.capture but adds buckets."""
    r_cls, r_conf, r_img = [], [], []
    r_ob = []                 # detection own-area bucket
    r_mb50, r_mb75 = [], []   # matched-GT bucket at IoU .5 / .75 (-1 if unmatched)
    ngt_rows = []             # per image: (num_classes, 3) GT counts by bucket
    img_id = 0
    for imgs, targets in dl:
        imgs = imgs.to(device, non_blocking=True)
        preds = model(imgs)
        batch = T.decode_predictions(preds, 0.01, img_size=img_size)
        for b in range(imgs.shape[0]):
            dets = T.nms(batch[b].cpu(), 0.45)
            gt = targets[b]
            ngt = np.zeros((num_classes, 3))
            if gt.shape[0] > 0:
                gc = gt[:, 0].numpy().astype(int)
                cx, cy = gt[:, 1] * img_size, gt[:, 2] * img_size
                w, h = gt[:, 3] * img_size, gt[:, 4] * img_size
                gt_boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
                gw = (w * h).numpy()
                gbucket = np.array([area_bucket(float(a)) for a in gw], int)
                for i in range(len(gc)):
                    ngt[gc[i], gbucket[i]] += 1
            else:
                gc = np.array([], int)
                gt_boxes = torch.zeros((0, 4))
                gbucket = np.array([], int)
            ngt_rows.append(ngt)

            N, M = dets.shape[0], gt_boxes.shape[0]
            if N > 0:
                pc = dets[:, 4].numpy()
                pcl = dets[:, 5].numpy().astype(int)
                db = dets[:, :4].numpy()
                darea = (db[:, 2] - db[:, 0]) * (db[:, 3] - db[:, 1])
                ob = np.array([area_bucket(float(a)) for a in darea], int)
                iou = T._iou_matrix(dets[:, :4], gt_boxes).numpy() if M > 0 else np.zeros((N, 0))
                order = np.argsort(-pc)
                mb = {0.5: -np.ones(N, int), 0.75: -np.ones(N, int)}
                for thr in (0.5, 0.75):
                    matched = np.zeros(M, bool)
                    for idx in order:
                        c = pcl[idx]
                        bi, bj = 0.0, -1
                        for j in range(M):
                            if matched[j] or int(gc[j]) != c:
                                continue
                            if iou[idx, j] > bi:
                                bi, bj = iou[idx, j], j
                        if bi >= thr and bj >= 0:
                            matched[bj] = True
                            mb[thr][idx] = gbucket[bj]
                for idx in range(N):
                    r_cls.append(pcl[idx]); r_conf.append(pc[idx]); r_img.append(img_id)
                    r_ob.append(ob[idx]); r_mb50.append(mb[0.5][idx]); r_mb75.append(mb[0.75][idx])
            img_id += 1
            if limit and img_id >= limit:
                break
        if limit and img_id >= limit:
            break
    return dict(cls=np.array(r_cls, np.int32), conf=np.array(r_conf, np.float32),
                img=np.array(r_img, np.int64), ob=np.array(r_ob, np.int8),
                mb50=np.array(r_mb50, np.int8), mb75=np.array(r_mb75, np.int8),
                ngt=np.array(ngt_rows, np.float64))


def prep_all(cap, num_classes, thr):
    """Per-class sorted arrays for the ALL-areas mAP (sanity gate, all-class mean)."""
    mb = cap["mb50"] if thr == 0.5 else cap["mb75"]
    hit = (mb != -1).astype(np.float64)
    ngt_all = cap["ngt"].sum(axis=2)  # (n_img, num_classes)
    per = []
    for c in range(num_classes):
        m = cap["cls"] == c
        o = np.argsort(-cap["conf"][m])
        per.append((hit[m][o], cap["img"][m][o], ngt_all[:, c]))
    return per


def map_all(per, mult):
    """Standard mAP averaged over ALL classes (matches paper convention)."""
    aps = []
    for hit, img, ngtc in per:
        if hit.size == 0:
            aps.append(0.0); continue
        w = mult[img]
        cumtp = np.cumsum(hit * w)
        cumpred = np.cumsum(w)
        tg = float((mult * ngtc).sum())
        p = cumtp / np.maximum(cumpred, 1e-9)
        r = cumtp / max(tg, 1.0)
        aps.append(T._compute_ap(r, p))
    return float(np.mean(aps)) if aps else 0.0


def prep_bucket(cap, num_classes, bucket, thr):
    """Per-class sorted arrays for one area bucket (COCO areaRng inclusion)."""
    mb = cap["mb50"] if thr == 0.5 else cap["mb75"]
    per = []
    for c in range(num_classes):
        sel = cap["cls"] == c
        mbc = mb[sel]; obc = cap["ob"][sel]; cf = cap["conf"][sel]; im = cap["img"][sel]
        is_tp = mbc == bucket
        inc = is_tp | ((mbc == -1) & (obc == bucket))  # TP in bucket, or FP with own area in bucket
        cf_i = cf[inc]; im_i = im[inc]; hit_i = is_tp[inc].astype(np.float64)
        o = np.argsort(-cf_i)
        per.append((hit_i[o], im_i[o], cap["ngt"][:, c, bucket]))
    return per


def map_bucket(per, mult):
    """AP for one bucket, averaged over classes that HAVE GT in the bucket (COCO)."""
    aps = []
    for hit, img, ngtc in per:
        tg = float((mult * ngtc).sum())
        if tg <= 0:
            continue  # class absent from this bucket -> excluded from the mean
        if hit.size == 0:
            aps.append(0.0); continue
        w = mult[img]
        cumtp = np.cumsum(hit * w)
        cumpred = np.cumsum(w)
        p = cumtp / np.maximum(cumpred, 1e-9)
        r = cumtp / max(tg, 1.0)
        aps.append(T._compute_ap(r, p))
    return float(np.mean(aps)) if aps else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--B", type=int, default=0, help="bootstrap resamples for ΔAP CIs (0 = skip)")
    ap.add_argument("--limit", type=int, default=None, help="cap #images (smoke test)")
    ap.add_argument("--variants", nargs="*", default=None)
    args = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    device = B.pick_device()
    ds_cfg = cfg.load_dataset_cfg(args.dataset)
    ncls = ds_cfg["num_classes"]
    split = ds_cfg["split_dir"]
    print(f"device={device}  dataset={args.dataset}  classes={ncls}  B={args.B}"
          + (f"  LIMIT={args.limit}" if args.limit else ""))

    test_ds = T.MothDataset(os.path.join(split, "test", "images"),
                            os.path.join(split, "test", "labels"), cfg.INPUT_SIZE)
    dl = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                    collate_fn=T._collate_fn, num_workers=0)

    variants = args.variants or [v for v in B.VARIANTS
                                 if os.path.exists(os.path.join(cfg.RUNS_DIR,
                                    f"{args.dataset}_{v}", "weights", "best.pt"))]
    print("variants:", variants)

    caps = {}
    for v in variants:
        ck = os.path.join(CACHE, f"{args.dataset}_{v}.npz")
        if os.path.exists(ck) and not args.limit:
            caps[v] = dict(np.load(ck))
            print(f"  [cache] {v}")
            continue
        model = B.build_model(v, args.dataset, ncls, device)
        if model is None:
            print(f"  [skip] {v} (no checkpoint)"); continue
        cap = capture_scale(model, dl, device, ncls, cfg.INPUT_SIZE, limit=args.limit)
        caps[v] = cap
        if not args.limit:
            np.savez_compressed(ck, **cap)
        # SANITY GATE: recomputed all-areas mAP vs stored test_metrics.json
        n_img = cap["ngt"].shape[0]
        pt = map_all(prep_all(cap, ncls, 0.5), np.ones(n_img))
        tm = os.path.join(cfg.RUNS_DIR, f"{args.dataset}_{v}", "test_metrics.json")
        ref = json.load(open(tm))["mAP@0.5"] if os.path.exists(tm) else None
        print(f"  {v:16s} mAP@0.5(recomputed all-areas)={pt:.4f}"
              + (f"  stored={ref:.4f}  Δ={pt-ref:+.4f}" if ref is not None else ""))
        del model
        if device == "mps":
            torch.mps.empty_cache()

    if "baseline" not in caps:
        print("no baseline capture — cannot compute deltas"); return

    # ---- GT census by bucket (from baseline; identical across variants) ----
    ngt_base = caps["baseline"]["ngt"]
    census = {BUCKETS[k]: int(ngt_base[:, :, k].sum()) for k in range(3)}
    n_img = ngt_base.shape[0]
    print(f"\nGT census ({args.dataset}, n_img={n_img}):  "
          + "  ".join(f"{b}={census[b]}" for b in BUCKETS))

    # ---- point estimates: AP_bucket @0.5 and @0.75 ----
    point = {}  # variant -> {thr -> {bucket -> ap}}
    for v in caps:
        point[v] = {}
        for thr in (0.5, 0.75):
            point[v][thr] = {BUCKETS[k]: map_bucket(prep_bucket(caps[v], ncls, k, thr), np.ones(n_img))
                             for k in range(3)}

    def fmt(x):
        return "  n/a " if (x != x) else f"{x:6.4f}"  # NaN check

    print(f"\n=== AP@0.5 by object scale ({args.dataset}) ===")
    print(f"{'variant':16s} {'AP_small':>9s} {'AP_med':>9s} {'AP_large':>9s}    "
          f"{'ΔS':>7s} {'ΔM':>7s} {'ΔL':>7s}")
    base05 = point["baseline"][0.5]
    for v in variants:
        if v not in point:
            continue
        p = point[v][0.5]
        ds = [p[b] - base05[b] for b in BUCKETS]
        dstr = "" if v == "baseline" else "   " + " ".join(
            (" n/a  " if (d != d) else f"{d:+6.3f}") for d in ds)
        print(f"{v:16s} {fmt(p['small'])} {fmt(p['medium'])} {fmt(p['large'])}{dstr}")

    # ---- optional paired bootstrap CIs on ΔAP_bucket @0.5 ----
    ci = {}
    if args.B > 0:
        rng = np.random.default_rng(0)
        mults = [np.bincount(rng.integers(0, n_img, n_img), minlength=n_img).astype(np.float64)
                 for _ in range(args.B)]
        boot = {}  # (variant, bucket) -> np.array over resamples
        for v in caps:
            for k in range(3):
                per = prep_bucket(caps[v], ncls, k, 0.5)
                boot[(v, k)] = np.array([map_bucket(per, m) for m in mults])
        for v in variants:
            if v == "baseline" or v not in caps:
                continue
            ci[v] = {}
            for k in range(3):
                d = boot[(v, k)] - boot[("baseline", k)]
                d = d[~np.isnan(d)]
                if d.size == 0:
                    ci[v][BUCKETS[k]] = None; continue
                dlo, dhi = np.percentile(d, [2.5, 97.5])
                ci[v][BUCKETS[k]] = [round(float(dlo), 4), round(float(dhi), 4),
                                     bool((dlo > 0) or (dhi < 0))]
        print(f"\n=== ΔAP@0.5 95% CI (paired image bootstrap, B={args.B}) ===")
        for v in variants:
            if v not in ci:
                continue
            row = "  ".join(f"{b}:[{ci[v][b][0]:+.3f},{ci[v][b][1]:+.3f}]"
                            f"{'*' if ci[v][b] and ci[v][b][2] else ''}"
                            if ci[v][b] else f"{b}:n/a" for b in BUCKETS)
            print(f"{v:16s} {row}")

    # ---- write JSON ----
    result = {
        "dataset": args.dataset, "device": device, "n_test_images": int(n_img),
        "area_thresholds_px2_at640": {"small": "<1024", "medium": "1024-9216", "large": ">=9216"},
        "convention": {
            "overall_mAP": "mean over ALL classes (matches paper evaluate_detection_metrics)",
            "AP_bucket": "mean over classes with GT in the bucket (COCO areaRng)"},
        "gt_counts_by_bucket": census,
        "variants": {},
    }
    for v in variants:
        if v not in caps:
            continue
        cap = caps[v]
        entry = {
            "mAP@0.5_all_recomputed": round(map_all(prep_all(cap, ncls, 0.5), np.ones(n_img)), 4),
            "AP@0.5": {b: (None if point[v][0.5][b] != point[v][0.5][b] else round(point[v][0.5][b], 4))
                       for b in BUCKETS},
            "AP@0.75": {b: (None if point[v][0.75][b] != point[v][0.75][b] else round(point[v][0.75][b], 4))
                        for b in BUCKETS},
        }
        tm = os.path.join(cfg.RUNS_DIR, f"{args.dataset}_{v}", "test_metrics.json")
        if os.path.exists(tm):
            entry["mAP@0.5_all_stored"] = json.load(open(tm))["mAP@0.5"]
        if v != "baseline":
            entry["delta_AP@0.5"] = {
                b: (None if (point[v][0.5][b] != point[v][0.5][b] or base05[b] != base05[b])
                    else round(point[v][0.5][b] - base05[b], 4)) for b in BUCKETS}
            if v in ci:
                entry["delta_ci95@0.5"] = ci[v]
        result["variants"][v] = entry

    outp = os.path.join(OUT, f"{args.dataset}_scale_ap.json")
    json.dump(result, open(outp, "w"), indent=2)
    print(f"\nsaved: {outp}")


if __name__ == "__main__":
    main()
