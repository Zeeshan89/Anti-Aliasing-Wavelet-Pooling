"""
Zero-retrain bootstrap confidence intervals for mAP@0.5 on already-trained
checkpoints (Tier-2 fallback for statistical significance).

For each variant we run inference on the test set ONCE, cache per-detection
(class, conf, hit@0.5, hit@0.75, image_id) plus per-image per-class GT counts,
then bootstrap by resampling IMAGES with replacement B times.  We reuse the
SAME image resamples across every variant so the baseline-vs-variant delta is
PAIRED, giving a CI on ΔmAP@0.5.

CAVEAT (must be stated in the paper): this captures evaluation / test-set
sampling variance only, NOT training-seed variance.  It is a LOWER BOUND on the
true uncertainty of ΔmAP.  A CI that already excludes zero here would only widen
under seed variance, so "excludes zero" is informative; "includes zero" is
expected for small effects and does not by itself prove significance.

Outputs (separate dir, nothing else touched):
    runs/bootstrap_ci/cache/{dataset}_{variant}.npz     per-variant capture
    runs/bootstrap_ci/{dataset}_ci.json                 CIs + paired deltas
Usage:
    python bootstrap_ci.py --dataset dota            # all present variants
    python bootstrap_ci.py --dataset dota --limit 200 --B 200   # smoke test
"""
import os, sys, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
import train as T

VARIANTS = ["baseline", "a", "blurpool", "gaussian", "blurpool_small", "gaussian_small"]
OUT = os.path.join(cfg.RUNS_DIR, "bootstrap_ci")
CACHE = os.path.join(OUT, "cache")


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(variant, dataset, num_classes, device):
    run_dir = os.path.join(cfg.RUNS_DIR, f"{dataset}_{variant.lower()}")
    weights = os.path.join(run_dir, "weights", "best.pt")
    if not os.path.exists(weights):
        return None
    wl_path = os.path.join(run_dir, "wavelet_layers.json")
    target_layers = json.load(open(wl_path)) if os.path.exists(wl_path) else None

    model = T.YOLOv8S(num_classes=num_classes)
    v = variant.lower()
    if v == "a":
        T.apply_wavelet_selective(model, target_layers=target_layers, dataset=dataset)
    elif v == "blurpool":
        T.apply_blurpool_selective(model, target_layers=target_layers, dataset=dataset)
    elif v == "blurpool_small":
        T.apply_blurpool_selective(model, target_layers=target_layers, dataset=dataset,
                                   filter_size=cfg.BLURPOOL_FILTER_SIZE_SMALL)
    elif v == "gaussian":
        T.apply_gaussian_selective(model, target_layers=target_layers, dataset=dataset)
    elif v == "gaussian_small":
        T.apply_gaussian_selective(model, target_layers=target_layers, dataset=dataset,
                                   kernel_size=cfg.GAUSSIAN_KERNEL_SIZE_SMALL,
                                   sigma=cfg.GAUSSIAN_SIGMA_SMALL)
    state = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def capture(model, dl, device, num_classes, img_size, limit=None):
    """One inference pass -> per-detection rows + per-image GT counts."""
    r_cls, r_conf, r_h50, r_h75, r_img = [], [], [], [], []
    ngt_rows = []
    img_id = 0
    for imgs, targets in dl:
        imgs = imgs.to(device, non_blocking=True)
        preds = model(imgs)
        batch = T.decode_predictions(preds, 0.01, img_size=img_size)
        for b in range(imgs.shape[0]):
            dets = T.nms(batch[b].cpu(), 0.45)
            gt = targets[b]
            ngt = np.zeros(num_classes)
            if gt.shape[0] > 0:
                gc = gt[:, 0].numpy().astype(int)
                cx, cy = gt[:, 1] * img_size, gt[:, 2] * img_size
                w, h = gt[:, 3] * img_size, gt[:, 4] * img_size
                gt_boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
                for c in range(num_classes):
                    ngt[c] = int((gc == c).sum())
            else:
                gc = np.array([], int); gt_boxes = torch.zeros((0, 4))
            ngt_rows.append(ngt)
            N, M = dets.shape[0], gt_boxes.shape[0]
            if N > 0:
                pc = dets[:, 4].numpy(); pcl = dets[:, 5].numpy().astype(int)
                iou = T._iou_matrix(dets[:, :4], gt_boxes).numpy() if M > 0 else np.zeros((N, 0))
                order = np.argsort(-pc)
                hit = {0.5: {}, 0.75: {}}
                for thr in (0.5, 0.75):
                    matched = np.zeros(M, bool)
                    for idx in order:
                        c = pcl[idx]; bi, bj = 0.0, -1
                        for j in range(M):
                            if matched[j] or int(gc[j]) != c:
                                continue
                            if iou[idx, j] > bi:
                                bi, bj = iou[idx, j], j
                        h = bi >= thr and bj >= 0
                        hit[thr][idx] = h
                        if h:
                            matched[bj] = True
                for idx in range(N):
                    r_cls.append(pcl[idx]); r_conf.append(pc[idx])
                    r_h50.append(hit[0.5][idx]); r_h75.append(hit[0.75][idx])
                    r_img.append(img_id)
            img_id += 1
            if limit and img_id >= limit:
                break
        if limit and img_id >= limit:
            break
    return dict(cls=np.array(r_cls, np.int32), conf=np.array(r_conf, np.float32),
                h50=np.array(r_h50, bool), h75=np.array(r_h75, bool),
                img=np.array(r_img, np.int64), ngt=np.array(ngt_rows, np.float64))


def prep(cap, num_classes):
    per = []
    for c in range(num_classes):
        m = cap["cls"] == c
        o = np.argsort(-cap["conf"][m])
        per.append((cap["h50"][m][o].astype(np.float64),
                    cap["img"][m][o], cap["ngt"][:, c]))
    return per


def map_at50(per, mult):
    aps = []
    for h50, img, ngt in per:
        if h50.size == 0:
            aps.append(0.0); continue
        w = mult[img]
        cumtp = np.cumsum(h50 * w)
        cumpred = np.cumsum(w)
        tg = float((mult * ngt).sum())
        p = cumtp / np.maximum(cumpred, 1e-9)
        r = cumtp / max(tg, 1.0)
        aps.append(T._compute_ap(r, p))
    return float(np.mean(aps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--variants", nargs="*", default=None)
    args = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    device = pick_device()
    ds_cfg = cfg.load_dataset_cfg(args.dataset)
    ncls = ds_cfg["num_classes"]
    split = ds_cfg["split_dir"]
    print(f"device={device}  dataset={args.dataset}  classes={ncls}  B={args.B}"
          + (f"  LIMIT={args.limit}" if args.limit else ""))

    test_ds = T.MothDataset(os.path.join(split, "test", "images"),
                            os.path.join(split, "test", "labels"), cfg.INPUT_SIZE)
    dl = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                    collate_fn=T._collate_fn, num_workers=0)

    variants = args.variants or [v for v in VARIANTS
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
        model = build_model(v, args.dataset, ncls, device)
        if model is None:
            print(f"  [skip] {v} (no checkpoint)"); continue
        cap = capture(model, dl, device, ncls, cfg.INPUT_SIZE, limit=args.limit)
        caps[v] = cap
        if not args.limit:
            np.savez_compressed(ck, **cap)
        # sanity: uniform-weight mAP vs stored test_metrics.json
        n_img = cap["ngt"].shape[0]
        pt = map_at50(prep(cap, ncls), np.ones(n_img))
        tm = os.path.join(cfg.RUNS_DIR, f"{args.dataset}_{v}", "test_metrics.json")
        ref = json.load(open(tm))["mAP@0.5"] if os.path.exists(tm) else None
        print(f"  {v:16s} mAP@0.5(recomputed)={pt:.4f}" +
              (f"  stored={ref:.4f}  Δ={pt-ref:+.4f}" if ref is not None else ""))
        del model
        if device == "mps":
            torch.mps.empty_cache()

    if "baseline" not in caps:
        print("no baseline capture — cannot compute deltas"); return

    n_img = caps["baseline"]["ngt"].shape[0]
    rng = np.random.default_rng(0)
    mults = [np.bincount(rng.integers(0, n_img, n_img), minlength=n_img).astype(np.float64)
             for _ in range(args.B)]

    boot = {}
    for v in caps:
        per = prep(caps[v], ncls)
        boot[v] = np.array([map_at50(per, m) for m in mults])
        boot[v + "__point"] = map_at50(per, np.ones(n_img))

    base = boot["baseline"]
    result = {"dataset": args.dataset, "B": args.B, "n_test_images": int(n_img),
              "device": device, "note": "test-set (evaluation) variance only; "
              "lower bound on uncertainty; excludes training-seed variance.",
              "variants": {}}
    print(f"\n{'variant':16s} {'mAP@0.5':>8s} {'CI2.5':>7s} {'CI97.5':>7s} "
          f"{'ΔmAP':>8s} {'Δ CI2.5':>8s} {'Δ CI97.5':>8s}  sig")
    for v in variants:
        if v not in caps:
            continue
        b = boot[v]; pt = boot[v + "__point"]
        lo, hi = np.percentile(b, [2.5, 97.5])
        entry = {"mAP@0.5": round(pt, 4),
                 "ci95": [round(float(lo), 4), round(float(hi), 4)]}
        if v != "baseline":
            d = b - base
            dlo, dhi = np.percentile(d, [2.5, 97.5])
            sig = (dlo > 0) or (dhi < 0)
            entry["delta_mAP"] = round(float(boot[v + "__point"] - boot["baseline__point"]), 4)
            entry["delta_ci95"] = [round(float(dlo), 4), round(float(dhi), 4)]
            entry["delta_excludes_zero"] = bool(sig)
            print(f"{v:16s} {pt:8.4f} {lo:7.4f} {hi:7.4f} "
                  f"{entry['delta_mAP']:+8.4f} {dlo:+8.4f} {dhi:+8.4f}  {'YES' if sig else 'no'}")
        else:
            print(f"{v:16s} {pt:8.4f} {lo:7.4f} {hi:7.4f}")
        result["variants"][v] = entry

    outp = os.path.join(OUT, f"{args.dataset}_ci.json")
    json.dump(result, open(outp, "w"), indent=2)
    print(f"\nsaved: {outp}")


if __name__ == "__main__":
    main()
