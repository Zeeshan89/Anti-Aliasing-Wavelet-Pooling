"""
DOTA-Devkit Task 2 (HBB) re-evaluation of the already-trained DOTA variants.

No retraining. For each variant in runs/dota_{variant}/weights/best.pt this script:
  1. Rebuilds the exact architecture (same anti-aliasing layers as training) and
     loads best.pt  — mirrors evaluate_variant_tta() in train.py.
  2. Runs inference on every 640x640 patch under dataset/dota/split/test/images/,
     decoding with the project's own decode_predictions() + per-patch nms().
  3. Unslices patch-pixel boxes back to original DOTA image coordinates using
     _window_origins() from preprocess_dota.py (patch=640, stride=320).
  4. Runs a per-source-image, per-class NMS (IoU=0.5) to dedupe boxes in the
     stride-320 overlap regions.
  5. Writes a DOTA Task 2 submission: one Task2_{classname}.txt per class with
     lines "<image_stem> <conf> <x1> <y1> <x2> <y2>".
  6. Runs the official DOTA-Devkit dota-v1.5_evaluation_task2.py voc_eval() against
     the val HBB ground truth and reports per-class AP + mAP.

Ground truth: DOTA_1_5/Val_dota/DOTA-v1.5_val_hbb/{stem}.txt (the test patches are
derived from the DOTA v1.5 *val* images, which have public HBB GT).

Caveat surfaced in the report: preprocess_dota.py drops empty (background) patches,
so the model is only run on windows that contain >=1 object of our 15 classes.
Recall is fair (every GT box lands in a kept patch), but precision is optimistic
relative to the full-image DOTA leaderboard. This applies equally to the project's
custom evaluator, so custom-vs-devkit remains apples-to-apples.

Model is 15-class (DOTA v1.0 class set). DOTA v1.5's 16th class 'container-crane' is
never predicted and is ignored per-class by the VOC-style evaluator — identical to
how the custom evaluator (and preprocessing) handle it.

Usage:
    python dota_task2_eval.py                       # all 6 variants
    python dota_task2_eval.py --variant baseline    # one variant
    python dota_task2_eval.py --variant baseline --limit 200   # quick smoke test
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import re
import json
import argparse
import importlib.util
import contextlib

import numpy as np
import cv2
import torch

import config as cfg
from model import YOLOv8S
from train import (
    apply_wavelet_selective,
    apply_blurpool_selective,
    apply_gaussian_selective,
    apply_bottom_heavy,
    decode_predictions,
    nms,
)
from preprocess_dota import _window_origins, PATCH, STRIDE

# ── Paths ────────────────────────────────────────────────────────────
PROJECT   = cfg.PROJECT_ROOT
TEST_IMG  = os.path.join(PROJECT, "dataset", "dota", "split", "test", "images")
ORIG_IMG  = os.path.join(PROJECT, "DOTA_1_5", "Val_dota", "images")
GT_HBB    = os.path.join(PROJECT, "DOTA_1_5", "Val_dota", "DOTA-v1.5_val_hbb")
DEVKIT    = os.path.join(PROJECT, "DOTA_devkit", "dota-v1.5_evaluation_task2.py")
SCRATCH   = os.path.join(PROJECT, "runs", "_dota_task2_work")

VARIANTS  = ["baseline", "a", "blurpool", "gaussian", "blurpool_small", "gaussian_small"]
CLASS_NAMES = cfg.DOTA_CLASS_NAMES          # 15 names, exact DOTA spelling
PATCH_RE  = re.compile(r"^(?P<stem>.+)__r(?P<row>\d+)_c(?P<col>\d+)\.png$")

CONF_THRESH = 0.05      # submission conf floor
PATCH_NMS   = 0.45      # per-patch NMS
MERGE_NMS   = 0.50      # cross-patch NMS per source image
BATCH       = 16


# ── Model building (mirrors evaluate_variant_tta in train.py) ────────
def build_model(variant, device):
    run_dir = os.path.join(cfg.RUNS_DIR, f"dota_{variant.lower()}")
    weights = os.path.join(run_dir, "weights", "best.pt")
    if not os.path.exists(weights):
        raise FileNotFoundError(weights)

    model = YOLOv8S(num_classes=len(CLASS_NAMES))
    wl_path = os.path.join(run_dir, "wavelet_layers.json")
    target_layers = None
    if os.path.exists(wl_path):
        with open(wl_path) as f:
            target_layers = json.load(f)

    v = variant.upper()
    if v in ("A", "B"):
        apply_wavelet_selective(model, target_layers=target_layers, dataset="dota")
    elif variant.lower() == "blurpool":
        apply_blurpool_selective(model, target_layers=target_layers, dataset="dota")
    elif variant.lower() == "blurpool_small":
        apply_blurpool_selective(model, target_layers=target_layers, dataset="dota",
                                 filter_size=cfg.BLURPOOL_FILTER_SIZE_SMALL)
    elif variant.lower() == "gaussian_small":
        apply_gaussian_selective(model, target_layers=target_layers, dataset="dota",
                                 kernel_size=cfg.GAUSSIAN_KERNEL_SIZE_SMALL,
                                 sigma=cfg.GAUSSIAN_SIGMA_SMALL)
    elif variant.lower() == "gaussian":
        apply_gaussian_selective(model, target_layers=target_layers, dataset="dota")
    if v == "B":
        apply_bottom_heavy(model)

    state = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


# ── Original-image size cache (for window-origin reconstruction) ─────
_size_cache = {}
def orig_size(stem):
    if stem not in _size_cache:
        p = os.path.join(ORIG_IMG, f"{stem}.png")
        im = cv2.imread(p)
        if im is None:
            raise FileNotFoundError(p)
        h, w = im.shape[:2]
        _size_cache[stem] = (w, h)
    return _size_cache[stem]


def patch_origin(stem, row, col):
    W, H = orig_size(stem)
    xs = _window_origins(W, PATCH, STRIDE)
    ys = _window_origins(H, PATCH, STRIDE)
    return xs[col], ys[row], W, H


# ── Inference over all patches for one variant ───────────────────────
@torch.no_grad()
def infer_variant(model, device, limit=None):
    files = sorted(f for f in os.listdir(TEST_IMG) if f.endswith(".png"))
    if limit:
        files = files[:limit]

    # per source stem -> list of [x1,y1,x2,y2,conf,cls]
    per_stem = {}
    batch_imgs, batch_meta = [], []

    def flush():
        if not batch_imgs:
            return
        x = torch.from_numpy(np.stack(batch_imgs)).to(device)
        preds = model(x)
        dets = decode_predictions(preds, conf_thresh=CONF_THRESH, img_size=cfg.INPUT_SIZE)
        for i, (stem, row, col) in enumerate(batch_meta):
            d = nms(dets[i].cpu(), PATCH_NMS)
            if d.shape[0] == 0:
                continue
            x0, y0, W, H = patch_origin(stem, row, col)
            d = d.clone()
            d[:, [0, 2]] += x0
            d[:, [1, 3]] += y0
            d[:, [0, 2]] = d[:, [0, 2]].clamp(0, W)
            d[:, [1, 3]] = d[:, [1, 3]].clamp(0, H)
            per_stem.setdefault(stem, []).append(d)
        batch_imgs.clear()
        batch_meta.clear()

    for n, fname in enumerate(files):
        m = PATCH_RE.match(fname)
        if not m:
            continue
        stem, row, col = m["stem"], int(m["row"]), int(m["col"])
        img = cv2.imread(os.path.join(TEST_IMG, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (cfg.INPUT_SIZE, cfg.INPUT_SIZE))
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        batch_imgs.append(img)
        batch_meta.append((stem, row, col))
        if len(batch_imgs) >= BATCH:
            flush()
        if (n + 1) % 2000 == 0:
            print(f"    ...{n + 1}/{len(files)} patches")
    flush()

    # cross-patch NMS per source image, per class (nms() is already per-class)
    merged = {}
    for stem, chunks in per_stem.items():
        alld = torch.cat(chunks, 0)
        merged[stem] = nms(alld, MERGE_NMS)
    return merged, files


# ── Write DOTA Task 2 submission files ───────────────────────────────
def write_submission(merged, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # open one file per class (ensures every class file exists, even if empty)
    handles = {c: open(os.path.join(out_dir, f"Task2_{name}.txt"), "w")
               for c, name in enumerate(CLASS_NAMES)}
    counts = {c: 0 for c in range(len(CLASS_NAMES))}
    try:
        for stem, dets in merged.items():
            for row in dets.tolist():
                x1, y1, x2, y2, conf, c = row
                c = int(c)
                handles[c].write(f"{stem} {conf:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")
                counts[c] += 1
    finally:
        for h in handles.values():
            h.close()
    return counts


# ── Prepare cleaned GT (strip header lines) + imageset once ──────────
def prepare_gt_and_imageset(stems):
    gt_out = os.path.join(SCRATCH, "gt_clean")
    os.makedirs(gt_out, exist_ok=True)
    for stem in stems:
        src = os.path.join(GT_HBB, f"{stem}.txt")
        dst = os.path.join(gt_out, f"{stem}.txt")
        lines = []
        if os.path.exists(src):
            with open(src) as f:
                for ln in f:
                    parts = ln.strip().split()
                    # object lines: 8 coords + name (+ optional difficult) >= 9 fields,
                    # and the 9th field must be a class name (skips imagesource:/gsd: headers)
                    if len(parts) >= 9 and parts[8] in CLASS_NAMES:
                        lines.append(ln.rstrip("\n"))
        with open(dst, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
    imageset = os.path.join(SCRATCH, "testset.txt")
    with open(imageset, "w") as f:
        f.write("\n".join(sorted(stems)) + "\n")
    return gt_out, imageset


# ── Load the (hyphen-named) devkit module ────────────────────────────
def load_devkit():
    spec = importlib.util.spec_from_file_location("dota_task2_devkit", DEVKIT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def evaluate(pred_dir, gt_dir, imageset, use_07_metric=True):
    devkit = load_devkit()
    detpath  = os.path.join(pred_dir, "Task2_{:s}.txt")
    annopath = os.path.join(gt_dir, "{:s}.txt")
    per_class, aps = {}, []
    for name in CLASS_NAMES:
        try:
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                rec, prec, ap = devkit.voc_eval(
                    detpath, annopath, imageset, name,
                    ovthresh=0.5, use_07_metric=use_07_metric)
            ap = float(ap)
            if not np.isfinite(ap):         # class absent from GT (npos=0) -> AP 0
                ap = 0.0
        except Exception as e:              # no detections / empty file -> AP 0
            ap = 0.0
        per_class[name] = round(ap, 4)
        aps.append(ap)
    return {"mAP": round(float(np.mean(aps)), 4), "per_class": per_class}


def run_variant(variant, device, limit=None):
    print(f"\n{'='*60}\n  DOTA Task 2 — variant: {variant}\n{'='*60}")
    model = build_model(variant, device)
    merged, files = infer_variant(model, device, limit=limit)
    stems = sorted({PATCH_RE.match(f)["stem"] for f in files if PATCH_RE.match(f)})
    print(f"  Inference done: {len(files)} patches, {len(stems)} source images")

    pred_dir = os.path.join(cfg.RUNS_DIR, f"dota_{variant.lower()}", "dota_task2_preds")
    counts = write_submission(merged, pred_dir)
    print(f"  Wrote submission -> {pred_dir}  (dets/class: "
          f"min={min(counts.values())}, max={max(counts.values())})")

    gt_dir, imageset = prepare_gt_and_imageset(stems)
    res07  = evaluate(pred_dir, gt_dir, imageset, use_07_metric=True)
    resC   = evaluate(pred_dir, gt_dir, imageset, use_07_metric=False)

    # pull the custom evaluator's number for side-by-side
    tm = os.path.join(cfg.RUNS_DIR, f"dota_{variant.lower()}", "test_metrics.json")
    custom = None
    if os.path.exists(tm):
        with open(tm) as f:
            custom = json.load(f).get("mAP@0.5")

    out = {
        "variant": variant,
        "custom_mAP@0.5": custom,
        "devkit_task2_mAP_voc07": res07["mAP"],
        "devkit_task2_mAP_cont":  resC["mAP"],
        "per_class_voc07": res07["per_class"],
        "n_source_images": len(stems),
    }
    out_path = os.path.join(cfg.RUNS_DIR, f"dota_{variant.lower()}", "dota_devkit_task2.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  custom mAP@0.5={custom}  |  devkit Task2 mAP (VOC07)={res07['mAP']}  "
          f"(continuous={resC['mAP']})")
    print(f"  Saved: {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None,
                    help="Single variant; default runs all 6.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N patches (quick smoke test).")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")
    os.makedirs(SCRATCH, exist_ok=True)

    variants = [args.variant] if args.variant else VARIANTS
    results = []
    for v in variants:
        try:
            results.append(run_variant(v, device, limit=args.limit))
        except FileNotFoundError as e:
            print(f"  [SKIP] {v}: missing {e}")

    print(f"\n{'='*72}\n  SUMMARY — DOTA Task 2 (HBB) vs custom evaluator\n{'='*72}")
    print(f"  {'variant':<16s} {'custom mAP@.5':>14s} {'devkit VOC07':>14s} {'devkit cont':>14s}")
    print("  " + "-" * 62)
    for r in results:
        print(f"  {r['variant']:<16s} {str(r['custom_mAP@0.5']):>14s} "
              f"{r['devkit_task2_mAP_voc07']:>14.4f} {r['devkit_task2_mAP_cont']:>14.4f}")
    print()


if __name__ == "__main__":
    main()
