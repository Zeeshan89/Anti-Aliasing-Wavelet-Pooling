"""
Train YOLOv10-L on moth / IP102 / DOTA with optional anti-aliasing injection.

Mirrors the experiment matrix from train.py (custom YOLOv8-S) so that paper
comparisons are apples-to-apples:
  - baseline           – vanilla Ultralytics YOLOv10-L
  - A                  – WaveletSmooth (Coif3) before the same 2 neck layers
  - blurpool           – BlurSmooth (size 3, default binomial)
  - blurpool_small     – BlurSmooth (size 2, ablation)
  - gaussian           – GaussianSmooth (7×7, σ=1.5)
  - gaussian_small     – GaussianSmooth (3×3, σ=0.5, ablation)
  - all                – baseline + A + blurpool + gaussian
  - ablation_small     – blurpool_small + gaussian_small

AA injection points: layer indices 17 (Conv P3→P4) and 20 (SCDown P4→P5) of
YOLOv10-L, the direct structural analogs of YOLOv8-S's neck.down_p3_p4 /
neck.down_p4_p5 .  Injection is done via forward-patching the layer's
`forward` method so Ultralytics' index-based routing and checkpoint
save/load logic are unaffected.

Outputs go to runs/yolov10l_<dataset>_<variant>/ with the same files our
existing plot_results.py expects:
  - weights/best.pt, weights/last.pt   (saved by Ultralytics)
  - test_metrics.json                  (our schema, written here)
  - done.flag                          (so re-runs skip)

Usage:
  python train_yolov10.py --dataset dota --variant baseline
  python train_yolov10.py --dataset dota --variant all
  python train_yolov10.py --dataset dota --variant ablation_small
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO

import config as cfg
from wavelet_pool import BlurSmooth, GaussianSmooth, WaveletSmooth

# YOLOv10-L neck downsampling indices (verified by introspecting the model:
# layer 17 = Conv stride-2, layer 20 = SCDown stride-2).  These are the
# direct analogs of YOLOv8-S's neck.down_p3_p4 / neck.down_p4_p5.
TARGET_LAYER_INDICES = [17, 20]


# ──────────────────────────────────────────────────────────────────────
# AA module factory + forward-patching injection
# ──────────────────────────────────────────────────────────────────────

def make_aa_module(variant: str) -> nn.Module:
    v = variant.lower()
    if v == "a":
        return WaveletSmooth()                                   # Coif3
    if v == "blurpool":
        return BlurSmooth(filter_size=cfg.BLURPOOL_FILTER_SIZE)
    if v == "blurpool_small":
        return BlurSmooth(filter_size=cfg.BLURPOOL_FILTER_SIZE_SMALL)
    if v == "gaussian":
        return GaussianSmooth(kernel_size=cfg.GAUSSIAN_KERNEL_SIZE,
                              sigma=cfg.GAUSSIAN_SIGMA)
    if v == "gaussian_small":
        return GaussianSmooth(kernel_size=cfg.GAUSSIAN_KERNEL_SIZE_SMALL,
                              sigma=cfg.GAUSSIAN_SIGMA_SMALL)
    raise ValueError(f"No AA module for variant '{variant}'")


def inject_aa(torch_model: nn.Module, variant: str,
              target_indices=TARGET_LAYER_INDICES):
    """
    Insert an AA pre-filter before each target layer's forward().

    Uses forward-patching (not nn.Sequential wrapping) to keep Ultralytics'
    routing-by-integer-index intact.  AA modules are stored as a child
    nn.ModuleList so their buffers go through .to(device) and appear in
    the saved state_dict.

    Idempotent: when called on a model restored from a resumed checkpoint,
    reuses the existing _aa_modules (so weights persist) and only re-applies
    forward patches to layers that don't already have them.

    Returns list of human-readable strings naming the modified layers.
    """
    if variant.lower() == "baseline":
        return []

    # Reuse _aa_modules if present (resumed ckpt), else create fresh.
    if hasattr(torch_model, "_aa_modules") and len(torch_model._aa_modules) == len(target_indices):
        aa_list = torch_model._aa_modules
        reused = True
    else:
        aa_list = nn.ModuleList([make_aa_module(variant) for _ in target_indices])
        torch_model.add_module("_aa_modules", aa_list)
        reused = False

    modified = []
    for aa_mod, idx in zip(aa_list, target_indices):
        layer = torch_model.model[idx]
        if getattr(layer, "_aa_patched", False):
            modified.append(f"layer[{idx}] ({type(layer).__name__})  [already patched]")
            continue
        original_forward = layer.forward

        def make_patched(aa, orig):
            def patched(x, *a, **kw):
                return orig(aa(x), *a, **kw)
            return patched

        layer.forward = make_patched(aa_mod, original_forward)
        layer._aa_patched = True
        modified.append(f"layer[{idx}] ({type(layer).__name__})"
                        + ("  [reattached]" if reused else ""))
    return modified


# ──────────────────────────────────────────────────────────────────────
# Metric extraction — convert Ultralytics DetMetrics → our JSON schema
# ──────────────────────────────────────────────────────────────────────

def extract_metrics(val_results, class_names) -> dict:
    box = val_results.box
    p, r = float(box.mp), float(box.mr)
    metrics = {
        "mAP@0.5":       float(box.map50),
        "mAP@0.75":      float(box.map75),
        "mAP@.5:.95":    float(box.map),
        "avg_precision": p,
        "avg_recall":    r,
        # Approximation of weighted_f1 — Ultralytics doesn't compute it
        # directly; we use the harmonic mean of class-averaged P/R.
        "weighted_f1":   float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0,
    }
    per_class = {}
    if getattr(box, "maps", None) is not None:
        # box.maps is per-class mAP@.5:.95.  Per-class mAP@0.5 isn't always
        # exposed; we store what we have so downstream code can adapt.
        for i, _ in enumerate(class_names):
            if i < len(box.maps):
                per_class[str(i)] = {"ap50_95": float(box.maps[i])}
    metrics["per_class"] = per_class
    return metrics


# ──────────────────────────────────────────────────────────────────────
# Training one variant
# ──────────────────────────────────────────────────────────────────────

def train_variant(variant: str, dataset: str, device: str,
                  epochs: int = None, batch: int = None,
                  imgsz: int = 640, lr0: float = 0.001):
    name = f"yolov10l_{dataset}_{variant.lower()}"
    run_dir = Path(cfg.RUNS_DIR) / name
    done_flag = run_dir / "done.flag"
    if done_flag.exists():
        print(f"  [SKIP] {name} already complete (done.flag exists)")
        return None

    ds_cfg = cfg.load_dataset_cfg(dataset)
    if epochs is None:
        epochs = ds_cfg["epochs"]
    if batch is None:
        batch = cfg.BATCH_SIZE
    data_yaml = os.path.join(ds_cfg["split_dir"], "data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(
            f"data.yaml not found for '{dataset}' at {data_yaml}. "
            f"Run preprocess_{dataset}.py first.")

    print("=" * 64)
    print(f"  YOLOv10-L  |  Dataset: {dataset}  |  Variant: {variant}")
    print(f"  Run dir   : {run_dir}")
    print(f"  Epochs    : {epochs}   Batch: {batch}   ImgSz: {imgsz}   LR: {lr0}")
    print(f"  Device    : {device}")
    print("=" * 64)

    # Resume detection: if a prior chained job got cut off by walltime, its
    # last.pt is here.  Load from it and pass resume=True so Ultralytics
    # restores epoch counter / optimizer state / lr schedule.
    last_pt = run_dir / "weights" / "last.pt"
    resuming = last_pt.exists()

    if resuming:
        print(f"  [resume] Found {last_pt} — continuing previous training run.")
        model = YOLO(str(last_pt))
    else:
        # Pretrained COCO weights.  First call downloads to the Ultralytics cache.
        model = YOLO("yolov10l.pt")

    # inject_aa is idempotent — on resume it reuses the restored _aa_modules
    # and only re-applies forward patches (which don't always survive pickle).
    modified = inject_aa(model.model, variant)
    if modified:
        tag = "Re-attached" if resuming else "Injected"
        print(f"  {tag} '{variant}' AA pre-filter at:")
        for m in modified:
            print(f"    -> {m}")
    else:
        print("  Baseline: no AA injected.")

    # Match our YOLOv8-S pipeline: NO augmentation (slicing/preprocessing is
    # the only data variation).  Set every Ultralytics augmentation knob to 0.
    no_aug = dict(
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.0,
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
        erasing=0.0, crop_fraction=1.0,
    )

    t0 = time.time()
    if resuming:
        # With resume=True, Ultralytics restores data/epochs/batch/optimizer
        # from the checkpoint's saved args — passing them again is unnecessary
        # and can confuse the resume logic.
        model.train(resume=True, plots=False, verbose=True)
    else:
        model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=str(cfg.RUNS_DIR),
            name=name,
            exist_ok=True,
            optimizer="AdamW",
            lr0=lr0,
            weight_decay=cfg.WEIGHT_DECAY,
            warmup_epochs=cfg.WARMUP_EPOCHS,
            patience=cfg.PATIENCE,
            plots=False,
            verbose=True,
            **no_aug,
        )
    train_min = (time.time() - t0) / 60
    print(f"\n  Training complete in {train_min:.1f} min")

    # Final eval on the test split.  The same model object (forward-patched)
    # is used, so AA stays active.
    print("\n  Running test-set evaluation...")
    val_results = model.val(
        data=data_yaml, split="test", device=device,
        project=str(cfg.RUNS_DIR), name=f"{name}_test_eval",
        exist_ok=True, plots=False, verbose=False,
    )
    metrics = extract_metrics(val_results, ds_cfg["class_names"])
    metrics_path = run_dir / "test_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"  Saved: {metrics_path}")
    print(f"  mAP@0.5 = {metrics['mAP@0.5']:.4f}   "
          f"mAP@0.75 = {metrics['mAP@0.75']:.4f}   "
          f"P = {metrics['avg_precision']:.4f}   "
          f"R = {metrics['avg_recall']:.4f}")

    done_flag.write_text(
        f"Trained {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Epochs: {epochs}\n"
        f"Variant: {variant}\n"
        f"AA layers: {modified}\n"
    )
    return metrics


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Train YOLOv10-L (multi-dataset)")
    p.add_argument("--dataset", type=str, default="dota",
                   choices=list(cfg.DATASET_CONFIGS.keys()))
    p.add_argument("--variant", type=str, default="baseline",
                   choices=["baseline", "A", "blurpool", "gaussian",
                            "blurpool_small", "gaussian_small",
                            "all", "ablation_small"])
    p.add_argument("--device", type=str, default=None,
                   help="cpu / 0 / cuda / mps.  Auto-detected if omitted.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override per-variant epoch count (default = dataset's cfg).")
    p.add_argument("--batch", type=int, default=None,
                   help=f"Override batch size (default = cfg.BATCH_SIZE = {cfg.BATCH_SIZE}).")
    args = p.parse_args()

    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = "0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if args.variant == "all":
        variants = ["baseline", "A", "blurpool", "gaussian"]
    elif args.variant == "ablation_small":
        variants = ["blurpool_small", "gaussian_small"]
    else:
        variants = [args.variant]

    for v in variants:
        train_variant(v, args.dataset, device,
                      epochs=args.epochs, batch=args.batch)

    print("\nAll done.")


if __name__ == "__main__":
    main()
