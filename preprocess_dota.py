"""
Preprocess DOTA v1.5 (using v1.0 labels — 15 classes) → 640×640 patches in YOLO format.

DOTA images are huge (up to 7145×5118) with many tiny objects. We slide a
640×640 window over each image with stride=320 (50% overlap), drop empty
patches, and write each non-empty patch + its YOLO-normalized labels to disk.
This produces ~50K training patches in which objects are 10–40 px — the
regime where anti-aliasing should matter most.

Test set comes from DOTA's Val_dota (the official Test_dota has no labels).

Input layout:
    DOTA_1_5/
      Train_dota/{images,labelTxt}/  -> sliced + 85/15 stratified split (train/val)
      Val_dota/{images,labelTxt}/    -> sliced -> test/

Annotation format (per .txt, after 2 header lines):
    x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
    (8-point oriented polygon → we convert to axis-aligned via min/max)

Output layout (matches preprocess_ip102.py):
    dataset/dota/split/
      train/{images,labels}/  *.png + *.txt
      val/{images,labels}/
      test/{images,labels}/
      data.yaml

Run:
    python preprocess_dota.py
"""

import glob
import os
import random
import shutil
from collections import Counter

import cv2
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

import config as cfg

random.seed(cfg.SEED)
np.random.seed(cfg.SEED)


# ─────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────
PATCH = 640
STRIDE = 320
MIN_VISIBILITY = 0.3       # keep box if (clipped_area / orig_area) >= this
VAL_FRAC = 0.15            # of Train_dota patches, stratified by primary class
DROP_EMPTY = True          # skip patches with no kept boxes


# ─────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────

def parse_dota_label(txt_path: str):
    """
    Parse one DOTA labelTxt file.
    Returns list of (cls_name, x1, y1, x2, y2) axis-aligned boxes (pixel coords).
    Returns [] if file is missing or has no valid objects.
    """
    if not os.path.exists(txt_path):
        return []
    boxes = []
    with open(txt_path) as f:
        for i, raw in enumerate(f):
            line = raw.strip()
            if i < 2:        # skip "imagesource:..." and "gsd:..."
                continue
            if not line:
                continue
            parts = line.split()
            # 8 polygon coords + class_name + difficulty (some files omit difficulty)
            if len(parts) < 9:
                continue
            try:
                pts = list(map(float, parts[:8]))
            except ValueError:
                continue
            cls_name = parts[8]
            xs = pts[0::2]
            ys = pts[1::2]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            if x2 - x1 <= 1 or y2 - y1 <= 1:
                continue
            boxes.append((cls_name, x1, y1, x2, y2))
    return boxes


# ─────────────────────────────────────────────────────────────────────
# Patch slicing
# ─────────────────────────────────────────────────────────────────────

def _window_origins(dim: int, patch: int, stride: int):
    """
    Yield window start indices along one axis.  Snaps the final window so its
    right/bottom edge lands exactly at `dim` (avoids OOB and ensures full
    coverage even when (dim - patch) is not divisible by stride).
    For dim <= patch we yield a single origin at 0 (the patch extractor pads/letterbox-handles upstream).
    """
    if dim <= patch:
        return [0]
    origins = list(range(0, dim - patch + 1, stride))
    last = dim - patch
    if origins[-1] != last:
        origins.append(last)
    return origins


def slice_image_with_boxes(img: np.ndarray, boxes, cls_to_id: dict,
                           patch: int = PATCH, stride: int = STRIDE,
                           min_vis: float = MIN_VISIBILITY):
    """
    Slide a `patch`×`patch` window over `img`, clip boxes to each window, keep
    boxes whose visible fraction ≥ min_vis, and emit (patch_array, yolo_labels)
    tuples.  Patches with zero kept boxes are returned with an empty label list;
    caller decides whether to drop them.

    `boxes` is list of (cls_name, x1, y1, x2, y2) in image pixel coords.
    Returns list of dicts: {"row", "col", "img": HxWx3 uint8, "labels": [(cid, xc, yc, w, h)]}.
    """
    H, W = img.shape[:2]
    out = []

    # Pre-compute box areas once
    box_data = []
    for (name, bx1, by1, bx2, by2) in boxes:
        if name not in cls_to_id:
            continue
        area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        if area <= 0:
            continue
        box_data.append((cls_to_id[name], bx1, by1, bx2, by2, area))

    ys = _window_origins(H, patch, stride)
    xs = _window_origins(W, patch, stride)

    for row, y0 in enumerate(ys):
        for col, x0 in enumerate(xs):
            x1w, y1w = x0, y0
            x2w, y2w = x0 + patch, y0 + patch

            # If the requested window extends past the image (only happens when
            # H<patch or W<patch, since _window_origins snaps the last origin
            # otherwise), crop what we have and zero-pad the remainder.
            crop_x2 = min(x2w, W)
            crop_y2 = min(y2w, H)
            crop = img[y1w:crop_y2, x1w:crop_x2]
            if crop.shape[0] != patch or crop.shape[1] != patch:
                tile = np.zeros((patch, patch, 3), dtype=img.dtype)
                tile[:crop.shape[0], :crop.shape[1]] = crop
            else:
                tile = crop

            labels = []
            for (cid, bx1, by1, bx2, by2, area) in box_data:
                ix1 = max(bx1, x1w)
                iy1 = max(by1, y1w)
                ix2 = min(bx2, x2w)
                iy2 = min(by2, y2w)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = (ix2 - ix1) * (iy2 - iy1)
                vis = inter / area
                if vis < min_vis:
                    continue
                # Convert to YOLO (normalized within patch frame)
                cx = ((ix1 + ix2) / 2.0 - x1w) / patch
                cy = ((iy1 + iy2) / 2.0 - y1w) / patch
                bw = (ix2 - ix1) / patch
                bh = (iy2 - iy1) / patch
                # Clamp to [0, 1]
                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)
                bw = min(max(bw, 0.0), 1.0)
                bh = min(max(bh, 0.0), 1.0)
                if bw <= 0 or bh <= 0:
                    continue
                labels.append((cid, cx, cy, bw, bh))

            out.append({"row": row, "col": col, "img": tile, "labels": labels})
    return out


# ─────────────────────────────────────────────────────────────────────
# Writing
# ─────────────────────────────────────────────────────────────────────

def _write_label_file(path: str, labels):
    with open(path, "w") as f:
        for (c, xc, yc, w, h) in labels:
            f.write(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def write_patch(split_dir: str, split_name: str, patch_id: str,
                tile: np.ndarray, labels) -> None:
    img_out = os.path.join(split_dir, split_name, "images", f"{patch_id}.png")
    lbl_out = os.path.join(split_dir, split_name, "labels", f"{patch_id}.txt")
    cv2.imwrite(img_out, tile)
    _write_label_file(lbl_out, labels)


def write_data_yaml(split_dir: str, class_names):
    data = {
        "path":  split_dir,
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    len(class_names),
        "names": class_names,
    }
    yaml_path = os.path.join(split_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def stratify_key(labels) -> str:
    """Primary class = most-common class id in the patch (ties → smallest id)."""
    if not labels:
        return "empty"
    cnt = Counter(c for (c, *_) in labels)
    return str(min(c for c, n in cnt.items() if n == max(cnt.values())))


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  DOTA v1.5 (v1.0 labels, 15 classes)  →  640×640 YOLO patches")
    print(f"  patch={PATCH}  stride={STRIDE}  min_visibility={MIN_VISIBILITY}")
    print("=" * 64)

    ds_cfg = cfg.load_dataset_cfg("dota")
    class_names = ds_cfg["class_names"]
    cls_to_id = {name: i for i, name in enumerate(class_names)}
    print(f"  Classes ({len(class_names)}): {class_names}")

    split_dir = ds_cfg["split_dir"]
    if os.path.exists(split_dir):
        print(f"  Removing existing split dir: {split_dir}")
        shutil.rmtree(split_dir)
    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(split_dir, sp, "images"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, sp, "labels"), exist_ok=True)

    sources = [
        ("trainval", os.path.join(cfg.DOTA_ROOT, "Train_dota", "images"),
                     os.path.join(cfg.DOTA_ROOT, "Train_dota", "labelTxt")),
        ("test",     os.path.join(cfg.DOTA_ROOT, "Val_dota",   "images"),
                     os.path.join(cfg.DOTA_ROOT, "Val_dota",   "labelTxt")),
    ]

    # Phase 1: slice everything, collect (patch_id, tile, labels, split-bucket)
    #          - trainval patches are held in memory descriptors and split below
    #          - test patches are written straight to disk
    trainval_patches = []   # list of (patch_id, tile, labels, primary_cls_key)
    test_patches_written = 0
    test_empty_dropped = 0

    for tag, img_dir, lbl_dir in sources:
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        print(f"\n  [{tag}] {len(img_paths)} source images from {img_dir}")
        local_patches = 0
        local_empty_dropped = 0
        instance_count = Counter()

        for i, img_path in enumerate(img_paths, 1):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"    !! failed to read {img_path}")
                continue
            boxes = parse_dota_label(os.path.join(lbl_dir, f"{stem}.txt"))
            patches = slice_image_with_boxes(img, boxes, cls_to_id)

            for p in patches:
                if DROP_EMPTY and not p["labels"]:
                    local_empty_dropped += 1
                    continue
                patch_id = f"{stem}__r{p['row']}_c{p['col']}"
                for (c, *_rest) in p["labels"]:
                    instance_count[c] += 1
                if tag == "trainval":
                    trainval_patches.append((patch_id, p["img"], p["labels"],
                                             stratify_key(p["labels"])))
                else:  # test goes straight to disk
                    write_patch(split_dir, "test", patch_id, p["img"], p["labels"])
                    test_patches_written += 1
                local_patches += 1

            if i % 100 == 0 or i == len(img_paths):
                print(f"    {i}/{len(img_paths)} images -> {local_patches} patches"
                      f" (dropped {local_empty_dropped} empty)", flush=True)

        if tag == "test":
            test_empty_dropped = local_empty_dropped
        else:
            trainval_empty_dropped = local_empty_dropped
        print(f"  [{tag}] kept {local_patches} patches across {len(class_names)} classes; "
              f"instance counts: {dict(sorted(instance_count.items()))}")

    # Phase 2: stratified split of trainval into train/val (85/15)
    print(f"\n  Total trainval patches: {len(trainval_patches)}")
    keys = [k for (_pid, _img, _lbl, k) in trainval_patches]
    counts = Counter(keys)
    safe_keys = [k if counts[k] >= 3 else "rare" for k in keys]
    train_idx, val_idx = train_test_split(
        list(range(len(trainval_patches))),
        test_size=VAL_FRAC,
        random_state=cfg.SEED,
        stratify=safe_keys,
    )
    print(f"  Stratified split: train={len(train_idx)}  val={len(val_idx)}")

    train_class_count = Counter()
    val_class_count   = Counter()
    test_class_count  = Counter()

    print("\n  Writing train/val patches to disk...")
    for i, idx in enumerate(train_idx, 1):
        pid, tile, labels, _ = trainval_patches[idx]
        write_patch(split_dir, "train", pid, tile, labels)
        for (c, *_r) in labels:
            train_class_count[c] += 1
        if i % 2000 == 0:
            print(f"    train: {i}/{len(train_idx)}", flush=True)
    print(f"    train: {len(train_idx)} written")

    for i, idx in enumerate(val_idx, 1):
        pid, tile, labels, _ = trainval_patches[idx]
        write_patch(split_dir, "val", pid, tile, labels)
        for (c, *_r) in labels:
            val_class_count[c] += 1
        if i % 2000 == 0:
            print(f"    val: {i}/{len(val_idx)}", flush=True)
    print(f"    val: {len(val_idx)} written")

    # Count test class instances by re-reading the labels we already wrote
    for p in glob.glob(os.path.join(split_dir, "test", "labels", "*.txt")):
        with open(p) as f:
            for line in f:
                s = line.split()
                if s:
                    try:
                        test_class_count[int(s[0])] += 1
                    except ValueError:
                        continue

    yaml_path = write_data_yaml(split_dir, class_names)

    # ── Final stats ──────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  DOTA preprocessing complete")
    print("=" * 64)
    print(f"  Output dir         : {split_dir}")
    print(f"  data.yaml          : {yaml_path}")
    print(f"  Patch size / stride: {PATCH} / {STRIDE}  (min_vis={MIN_VISIBILITY})")
    print(f"  Train patches      : {len(train_idx):>7d}")
    print(f"  Val   patches      : {len(val_idx):>7d}")
    print(f"  Test  patches      : {test_patches_written:>7d}  (from Val_dota)")
    total = len(train_idx) + len(val_idx) + test_patches_written
    print(f"  TOTAL patches      : {total:>7d}")
    print(f"  Empty patches dropped — trainval: {trainval_empty_dropped}, "
          f"test: {test_empty_dropped}")
    print("\n  Per-class instance counts:")
    print(f"  {'idx':<4} {'class':<22} {'train':>8} {'val':>8} {'test':>8}")
    for cid, name in enumerate(class_names):
        print(f"  {cid:<4} {name:<22} "
              f"{train_class_count[cid]:>8d} "
              f"{val_class_count[cid]:>8d} "
              f"{test_class_count[cid]:>8d}")
    print(f"  {'':4} {'TOTAL':<22} "
          f"{sum(train_class_count.values()):>8d} "
          f"{sum(val_class_count.values()):>8d} "
          f"{sum(test_class_count.values()):>8d}")
    print()


if __name__ == "__main__":
    main()
