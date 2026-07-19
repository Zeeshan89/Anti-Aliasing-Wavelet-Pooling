"""
Preprocess images, apply augmentation, and split into train/val/test sets.

Pipeline:
  1. Read raw images + YOLO labels from  dataset/raw/
  2. Apply offline augmentation (Albumentations) to expand the training data
  3. Stratified split into train / val / test
  4. Write out the final directory structure expected by Ultralytics YOLO:
        dataset/split/
            train/images/  train/labels/
            val/images/    val/labels/
            test/images/   test/labels/
            data.yaml
  5. Augmentation is applied ONLY to the training split
     (val and test remain un-augmented for honest evaluation)

Run:
    python preprocess_augment_split.py
"""

import os
import glob
import shutil
import random
import yaml
import cv2
import numpy as np
import albumentations as A
from sklearn.model_selection import train_test_split
import config as cfg

random.seed(cfg.SEED)
np.random.seed(cfg.SEED)


# ─────────────────────────────────────────────────────────────────────
# 1.  Discover raw data
# ─────────────────────────────────────────────────────────────────────
def _discover_samples(raw_dir):
    """Return list of (image_path, label_path) pairs from the raw dataset."""
    img_dir = os.path.join(raw_dir, "images")
    lbl_dir = os.path.join(raw_dir, "labels")
    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    samples = []
    for img_path in images:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, f"{stem}.txt")
        if os.path.exists(lbl_path):
            samples.append((img_path, lbl_path))
    return samples

def _discover_presplit(dataset_dir, split_name):
    """Load already-split image-label pairs, handling jpg/bmp/png."""
    img_dir = os.path.join(dataset_dir, "images", split_name)
    lbl_dir = os.path.join(dataset_dir, "labels", split_name)
    samples = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".bmp", ".png")):
            continue
        stem = os.path.splitext(fname)[0]
        lbl_path = os.path.join(lbl_dir, f"{stem}.txt")
        if os.path.exists(lbl_path):
            samples.append((os.path.join(img_dir, fname), lbl_path))
    return samples

def _read_yolo_labels(label_path):
    """Parse YOLO label file → list of (cls_id, x_c, y_c, w, h)."""
    labels = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                labels.append((cls_id, *coords))
    return labels


def _stratification_label(label_path):
    """
    Return a stratification key for the image.
    For single-class images: the class id (e.g. '0', '1', '2', '3').
    For multi-class images: sorted class set as string (e.g. '0_2', '1_3').
    This ensures multi-class images are spread across all splits.
    """
    labels = _read_yolo_labels(label_path)
    if not labels:
        return "empty"
    classes = sorted(set(lbl[0] for lbl in labels))
    return "_".join(str(c) for c in classes)


# ─────────────────────────────────────────────────────────────────────
# 2.  Augmentation pipeline
# ─────────────────────────────────────────────────────────────────────
def _build_augmentation_pipeline():
    """Build Albumentations pipeline matching the proposal's augmentation plan."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.4),
            A.RandomScale(scale_limit=0.3, p=0.3),
            A.PadIfNeeded(min_height=cfg.IMAGE_HEIGHT, min_width=cfg.IMAGE_WIDTH,
              border_mode=cv2.BORDER_CONSTANT, p=1.0),
            A.CenterCrop(height=cfg.IMAGE_HEIGHT, width=cfg.IMAGE_WIDTH, p=1.0),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_ids"],
            min_visibility=0.3,
        ),
    )


def _augment_sample(img_path, label_path, aug_transform, n_augments):
    """Return list of (augmented_image_np, labels_list) tuples."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    labels = _read_yolo_labels(label_path)

    bboxes = [(x, y, w, h) for (_, x, y, w, h) in labels]
    cls_ids = [c for (c, *_) in labels]

    augmented = []
    for _ in range(n_augments):
        try:
            result = aug_transform(image=img, bboxes=bboxes, class_ids=cls_ids)
            new_labels = []
            for cid, bbox in zip(result["class_ids"], result["bboxes"]):
                new_labels.append((cid, *bbox))
            augmented.append((result["image"], new_labels))
        except Exception:
            # some augmentations may fail on edge cases – skip
            continue
    return augmented


# ─────────────────────────────────────────────────────────────────────
# 3.  Split
# ─────────────────────────────────────────────────────────────────────
def _stratified_split(samples):
    """Split samples into train / val / test using stratification by class composition."""
    from collections import Counter
    strat_labels = [_stratification_label(lp) for (_, lp) in samples]

    # Merge rare strata (< 3 samples) into a catch-all bucket so
    # train_test_split doesn't fail on too-small groups.
    counts = Counter(strat_labels)
    strat_safe = [sl if counts[sl] >= 3 else "rare" for sl in strat_labels]

    # first split off test
    train_val, test, strat_tv, _ = train_test_split(
        samples, strat_safe,
        test_size=cfg.TEST_RATIO,
        random_state=cfg.SEED,
        stratify=strat_safe,
    )
    # then split train_val into train and val
    relative_val = cfg.VAL_RATIO / (cfg.TRAIN_RATIO + cfg.VAL_RATIO)
    strat_tv_counts = Counter(strat_tv)
    strat_tv_safe = [s if strat_tv_counts[s] >= 2 else "rare" for s in strat_tv]
    train, val, _, _ = train_test_split(
        train_val, strat_tv_safe,
        test_size=relative_val,
        random_state=cfg.SEED,
        stratify=strat_tv_safe,
    )
    return train, val, test


# ─────────────────────────────────────────────────────────────────────
# 4.  Write split to disk
# ─────────────────────────────────────────────────────────────────────
def _write_split(split_name, samples, split_dir, aug_transform=None, n_augments=0):
    """Copy (and optionally augment) images+labels into split_dir/<split_name>/."""
    img_out = os.path.join(split_dir, split_name, "images")
    lbl_out = os.path.join(split_dir, split_name, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    count = 0
    for img_path, lbl_path in samples:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        
        # copy original
        img_cv = cv2.imread(img_path)
        cv2.imwrite(os.path.join(img_out, f"{stem}.jpg"), img_cv)
        shutil.copy2(lbl_path, os.path.join(lbl_out, f"{stem}.txt"))
        count += 1

        # augment (training only)
        if aug_transform and n_augments > 0:
            augs = _augment_sample(img_path, lbl_path, aug_transform, n_augments)
            for j, (aug_img, aug_labels) in enumerate(augs):
                aug_stem = f"{stem}_aug{j}"
                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(img_out, f"{aug_stem}.jpg"), aug_img_bgr)
                with open(os.path.join(lbl_out, f"{aug_stem}.txt"), "w") as f:
                    for lbl in aug_labels:
                        f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")
                count += 1

    return count


def _write_data_yaml(split_dir):
    """Write the data.yaml expected by Ultralytics YOLO."""
    data = {
        "path": split_dir,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": cfg.NUM_CLASSES,
        "names": cfg.CLASS_NAMES,
    }
    yaml_path = os.path.join(split_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    return yaml_path


# ─────────────────────────────────────────────────────────────────────
# 5.  Main
# ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Preprocess  →  Re-Split  →  Augment  →  Write")
    print("=" * 60)

    # ── Pool ALL images from original train/val/test into one set ──
    all_samples = []
    for split_name in ["train", "val", "test"]:
        samples = _discover_presplit(cfg.ORIGINAL_DATASET_DIR, split_name)
        all_samples.extend(samples)
        print(f"  Original {split_name}: {len(samples)} images")

    # Deduplicate by filename stem (in case of overlap)
    seen = {}
    unique_samples = []
    for img_path, lbl_path in all_samples:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        if stem not in seen:
            seen[stem] = True
            unique_samples.append((img_path, lbl_path))
    all_samples = unique_samples
    print(f"\n  Total unique images pooled: {len(all_samples)}")

    if len(all_samples) == 0:
        print("ERROR: No samples found. Check ORIGINAL_DATASET_DIR.")
        return

    # ── Stratified re-split (70/18/12) ────────────────────────────
    train_samples, val_samples, test_samples = _stratified_split(all_samples)
    print(f"\n  New split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

    # Show multi-class distribution per split
    for name, s in [("Train", train_samples), ("Val", val_samples), ("Test", test_samples)]:
        mc = sum(1 for (_, lp) in s if len(set(lbl[0] for lbl in _read_yolo_labels(lp))) > 1)
        print(f"    {name}: {mc} multi-class images ({mc*100//len(s)}%)")

    # Wipe and recreate split output
    if os.path.exists(cfg.SPLIT_DIR):
        shutil.rmtree(cfg.SPLIT_DIR)

    aug = _build_augmentation_pipeline()

    # Augment train only; val/test copied as-is
    n_train = _write_split("train", train_samples, cfg.SPLIT_DIR,
                           aug_transform=aug, n_augments=cfg.AUG_PER_IMAGE)
    n_val   = _write_split("val",   val_samples,   cfg.SPLIT_DIR)
    n_test  = _write_split("test",  test_samples,  cfg.SPLIT_DIR)

    print(f"\n  Final counts written to disk:")
    print(f"    Train (original + augmented): {n_train}")
    print(f"    Val   (original only)       : {n_val}")
    print(f"    Test  (original only)       : {n_test}")

    yaml_path = _write_data_yaml(cfg.SPLIT_DIR)
    print(f"\n  YOLO data config: {yaml_path}")
    print("Done.")

if __name__ == "__main__":
    main()
