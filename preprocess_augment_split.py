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


def _dominant_class(label_path):
    """Return the most-frequent class id in the label file (for stratification)."""
    labels = _read_yolo_labels(label_path)
    if not labels:
        return 0
    counts = {}
    for lbl in labels:
        counts[lbl[0]] = counts.get(lbl[0], 0) + 1
    return max(counts, key=counts.get)


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
    """Split samples into train / val / test using stratification by dominant class."""
    strat_labels = [_dominant_class(lp) for (_, lp) in samples]

    # first split off test
    train_val, test, strat_tv, _ = train_test_split(
        samples, strat_labels,
        test_size=cfg.TEST_RATIO,
        random_state=cfg.SEED,
        stratify=strat_labels,
    )
    # then split train_val into train and val
    relative_val = cfg.VAL_RATIO / (cfg.TRAIN_RATIO + cfg.VAL_RATIO)
    train, val, _, _ = train_test_split(
        train_val, strat_tv,
        test_size=relative_val,
        random_state=cfg.SEED,
        stratify=strat_tv,
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
    print("  Preprocess  →  Augment  →  Write Pre-Split Data")
    print("=" * 60)

    train_samples = _discover_presplit(cfg.ORIGINAL_DATASET_DIR, "train")
    val_samples   = _discover_presplit(cfg.ORIGINAL_DATASET_DIR, "val")
    test_samples  = _discover_presplit(cfg.ORIGINAL_DATASET_DIR, "test")

    print(f"\nFound: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    if len(train_samples) == 0:
        print("ERROR: No training samples found. Check ORIGINAL_DATASET_DIR.")
        return

    # Wipe and recreate split output
    if os.path.exists(cfg.SPLIT_DIR):
        shutil.rmtree(cfg.SPLIT_DIR)

    aug = _build_augmentation_pipeline()

    # Augment train only; val/test copied as-is
    n_train = _write_split("train", train_samples, cfg.SPLIT_DIR,
                           aug_transform=aug, n_augments=cfg.AUG_PER_IMAGE)
    n_val   = _write_split("val",   val_samples,   cfg.SPLIT_DIR)
    n_test  = _write_split("test",  test_samples,  cfg.SPLIT_DIR)

    print(f"\nFinal counts written to disk:")
    print(f"  Train (original + augmented): {n_train}")
    print(f"  Val   (original only)       : {n_val}")
    print(f"  Test  (original only)       : {n_test}")

    yaml_path = _write_data_yaml(cfg.SPLIT_DIR)
    print(f"\nYOLO data config: {yaml_path}")
    print("Done.")

if __name__ == "__main__":
    main()
