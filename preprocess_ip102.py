"""
Preprocess IP102 (insect pest) dataset → YOLO format + train/val/test split.

IP102_v1_1_dataset is in Pascal VOC format:
    IP102_v1_1_dataset/
      Annotations/*.xml         one XML per image (VOC format)
      JPEGImages/*.jpg          one image per stem
      Main/trainval.txt         training+validation image stems
      Main/test.txt             test image stems
      classes.txt               1-indexed list of 102 class names

XML class IDs use <name>N</name> where N is the 0-indexed class (matches
YOLO).  We split trainval.txt into train/val (85/15, stratified by primary
class) and respect IP102's predefined test split as-is.

Output layout matches the moth pipeline so train.py can consume it:
    dataset/ip102/split/
      train/{images,labels}/
      val/{images,labels}/
      test/{images,labels}/
      data.yaml

Run:
    python preprocess_ip102.py
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET
from collections import Counter

import yaml
import cv2
import numpy as np
import albumentations as A
from sklearn.model_selection import train_test_split

import config as cfg

random.seed(cfg.SEED)
np.random.seed(cfg.SEED)


# ─────────────────────────────────────────────────────────────────────
# Augmentation pipeline (used only for train split)
# ─────────────────────────────────────────────────────────────────────
def _build_ip102_aug():
    """
    IP102-specific Albumentations pipeline.  Unlike the moth pipeline
    (which crops/pads to a fixed 640x640 size on disk), this preserves
    each image's native resolution so the YOLO dataloader can resize at
    load-time.  Focuses on colour/lighting/geometry variation, which is
    what IP102's varied field-photo conditions actually need.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.4),
            A.Rotate(limit=25, border_mode=cv2.BORDER_CONSTANT, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=25, val_shift_limit=20, p=0.5),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.25),
            A.MotionBlur(blur_limit=5, p=0.15),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_ids"],
            min_visibility=0.4,
        ),
    )


def _apply_aug(img, labels, transform, n_augments):
    """
    Apply Albumentations augmentation `n_augments` times to one (img, labels)
    sample.  Returns list of (aug_img, aug_labels) tuples.  Skips silently
    if a particular augmentation drops all bboxes (rare).
    """
    bboxes  = [(x, y, w, h) for (_, x, y, w, h) in labels]
    cls_ids = [c for (c, *_) in labels]
    out = []
    for _ in range(n_augments):
        try:
            r = transform(image=img, bboxes=bboxes, class_ids=cls_ids)
            new_labels = [(c, *b) for c, b in zip(r["class_ids"], r["bboxes"])]
            if new_labels:
                out.append((r["image"], new_labels))
        except Exception:
            continue
    return out


# ─────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────

def parse_classes(classes_txt: str) -> list:
    """
    Read classes.txt → list of 102 class names (0-indexed internally).
    File format: "1  rice leaf roller", "2  rice leaf caterpillar", ...
    """
    names = []
    with open(classes_txt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[0].lstrip("-").isdigit():
                names.append(parts[1].strip())
            else:
                names.append(line)
    return names


def parse_voc_xml(xml_path: str):
    """
    Parse one VOC XML annotation.
    Returns:  (img_w, img_h, [(cls_id, x_c, y_c, w, h), ...]) in YOLO-normalized form.
    Returns None if the file is missing or has no valid objects.
    """
    try:
        tree = ET.parse(xml_path)
    except (ET.ParseError, FileNotFoundError):
        return None

    root = tree.getroot()
    size = root.find("size")
    if size is None:
        return None
    try:
        W = int(size.findtext("width"))
        H = int(size.findtext("height"))
    except (TypeError, ValueError):
        return None
    if W <= 0 or H <= 0:
        return None

    labels = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name is None:
            continue
        # <name> may be a numeric id ("0".."101") or a class name string.
        try:
            cls_id = int(name)
        except ValueError:
            # numeric expected; if not, skip
            continue

        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        try:
            xmin = float(bbox.findtext("xmin"))
            ymin = float(bbox.findtext("ymin"))
            xmax = float(bbox.findtext("xmax"))
            ymax = float(bbox.findtext("ymax"))
        except (TypeError, ValueError):
            continue

        # clamp and reject degenerate boxes
        xmin = max(0.0, min(xmin, W - 1))
        ymin = max(0.0, min(ymin, H - 1))
        xmax = max(0.0, min(xmax, W))
        ymax = max(0.0, min(ymax, H))
        if xmax <= xmin or ymax <= ymin:
            continue

        x_c = (xmin + xmax) / 2.0 / W
        y_c = (ymin + ymax) / 2.0 / H
        w   = (xmax - xmin) / W
        h   = (ymax - ymin) / H
        if w <= 0 or h <= 0 or w > 1 or h > 1:
            continue

        labels.append((cls_id, x_c, y_c, w, h))

    if not labels:
        return None
    return (W, H, labels)


def load_split_ids(txt_path: str) -> list:
    """Read trainval.txt / test.txt → list of image stems."""
    ids = []
    with open(txt_path) as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(s)
    return ids


def stratify_key(labels) -> str:
    """Stratification bucket: sorted unique class IDs joined by '_'."""
    if not labels:
        return "empty"
    classes = sorted({lbl[0] for lbl in labels})
    return "_".join(str(c) for c in classes)


# ─────────────────────────────────────────────────────────────────────
# Writing
# ─────────────────────────────────────────────────────────────────────

def _write_label_file(lbl_path: str, labels: list):
    with open(lbl_path, "w") as f:
        for (c, xc, yc, w, h) in labels:
            f.write(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def write_yolo_sample(out_split_dir: str, split_name: str, stem: str,
                      src_img_path: str, labels: list,
                      aug_transform=None, n_augments: int = 0) -> int:
    """
    Copy image + YOLO label.  If `aug_transform` is provided (train split only),
    also write `n_augments` augmented copies as `{stem}_aug{j}.jpg/.txt`.
    Returns total samples written (original + augmented).
    """
    img_out = os.path.join(out_split_dir, split_name, "images", f"{stem}.jpg")
    lbl_out = os.path.join(out_split_dir, split_name, "labels", f"{stem}.txt")

    # original (always)
    shutil.copy2(src_img_path, img_out)
    _write_label_file(lbl_out, labels)
    count = 1

    if aug_transform is None or n_augments <= 0:
        return count

    # augmented copies
    img = cv2.imread(src_img_path)
    if img is None:
        return count
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for j, (aug_img, aug_labels) in enumerate(_apply_aug(img_rgb, labels, aug_transform, n_augments)):
        aug_stem = f"{stem}_aug{j}"
        aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_split_dir, split_name, "images", f"{aug_stem}.jpg"), aug_img_bgr)
        _write_label_file(os.path.join(out_split_dir, split_name, "labels", f"{aug_stem}.txt"),
                          aug_labels)
        count += 1
    return count


def write_data_yaml(split_dir: str, class_names: list):
    """Write YOLO data.yaml for the IP102 split."""
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


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

VAL_FRAC = 0.15        # of trainval (stratified by primary class)
AUG_PER_IMG = 2        # train-only augmented copies (0 = no augmentation)


def main():
    print("=" * 60)
    print("  IP102  →  VOC XML → YOLO  →  Train/Val/Test split")
    print("=" * 60)

    ann_dir = os.path.join(cfg.IP102_ROOT, "Annotations")
    img_dir = os.path.join(cfg.IP102_ROOT, "JPEGImages")
    main_dir = os.path.join(cfg.IP102_ROOT, "Main")
    classes_txt = os.path.join(cfg.IP102_ROOT, "classes.txt")

    for p, kind in [(ann_dir, "Annotations dir"),
                    (img_dir, "JPEGImages dir"),
                    (main_dir, "Main dir"),
                    (classes_txt, "classes.txt")]:
        if not os.path.exists(p):
            print(f"ERROR: missing {kind}: {p}")
            return

    class_names = parse_classes(classes_txt)
    print(f"  Classes loaded: {len(class_names)} (first: '{class_names[0]}', last: '{class_names[-1]}')")

    # ── Load IP102's predefined splits ────────────────────────────────
    trainval_ids = load_split_ids(os.path.join(main_dir, "trainval.txt"))
    test_ids     = load_split_ids(os.path.join(main_dir, "test.txt"))
    print(f"  IP102 splits: trainval={len(trainval_ids)}  test={len(test_ids)}")

    # ── Parse all XMLs, keep only ones with valid annotations ────────
    print("\n  Parsing annotations...")
    parsed = {}    # stem -> (W, H, labels)
    skipped = 0
    for stem in trainval_ids + test_ids:
        xml_path = os.path.join(ann_dir, f"{stem}.xml")
        img_path = os.path.join(img_dir, f"{stem}.jpg")
        if not (os.path.exists(xml_path) and os.path.exists(img_path)):
            skipped += 1
            continue
        info = parse_voc_xml(xml_path)
        if info is None:
            skipped += 1
            continue
        parsed[stem] = info
    print(f"  Parsed: {len(parsed)} usable  |  Skipped (missing/empty): {skipped}")

    trainval_ids = [s for s in trainval_ids if s in parsed]
    test_ids     = [s for s in test_ids     if s in parsed]
    print(f"  After validation: trainval={len(trainval_ids)}  test={len(test_ids)}")

    # ── Stratified train/val split within trainval ───────────────────
    strat_labels = [stratify_key(parsed[s][2]) for s in trainval_ids]
    counts = Counter(strat_labels)
    strat_safe = [s if counts[s] >= 3 else "rare" for s in strat_labels]
    train_ids, val_ids, _, _ = train_test_split(
        trainval_ids, strat_safe,
        test_size=VAL_FRAC,
        random_state=cfg.SEED,
        stratify=strat_safe,
    )
    print(f"\n  Split (within trainval): Train={len(train_ids)}  Val={len(val_ids)}")

    # ── Recreate output split directory ──────────────────────────────
    split_dir = cfg.IP102_SPLIT_DIR
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)
    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(split_dir, sp, "images"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, sp, "labels"), exist_ok=True)

    # ── Write samples ────────────────────────────────────────────────
    aug_transform = _build_ip102_aug() if AUG_PER_IMG > 0 else None
    if aug_transform is not None:
        print(f"\n  Writing samples (train: {AUG_PER_IMG}× Albumentations augmentation)...")
    else:
        print("\n  Writing samples to disk (no augmentation)...")

    for split_name, stems in (("train", train_ids),
                              ("val", val_ids),
                              ("test", test_ids)):
        total = 0
        # Augmentation applied to TRAIN only; val and test stay clean.
        aug = aug_transform if split_name == "train" else None
        n_aug = AUG_PER_IMG if split_name == "train" else 0
        for stem in stems:
            _, _, labels = parsed[stem]
            src_img = os.path.join(img_dir, f"{stem}.jpg")
            total += write_yolo_sample(split_dir, split_name, stem, src_img, labels,
                                       aug_transform=aug, n_augments=n_aug)
        suffix = f" ({len(stems)} originals + {total - len(stems)} augmented)" if aug else ""
        print(f"    {split_name}: {total} samples written{suffix}")

    # ── Write data.yaml ──────────────────────────────────────────────
    yaml_path = write_data_yaml(split_dir, class_names)
    print(f"\n  YOLO data.yaml: {yaml_path}")

    # ── Quick stats: class distribution in train ─────────────────────
    cls_counter = Counter()
    for stem in train_ids:
        for c, *_ in parsed[stem][2]:
            cls_counter[c] += 1
    nonzero = sum(1 for c in range(len(class_names)) if cls_counter[c] > 0)
    rarest = min(cls_counter.values()) if cls_counter else 0
    commonest = max(cls_counter.values()) if cls_counter else 0
    print(f"\n  Train class coverage: {nonzero}/{len(class_names)} classes have ≥1 sample")
    print(f"  Train sample count per class:  min={rarest}  max={commonest}")
    print("\nDone.")


if __name__ == "__main__":
    main()
