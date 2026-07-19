"""
Preprocess DIOR (Dataset for Object Detection in Optical Remote Sensing Images)
→ YOLO format.

DIOR ships in Pascal VOC layout with the official 3-split already provided:
  DIOR_dataset/
    Annotations/*.xml        VOC XML for all 23,463 images
    ImageSets/Main/
      train.txt              5,862 stems (IDs 00001-05862)
      val.txt                5,863 stems (IDs 05863-11725)
      test.txt              11,738 stems (IDs 11726-23463)
    JPEGImages-trainval/     11,725 JPGs (IDs 00001-11725)
    JPEGImages-test/         11,737 JPGs (IDs 11726-23463; 19362.jpg is missing)

Simpler than DOTA and IP102:
  * no patching — images are already 800x800
  * no re-splitting — use official train/val/test
  * class names are strings in <name>; we build cls_name -> id from a fixed
    alphabetically-sorted list preserving original case (config.DIOR_CLASS_NAMES)

Output layout matches the other preprocessors so train.py can consume it:
    dataset/dior/split/
      train/{images,labels}/
      val/{images,labels}/
      test/{images,labels}/
      data.yaml

Run:
    python preprocess_dior.py
"""

import os
import shutil
import xml.etree.ElementTree as ET
from collections import Counter

import yaml

import config as cfg


IMAGE_EXT = ".jpg"


# ─────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────

def parse_voc_xml(xml_path: str, cls_to_id: dict):
    """
    Parse one DIOR VOC XML annotation.
    Returns  (img_w, img_h, [(cls_id, x_c, y_c, w, h), ...]) in YOLO-normalized form.
    Returns None if unreadable or contains no valid objects.
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
        cname = obj.findtext("name")
        if cname is None or cname not in cls_to_id:
            continue
        cid = cls_to_id[cname]
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            xmin = float(bb.findtext("xmin"))
            ymin = float(bb.findtext("ymin"))
            xmax = float(bb.findtext("xmax"))
            ymax = float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        xmin = max(0.0, min(xmin, W - 1))
        ymin = max(0.0, min(ymin, H - 1))
        xmax = max(0.0, min(xmax, W))
        ymax = max(0.0, min(ymax, H))
        if xmax <= xmin or ymax <= ymin:
            continue
        xc = (xmin + xmax) / 2.0 / W
        yc = (ymin + ymax) / 2.0 / H
        w  = (xmax - xmin) / W
        h  = (ymax - ymin) / H
        if w <= 0 or h <= 0 or w > 1 or h > 1:
            continue
        labels.append((cid, xc, yc, w, h))
    if not labels:
        return None
    return (W, H, labels)


def load_split_ids(txt_path: str):
    ids = []
    with open(txt_path) as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(s)
    return ids


# ─────────────────────────────────────────────────────────────────────
# Writing
# ─────────────────────────────────────────────────────────────────────

def _write_label_file(path: str, labels):
    with open(path, "w") as f:
        for (c, xc, yc, w, h) in labels:
            f.write(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def write_sample(split_dir, split_name, stem, src_img_path, labels):
    img_out = os.path.join(split_dir, split_name, "images", f"{stem}{IMAGE_EXT}")
    lbl_out = os.path.join(split_dir, split_name, "labels", f"{stem}.txt")
    shutil.copy2(src_img_path, img_out)
    _write_label_file(lbl_out, labels)


def write_data_yaml(split_dir, class_names):
    data = {
        "path":  split_dir,
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    len(class_names),
        "names": class_names,
    }
    p = os.path.join(split_dir, "data.yaml")
    with open(p, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return p


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  DIOR  →  VOC XML → YOLO  (official train/val/test)")
    print("=" * 64)

    ds_cfg = cfg.load_dataset_cfg("dior")
    class_names = ds_cfg["class_names"]
    cls_to_id = {n: i for i, n in enumerate(class_names)}
    print(f"  Classes ({len(class_names)}): {class_names}")

    ann_dir       = os.path.join(cfg.DIOR_ROOT, "Annotations")
    trainval_dir  = os.path.join(cfg.DIOR_ROOT, "JPEGImages-trainval")
    test_dir      = os.path.join(cfg.DIOR_ROOT, "JPEGImages-test")
    sets_dir      = os.path.join(cfg.DIOR_ROOT, "ImageSets", "Main")

    for path, kind in [(ann_dir, "Annotations"), (trainval_dir, "JPEGImages-trainval"),
                       (test_dir, "JPEGImages-test"), (sets_dir, "ImageSets/Main")]:
        if not os.path.isdir(path):
            print(f"ERROR: missing {kind} at {path}")
            return

    train_ids = load_split_ids(os.path.join(sets_dir, "train.txt"))
    val_ids   = load_split_ids(os.path.join(sets_dir, "val.txt"))
    test_ids  = load_split_ids(os.path.join(sets_dir, "test.txt"))
    print(f"  Splits: train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    split_dir = ds_cfg["split_dir"]
    if os.path.exists(split_dir):
        print(f"  Removing existing split dir: {split_dir}")
        shutil.rmtree(split_dir)
    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(split_dir, sp, "images"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, sp, "labels"), exist_ok=True)

    # Parse XMLs on demand as we write each split
    unknown_names = Counter()
    written = {"train": 0, "val": 0, "test": 0}
    missing_img = {"train": 0, "val": 0, "test": 0}
    missing_ann = {"train": 0, "val": 0, "test": 0}
    empty_ann   = {"train": 0, "val": 0, "test": 0}
    class_counts = {"train": Counter(), "val": Counter(), "test": Counter()}

    for split_name, ids, img_dir in (("train", train_ids, trainval_dir),
                                     ("val",   val_ids,   trainval_dir),
                                     ("test",  test_ids,  test_dir)):
        print(f"\n  Processing {split_name} ({len(ids)} ids)...")
        for i, stem in enumerate(ids, 1):
            xml_path = os.path.join(ann_dir, f"{stem}.xml")
            img_path = os.path.join(img_dir, f"{stem}{IMAGE_EXT}")
            if not os.path.exists(xml_path):
                missing_ann[split_name] += 1
                continue
            if not os.path.exists(img_path):
                missing_img[split_name] += 1
                continue

            # Sanity: track unknown class names once
            try:
                for obj in ET.parse(xml_path).getroot().findall("object"):
                    cname = obj.findtext("name")
                    if cname and cname not in cls_to_id:
                        unknown_names[cname] += 1
            except ET.ParseError:
                pass

            info = parse_voc_xml(xml_path, cls_to_id)
            if info is None:
                empty_ann[split_name] += 1
                continue
            W, H, labels = info

            write_sample(split_dir, split_name, stem, img_path, labels)
            written[split_name] += 1
            for (c, *_r) in labels:
                class_counts[split_name][c] += 1

            if i % 1000 == 0 or i == len(ids):
                print(f"    {i}/{len(ids)}  written={written[split_name]}", flush=True)

    yaml_path = write_data_yaml(split_dir, class_names)

    # ── Report ────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  DIOR preprocessing complete")
    print("=" * 64)
    print(f"  Output dir : {split_dir}")
    print(f"  data.yaml  : {yaml_path}")
    print(f"\n  Written per split (originals only, no augmentation):")
    for sp in ("train", "val", "test"):
        expected = {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)}[sp]
        print(f"    {sp:<5}: {written[sp]:>6d} / {expected:>6d}   "
              f"missing_img={missing_img[sp]}  missing_ann={missing_ann[sp]}  "
              f"empty_ann={empty_ann[sp]}")
    if unknown_names:
        print(f"\n  ⚠ Unknown class names encountered (not in DIOR_CLASS_NAMES):")
        for n, c in unknown_names.most_common():
            print(f"    {n!r}: {c} objects — objects with these names were skipped")

    print(f"\n  Per-class instance counts:")
    print(f"  {'idx':<4} {'class':<26} {'train':>8} {'val':>8} {'test':>8}")
    for cid, cname in enumerate(class_names):
        print(f"  {cid:<4} {cname:<26} "
              f"{class_counts['train'][cid]:>8d} "
              f"{class_counts['val'][cid]:>8d} "
              f"{class_counts['test'][cid]:>8d}")
    print(f"  {'':4} {'TOTAL':<26} "
          f"{sum(class_counts['train'].values()):>8d} "
          f"{sum(class_counts['val'].values()):>8d} "
          f"{sum(class_counts['test'].values()):>8d}")
    print()


if __name__ == "__main__":
    main()
