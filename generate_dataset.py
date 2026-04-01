"""
Generate a synthetic dummy dataset of sticky-trap moth images in YOLO format.

Each image simulates a sticky trap background with moth objects (whole moths,
head/thorax fragments, wing fragments, abdomen fragments) drawn as simple
geometric shapes.  Every image gets a matching .txt label file in YOLO format:
    <class_id> <x_center> <y_center> <width> <height>   (normalised 0-1)

Run:
    python generate_dataset.py
"""

import os
import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import config as cfg


# ── reproducibility ──────────────────────────────────────────────────
random.seed(cfg.SEED)
np.random.seed(cfg.SEED)


# ── helpers ──────────────────────────────────────────────────────────
def _random_color(base, var=30):
    """Return a slightly randomised colour tuple around *base*."""
    return tuple(max(0, min(255, b + random.randint(-var, var))) for b in base)


def _sticky_trap_background(w, h):
    """Create a yellowish-brown sticky-trap looking background."""
    base = _random_color((200, 180, 130), var=20)
    img = Image.new("RGB", (w, h), base)
    draw = ImageDraw.Draw(img)
    # add some random noise / speckles to mimic glue residue
    for _ in range(random.randint(100, 300)):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        r = random.randint(1, 3)
        c = _random_color((170, 155, 110), var=25)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=c)
    # slight blur to emulate real trap texture
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    return img


def _draw_whole_moth(draw, cx, cy, size):
    """Elliptical body + triangular wings → whole moth silhouette."""
    body_color = _random_color((80, 60, 40))
    wing_color = _random_color((110, 90, 60))
    bw, bh = size // 4, size // 2
    # body
    draw.ellipse([cx - bw, cy - bh, cx + bw, cy + bh], fill=body_color)
    # left wing
    pts_l = [(cx - bw, cy - bh // 2),
             (cx - size // 2, cy),
             (cx - bw, cy + bh // 2)]
    draw.polygon(pts_l, fill=wing_color)
    # right wing
    pts_r = [(cx + bw, cy - bh // 2),
             (cx + size // 2, cy),
             (cx + bw, cy + bh // 2)]
    draw.polygon(pts_r, fill=wing_color)
    half = size // 2
    return (cx - half, cy - half, cx + half, cy + half)


def _draw_head_thorax(draw, cx, cy, size):
    """Small circle (head) + ellipse (thorax)."""
    color = _random_color((70, 50, 35))
    r = size // 4
    # head
    draw.ellipse([cx - r, cy - size // 3, cx + r, cy], fill=color)
    # thorax
    draw.ellipse([cx - r, cy, cx + r, cy + size // 3], fill=_random_color((90, 70, 50)))
    return (cx - r, cy - size // 3, cx + r, cy + size // 3)


def _draw_wing_fragment(draw, cx, cy, size):
    """A single triangular wing shape."""
    color = _random_color((120, 100, 70))
    half = size // 2
    angle = random.uniform(0, 2 * math.pi)
    pts = []
    for i in range(3):
        a = angle + i * (2 * math.pi / 3)
        pts.append((cx + int(half * math.cos(a)), cy + int(half * math.sin(a))))
    draw.polygon(pts, fill=color)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _draw_abdomen_fragment(draw, cx, cy, size):
    """Elongated ellipse representing abdomen."""
    color = _random_color((85, 65, 45))
    w = size // 5
    h = size // 2
    draw.ellipse([cx - w, cy - h, cx + w, cy + h], fill=color)
    return (cx - w, cy - h, cx + w, cy + h)


DRAW_FNS = {
    0: _draw_whole_moth,
    1: _draw_head_thorax,
    2: _draw_wing_fragment,
    3: _draw_abdomen_fragment,
}


def _clamp_bbox(bbox, w, h):
    x1 = max(0, bbox[0])
    y1 = max(0, bbox[1])
    x2 = min(w - 1, bbox[2])
    y2 = min(h - 1, bbox[3])
    return (x1, y1, x2, y2)


def _bbox_to_yolo(bbox, img_w, img_h):
    """Convert (x1,y1,x2,y2) → normalised (x_c, y_c, w, h)."""
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    if bw < 3 or bh < 3:
        return None
    x_c = (x1 + x2) / 2.0 / img_w
    y_c = (y1 + y2) / 2.0 / img_h
    return (x_c, y_c, bw / img_w, bh / img_h)


def generate_dataset():
    """Generate the full synthetic dataset in YOLO format."""
    img_dir = os.path.join(cfg.RAW_DATASET_DIR, "images")
    lbl_dir = os.path.join(cfg.RAW_DATASET_DIR, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    W, H = cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT
    class_counts = {i: 0 for i in range(cfg.NUM_CLASSES)}
    total = cfg.NUM_SYNTHETIC_IMAGES

    print(f"Generating {total} synthetic sticky-trap images ({W}x{H}) ...")

    for idx in range(total):
        img = _sticky_trap_background(W, H)
        draw = ImageDraw.Draw(img)
        n_objects = random.randint(*cfg.OBJECTS_PER_IMAGE)
        labels = []

        for _ in range(n_objects):
            cls_id = random.randint(0, cfg.NUM_CLASSES - 1)
            size = random.randint(25, 80)  # small objects on purpose
            cx = random.randint(size, W - size)
            cy = random.randint(size, H - size)
            bbox = DRAW_FNS[cls_id](draw, cx, cy, size)
            bbox = _clamp_bbox(bbox, W, H)
            yolo = _bbox_to_yolo(bbox, W, H)
            if yolo:
                labels.append(f"{cls_id} {yolo[0]:.6f} {yolo[1]:.6f} {yolo[2]:.6f} {yolo[3]:.6f}")
                class_counts[cls_id] += 1

        # rotate entire image randomly for variety
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            img = img.rotate(angle, expand=False)

        fname = f"trap_{idx:04d}"
        img.save(os.path.join(img_dir, f"{fname}.jpg"), quality=95)
        with open(os.path.join(lbl_dir, f"{fname}.txt"), "w") as f:
            f.write("\n".join(labels))

    print("Done. Class distribution:")
    for cid, name in enumerate(cfg.CLASS_NAMES):
        print(f"  [{cid}] {name:20s} : {class_counts[cid]} instances")
    print(f"Images saved to: {img_dir}")
    print(f"Labels saved to: {lbl_dir}")


if __name__ == "__main__":
    generate_dataset()
