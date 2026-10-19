"""
SODA-D (COCO json) -> YOLO tiles, matching the SODA-D protocol's 800x800 / 150 px overlap patches.

Same convention as the DOTA patches used in the paper: evaluation is on tiles
(not merged full-image predictions), stated in the experimental setup.

  - ignore regions (category 'ignore' or ignore=1) are dropped
  - a box is kept in a tile if >= 50% of its area lies inside; it is clipped to the tile
  - tiles without any kept box are kept for val/test, and for train only 1 in 10 (background)

Usage:
  python preprocess_sodad.py --train-json SODA_D_train.json
"""

import argparse
import json
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from PIL import Image

SODA = "/lustre/orion/cli138/proj-shared/z7a/tiny_ovd_bench_project/datasets/sodad"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "sodad", "split")
TILE, OVERLAP, MIN_VIS = 800, 150, 0.5


def starts(n):
    step = TILE - OVERLAP
    s = list(range(0, max(n - TILE, 0) + 1, step))
    if s[-1] + TILE < n:
        s.append(n - TILE)
    return s


def tile_image(job):
    split, im, anns, cmap, keep_empty = job
    src = os.path.join(SODA, "images", os.path.basename(im["file_name"]))
    stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
    W, H = im["width"], im["height"]
    img = None
    n = 0
    rng = random.Random(im["id"])
    for y0 in starts(H):
        for x0 in starts(W):
            x1, y1 = min(x0 + TILE, W), min(y0 + TILE, H)
            tw, th = x1 - x0, y1 - y0
            lines = []
            for a in anns:
                bx, by, bw, bh = a["bbox"]
                ix0, iy0 = max(bx, x0), max(by, y0)
                ix1, iy1 = min(bx + bw, x1), min(by + bh, y1)
                if ix1 <= ix0 or iy1 <= iy0 or bw * bh <= 0:
                    continue
                if (ix1 - ix0) * (iy1 - iy0) / (bw * bh) < MIN_VIS:
                    continue
                cx, cy = (ix0 + ix1) / 2 - x0, (iy0 + iy1) / 2 - y0
                lines.append(f"{cmap[a['category_id']]} {cx / tw:.6f} {cy / th:.6f} "
                             f"{(ix1 - ix0) / tw:.6f} {(iy1 - iy0) / th:.6f}")
            if not lines and not (keep_empty or rng.random() < 0.1):
                continue
            name = f"{stem}__{x0}_{y0}"
            if img is None:
                img = Image.open(src).convert("RGB")
            img.crop((x0, y0, x1, y1)).save(os.path.join(OUT, split, "images", name + ".jpg"), quality=95)
            with open(os.path.join(OUT, split, "labels", name + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    jsons = {"train": args.train_json,
             "val": os.path.join(SODA, "annotations", "val.json"),
             "test": os.path.join(SODA, "annotations", "test.json")}
    names = None
    for split, path in jsons.items():
        j = json.load(open(path))
        cats = [c for c in j["categories"] if c["name"] != "ignore"]
        if names is None:
            names = {i: c["name"] for i, c in enumerate(sorted(cats, key=lambda c: c["id"]))}
        cmap = {c["id"]: i for i, c in enumerate(sorted(cats, key=lambda c: c["id"]))}
        by_img = defaultdict(list)
        for a in j["annotations"]:
            if a.get("ignore", 0) or a.get("iscrowd", 0) or a["category_id"] not in cmap:
                continue
            by_img[a["image_id"]].append(a)
        for k in ("images", "labels"):
            os.makedirs(os.path.join(OUT, split, k), exist_ok=True)
        jobs = [(split, im, by_img[im["id"]], cmap, split != "train") for im in j["images"]]
        with ProcessPoolExecutor(args.workers) as ex:
            tiles = sum(ex.map(tile_image, jobs, chunksize=8))
        print(f"{split}: {len(j['images'])} images -> {tiles} tiles, "
              f"{sum(map(len, by_img.values()))} objects")

    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write(f"path: {OUT}\ntrain: train/images\nval: val/images\ntest: test/images\n"
                f"nc: {len(names)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in names.items()))


if __name__ == "__main__":
    main()
