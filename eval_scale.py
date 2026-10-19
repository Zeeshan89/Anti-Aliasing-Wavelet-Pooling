"""
Scale-stratified COCO evaluation for any Ultralytics detector on a YOLO-format split.

Predictions come from Ultralytics' own validator (save_json=True), so they are
exactly the detections behind `model.val()`; they are re-scored with pycocotools
over many area ranges in one pass.  Ported from
tiny_ovd_bench_project/src/eval/cocoeval_size.py.

Reported (all AP = mAP@[.5:.95] unless suffixed):
  AP, AP50, AP75
  COCO buckets:   AP_S (<32^2), AP_M (32^2-96^2), AP_L (>96^2)
  AI-TOD buckets: AP_VT (<8^2), AP_T (8^2-16^2), AP_aS (16^2-32^2)
  AP_bin_<lo>_<hi>: log-spaced bins of sqrt(area) in original-image px
  AP_iw:          instance-weighted AP (class AP weighted by #GT)
  per_class:      AP / AP50 per class, and per class x {S, M, L}
  n_gt:           #GT per bucket and per bin (weights for the pooled regression)
  ul_map, ul_map50: Ultralytics' own numbers for the sanity gate

Usage:
  python eval_scale.py --weights runs_ul/<run>/weights/best.pt --data <data.yaml> --split test
"""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path

import numpy as np

# sqrt(area) bin edges in original-image pixels (log-spaced, factor sqrt(2)).
BIN_EDGES = [0, 4, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256, 1e5]

BUCKETS = {
    "all": (0, 1e10),
    "VT": (0, 8 ** 2), "T": (8 ** 2, 16 ** 2), "aS": (16 ** 2, 32 ** 2),
    "S": (0, 32 ** 2), "M": (32 ** 2, 96 ** 2), "L": (96 ** 2, 1e10),
}


def _area_ranges():
    rng = dict(BUCKETS)
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        rng[f"bin_{lo}_{int(hi) if hi < 1e5 else 'inf'}"] = (lo ** 2, hi ** 2)
    return rng


def build_coco_gt(data_yaml: str, split: str, out_json: str) -> str:
    """COCO GT json from the same image list / label cache Ultralytics uses."""
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(data_yaml)
    with contextlib.redirect_stdout(io.StringIO()):
        ds = YOLODataset(img_path=data[split], imgsz=640, augment=False,
                         data=data, task="detect")
    images, anns = [], []
    for lab in ds.labels:
        stem = Path(lab["im_file"]).stem
        img_id = int(stem) if stem.isnumeric() else stem
        h, w = lab["shape"]
        images.append({"id": img_id, "file_name": os.path.basename(lab["im_file"]),
                       "height": int(h), "width": int(w)})
        for c, (xc, yc, bw, bh) in zip(lab["cls"].reshape(-1), lab["bboxes"]):
            bw, bh = float(bw) * w, float(bh) * h
            anns.append({"id": len(anns) + 1, "image_id": img_id,
                         "category_id": int(c) + 1,   # Ultralytics class_map is 1-based
                         "bbox": [float(xc) * w - bw / 2, float(yc) * h - bh / 2, bw, bh],
                         "area": bw * bh, "iscrowd": 0})
    cats = [{"id": i + 1, "name": n} for i, n in data["names"].items()]
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    json.dump({"images": images, "annotations": anns, "categories": cats}, open(out_json, "w"))
    return out_json


def score(gt_json: str, dt_json: str, max_dets: int = 300) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(gt_json)
        dt_list = json.load(open(dt_json))
        if not dt_list:
            return {"AP": 0.0, "empty": True}
        dt = gt.loadRes(dt_list)
        E = COCOeval(gt, dt, iouType="bbox")
        rng = _area_ranges()
        E.params.areaRng = [list(v) for v in rng.values()]
        E.params.areaRngLbl = list(rng.keys())
        E.params.maxDets = [1, 10, max_dets]
        E.evaluate()
        E.accumulate()

    prec = E.eval["precision"]          # [T, R, K, A, M]
    a_idx = {lbl: i for i, lbl in enumerate(E.params.areaRngLbl)}
    t50 = int(np.where(np.isclose(E.params.iouThrs, 0.5))[0][0])
    t75 = int(np.where(np.isclose(E.params.iouThrs, 0.75))[0][0])

    def ap(area="all", t=None, k=None):
        p = prec[:, :, :, a_idx[area], -1] if t is None else prec[t, :, :, a_idx[area], -1]
        if k is not None:
            p = p[..., k]
        p = p[p > -1]
        return float(p.mean()) if p.size else float("nan")

    # #GT per category per area range (non-crowd)
    cat_ids = E.params.catIds
    n_gt = {lbl: np.zeros(len(cat_ids), int) for lbl in rng}
    for ann in gt.dataset["annotations"]:
        k = cat_ids.index(ann["category_id"])
        for lbl, (lo, hi) in rng.items():
            if lo <= ann["area"] < hi:
                n_gt[lbl][k] += 1

    out = {"AP": ap(), "AP50": ap(t=t50), "AP75": ap(t=t75), "max_dets": max_dets}
    for lbl in rng:
        if lbl != "all":
            out[f"AP_{lbl}"] = ap(lbl)
            out[f"AP50_{lbl}"] = ap(lbl, t=t50)
    class_ap = np.array([ap(k=k) for k in range(len(cat_ids))])
    valid = ~np.isnan(class_ap)
    w = n_gt["all"][valid]
    out["AP_iw"] = float((class_ap[valid] * w).sum() / w.sum()) if w.sum() else float("nan")
    out["n_gt"] = {lbl: int(v.sum()) for lbl, v in n_gt.items()}
    names = {c["id"]: c["name"] for c in gt.dataset["categories"]}
    out["per_class"] = {
        names[cid]: {"AP": ap(k=k), "AP50": ap(t=t50, k=k), "n_gt": int(n_gt["all"][k]),
                     **{f"AP_{b}": ap(b, k=k) for b in ("S", "M", "L")}}
        for k, cid in enumerate(cat_ids)
    }
    return out


def evaluate(model, data_yaml: str, split: str, out_dir: str, max_det: int = 300,
             device=None, batch: int = 16, imgsz: int = 640) -> dict:
    """Run Ultralytics val (save_json) then scale-stratified COCO scoring."""
    out_dir = Path(out_dir)
    res = model.val(data=data_yaml, split=split, save_json=True, max_det=max_det,
                    conf=0.001, batch=batch, imgsz=imgsz, device=device, plots=False,
                    project=str(out_dir), name=f"val_{split}", exist_ok=True, verbose=False)
    dt_json = Path(res.save_dir) / "predictions.json"
    from ultralytics.data.utils import check_det_dataset
    data = check_det_dataset(data_yaml)
    split_path = str(data[split])
    if "coco" in split_path and split_path.endswith(f"{os.sep}val2017.txt"):
        # Ultralytics writes COCO-91 category ids here; score against the official GT.
        gt_json = str(Path(data["path"]) / "annotations" / "instances_val2017.json")
    else:
        gt_json = build_coco_gt(data_yaml, split, str(out_dir / f"gt_{split}.json"))
    m = score(gt_json, str(dt_json), max_dets=max_det)
    m["ul_map"] = float(res.box.map)
    m["ul_map50"] = float(res.box.map50)
    m["split"] = split
    (out_dir / f"scale_metrics_{split}.json").write_text(json.dumps(m, indent=2))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None, help="default: weights' run dir")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    import aa_inject  # noqa: F401  (AAWrap must be importable to unpickle checkpoints)
    from ultralytics import RTDETR, YOLO
    model = RTDETR(args.weights) if "rtdetr" in args.weights else YOLO(args.weights)
    out = args.out or str(Path(args.weights).parent.parent)
    m = evaluate(model, args.data, args.split, out, args.max_det, args.device, args.batch)
    keys = ["AP", "AP50", "AP_S", "AP_M", "AP_L", "AP_VT", "AP_T", "AP_aS", "AP_iw", "ul_map", "ul_map50"]
    print(json.dumps({k: round(m[k], 4) for k in keys if k in m}, indent=1))
    print(f"sanity |AP - ul_map| = {abs(m['AP'] - m['ul_map']):.4f}")


if __name__ == "__main__":
    main()
