"""
Unified Ultralytics training for the AA revision experiments (YOLOv8s / YOLOv10-L / RT-DETR-L).

Supersedes train_yolov10.py, whose forward-patched AA was dropped by Ultralytics'
model rebuild (see aa_inject.py).  AA is injected inside the trainer's get_model().

Variants (see aa_inject.parse_variant):
  baseline | a | blurpool | gaussian | blurpool_small | gaussian_small
  sel_p5_<filter> | sel_p45_<filter> | all_<filter> | fm_blurpool | fm_gaussian

Recipes:
  standard - Ultralytics default hyper-parameters (mosaic, HSV, flips, SGD/auto, ...)
  noaug    - the WACV isolation protocol (AdamW 1e-3, every augmentation off)

Outputs: <runs-dir>/<model>_<dataset>_<recipe>_<variant>_s<seed>/
  weights/{best,last}.pt   aa_spec.json   run_meta.json
  scale_metrics_test.json  (eval_scale.py)   train.done   done.flag

Re-running the same command resumes from weights/last.pt (chained Frontier jobs)
and skips finished runs.

Examples:
  python train_ul.py --model yolov8s --dataset dota --variant baseline --seed 0
  python train_ul.py --model yolov8s --dataset dior --variant sel_p5_gaussian --seed 1
  python train_ul.py --model rtdetr-l --data /mnt/bb/$USER/data/dota/data.yaml --dataset dota --variant fm_gaussian
"""

import argparse
import json
import os
import time
from pathlib import Path

import config as cfg
import aa_inject
import eval_scale

NO_AUG = dict(
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.0,
    mosaic=0.0, mixup=0.0, copy_paste=0.0,
    erasing=0.0,
)


def recipe_args(recipe: str, model_key: str) -> dict:
    if recipe == "noaug":
        return dict(optimizer="AdamW", lr0=1e-3, weight_decay=cfg.WEIGHT_DECAY,
                    warmup_epochs=cfg.WARMUP_EPOCHS, **NO_AUG)
    if model_key.startswith("rtdetr"):
        # RT-DETR diverges with Ultralytics' 'auto' SGD lr0=0.01 on long runs.
        return dict(optimizer="AdamW", lr0=1e-4, weight_decay=1e-4)
    return {}   # Ultralytics defaults


def resolve_data(dataset: str, data: str = None) -> str:
    if data:
        return data
    if dataset in cfg.DATASET_CONFIGS:
        return os.path.join(cfg.load_dataset_cfg(dataset)["split_dir"], "data.yaml")
    return os.path.join(cfg.PROJECT_ROOT, "dataset", dataset, "split", "data.yaml")


def object_sizes(data_yaml: str, split: str = "train", imgsz: int = 640):
    """sqrt(box area) of every training object, in letterboxed-imgsz pixels."""
    import contextlib
    import io
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(data_yaml)
    with contextlib.redirect_stdout(io.StringIO()):
        ds = YOLODataset(img_path=data[split], imgsz=imgsz, augment=False,
                         data=data, task="detect")
    sizes = []
    for lab in ds.labels:
        h, w = lab["shape"]
        r = imgsz / max(h, w)
        for bw, bh in lab["bboxes"][:, 2:4]:
            sizes.append(float((bw * w * bh * h) ** 0.5 * r))
    return sizes


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--model", required=True, choices=list(aa_inject.WEIGHTS))
    p.add_argument("--dataset", required=True, help="dior | dota | ip102 | aitod | ...")
    p.add_argument("--data", default=None, help="data.yaml override (e.g. NVMe copy)")
    p.add_argument("--variant", default="baseline")
    p.add_argument("--layers", default=None, help="comma-separated layer indices override")
    p.add_argument("--recipe", default="standard", choices=["standard", "noaug"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None, help="default 100 (standard) / 50 (noaug)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-det", type=int, default=300, help="use 1500 for AI-TOD")
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--fraction", type=float, default=1.0, help="train subset (smoke tests)")
    p.add_argument("--runs-dir", default=os.path.join(cfg.PROJECT_ROOT, "runs_ul"))
    args = p.parse_args()

    from ultralytics import RTDETR, YOLO
    is_detr = args.model.startswith("rtdetr")
    Model = RTDETR if is_detr else YOLO
    Trainer = aa_inject.AARTDETRTrainer if is_detr else aa_inject.AADetectionTrainer

    name = f"{args.model}_{args.dataset}_{args.recipe}_{args.variant.lower()}_s{args.seed}"
    run_dir = Path(args.runs_dir) / name
    if (run_dir / "done.flag").exists():
        print(f"[SKIP] {name}: done.flag exists")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = resolve_data(args.dataset, args.data)
    epochs = args.epochs or (100 if args.recipe == "standard" else 50)

    # AA spec: computed once, frozen in aa_spec.json so resumed jobs use the same layers.
    spec_path = run_dir / "aa_spec.json"
    if spec_path.exists():
        spec = json.loads(spec_path.read_text())
    else:
        probe = Model(aa_inject.WEIGHTS[args.model]).model.float()
        layers = [int(x) for x in args.layers.split(",")] if args.layers else None
        sizes = None
        if args.variant.lower().startswith("fm_"):
            sizes = object_sizes(data_yaml, "train", args.imgsz)
        spec = aa_inject.build_spec(probe, args.model, args.variant, layers,
                                    object_sizes=sizes, imgsz=args.imgsz)
        spec = {str(k): v for k, v in spec.items()}
        spec_path.write_text(json.dumps(spec, indent=2))
        (run_dir / "run_meta.json").write_text(json.dumps({
            "args": vars(args), "data_yaml": data_yaml, "epochs": epochs,
            "aa_spec": spec, "layer_report": aa_inject.layer_report(probe, args.imgsz),
        }, indent=2))
        del probe
    os.environ[aa_inject.ENV_LAYERS] = json.dumps(spec)   # read by the trainer (and DDP children)

    print("=" * 70)
    print(f"  {name}\n  data: {data_yaml}\n  epochs={epochs} batch={args.batch} device={args.device}")
    for i, s in spec.items():
        print(f"  AA layer {i}: {s}")
    if not spec:
        print("  baseline: no AA")
    print("=" * 70)

    t0 = time.time()
    last = run_dir / "weights" / "last.pt"
    if not (run_dir / "train.done").exists():
        if last.exists():
            print(f"[resume] {last}")
            Model(str(last)).train(resume=True, trainer=Trainer, device=args.device,
                                   workers=args.workers)
        else:
            Model(aa_inject.WEIGHTS[args.model]).train(
                trainer=Trainer, data=data_yaml, epochs=epochs, batch=args.batch,
                imgsz=args.imgsz, device=args.device, workers=args.workers,
                seed=args.seed, deterministic=False, fraction=args.fraction,
                project=str(run_dir.parent), name=name, exist_ok=True,
                max_det=args.max_det, plots=False, verbose=True,
                **recipe_args(args.recipe, args.model),
            )
        (run_dir / "train.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    train_min = (time.time() - t0) / 60

    best = run_dir / "weights" / "best.pt"
    model = Model(str(best if best.exists() else last))
    wrapped = [i for i, m in enumerate(model.model.model) if isinstance(m, aa_inject.AAWrap)]
    if sorted(wrapped) != sorted(int(i) for i in spec):
        raise RuntimeError(f"checkpoint AA layers {wrapped} != spec {list(spec)}")

    m = eval_scale.evaluate(model, data_yaml, "test", str(run_dir), max_det=args.max_det,
                            device=args.device, batch=args.batch, imgsz=args.imgsz)
    print(f"  test AP={m['AP']:.4f} AP50={m['AP50']:.4f} AP_S={m['AP_S']:.4f} "
          f"AP_M={m['AP_M']:.4f} AP_L={m['AP_L']:.4f} (ultralytics mAP={m['ul_map']:.4f})")
    (run_dir / "done.flag").write_text(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}\ntrain_min_this_job={train_min:.1f}\n"
        f"aa_layers={wrapped}\n")


if __name__ == "__main__":
    main()
