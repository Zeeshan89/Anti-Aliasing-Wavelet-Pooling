"""
Cost of each AA variant (plan 1.4, reviewer Mn63 #3): params, GFLOPs, latency.

"Parameter-free" filters still cost compute.  FLOPs are counted with
torch.utils.flop_counter.FlopCounterMode, which sees the functional conv2d /
conv_transpose2d calls inside the filters (thop does not).

Two model families:
  wacv - the custom YOLOv8-S from model.py with the paper's layer selections
  ul   - Ultralytics YOLOv8s / YOLOv10-L / RT-DETR-L with aa_inject specs (revision)

Latency: batch 1, 640x640, median of --iters runs after warm-up.  On a login
node the CPU numbers are indicative only; run with --device cuda on a GCD for
the paper's GPU column (FP16).

Usage:
  python analysis/cost_table.py --device cpu --iters 20
  python analysis/cost_table.py --device cuda --half --iters 200     # on a compute node
"""

import argparse
import contextlib
import csv
import io
import json
import os
import sys
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

WACV_VARIANTS = ["baseline", "a", "blurpool", "gaussian", "blurpool_small", "gaussian_small"]
UL_VARIANTS = ["baseline", "a", "blurpool", "gaussian", "blurpool_small", "gaussian_small",
               "sel_p5_gaussian", "sel_p5_blurpool"]


def wacv_model(variant, dataset):
    import config as cfg
    import train as T
    from model import YOLOv8S
    nc = {"dior": 20, "dota": 15, "ip102": 102}[dataset]
    m = YOLOv8S(num_classes=nc)
    layers = json.load(open(os.path.join(ROOT, "runs", "runs", f"{dataset}_a", "wavelet_layers.json")))
    with contextlib.redirect_stdout(io.StringIO()):
        if variant == "a":
            T.apply_wavelet_selective(m, target_layers=layers, dataset=dataset)
        elif variant == "blurpool":
            T.apply_blurpool_selective(m, layers, dataset=dataset)
        elif variant == "blurpool_small":
            T.apply_blurpool_selective(m, layers, dataset=dataset, filter_size=cfg.BLURPOOL_FILTER_SIZE_SMALL)
        elif variant == "gaussian":
            T.apply_gaussian_selective(m, layers, dataset=dataset)
        elif variant == "gaussian_small":
            T.apply_gaussian_selective(m, layers, dataset=dataset, kernel_size=cfg.GAUSSIAN_KERNEL_SIZE_SMALL,
                                       sigma=cfg.GAUSSIAN_SIGMA_SMALL)
    return m


def ul_model(model_key, variant):
    import aa_inject as A
    from ultralytics import RTDETR, YOLO
    w = A.WEIGHTS[model_key]
    m = (RTDETR(w) if model_key.startswith("rtdetr") else YOLO(w)).model.float()
    A.wrap_layers(m, A.build_spec(m, model_key, variant))
    return m


def measure(m, device, half, iters, imgsz=640):
    m = m.to(device).eval()
    x = torch.zeros(1, 3, imgsz, imgsz, device=device)
    if half:
        m, x = m.half(), x.half()
    params = sum(p.numel() for p in m.parameters())
    with torch.no_grad():
        fc = FlopCounterMode(display=False)
        with fc:
            m(x)
        gflops = fc.get_total_flops() / 1e9
        for _ in range(max(3, iters // 10)):
            m(x)
        times = []
        for _ in range(iters):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t = time.perf_counter()
            m(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t) * 1e3)
    times.sort()
    return params, gflops, times[len(times) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--family", default="wacv,ul")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    rows = []
    jobs = []
    if "wacv" in args.family:
        jobs += [("wacv-yolov8s", ds, v, lambda v=v, ds=ds: wacv_model(v, ds))
                 for ds in ("dota", "ip102") for v in WACV_VARIANTS]
    if "ul" in args.family:
        jobs += [(mk, "-", v, lambda v=v, mk=mk: ul_model(mk, v))
                 for mk in ("yolov8s", "yolov10l", "rtdetr-l") for v in UL_VARIANTS]
    base = {}
    for family, ds, v, build in jobs:
        with contextlib.redirect_stdout(io.StringIO()):
            m = build()
        p, g, lat = measure(m, args.device, args.half, args.iters)
        key = (family, ds)
        if v == "baseline":
            base[key] = (g, lat)
        g0, l0 = base[key]
        rows.append({"model": family, "layers_from": ds, "variant": v, "params_M": round(p / 1e6, 3),
                     "GFLOPs": round(g, 2), "dGFLOPs_pct": round(100 * (g - g0) / g0, 2),
                     f"latency_ms_{args.device}{'_fp16' if args.half else ''}": round(lat, 2),
                     "dlatency_pct": round(100 * (lat - l0) / l0, 1)})
        print(rows[-1], flush=True)
        del m

    os.makedirs("analysis/out", exist_ok=True)
    out = f"analysis/out/cost_table_{args.device}{'_fp16' if args.half else ''}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved", out)


if __name__ == "__main__":
    main()
