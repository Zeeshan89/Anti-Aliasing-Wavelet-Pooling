"""
Anti-aliasing (AA) injection for Ultralytics detectors (YOLOv8 / YOLOv10 / RT-DETR).

Why not forward-patching (as in train_yolov10.py)?
  Ultralytics' Model.train() rebuilds a fresh DetectionModel from the YAML and
  only copies weights (ultralytics/engine/model.py: trainer.get_model(...)).
  Instance-level forward patches are silently dropped, so the network that is
  actually trained has NO AA.  Patched closures also cannot be pickled into
  best.pt / last.pt.

Here a target layer model.model[i] is *replaced* by AAWrap(aa, layer), which
copies the routing attributes (f, i, type, np) so Ultralytics' index-based
graph is unchanged.  Injection happens inside the trainer's get_model(), so it
survives the rebuild, resume, EMA and DDP (config is read from env vars, which
DDP subprocesses inherit).

Also provides:
  - layer_report(): input stride, footprint and which heads (P3/P4/P5) every
    stride-2 layer feeds  (plan item 0.2)
  - fm_kernel():    footprint-matched kernel size per layer

CLI (report only, CPU):
  python aa_inject.py --weights yolov8s.pt
  python aa_inject.py --weights yolov8s.pt --variant gaussian
"""

import argparse
import json
import math
import os

import torch
import torch.nn as nn

import config as cfg
from wavelet_pool import BlurSmooth, GaussianSmooth, WaveletSmooth

ENV_VARIANT = "AA_VARIANT"
ENV_LAYERS = "AA_LAYERS"        # JSON: {"<layer idx>": {"filter": ..., "k": ..., "sigma": ...}}

# Fixed filters used in the WACV paper (sizes from config.py).
BASE_FILTERS = ("a", "blurpool", "gaussian", "blurpool_small", "gaussian_small")

# Effective spatial support (taps) of each fixed filter, used for the footprint.
# Paper convention (Table 1): full spatial support, so Coiflet-3 counts all 18 taps.
FILTER_TAPS = {
    "blurpool": cfg.BLURPOOL_FILTER_SIZE,
    "blurpool_small": cfg.BLURPOOL_FILTER_SIZE_SMALL,
    "gaussian": cfg.GAUSSIAN_KERNEL_SIZE,
    "gaussian_small": cfg.GAUSSIAN_KERNEL_SIZE_SMALL,
    "a": 18,
}

# Default injection layers = PAN bottom-up downsampling convs, the analogs of
# the custom YOLOv8-S neck.down_p3_p4 / neck.down_p4_p5 used in the paper.
# Verify with `python aa_inject.py --weights <model>.pt` (prints the table).
DEFAULT_LAYERS = {
    "yolov8s": [16, 19],
    "yolov10l": [17, 20],
    "rtdetr-l": [22, 25],
}

# Pretrained checkpoints (pre-download on a login node; compute nodes are offline).
WEIGHTS = {
    "yolov8s": "yolov8s.pt",
    "yolov10l": "yolov10l.pt",
    "rtdetr-l": "../tiny_ovd_bench_project/src/rtdetr-l.pt",
}

# Approximate object-size range (sqrt(area), px at imgsz) served by each head,
# FCOS-style.  Used only to pick footprint-matched kernel sizes.
HEAD_SIZE_RANGE = {"P3": (0, 64), "P4": (64, 128), "P5": (128, 1e9)}


# ──────────────────────────────────────────────────────────────────────
# Filters and wrapper
# ──────────────────────────────────────────────────────────────────────

def make_filter(name: str, k: int = None, sigma: float = None) -> nn.Module:
    if name == "a":
        return WaveletSmooth()
    if name == "blurpool":
        return BlurSmooth(filter_size=k or cfg.BLURPOOL_FILTER_SIZE)
    if name == "blurpool_small":
        return BlurSmooth(filter_size=cfg.BLURPOOL_FILTER_SIZE_SMALL)
    if name == "gaussian":
        return GaussianSmooth(kernel_size=k or cfg.GAUSSIAN_KERNEL_SIZE,
                              sigma=sigma or cfg.GAUSSIAN_SIGMA)
    if name == "gaussian_small":
        return GaussianSmooth(kernel_size=cfg.GAUSSIAN_KERNEL_SIZE_SMALL,
                              sigma=cfg.GAUSSIAN_SIGMA_SMALL)
    raise ValueError(f"unknown filter '{name}'")


class AAWrap(nn.Module):
    """layer(aa(x)), transparent to Ultralytics routing."""

    def __init__(self, aa: nn.Module, layer: nn.Module):
        super().__init__()
        self.aa = aa
        self.layer = layer
        for attr in ("f", "i", "type", "np"):
            if hasattr(layer, attr):
                setattr(self, attr, getattr(layer, attr))

    def forward(self, x, *args, **kwargs):
        return self.layer(self.aa(x), *args, **kwargs)


def wrap_layers(model: nn.Module, spec: dict) -> list:
    """spec: {idx: {"filter": name, "k": int|None, "sigma": float|None}}."""
    seq = model.model
    done = []
    for idx, s in sorted(((int(i), s) for i, s in spec.items())):
        if isinstance(seq[idx], AAWrap):
            continue
        seq[idx] = AAWrap(make_filter(s["filter"], s.get("k"), s.get("sigma")), seq[idx])
        done.append(f"model.{idx} ({type(seq[idx].layer).__name__}) <- {s}")
    return done


def load_weights(model: nn.Module, weights) -> int:
    """Load a (possibly un-wrapped) checkpoint into a (possibly wrapped) model."""
    from ultralytics.utils.torch_utils import intersect_dicts
    if weights is None:
        return 0
    src = weights.float().state_dict() if isinstance(weights, nn.Module) else weights
    dst = model.state_dict()
    wrapped = {int(i) for i, m in enumerate(model.model) if isinstance(m, AAWrap)}
    remapped = {}
    for key, v in src.items():
        parts = key.split(".")
        if (len(parts) > 2 and parts[0] == "model" and parts[1].isdigit()
                and int(parts[1]) in wrapped and parts[2] not in ("layer", "aa")):
            key = ".".join(parts[:2] + ["layer"] + parts[2:])
        remapped[key] = v
    csd = intersect_dicts(remapped, dst)
    model.load_state_dict(csd, strict=False)
    return len(csd)


# ──────────────────────────────────────────────────────────────────────
# Graph analysis: strides, footprints, which head each layer feeds
# ──────────────────────────────────────────────────────────────────────

def _head_inputs(model):
    head = model.model[-1]
    return list(head.f) if isinstance(head.f, (list, tuple)) else [head.f]


def _reach(model):
    """reach[i] = set of top-level indices that consume (transitively) layer i."""
    n = len(model.model)
    consumers = {i: set() for i in range(n)}
    for j, m in enumerate(model.model):
        srcs = m.f if isinstance(m.f, (list, tuple)) else [m.f]
        for s in srcs:
            s = j + s if s < 0 else s
            if s >= 0:
                consumers[s].add(j)
    reach = {}
    for i in reversed(range(n)):
        r = set(consumers[i])
        for c in consumers[i]:
            r |= reach[c]
        reach[i] = r
    return reach


@torch.no_grad()
def layer_report(model: nn.Module, imgsz: int = 640) -> list:
    """One row per stride-2 top-level layer: idx, type, in/out stride, heads fed."""
    sizes = {}
    hooks = []
    for i, m in enumerate(model.model):
        def hook(mod, inp, out, i=i):
            x = inp[0] if isinstance(inp, tuple) else inp
            if isinstance(x, (list, tuple)):
                x = x[0]
            y = out[0] if isinstance(out, (list, tuple)) else out
            if torch.is_tensor(x) and torch.is_tensor(y) and x.dim() == 4 and y.dim() == 4:
                sizes[i] = (x.shape[-1], y.shape[-1])
        hooks.append(m.register_forward_hook(hook))
    p = next(model.parameters())
    was_training = model.training
    model.eval()
    model(torch.zeros(1, 3, imgsz, imgsz, device=p.device, dtype=p.dtype))
    model.train(was_training)
    for h in hooks:
        h.remove()

    head_in = _head_inputs(model)
    # Name heads by their output stride (8 -> P3, 16 -> P4, 32 -> P5, ...)
    head_names = {}
    for h in head_in:
        stride = imgsz // sizes[h][1]
        head_names[h] = f"P{int(round(math.log2(stride)))}"
    reach = _reach(model)

    rows = []
    for i, m in enumerate(model.model[:-1]):
        if i not in sizes:
            continue
        w_in, w_out = sizes[i]
        if w_out * 2 != w_in:
            continue
        heads = sorted(head_names[h] for h in head_in if h == i or h in reach[i])
        layer = m.layer if isinstance(m, AAWrap) else m
        rows.append({
            "idx": i,
            "type": type(layer).__name__,
            "in_stride": imgsz // w_in,
            "out_stride": imgsz // w_out,
            "heads": heads,
            "wrapped": isinstance(m, AAWrap),
        })
    return rows


def footprint(filter_name: str, in_stride: int, k: int = None) -> int:
    """Filter footprint in input pixels = taps x input stride at the layer."""
    taps = k if k is not None else FILTER_TAPS[filter_name]
    return int(taps * in_stride)


def fm_kernel(in_stride: int, target_px: float, k_max: int = 7) -> int:
    """Largest kernel with footprint <= target_px (0 -> do not filter)."""
    k = int(target_px // in_stride)
    return min(k, k_max) if k >= 2 else 0


# ──────────────────────────────────────────────────────────────────────
# Variant -> per-layer spec
# ──────────────────────────────────────────────────────────────────────

def parse_variant(variant: str):
    """Returns (placement, filter).

    baseline                         -> (None, None)
    a | blurpool | gaussian | *_small -> ("default", filter)
    sel_p5_<filter>                  -> layers feeding only P5
    sel_p45_<filter>                 -> layers not feeding P3
    all_<filter>                     -> every stride-2 layer (incl. backbone)
    fm_blurpool | fm_gaussian        -> footprint-matched kernels on default layers
    """
    v = variant.lower()
    if v == "baseline":
        return None, None
    if v in BASE_FILTERS:
        return "default", v
    for pre in ("sel_p5_", "sel_p45_", "all_"):
        if v.startswith(pre) and v[len(pre):] in BASE_FILTERS:
            return pre.rstrip("_"), v[len(pre):]
    if v in ("fm_blurpool", "fm_gaussian"):
        return "fm", v[3:]
    raise ValueError(f"unknown variant '{variant}'")


def build_spec(model, model_key: str, variant: str, layers=None,
               object_sizes=None, imgsz: int = 640) -> dict:
    """Decide which layers get which filter.  Returns {idx: spec}."""
    placement, filt = parse_variant(variant)
    if placement is None:
        return {}
    rows = {r["idx"]: r for r in layer_report(model, imgsz)}

    if layers:
        idxs = list(layers)
    elif placement == "default":
        idxs = DEFAULT_LAYERS.get(model_key) or \
            [i for i, r in rows.items() if "P3" not in r["heads"]]
    elif placement == "sel_p5":
        idxs = [i for i, r in rows.items() if r["heads"] == ["P5"]]
    elif placement == "sel_p45":
        idxs = [i for i, r in rows.items() if "P3" not in r["heads"]]
    elif placement == "all":
        idxs = list(rows)
    elif placement == "fm":
        idxs = DEFAULT_LAYERS.get(model_key) or \
            [i for i, r in rows.items() if "P3" not in r["heads"]]
    missing = [i for i in idxs if i not in rows]
    if missing:
        raise ValueError(f"layers {missing} are not stride-2 layers; valid: {sorted(rows)}")

    spec = {}
    for i in idxs:
        r = rows[i]
        if placement == "fm":
            if object_sizes is None:
                raise ValueError("fm variants need object_sizes")
            lo, hi = HEAD_SIZE_RANGE[r["heads"][0]]           # finest head fed
            sel = [s for s in object_sizes if lo <= s < hi] or list(object_sizes)
            target = float(torch.tensor(sel).quantile(0.25))
            k = fm_kernel(r["in_stride"], target)
            if k == 0:
                continue
            s = {"filter": filt, "k": k, "target_px": round(target, 1)}
            if filt == "gaussian":
                s["sigma"] = round(max(0.5, 0.2 * k), 2)
        else:
            s = {"filter": filt}
        s["in_stride"] = r["in_stride"]
        s["heads"] = r["heads"]
        s["footprint_px"] = footprint(filt, r["in_stride"], s.get("k"))
        spec[i] = s
    return spec


# ──────────────────────────────────────────────────────────────────────
# Trainers (AA injected inside get_model so it survives Ultralytics' rebuild)
# ──────────────────────────────────────────────────────────────────────

def _env_spec():
    raw = os.environ.get(ENV_LAYERS, "")
    return {int(k): v for k, v in json.loads(raw).items()} if raw else {}


class _AAMixin:
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=None, verbose=verbose)
        spec = _env_spec()
        done = wrap_layers(model, spec)
        n = load_weights(model, weights)
        if verbose and (done or weights is not None):
            from ultralytics.utils import LOGGER
            for d in done:
                LOGGER.info(f"AA: {d}")
            if weights is not None:
                LOGGER.info(f"AA: transferred {n}/{len(model.state_dict())} tensors")
        return model


def detection_trainer():
    from ultralytics.models.yolo.detect import DetectionTrainer

    class AADetectionTrainer(_AAMixin, DetectionTrainer):
        pass
    return AADetectionTrainer


def rtdetr_trainer():
    from ultralytics.models.rtdetr.train import RTDETRTrainer

    class AARTDETRTrainer(_AAMixin, RTDETRTrainer):
        pass
    return AARTDETRTrainer


# Module-level names so Ultralytics' DDP launcher can import them.
try:
    AADetectionTrainer = detection_trainer()
    AARTDETRTrainer = rtdetr_trainer()
    AADetectionTrainer.__module__ = AARTDETRTrainer.__module__ = __name__
    AADetectionTrainer.__qualname__ = AADetectionTrainer.__name__ = "AADetectionTrainer"
    AARTDETRTrainer.__qualname__ = AARTDETRTrainer.__name__ = "AARTDETRTrainer"
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────
# CLI: print the layer / head / footprint table
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8s.pt")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO, RTDETR
    w = args.weights
    m = (RTDETR(w) if "rtdetr" in w else YOLO(w)).model.float()
    key = os.path.splitext(os.path.basename(w))[0]
    print(f"\n{key}: stride-2 layers (input stride, heads fed)")
    print(f"{'idx':>4} {'type':<12} {'in_s':>5} {'out_s':>6}  heads   footprint px (blur3/gauss7/blur2/gauss3)")
    for r in layer_report(m, args.imgsz):
        fps = "/".join(str(footprint(f, r["in_stride"])) for f in
                       ("blurpool", "gaussian", "blurpool_small", "gaussian_small"))
        default = "  <- default" if r["idx"] in (DEFAULT_LAYERS.get(key) or []) else ""
        print(f"{r['idx']:>4} {r['type']:<12} {r['in_stride']:>5} {r['out_stride']:>6}  "
              f"{','.join(r['heads']):<7} {fps}{default}")
    if args.variant:
        spec = build_spec(m, key, args.variant, object_sizes=[16, 24, 32, 64, 128, 256])
        print(f"\nspec for {args.variant}: {json.dumps(spec, indent=1)}")


if __name__ == "__main__":
    main()
