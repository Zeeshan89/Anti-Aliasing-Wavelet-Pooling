"""
Explicit YOLOv8-S architecture in pure PyTorch.

Every layer is defined transparently — no black-box library calls.
The architecture follows the publicly documented YOLOv8-Small structure:
  Backbone:  CSPDarknet (5 stages, each with Conv(stride=2) down-sampling)
  Neck:      PANet / FPN with C2f blocks and up-sampling + concat
  Head:      Decoupled anchor-free detection head (classification + regression)

We expose each down-sampling point so it can be individually replaced with
WaveletPool for the anti-aliasing variants.

References:
  - Ultralytics YOLOv8 (open-source, AGPL-3.0)
  - Architecture spec: https://github.com/ultralytics/ultralytics

Weight loading:
  Pretrained weights from Ultralytics can be loaded via `load_pretrained()`.
  For modified variants (WaveletPool), only matching layers are loaded;
  new/changed layers are randomly initialised.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
#  Basic Building Blocks
# =====================================================================

class ConvBnSiLU(nn.Module):
    """Conv2d → BatchNorm2d → SiLU.  The fundamental YOLO building block."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck: 3×3 conv → 3×3 conv with optional residual."""

    def __init__(self, in_ch, out_ch, shortcut=True, expansion=0.5):
        super().__init__()
        hidden = int(out_ch * expansion)
        self.cv1 = ConvBnSiLU(in_ch, hidden, 3)
        self.cv2 = ConvBnSiLU(hidden, out_ch, 3)
        self.add = shortcut and (in_ch == out_ch)

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """
    CSP Bottleneck with 2 convolutions  (YOLOv8's core repeating block).
    Splits channels, runs N bottlenecks on one half, concatenates, projects.
    """

    def __init__(self, in_ch, out_ch, n_bottlenecks=1, shortcut=True, expansion=0.5):
        super().__init__()
        self.hidden = int(out_ch * expansion)
        self.cv1 = ConvBnSiLU(in_ch, 2 * self.hidden, 1)
        self.cv2 = ConvBnSiLU((2 + n_bottlenecks) * self.hidden, out_ch, 1)
        self.bottlenecks = nn.ModuleList(
            Bottleneck(self.hidden, self.hidden, shortcut=shortcut, expansion=1.0)
            for _ in range(n_bottlenecks)
        )

    def forward(self, x):
        y = self.cv1(x)
        # split into two chunks of size self.hidden
        chunks = [y[:, :self.hidden, :, :], y[:, self.hidden:, :, :]]
        for bn in self.bottlenecks:
            chunks.append(bn(chunks[-1]))
        return self.cv2(torch.cat(chunks, dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling — Fast  (kernel=5, three sequential max-pools)."""

    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        hidden = in_ch // 2
        self.cv1 = ConvBnSiLU(in_ch, hidden, 1)
        self.cv2 = ConvBnSiLU(hidden * 4, out_ch, 1)
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.cv2(torch.cat([x, p1, p2, p3], dim=1))


# =====================================================================
#  Backbone  –  CSPDarknet  (YOLOv8-S channel widths)
# =====================================================================
#
# YOLOv8-S channel plan:
#   Stage 0 (stem):  3 → 32,  stride 2      (P1: 320×320)
#   Stage 1:        32 → 64,  stride 2      (P2: 160×160)   + C2f(n=1)
#   Stage 2:        64 → 128, stride 2      (P3: 80×80)     + C2f(n=2)
#   Stage 3:       128 → 256, stride 2      (P4: 40×40)     + C2f(n=2)
#   Stage 4:       256 → 512, stride 2      (P5: 20×20)     + C2f(n=1) + SPPF
#

class CSPDarknet(nn.Module):
    """
    YOLOv8-S backbone.  All 5 down-sampling convolutions are stored as
    named attributes (`down0` … `down4`) so they can be individually
    swapped with WaveletPool.
    """

    def __init__(self, width_mult=0.5):
        super().__init__()
        # Channel widths (YOLOv8-S uses width_mult=0.5 of the base)
        base = [64, 128, 256, 512, 1024]
        ch = [max(int(c * width_mult), 1) for c in base]  # [32, 64, 128, 256, 512]

        # ── Stage 0  (stem) ────────────────────────────────────────
        self.down0 = ConvBnSiLU(3, ch[0], 3, stride=2)             # 640→320

        # ── Stage 1 ───────────────────────────────────────────────
        self.down1 = ConvBnSiLU(ch[0], ch[1], 3, stride=2)         # 320→160
        self.c2f_1 = C2f(ch[1], ch[1], n_bottlenecks=1)

        # ── Stage 2 ───────────────────────────────────────────────
        self.down2 = ConvBnSiLU(ch[1], ch[2], 3, stride=2)         # 160→80
        self.c2f_2 = C2f(ch[2], ch[2], n_bottlenecks=2)

        # ── Stage 3 ───────────────────────────────────────────────
        self.down3 = ConvBnSiLU(ch[2], ch[3], 3, stride=2)         # 80→40
        self.c2f_3 = C2f(ch[3], ch[3], n_bottlenecks=2)

        # ── Stage 4 ───────────────────────────────────────────────
        self.down4 = ConvBnSiLU(ch[3], ch[4], 3, stride=2)         # 40→20
        self.c2f_4 = C2f(ch[4], ch[4], n_bottlenecks=1)
        self.sppf  = SPPF(ch[4], ch[4])

        self.out_channels = [ch[2], ch[3], ch[4]]  # P3, P4, P5

    def forward(self, x):
        """Returns multi-scale features [P3, P4, P5]."""
        x = self.down0(x)                     # → P1  (B, 32, 320, 320)
        x = self.c2f_1(self.down1(x))         # → P2  (B, 64, 160, 160)
        p3 = self.c2f_2(self.down2(x))        # → P3  (B, 128, 80, 80)
        p4 = self.c2f_3(self.down3(p3))       # → P4  (B, 256, 40, 40)
        p5 = self.sppf(self.c2f_4(self.down4(p4)))  # → P5  (B, 512, 20, 20)
        return [p3, p4, p5]


# =====================================================================
#  Neck  –  PANet / BiFPN-style feature pyramid
# =====================================================================

class Neck(nn.Module):
    """
    Bottom-up + top-down feature fusion neck.
    Takes [P3, P4, P5] and produces fused [F3, F4, F5].
    """

    def __init__(self, ch_p3, ch_p4, ch_p5):
        super().__init__()
        # ── Top-down (P5 → P4 → P3) ──────────────────────────────
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # No lateral reduction — concat full channels (matches Ultralytics)
        self.c2f_td_p4  = C2f(ch_p5 + ch_p4, ch_p4, n_bottlenecks=1, shortcut=False)
        self.c2f_td_p3  = C2f(ch_p4 + ch_p3, ch_p3, n_bottlenecks=1, shortcut=False)

        # ── Bottom-up (P3 → P4 → P5) ─────────────────────────────
        self.down_p3_p4 = ConvBnSiLU(ch_p3, ch_p3, 3, stride=2)
        self.c2f_bu_p4  = C2f(ch_p3 + ch_p4, ch_p4, n_bottlenecks=1, shortcut=False)

        self.down_p4_p5 = ConvBnSiLU(ch_p4, ch_p4, 3, stride=2)
        self.c2f_bu_p5  = C2f(ch_p4 + ch_p5, ch_p5, n_bottlenecks=1, shortcut=False)

    def forward(self, features):
        p3, p4, p5 = features

        # top-down
        td_p4 = self.c2f_td_p4(torch.cat([self.up(p5), p4], 1))
        td_p3 = self.c2f_td_p3(torch.cat([self.up(td_p4), p3], 1))

        # bottom-up
        bu_p4 = self.c2f_bu_p4(torch.cat([self.down_p3_p4(td_p3), td_p4], 1))
        bu_p5 = self.c2f_bu_p5(torch.cat([self.down_p4_p5(bu_p4), p5], 1))

        return [td_p3, bu_p4, bu_p5]


# =====================================================================
#  Detection Head  –  Anchor-Free Decoupled Head
# =====================================================================

class DetectHead(nn.Module):
    """
    Anchor-free decoupled head (per-scale), producing:
      - class probabilities  (B, num_classes, H, W)
      - bounding-box regression  (B, 4, H, W)  [x_c, y_c, w, h normalised]

    Applied independently to each FPN level.
    """

    def __init__(self, in_ch, num_classes, reg_max=1, dropout=0.1):
        super().__init__()
        hidden = max(in_ch, 64)
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.drop = nn.Dropout2d(p=dropout)

        # classification branch
        self.cls_convs = nn.Sequential(
            ConvBnSiLU(in_ch, hidden, 3),
            ConvBnSiLU(hidden, hidden, 3),
        )
        self.cls_pred = nn.Conv2d(hidden, num_classes, 1)

        # regression branch  (outputs 4 channels: x_c, y_c, w, h)
        self.reg_convs = nn.Sequential(
            ConvBnSiLU(in_ch, hidden, 3),
            ConvBnSiLU(hidden, hidden, 3),
        )
        self.reg_pred = nn.Conv2d(hidden, 4, 1)

    def forward(self, x):
        cls_out = self.cls_pred(self.drop(self.cls_convs(x)))
        reg_out = self.reg_pred(self.drop(self.reg_convs(x)))
        return cls_out, reg_out


# =====================================================================
#  Full Model  –  YOLOv8-S Moth Detector
# =====================================================================

class YOLOv8S(nn.Module):
    """
    Complete YOLOv8-S object detection model.

    Architecture:
        Backbone (CSPDarknet)  →  Neck (PANet)  →  Detect Heads (×3 scales)

    All stride-2 down-sampling layers are accessible individually via the
    backbone's `down0`–`down4` attributes, enabling targeted replacement
    with WaveletPool.
    """

    def __init__(self, num_classes=4, reg_max=1, width_mult=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

        # ── backbone ──────────────────────────────────────────────
        self.backbone = CSPDarknet(width_mult=width_mult)
        ch_p3, ch_p4, ch_p5 = self.backbone.out_channels

        # ── neck / FPN ────────────────────────────────────────────
        self.neck = Neck(ch_p3, ch_p4, ch_p5)

        # ── detection heads (one per scale) ───────────────────────
        self.head_p3 = DetectHead(ch_p3, num_classes, reg_max)   # 80×80
        self.head_p4 = DetectHead(ch_p4, num_classes, reg_max)   # 40×40
        self.head_p5 = DetectHead(ch_p5, num_classes, reg_max)   # 20×20

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming init for conv, constant init for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input image tensor
        Returns:
            List of (cls_out, reg_out) tuples for each scale.
            cls_out: (B, num_classes, Hi, Wi)
            reg_out: (B, 4*reg_max, Hi, Wi)
        """
        features = self.backbone(x)     # [P3, P4, P5]
        fused = self.neck(features)     # [F3, F4, F5]

        out_p3 = self.head_p3(fused[0])
        out_p4 = self.head_p4(fused[1])
        out_p5 = self.head_p5(fused[2])

        return [out_p3, out_p4, out_p5]

    def get_downsample_layers(self):
        """
        Return a dict of all stride-2 down-sampling layers in the backbone
        and neck, keyed by their attribute path.

        This is used by:
         - The aliasing analysis to inspect each layer individually
         - The WaveletPool replacement logic for selective substitution
        """
        layers = {}
        # backbone down-sampling
        for i in range(5):
            name = f"backbone.down{i}"
            layers[name] = getattr(self.backbone, f"down{i}")
        # neck down-sampling
        layers["neck.down_p3_p4"] = self.neck.down_p3_p4
        layers["neck.down_p4_p5"] = self.neck.down_p4_p5
        return layers

    def count_parameters(self):
        """Return (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, train


# =====================================================================
#  Utility: Load Ultralytics pretrained weights (partial match)
# =====================================================================

# Ultralytics YOLOv8-S module index → our attribute path
_UL_KEY_MAP = {
    'model.0':  'backbone.down0',
    'model.1':  'backbone.down1',
    'model.2':  'backbone.c2f_1',
    'model.3':  'backbone.down2',
    'model.4':  'backbone.c2f_2',
    'model.5':  'backbone.down3',
    'model.6':  'backbone.c2f_3',
    'model.7':  'backbone.down4',
    'model.8':  'backbone.c2f_4',
    'model.9':  'backbone.sppf',
    'model.12': 'neck.c2f_td_p4',
    'model.15': 'neck.c2f_td_p3',
    'model.16': 'neck.down_p3_p4',
    'model.18': 'neck.c2f_bu_p4',
    'model.19': 'neck.down_p4_p5',
    'model.21': 'neck.c2f_bu_p5',
}


def _translate_ul_key(ul_key: str) -> str:
    """Translate an Ultralytics key name to our model's key name."""
    parts = ul_key.split('.')
    prefix = '.'.join(parts[:2])          # e.g. 'model.0'
    rest = '.'.join(parts[2:])            # e.g. 'conv.weight'
    own_prefix = _UL_KEY_MAP.get(prefix)
    if own_prefix is None:
        return None
    # Ultralytics uses '.m.' for bottleneck list, we use '.bottlenecks.'
    rest = rest.replace('.m.', '.bottlenecks.').replace('m.', 'bottlenecks.')
    return f"{own_prefix}.{rest}" if rest else own_prefix


def load_pretrained_ultralytics(model: YOLOv8S, weights_path: str):
    """
    Load weights from an Ultralytics .pt file into our explicit model.
    Maps Ultralytics key names (model.0, model.1, ...) to our named
    attributes (backbone.down0, backbone.c2f_1, ...).  Only loads layers
    whose shapes match exactly; others are skipped.
    """
    import os
    if not os.path.exists(weights_path):
        print(f"  [WARN] Weights file not found: {weights_path}")
        print(f"  Training from scratch (random init).")
        return 0

    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        pretrained_sd = state["model"].float().state_dict()
    elif hasattr(state, "state_dict"):
        pretrained_sd = state.state_dict()
    else:
        pretrained_sd = state

    own_sd = model.state_dict()
    loaded, skipped = 0, 0
    for ul_key, v in pretrained_sd.items():
        own_key = _translate_ul_key(ul_key)
        if own_key is not None and own_key in own_sd:
            if own_sd[own_key].shape == v.shape:
                own_sd[own_key] = v
                loaded += 1
            else:
                skipped += 1
        # Keys not in _UL_KEY_MAP (detect head, unused layers) are silently ignored
    model.load_state_dict(own_sd, strict=False)
    total = len(own_sd)
    print(f"  Loaded {loaded}/{total} pretrained weight tensors (shape-matched).")
    if skipped:
        print(f"  Skipped {skipped} tensors (shape mismatch — expected for modified layers).")
    return loaded


# =====================================================================
#  Self-test
# =====================================================================

if __name__ == "__main__":
    print("YOLOv8-S Architecture Self-Test")
    print("=" * 50)

    model = YOLOv8S(num_classes=4)
    total, trainable = model.count_parameters()
    print(f"  Parameters: {total:,} total, {trainable:,} trainable")

    x = torch.randn(1, 3, 640, 640)
    outputs = model(x)

    print(f"\n  Input: {x.shape}")
    for i, (cls_out, reg_out) in enumerate(outputs):
        print(f"  Scale P{i+3}: cls={cls_out.shape}, reg={reg_out.shape}")

    # list all down-sampling layers
    print(f"\n  Down-sampling layers (replaceable with WaveletPool):")
    for name, layer in model.get_downsample_layers().items():
        print(f"    {name}: {layer.__class__.__name__}")

    # gradient check
    loss = sum(c.sum() + r.sum() for c, r in outputs)
    loss.backward()
    print(f"\n  Backward pass: OK ✓")
    print("All tests passed ✓")
