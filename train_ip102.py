"""
Train the moth detection model — fully transparent, no black-box library.

All architecture code lives in model.py (every layer visible).
WaveletPool is applied SELECTIVELY based on the aliasing analysis results.

Variants:
  baseline  –  Standard YOLOv8-S (no anti-aliasing)
  A         –  YOLOv8-S + WaveletSmooth (Coiflet-3) on high-aliasing layers
  B         –  Bottom-heavy YOLOv8-S + WaveletSmooth (selective)
  blurpool  –  YOLOv8-S + BlurSmooth (binomial pre-filter) on same layers as A
  gaussian  –  YOLOv8-S + GaussianSmooth pre-filter on same layers as A

Datasets (--dataset):
  moth      –  custom 4-class moth detection dataset (default)
  ip102     –  IP102 insect pest detection (102 classes)

Run:
    python train.py --dataset moth   --variant baseline
    python train.py --dataset ip102  --variant A
    python train.py --dataset ip102  --variant all   # baseline + A + blurpool + gaussian
"""

import argparse
import os
import time
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
import cv2

import config as cfg
from model import YOLOv8S
from wavelet_pool import WaveletPool, WaveletSmooth, BlurSmooth, GaussianSmooth
from model import load_pretrained_ultralytics

# =====================================================================
#  Dataset
# =====================================================================

class MothDataset(Dataset):
    """
    YOLO-format dataset loader.
    Reads images and label .txt files; returns tensors suitable for training.
    """

    def __init__(self, images_dir, labels_dir, img_size=640):
        self.img_size = img_size
        self.samples = []
        for fname in sorted(os.listdir(images_dir)):
            if not fname.endswith(".jpg"):
                continue
            img_path = os.path.join(images_dir, fname)
            lbl_path = os.path.join(labels_dir, fname.replace(".jpg", ".txt"))
            if os.path.exists(lbl_path):
                self.samples.append((img_path, lbl_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]

        # load & resize image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = torch.from_numpy(img)

        # load labels  [cls, x_c, y_c, w, h]  (normalised)
        labels = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append([float(x) for x in parts])
        if labels:
            labels = torch.tensor(labels, dtype=torch.float32)
        else:
            labels = torch.zeros((0, 5), dtype=torch.float32)

        return img, labels


def _collate_fn(batch):
    """Custom collate: images stacked, labels kept as list (variable length)."""
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, list(labels)


# =====================================================================
#  Loss  –  Simplified multi-task detection loss
# =====================================================================

class DetectionLoss(nn.Module):
    """
    Multi-scale detection loss (classification + regression).

    For each scale:
      1. Assign GT to ALL scales (center cell + small neighbourhood)
      2. Classification: Focal BCE loss with label smoothing
      3. Regression: CIoU loss for better box localization
    """

    def __init__(self, num_classes, strides=(8, 16, 32),
                 focal_alpha=0.25, focal_gamma=3.0,
                 cls_weight=0.5, reg_weight=7.5,
                 label_smoothing=0.05):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.label_smoothing = label_smoothing

    def _focal_bce(self, pred, target):
        """Focal Binary Cross-Entropy with label smoothing.
        Returns the SUM (not mean) so the caller can normalize by num_positives."""
        target = target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = nn.functional.binary_cross_entropy_with_logits(
            pred, target, reduction='none')
        p = torch.sigmoid(pred)
        pt = p * target + (1 - p) * (1 - target)
        focal_w = (1 - pt) ** self.focal_gamma
        alpha_w = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        return (alpha_w * focal_w * bce).sum()

    @staticmethod
    def _ciou_loss(pred_boxes, gt_boxes):
        """
        Complete IoU loss (CIoU) for bounding box regression.
        Inputs: (N, 4) tensors of [x1, y1, x2, y2] in absolute coords.
        Returns: mean CIoU loss (1 - CIoU).
        """
        # intersection
        x1 = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
        y1 = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
        x2 = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
        y2 = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])
        inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

        # union
        a_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        a_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        union = a_pred + a_gt - inter + 1e-7

        iou = inter / union

        # enclosing box
        ex1 = torch.min(pred_boxes[:, 0], gt_boxes[:, 0])
        ey1 = torch.min(pred_boxes[:, 1], gt_boxes[:, 1])
        ex2 = torch.max(pred_boxes[:, 2], gt_boxes[:, 2])
        ey2 = torch.max(pred_boxes[:, 3], gt_boxes[:, 3])
        c_diag_sq = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + 1e-7

        # center distance
        pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
        gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        rho_sq = (pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2

        # aspect ratio penalty
        pred_w = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(1e-6)
        pred_h = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(1e-6)
        gt_w = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(1e-6)
        gt_h = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(1e-6)
        v = (4 / (3.14159 ** 2)) * (torch.atan(gt_w / gt_h) - torch.atan(pred_w / pred_h)) ** 2
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-7)

        ciou = iou - rho_sq / c_diag_sq - alpha * v
        return (1 - ciou).sum()

    def forward(self, predictions, targets_list):
        device = predictions[0][0].device
        B = predictions[0][0].shape[0]
        total_cls = torch.tensor(0.0, device=device)
        total_reg = torch.tensor(0.0, device=device)
        n_pos = 0
        total_pos_cells = 0  # for focal loss normalization

        n_scales_with_pos = 0

        # Convert each per-image target tensor to a plain Python list once
        # (reused across all 3 scales). This avoids 0-dim CUDA-tensor syncs
        # inside the inner loop and lets us build cell tensors via a single
        # numpy->torch H2D copy at the end.
        targets_rows = [t.tolist() for t in targets_list]

        for scale_idx, (cls_out, reg_out) in enumerate(predictions):
            stride = self.strides[scale_idx]
            _, _, fH, fW = cls_out.shape

            # build grid once per scale
            gy_grid, gx_grid = torch.meshgrid(
                torch.arange(fH, device=device, dtype=torch.float32),
                torch.arange(fW, device=device, dtype=torch.float32),
                indexing='ij')

            scale_has_pos = False

            for b in range(B):
                tgt_rows = targets_rows[b]
                if not tgt_rows:
                    continue

                cls_target = torch.zeros(self.num_classes, fH, fW, device=device)
                # Per-cell GT storage: use lists to handle multiple GTs per cell
                cell_gt = {}  # (gy, gx) -> (gt_cx, gt_cy, gt_w, gt_h)

                for row in tgt_rows:
                    cls_id_f, cx, cy, bw, bh = row
                    cls_id = int(cls_id_f)

                    # Assign GT to ALL scales — center cell + small radius
                    gx_c = max(0, min(fW - 1, int(cx * cfg.INPUT_SIZE / stride)))
                    gy_c = max(0, min(fH - 1, int(cy * cfg.INPUT_SIZE / stride)))

                    # Assign center cell + 1 neighbour radius (3x3 around center)
                    radius = 1
                    gx_lo = max(0, gx_c - radius)
                    gx_hi = min(fW - 1, gx_c + radius)
                    gy_lo = max(0, gy_c - radius)
                    gy_hi = min(fH - 1, gy_c + radius)

                    # GT box in absolute pixel coords
                    gt_cx_abs = cx * cfg.INPUT_SIZE
                    gt_cy_abs = cy * cfg.INPUT_SIZE
                    gt_w_abs = bw * cfg.INPUT_SIZE
                    gt_h_abs = bh * cfg.INPUT_SIZE

                    for gy_i in range(gy_lo, gy_hi + 1):
                        for gx_i in range(gx_lo, gx_hi + 1):
                            cls_target[cls_id, gy_i, gx_i] = 1.0
                            # Only keep the closest GT per cell (by distance)
                            key = (gy_i, gx_i)
                            cell_center_x = (gx_i + 0.5) * stride
                            cell_center_y = (gy_i + 0.5) * stride
                            dist = (gt_cx_abs - cell_center_x)**2 + (gt_cy_abs - cell_center_y)**2
                            if key not in cell_gt or dist < cell_gt[key][4]:
                                cell_gt[key] = (gt_cx_abs, gt_cy_abs, gt_w_abs, gt_h_abs, dist)
                            n_pos += 1

                # classification loss (focal BCE — returns sum, normalized later)
                total_cls += self._focal_bce(cls_out[b], cls_target)

                # regression loss (CIoU on positive cells)
                if cell_gt:
                    scale_has_pos = True
                    total_pos_cells += len(cell_gt)

                    # Build one CPU array of (gy, gx, gt_cx, gt_cy, gt_w, gt_h)
                    # per cell, then issue ONE H2D copy instead of six tiny ones.
                    cells_arr = np.asarray(
                        [(k[0], k[1], v[0], v[1], v[2], v[3])
                         for k, v in cell_gt.items()],
                        dtype=np.float32,
                    )
                    cells_t = torch.from_numpy(cells_arr).to(device, non_blocking=True)
                    gy_idx  = cells_t[:, 0].long()
                    gx_idx  = cells_t[:, 1].long()
                    gt_cx_t = cells_t[:, 2]
                    gt_cy_t = cells_t[:, 3]
                    gt_w_t  = cells_t[:, 4]
                    gt_h_t  = cells_t[:, 5]

                    # decode pred boxes: use sigmoid for center offset (constrained to [0,1])
                    tx = torch.sigmoid(reg_out[b, 0, gy_idx, gx_idx])
                    ty = torch.sigmoid(reg_out[b, 1, gy_idx, gx_idx])
                    tw = reg_out[b, 2, gy_idx, gx_idx]
                    th = reg_out[b, 3, gy_idx, gx_idx]
                    pred_cx = (gx_idx.float() + tx) * stride
                    pred_cy = (gy_idx.float() + ty) * stride
                    pred_w = torch.exp(tw.clamp(max=5.0)) * stride
                    pred_h = torch.exp(th.clamp(max=5.0)) * stride
                    pred_boxes = torch.stack([
                        pred_cx - pred_w / 2, pred_cy - pred_h / 2,
                        pred_cx + pred_w / 2, pred_cy + pred_h / 2], dim=1)

                    gt_boxes = torch.stack([
                        gt_cx_t - gt_w_t / 2, gt_cy_t - gt_h_t / 2,
                        gt_cx_t + gt_w_t / 2, gt_cy_t + gt_h_t / 2], dim=1)

                    total_reg += self._ciou_loss(pred_boxes, gt_boxes)

            if scale_has_pos:
                n_scales_with_pos += 1

        n_scales = len(predictions)
        # Both losses normalized by total positive cells (standard detector practice)
        num_pos = max(total_pos_cells, 1)
        cls_loss = total_cls / num_pos
        reg_loss = total_reg / num_pos
        loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        return loss, {
            "cls_loss": cls_loss.item(),
            "reg_loss": reg_loss.item(),
            "n_positives": n_pos,
        }


# =====================================================================
#  Selective WaveletPool Application (Non-Invasive)
# =====================================================================

def apply_wavelet_selective(model: YOLOv8S, target_layers: list = None,
                            dataset: str = "moth"):
    """
    Insert WaveletSmooth anti-aliasing pre-filters BEFORE target
    down-sampling layers.  The original stride-2 conv is left completely
    untouched so ALL pretrained weights remain valid.

    WaveletSmooth has ZERO learnable parameters — it only contains fixed
    wavelet filter buffers.  This means:
      - No weight initialization issues
      - No feature distribution shift for downstream layers
      - No special unfreezing needed

    Returns list of modified layer names.
    """
    modified = []

    if target_layers is None:
        # auto-detect from aliasing analysis on the specified dataset
        try:
            from aliasing_analysis import run_analysis
            scores, target_layers = run_analysis(model=model, dataset=dataset)
        except Exception as e:
            print(f"  [WARN] Aliasing analysis failed ({e}). Replacing all backbone layers.")
            target_layers = [f"backbone.down{i}" for i in range(5)]

    for name in target_layers:
        parts = name.split(".")
        # navigate to parent
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr_name = parts[-1]
        old_layer = getattr(parent, attr_name)

        # Insert WaveletSmooth BEFORE the existing stride-2 conv.
        # The original layer (with pretrained weights) is preserved intact.
        new_block = nn.Sequential(
            WaveletSmooth(wavelet=cfg.WAVELET_NAME),
            old_layer,   # original ConvBnSiLU(stride=2) — untouched
        )
        setattr(parent, attr_name, new_block)
        modified.append(name)

    return modified


def apply_bottom_heavy(model: YOLOv8S):
    """
    Bottom-heavy modification: keep more computation at high resolution.

    Concretely for YOLOv8-S:
      - Stage 2 (P3, 80x80): increase C2f bottlenecks from 2 → 4
      - Stage 4 (P5, 20x20): decrease C2f bottlenecks from 1 → 1 (keep semantic capacity)
    This shifts compute toward early layers where spatial detail matters.

    Partial weight transfer: copies pretrained cv1 + bottleneck weights from
    the original C2f blocks into the new ones so training starts from a
    better initialisation.
    """
    from model import C2f
    ch = model.backbone.out_channels  # [128, 256, 512]

    # ── Save pretrained weights from original blocks ──────────────
    old_c2f_2 = model.backbone.c2f_2  # C2f(128, 128, n_bottlenecks=2)
    old_c2f_4 = model.backbone.c2f_4  # C2f(512, 512, n_bottlenecks=1)
    old_c2f_2_sd = {k: v.clone() for k, v in old_c2f_2.state_dict().items()}
    old_c2f_4_sd = {k: v.clone() for k, v in old_c2f_4.state_dict().items()}

    # ── Create new blocks ────────────────────────────────────────
    # P3 gets more capacity for spatial detail; P5 keeps 1 bottleneck
    # for semantic features (0 bottlenecks destroys P5 → near-zero mAP)
    new_c2f_2 = C2f(ch[0], ch[0], n_bottlenecks=4)
    new_c2f_4 = C2f(ch[2], ch[2], n_bottlenecks=1)  # was 0 → keep 1

    # ── Transfer matching weights: c2f_2 (2→4 bottlenecks) ──────
    # cv1 is identical (same in/out channels)
    new_sd = new_c2f_2.state_dict()
    transferred = 0
    for key in old_c2f_2_sd:
        if key.startswith("cv1.") or key.startswith("bottlenecks.0.") or key.startswith("bottlenecks.1."):
            if key in new_sd and old_c2f_2_sd[key].shape == new_sd[key].shape:
                new_sd[key] = old_c2f_2_sd[key]
                transferred += 1
    # cv2 has different input channels (2+4 vs 2+2 bottlenecks), skip it
    new_c2f_2.load_state_dict(new_sd)
    print(f"    c2f_2: transferred {transferred} pretrained tensors (cv1 + 2 bottlenecks)")

    # ── Transfer matching weights: c2f_4 (1→0 bottlenecks) ──────
    new_sd_4 = new_c2f_4.state_dict()
    transferred_4 = 0
    for key in old_c2f_4_sd:
        if key in new_sd_4 and old_c2f_4_sd[key].shape == new_sd_4[key].shape:
            new_sd_4[key] = old_c2f_4_sd[key]
            transferred_4 += 1
    new_c2f_4.load_state_dict(new_sd_4)
    print(f"    c2f_4: transferred {transferred_4} pretrained tensors")

    model.backbone.c2f_2 = new_c2f_2
    model.backbone.c2f_4 = new_c2f_4

    total, _ = model.count_parameters()
    print(f"  [Bottom-Heavy] Shifted compute to early layers. Params: {total:,}")
    return model


# ─────────────────────────────────────────────────────────────────────
# Shared helper: resolve target layers for any selective variant
# ─────────────────────────────────────────────────────────────────────

def _load_or_detect_target_layers(model, dataset: str = "moth"):
    """
    Return the list of layer names that should receive an anti-aliasing
    pre-filter for the given dataset.  Tries to reuse the dataset-specific
    Variant A layer selection (so all three anti-aliasing techniques target
    exactly the same 2 layers within a dataset for a fair comparison).
    Falls back to running aliasing_analysis on that dataset.
    """
    saved = os.path.join(cfg.RUNS_DIR, f"{dataset}_a", "wavelet_layers.json")
    if os.path.exists(saved):
        with open(saved) as f:
            layers = json.load(f)
        print(f"  Reusing layer selection from {dataset}_a: {layers}")
        return layers
    print(f"  No saved layer selection for '{dataset}' — running aliasing analysis...")
    try:
        from aliasing_analysis import run_analysis
        _, layers = run_analysis(model=model, dataset=dataset)
        return layers
    except Exception as e:
        print(f"  [WARN] Aliasing analysis failed ({e}). Using default layers.")
        return ["backbone.down3", "neck.down_p3_p4"]


def apply_blurpool_selective(model: YOLOv8S, target_layers: list = None,
                             dataset: str = "moth"):
    """
    Insert BlurSmooth anti-aliasing pre-filters BEFORE target down-sampling
    layers (BlurPool strategy, Zhang 2019).  Identical placement logic to
    apply_wavelet_selective — zero learnable parameters, pretrained weights intact.

    Returns list of modified layer names.
    """
    modified = []

    if target_layers is None:
        try:
            from aliasing_analysis import run_analysis
            _, target_layers = run_analysis(model=model, dataset=dataset)
        except Exception as e:
            print(f"  [WARN] Aliasing analysis failed ({e}). Using default layers.")
            target_layers = ["backbone.down3", "neck.down_p3_p4"]

    for name in target_layers:
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr_name = parts[-1]
        old_layer = getattr(parent, attr_name)
        new_block = nn.Sequential(BlurSmooth(), old_layer)
        setattr(parent, attr_name, new_block)
        modified.append(name)

    return modified


def apply_gaussian_selective(model: YOLOv8S, target_layers: list = None,
                             dataset: str = "moth"):
    """
    Insert GaussianSmooth anti-aliasing pre-filters BEFORE target down-sampling
    layers.  Identical placement logic to apply_wavelet_selective — zero
    learnable parameters, pretrained weights intact.

    Returns list of modified layer names.
    """
    modified = []

    if target_layers is None:
        try:
            from aliasing_analysis import run_analysis
            _, target_layers = run_analysis(model=model, dataset=dataset)
        except Exception as e:
            print(f"  [WARN] Aliasing analysis failed ({e}). Using default layers.")
            target_layers = ["backbone.down3", "neck.down_p3_p4"]

    for name in target_layers:
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr_name = parts[-1]
        old_layer = getattr(parent, attr_name)
        new_block = nn.Sequential(GaussianSmooth(), old_layer)
        setattr(parent, attr_name, new_block)
        modified.append(name)

    return modified


# =====================================================================
#  Evaluation
# =====================================================================

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    """Run evaluation; return average loss."""
    model.eval()
    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0
    n_batches = 0

    for imgs, targets in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            preds = model(imgs)
            loss, details = loss_fn(preds, targets)
        total_loss += loss.item()
        total_cls += details["cls_loss"]
        total_reg += details["reg_loss"]
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "cls_loss": total_cls / n,
        "reg_loss": total_reg / n,
    }


# =====================================================================
#  Detection Metrics  (mAP, Precision, Recall, F1, Equivariance)
# =====================================================================

def _iou_1_vs_N(box, boxes):
    """IoU of one box [x1,y1,x2,y2] against N boxes."""
    x1 = torch.max(box[0], boxes[:, 0])
    y1 = torch.max(box[1], boxes[:, 1])
    x2 = torch.min(box[2], boxes[:, 2])
    y2 = torch.min(box[3], boxes[:, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    a1 = (box[2] - box[0]) * (box[3] - box[1])
    a2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a1 + a2 - inter + 1e-7)


def _iou_matrix(b1, b2):
    """IoU matrix (M,N) between two sets of [x1,y1,x2,y2] boxes."""
    if b1.shape[0] == 0 or b2.shape[0] == 0:
        return torch.zeros((b1.shape[0], b2.shape[0]))
    x1 = torch.max(b1[:, None, 0], b2[None, :, 0])
    y1 = torch.max(b1[:, None, 1], b2[None, :, 1])
    x2 = torch.min(b1[:, None, 2], b2[None, :, 2])
    y2 = torch.min(b1[:, None, 3], b2[None, :, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    a1 = ((b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1]))[:, None]
    a2 = ((b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1]))[None, :]
    return inter / (a1 + a2 - inter + 1e-7)


def decode_predictions(predictions, conf_thresh=0.01, strides=(8, 16, 32),
                       img_size=640):
    """Decode raw model outputs into per-image detections [x1,y1,x2,y2,conf,cls].

    Regression encoding: cell-relative offsets (tx, ty) + log-space sizes (tw, th).
      cx = (gx + tx) * stride
      cy = (gy + ty) * stride
      w  = exp(tw) * stride
      h  = exp(th) * stride
    """
    B = predictions[0][0].shape[0]
    all_dets = [[] for _ in range(B)]
    for si, (cls_out, reg_out) in enumerate(predictions):
        stride = strides[si]
        _, _, fH, fW = cls_out.shape
        cls_prob = torch.sigmoid(cls_out)
        max_conf, cls_id = cls_prob.max(dim=1)

        # build grid once per scale
        gy_grid, gx_grid = torch.meshgrid(
            torch.arange(fH, device=reg_out.device, dtype=torch.float32),
            torch.arange(fW, device=reg_out.device, dtype=torch.float32),
            indexing='ij')

        for b in range(B):
            mask = max_conf[b] > conf_thresh
            if not mask.any():
                continue
            confs = max_conf[b][mask]
            classes = cls_id[b][mask].float()
            tx = torch.sigmoid(reg_out[b, 0][mask])
            ty = torch.sigmoid(reg_out[b, 1][mask])
            tw = reg_out[b, 2][mask]
            th = reg_out[b, 3][mask]
            cx = (gx_grid[mask] + tx) * stride
            cy = (gy_grid[mask] + ty) * stride
            w  = torch.exp(tw.clamp(max=5.0)) * stride
            h  = torch.exp(th.clamp(max=5.0)) * stride
            dets = torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2,
                                confs, classes], dim=1)
            all_dets[b].append(dets)
    result = []
    for b in range(B):
        result.append(torch.cat(all_dets[b], 0) if all_dets[b]
                      else torch.zeros((0, 6)))
    return result


# =====================================================================
#  Test-Time Augmentation (TTA)
# =====================================================================

@torch.no_grad()
def _tta_decode(model, imgs, conf_thresh=0.01, img_size=640, strides=(8, 16, 32)):
    """
    Test-Time Augmentation: forward `imgs` through `model` under multiple
    spatial augmentations, reverse the spatial transforms on the predicted
    bounding boxes, and concatenate all detections per image.  The caller
    is expected to run NMS on the merged set to deduplicate.

    Augmentations applied (4 total forward passes per batch):
      1. Original
      2. Horizontal flip
      3. Vertical flip
      4. Upscale by ~1.25x (forward at larger resolution, rescale boxes back)

    Returns: list[Tensor(N, 6)] of detections per image, in `img_size` coords
             with (x1, y1, x2, y2, conf, cls) layout — same as decode_predictions.
    """
    B = imgs.shape[0]
    _, _, H, W = imgs.shape

    # Container of per-image detection lists
    all_dets = [[] for _ in range(B)]

    def _accumulate(dets_list):
        for b in range(B):
            d = dets_list[b]
            if d.shape[0] > 0:
                # clip to image bounds
                d[:, 0].clamp_(0, W)
                d[:, 1].clamp_(0, H)
                d[:, 2].clamp_(0, W)
                d[:, 3].clamp_(0, H)
                all_dets[b].append(d)

    # 1. Original
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        preds = model(imgs)
    _accumulate(decode_predictions(preds, conf_thresh, strides=strides, img_size=img_size))

    # 2. Horizontal flip (mirror across vertical axis)
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        preds_h = model(torch.flip(imgs, dims=[3]))
    dets_h = decode_predictions(preds_h, conf_thresh, strides=strides, img_size=img_size)
    for b in range(B):
        d = dets_h[b]
        if d.shape[0] > 0:
            d = d.clone()
            x1_new = W - d[:, 2]
            x2_new = W - d[:, 0]
            d[:, 0] = x1_new
            d[:, 2] = x2_new
            dets_h[b] = d
    _accumulate(dets_h)

    # 3. Vertical flip (mirror across horizontal axis)
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        preds_v = model(torch.flip(imgs, dims=[2]))
    dets_v = decode_predictions(preds_v, conf_thresh, strides=strides, img_size=img_size)
    for b in range(B):
        d = dets_v[b]
        if d.shape[0] > 0:
            d = d.clone()
            y1_new = H - d[:, 3]
            y2_new = H - d[:, 1]
            d[:, 1] = y1_new
            d[:, 3] = y2_new
            dets_v[b] = d
    _accumulate(dets_v)

    # 4. Upscale (~1.25x → preserves divisibility by 32; e.g. 640 → 800)
    new_size = ((int(img_size * 1.25) + 31) // 32) * 32
    if new_size != img_size:
        imgs_up = torch.nn.functional.interpolate(
            imgs, size=(new_size, new_size), mode="bilinear", align_corners=False
        )
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            preds_up = model(imgs_up)
        dets_up = decode_predictions(preds_up, conf_thresh, strides=strides, img_size=new_size)
        scale = img_size / new_size
        for b in range(B):
            d = dets_up[b]
            if d.shape[0] > 0:
                d = d.clone()
                d[:, :4] = d[:, :4] * scale
                dets_up[b] = d
        _accumulate(dets_up)

    # Concatenate per-image
    result = []
    for b in range(B):
        if all_dets[b]:
            result.append(torch.cat(all_dets[b], dim=0))
        else:
            result.append(torch.zeros((0, 6)))
    return result


def nms(dets, iou_thresh=0.45):
    """Per-class greedy NMS."""
    if dets.shape[0] == 0:
        return dets
    keep = []
    for cid in dets[:, 5].unique():
        cd = dets[dets[:, 5] == cid]
        cd = cd[cd[:, 4].argsort(descending=True)]
        while cd.shape[0] > 0:
            keep.append(cd[0])
            if cd.shape[0] == 1:
                break
            ious = _iou_1_vs_N(cd[0, :4], cd[1:, :4])
            cd = cd[1:][ious < iou_thresh]
    return torch.stack(keep, 0) if keep else torch.zeros((0, 6))


def _compute_ap(recalls, precisions):
    """All-point interpolated AP (COCO-style, more accurate than 11-point)."""
    # Prepend sentinel values
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    # Make precision monotonically decreasing (right to left)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    # Find points where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


@torch.no_grad()
def evaluate_detection_metrics(model, dataloader, device, num_classes=4,
                               conf_thresh=0.01, nms_thresh=0.45,
                               img_size=640, use_tta=False):
    """
    Compute mAP@0.5, mAP@0.75, per-class AP/P/R/F1, weighted-F1.

    When use_tta=True, each image is forwarded under multiple spatial
    augmentations (original + h-flip + v-flip + 1.25x upscale) and the
    predictions are merged before NMS.  See _tta_decode() for details.
    Expect TTA to be ~4x slower than the standard path.
    """
    model.eval()
    # per-class accumulators
    tp50  = {c: [] for c in range(num_classes)}
    tp75  = {c: [] for c in range(num_classes)}
    confs = {c: [] for c in range(num_classes)}
    n_gt  = {c: 0  for c in range(num_classes)}

    for imgs, targets in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        if use_tta:
            batch_dets = _tta_decode(model, imgs, conf_thresh, img_size=img_size)
        else:
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                preds = model(imgs)
            batch_dets = decode_predictions(preds, conf_thresh, img_size=img_size)

        for b in range(imgs.shape[0]):
            dets = nms(batch_dets[b].cpu(), nms_thresh)
            gt = targets[b]
            if gt.shape[0] > 0:
                gt_cls = gt[:, 0].numpy().astype(int)
                cx, cy = gt[:, 1]*img_size, gt[:, 2]*img_size
                w, h   = gt[:, 3]*img_size, gt[:, 4]*img_size
                gt_boxes = torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], 1)
                for c in range(num_classes):
                    n_gt[c] += int((gt_cls == c).sum())
            else:
                gt_cls = np.array([], dtype=int)
                gt_boxes = torch.zeros((0, 4))

            N = dets.shape[0]
            M = gt_boxes.shape[0]
            if N == 0:
                continue
            pred_conf = dets[:, 4].numpy()
            pred_cls  = dets[:, 5].numpy().astype(int)
            iou_mat = _iou_matrix(dets[:, :4], gt_boxes).numpy() if M > 0 else np.zeros((N, 0))

            for iou_thr, tp_dict in [(0.5, tp50), (0.75, tp75)]:
                gt_matched = np.zeros(M, dtype=bool)
                order = np.argsort(-pred_conf)
                for idx in order:
                    c = pred_cls[idx]
                    best_iou, best_j = 0, -1
                    for j in range(M):
                        if gt_matched[j] or int(gt_cls[j]) != c:
                            continue
                        if iou_mat[idx, j] > best_iou:
                            best_iou, best_j = iou_mat[idx, j], j
                    hit = best_iou >= iou_thr and best_j >= 0
                    tp_dict[c].append(hit)
                    if iou_thr == 0.5:
                        confs[c].append(pred_conf[idx])
                    if hit:
                        gt_matched[best_j] = True

    # ── aggregate per-class ──────────────────────────────────────
    per_class = {}
    aps50, aps75, weights = [], [], []
    for c in range(num_classes):
        t50 = np.array(tp50[c], dtype=bool)
        t75 = np.array(tp75[c], dtype=bool)
        cf  = np.array(confs[c])
        tg  = n_gt[c]
        if tg == 0 and len(cf) == 0:
            per_class[c] = dict(ap50=0, ap75=0, precision=0, recall=0, f1=0, n_gt=0)
            aps50.append(0); aps75.append(0); weights.append(0)
            continue
        weights.append(tg)
        if len(cf) == 0:
            # GT exists but no detections for this class
            per_class[c] = dict(ap50=0, ap75=0, precision=0, recall=0, f1=0, n_gt=int(tg))
            aps50.append(0); aps75.append(0)
            continue
        order = np.argsort(-cf)
        t50, t75 = t50[order], t75[order]
        cum50 = np.cumsum(t50)
        cum75 = np.cumsum(t75)
        n = np.arange(1, len(t50) + 1)
        p50, r50 = cum50 / n, cum50 / max(tg, 1)
        p75, r75 = cum75 / n, cum75 / max(tg, 1)
        ap50 = _compute_ap(r50, p50)
        ap75 = _compute_ap(r75, p75)
        aps50.append(ap50); aps75.append(ap75)
        # Find precision/recall at the optimal F1 operating point
        f1_curve = 2 * p50 * r50 / (p50 + r50 + 1e-7)
        best_idx = np.argmax(f1_curve)
        prec = float(p50[best_idx])
        rec  = float(r50[best_idx])
        f1   = float(f1_curve[best_idx])
        per_class[c] = dict(ap50=round(ap50,4), ap75=round(ap75,4),
                            precision=round(prec,4), recall=round(rec,4),
                            f1=round(f1,4), n_gt=int(tg))

    tw = sum(weights)
    wf1 = sum(per_class[c]["f1"] * weights[c] for c in range(num_classes)) / tw if tw else 0
    active = [c for c in range(num_classes) if per_class[c]["n_gt"] > 0]
    avg_prec = float(np.mean([per_class[c]["precision"] for c in active])) if active else 0
    avg_rec  = float(np.mean([per_class[c]["recall"] for c in active])) if active else 0
    return {
        "mAP@0.5":  round(float(np.mean(aps50)), 4),
        "mAP@0.75": round(float(np.mean(aps75)), 4),
        "weighted_f1": round(float(wf1), 4),
        "avg_precision": round(avg_prec, 4),
        "avg_recall": round(avg_rec, 4),
        "per_class": per_class,
    }


@torch.no_grad()
def compute_equivariance_error(model, dataloader, device,
                               shifts=((2,0),(0,2),(2,2))):
    """
    Shift-equivariance error per downsample layer.
    Error = ||f(shift(x)) − shift(f(x))|| / ||f(x)||
    Lower is better; WaveletPool should reduce this.
    """
    model.eval()
    ds_layers = model.get_downsample_layers()
    imgs, _ = next(iter(dataloader))
    imgs = imgs.to(device)

    # capture inputs to each downsample layer
    feat_in = {}
    hooks = []
    for name, layer in ds_layers.items():
        def _hook(mod, inp, out, n=name):
            feat_in[n] = inp[0].detach()
        hooks.append(layer.register_forward_hook(_hook))
    _ = model(imgs)
    for h in hooks:
        h.remove()

    errors = {}
    for name, layer in ds_layers.items():
        x = feat_in[name]
        f_x = layer(x)
        layer_err = []
        for dx, dy in shifts:
            shifted_x = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
            f_shifted = layer(shifted_x)
            shifted_f = torch.roll(f_x, shifts=(dy//2, dx//2), dims=(2, 3))
            diff = (f_shifted - shifted_f).pow(2).mean()
            norm = f_x.pow(2).mean() + 1e-7
            layer_err.append(float((diff / norm).sqrt()))
        errors[name] = round(float(np.mean(layer_err)), 4)
    return errors


# =====================================================================
#  TTA-only Evaluation (loads existing best.pt, no retraining)
# =====================================================================

def evaluate_variant_tta(variant: str, dataset: str = "moth", device: str = "cpu"):
    """
    Load the best.pt checkpoint for an already-trained variant and run
    test-set evaluation with Test-Time Augmentation enabled.

    Results are saved to `runs/{dataset}_{variant}/test_metrics_tta.json`
    (preserving the original `test_metrics.json` so you can A/B compare).
    """
    print(f"\n{'='*60}")
    print(f"  TTA Evaluation — {dataset}_{variant}")
    print(f"{'='*60}")

    ds_cfg = cfg.load_dataset_cfg(dataset)
    split_dir   = ds_cfg["split_dir"]
    num_classes = ds_cfg["num_classes"]
    class_names = ds_cfg["class_names"]

    run_dir = os.path.join(cfg.RUNS_DIR, f"{dataset}_{variant.lower()}")
    weights_path = os.path.join(run_dir, "weights", "best.pt")
    wl_path = os.path.join(run_dir, "wavelet_layers.json")

    if not os.path.exists(weights_path):
        print(f"  ERROR: no checkpoint at {weights_path}")
        return None

    # Build the architecture WITH the same anti-aliasing modifications that
    # were used during training, so the state_dict loads cleanly.
    model = YOLOv8S(num_classes=num_classes)

    target_layers = None
    if os.path.exists(wl_path):
        with open(wl_path) as f:
            target_layers = json.load(f)

    v = variant.upper()
    if v in ("A", "B"):
        apply_wavelet_selective(model, target_layers=target_layers, dataset=dataset)
    elif variant.lower() == "blurpool":
        apply_blurpool_selective(model, target_layers=target_layers, dataset=dataset)
    elif variant.lower() == "gaussian":
        apply_gaussian_selective(model, target_layers=target_layers, dataset=dataset)
    if v == "B":
        apply_bottom_heavy(model)

    state = torch.load(weights_path, map_location=device, weights_only=True) \
            if hasattr(torch, "load") else torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(f"  Loaded {weights_path}")

    # Test dataloader (same as training)
    test_ds = MothDataset(
        os.path.join(split_dir, "test", "images"),
        os.path.join(split_dir, "test", "labels"),
        cfg.INPUT_SIZE,
    )
    test_dl = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                         collate_fn=_collate_fn, num_workers=0)
    print(f"  Test images: {len(test_ds)}  (batch={cfg.BATCH_SIZE})")

    # Run TTA evaluation
    t0 = time.time()
    det_metrics = evaluate_detection_metrics(
        model, test_dl, device, num_classes=num_classes,
        img_size=cfg.INPUT_SIZE, use_tta=True,
    )
    elapsed = time.time() - t0
    print(f"  TTA eval finished in {elapsed:.1f}s")

    # Pull baseline (non-TTA) numbers for printing the delta
    tm_orig_path = os.path.join(run_dir, "test_metrics.json")
    orig = {}
    if os.path.exists(tm_orig_path):
        with open(tm_orig_path) as f:
            orig = json.load(f)

    def _d(k):
        if k in orig:
            delta = det_metrics[k] - orig[k]
            return f" (baseline={orig[k]:.4f}, Δ={delta:+.4f})"
        return ""

    print(f"\n  TTA Results:")
    print(f"    mAP@0.5     = {det_metrics['mAP@0.5']:.4f}{_d('mAP@0.5')}")
    print(f"    mAP@0.75    = {det_metrics['mAP@0.75']:.4f}{_d('mAP@0.75')}")
    print(f"    weighted_f1 = {det_metrics['weighted_f1']:.4f}{_d('weighted_f1')}")
    print(f"    avg_prec    = {det_metrics['avg_precision']:.4f}{_d('avg_precision')}")
    print(f"    avg_recall  = {det_metrics['avg_recall']:.4f}{_d('avg_recall')}")

    # Save separately so non-TTA results stay intact
    det_metrics["tta_enabled"] = True
    det_metrics["tta_eval_time_sec"] = round(elapsed, 1)
    out_path = os.path.join(run_dir, "test_metrics_tta.json")
    with open(out_path, "w") as f:
        json.dump(det_metrics, f, indent=2)
    print(f"\n  Saved: {out_path}")
    return det_metrics


# =====================================================================
#  Training Loop
# =====================================================================

def train_variant(variant: str, dataset: str = "moth", device: str = "cpu",
                  session_epochs: int | None = None):
    """Train one variant. If a checkpoint exists in the run dir, resume from it.

    `session_epochs` caps how many epochs run in this invocation (so a 2-hour
    SLURM job can do a chunk and the next job picks up). When the target
    `epochs` is reached, the final test eval runs and a `done.flag` is written.
    """
    print(f"\n{'='*60}")
    print(f"  Training variant: {variant.upper()}  |  Dataset: {dataset}")
    print(f"{'='*60}")

    # ── Resolve dataset-specific config ───────────────────────────
    ds_cfg = cfg.load_dataset_cfg(dataset)
    split_dir   = ds_cfg["split_dir"]
    num_classes = ds_cfg["num_classes"]
    class_names = ds_cfg["class_names"]
    epochs      = ds_cfg["epochs"]
    print(f"  Classes: {num_classes}   Split dir: {split_dir}")
    print(f"  Epochs:  {epochs}")

    # ── Build model ──────────────────────────────────────────────
    model = YOLOv8S(num_classes=num_classes)
    total_p, _ = model.count_parameters()
    print(f"  Base model parameters: {total_p:,}")

    # ── Pretrained weights (all variants) ──────────────────────────
    # Load COCO pretrained weights first; modified layers (WaveletPool,
    # bottom-heavy C2f blocks) will be re-initialised afterwards but
    # all other layers retain strong pretrained features.
    pretrained_path = os.path.join(cfg.PROJECT_ROOT, "yolov8s.pt")
    loaded = load_pretrained_ultralytics(model, pretrained_path)
    if loaded > 0:
        print(f"  Transfer learning: {loaded} layers loaded from COCO pretrained weights")
    # ─────────────────────────────────────────────────────────────

    # ── Freeze backbone for warm-up phase ────────────────────────
    # Must freeze BEFORE building the optimizer so frozen params are
    # excluded from the optimizer's parameter group entirely.
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("  Backbone frozen for warm-up phase")

    replaced = []
    if variant.upper() in ("A", "B"):
        print("\n  Running aliasing analysis for selective WaveletPool application...")
        replaced = apply_wavelet_selective(model, dataset=dataset)
        print(f"  Applied WaveletSmooth pre-filter to {len(replaced)} layer(s):")
        for r in replaced:
            print(f"    -> {r}  (pretrained weights preserved)")

    elif variant.lower() == "blurpool":
        print("\n  Applying BlurSmooth pre-filter (selective, same layers as Variant A)...")
        target_layers = _load_or_detect_target_layers(model, dataset=dataset)
        replaced = apply_blurpool_selective(model, target_layers, dataset=dataset)
        print(f"  Applied BlurSmooth pre-filter to {len(replaced)} layer(s):")
        for r in replaced:
            print(f"    -> {r}  (pretrained weights preserved)")

    elif variant.lower() == "gaussian":
        print("\n  Applying GaussianSmooth pre-filter (selective, same layers as Variant A)...")
        target_layers = _load_or_detect_target_layers(model, dataset=dataset)
        replaced = apply_gaussian_selective(model, target_layers, dataset=dataset)
        print(f"  Applied GaussianSmooth pre-filter to {len(replaced)} layer(s):")
        for r in replaced:
            print(f"    -> {r}  (pretrained weights preserved)")

    if variant.upper() == "B":
        model = apply_bottom_heavy(model)
        # Unfreeze the new C2f blocks — they have random weights and
        # must train from epoch 1 (rest of backbone stays frozen)
        for name in ["c2f_2", "c2f_4"]:
            block = getattr(model.backbone, name)
            for param in block.parameters():
                param.requires_grad = True
            print(f"    (unfrozen: backbone.{name} — new block, needs training)")

    total_p, train_p = model.count_parameters()
    print(f"  Final parameters: {total_p:,} total, {train_p:,} trainable")

    model = model.to(device)

    # ── Datasets (paths come from dataset-specific split_dir) ───
    train_ds = MothDataset(
        os.path.join(split_dir, "train", "images"),
        os.path.join(split_dir, "train", "labels"),
        cfg.INPUT_SIZE,
    )
    val_ds = MothDataset(
        os.path.join(split_dir, "val", "images"),
        os.path.join(split_dir, "val", "labels"),
        cfg.INPUT_SIZE,
    )
    test_ds = MothDataset(
        os.path.join(split_dir, "test", "images"),
        os.path.join(split_dir, "test", "labels"),
        cfg.INPUT_SIZE,
    )

    dl_kwargs = dict(
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                          collate_fn=_collate_fn, **dl_kwargs)
    val_dl = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                        collate_fn=_collate_fn, **dl_kwargs)
    test_dl = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                         collate_fn=_collate_fn, **dl_kwargs)

    print(f"  Train: {len(train_ds)} images, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # ── Loss & Optimizer ─────────────────────────────────────────
    loss_fn = DetectionLoss(num_classes=num_classes)

    # optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE,
    #                         weight_decay=cfg.WEIGHT_DECAY)

    # Only pass trainable (non-frozen) params to optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )

    # Cosine annealing with warm-up
    def lr_lambda(epoch):
        if epoch < cfg.WARMUP_EPOCHS:
            return (epoch + 1) / cfg.WARMUP_EPOCHS
        progress = (epoch - cfg.WARMUP_EPOCHS) / max(epochs - cfg.WARMUP_EPOCHS, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training ─────────────────────────────────────────────────
    run_dir = os.path.join(cfg.RUNS_DIR, f"{dataset}_{variant.lower()}")
    os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
    ckpt_path = os.path.join(run_dir, "weights", "checkpoint.pt")
    done_flag = os.path.join(run_dir, "done.flag")

    best_val_loss = float("inf")
    patience_counter = 0
    history = []
    start_epoch = 0

    # ── Resume from checkpoint if present ────────────────────────
    if os.path.exists(ckpt_path):
        # weights_only=False: PyTorch 2.6+ default is True (refuses numpy
        # scalars in history list). We saved this file ourselves, so trusted.
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch       = ckpt["epoch"] + 1
        best_val_loss     = ckpt["best_val_loss"]
        patience_counter  = ckpt["patience_counter"]
        history           = ckpt["history"]

        # If we're resuming past warm-up, replicate the post-warmup
        # optimizer/scheduler construction the loop normally does at
        # epoch == WARMUP_EPOCHS, then restore saved state.
        if start_epoch > cfg.WARMUP_EPOCHS:
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(
                model.parameters(),
                lr=cfg.LEARNING_RATE * 0.1,
                weight_decay=cfg.WEIGHT_DECAY
            )
            remaining = epochs - cfg.WARMUP_EPOCHS
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)

        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        print(f"  Resumed from checkpoint: epoch {start_epoch}/{epochs}")

    # ── Determine where this session stops ───────────────────────
    if session_epochs is None:
        end_epoch = epochs
    else:
        end_epoch = min(start_epoch + session_epochs, epochs)

    if start_epoch >= epochs:
        print(f"  Target {epochs} epochs already reached. Going straight to test eval.")

    print(f"\n  Starting training: epochs {start_epoch+1}-{end_epoch}/{epochs}, "
          f"batch={cfg.BATCH_SIZE}, lr={cfg.LEARNING_RATE}, device={device}")
    print(f"  Early stopping patience: {cfg.PATIENCE}")
    print()

    for epoch in range(start_epoch, end_epoch):


        # ── Unfreeze backbone after warm-up ───────────────────────
        if epoch == cfg.WARMUP_EPOCHS:
            for param in model.backbone.parameters():
                param.requires_grad = True
            # Rebuild optimizer so it picks up the newly unfrozen params
            optimizer = optim.AdamW(
                model.parameters(),
                lr=cfg.LEARNING_RATE * 0.1,   # 10x smaller LR for backbone fine-tuning
                weight_decay=cfg.WEIGHT_DECAY
            )
            # Rebuild scheduler for remaining epochs
            remaining = epochs - epoch
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)
            print(f"\n  Epoch {epoch+1}: Backbone unfrozen — full fine-tuning begins (lr={cfg.LEARNING_RATE * 0.1})")
        # ──────────────────────────────────────────────────────────


        model.train()
        epoch_loss = 0.0
        epoch_cls = 0.0
        epoch_reg = 0.0
        t0 = time.time()

        for batch_idx, (imgs, targets) in enumerate(train_dl):
            imgs = imgs.to(device, non_blocking=True)
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                preds = model(imgs)
                loss, details = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_cls += details["cls_loss"]
            epoch_reg += details["reg_loss"]

        scheduler.step()
        n_batches = max(len(train_dl), 1)
        epoch_loss /= n_batches
        epoch_cls /= n_batches
        epoch_reg /= n_batches
        elapsed = time.time() - t0

        # ── Validation ────────────────────────────────────────────
        val_metrics = evaluate(model, val_dl, loss_fn, device)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1:3d}/{epochs}  "
              f"train_loss={epoch_loss:.4f} (cls={epoch_cls:.4f} reg={epoch_reg:.4f})  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"lr={lr_now:.6f}  {elapsed:.1f}s")

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "val_loss": val_metrics["loss"],
            "lr": lr_now,
        }

        # ── Periodic detection metrics on val (every 10 epochs + last) ──
        eval_freq = 10
        if (epoch + 1) % eval_freq == 0 or epoch == epochs - 1:
            val_det = evaluate_detection_metrics(
                model, val_dl, device, num_classes=num_classes, img_size=cfg.INPUT_SIZE)
            epoch_record["val_mAP50"] = val_det["mAP@0.5"]
            epoch_record["val_mAP75"] = val_det["mAP@0.75"]
            epoch_record["val_wF1"] = val_det["weighted_f1"]
            epoch_record["val_avgP"] = val_det["avg_precision"]
            epoch_record["val_avgR"] = val_det["avg_recall"]
            print(f"    -> val mAP@.5={val_det['mAP@0.5']:.4f} "
                  f"P={val_det['avg_precision']:.3f} R={val_det['avg_recall']:.3f}")

        history.append(epoch_record)

        # ── Checkpointing & Early Stopping ────────────────────────
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "weights", "best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1} (patience={cfg.PATIENCE})")
                break

        # ── Resumable checkpoint (every epoch) ────────────────────
        torch.save({
            "model":            model.state_dict(),
            "optimizer":        optimizer.state_dict(),
            "scheduler":        scheduler.state_dict(),
            "epoch":            epoch,
            "best_val_loss":    best_val_loss,
            "patience_counter": patience_counter,
            "history":          history,
        }, ckpt_path)

    # save last weights
    torch.save(model.state_dict(), os.path.join(run_dir, "weights", "last.pt"))

    # ── Persist partial history every session so plotting/inspection works ──
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(run_dir, "wavelet_layers.json"), "w") as f:
        json.dump(replaced, f)

    # ── If we haven't reached the target epoch count, exit and let
    #    the next job continue from the checkpoint.
    last_completed = end_epoch  # because range(start_epoch, end_epoch)
    if last_completed < epochs:
        print(f"\n  Session complete. Trained through epoch {last_completed}/{epochs}.")
        print(f"  Resubmit `sbatch Run_gpu_frontier_script.sh` to continue.")
        return history, None

    # ── Test evaluation (only when target epochs reached) ────────
    print(f"\n  Loading best weights and evaluating on TEST set ...")
    model.load_state_dict(torch.load(os.path.join(run_dir, "weights", "best.pt"),
                                     map_location=device))
    test_metrics = evaluate(model, test_dl, loss_fn, device)
    print(f"  TEST  loss={test_metrics['loss']:.4f}  "
          f"cls={test_metrics['cls_loss']:.4f}  reg={test_metrics['reg_loss']:.4f}")

    # ── Detection metrics (mAP, P/R/F1, equivariance) ────────────
    print(f"\n  Computing detection metrics (mAP, P/R/F1) ...")
    det_metrics = evaluate_detection_metrics(
        model, test_dl, device, num_classes=num_classes, img_size=cfg.INPUT_SIZE)
    test_metrics.update(det_metrics)

    print(f"  mAP@0.5={det_metrics['mAP@0.5']:.4f}  "
          f"mAP@0.75={det_metrics['mAP@0.75']:.4f}  "
          f"weighted-F1={det_metrics['weighted_f1']:.4f}")
    print(f"  Avg-Precision={det_metrics['avg_precision']:.4f}  "
          f"Avg-Recall={det_metrics['avg_recall']:.4f}")
    # For datasets with many classes (e.g. IP102 with 102), only show the
    # top-20 by GT count to keep the console output readable.
    show_top = 20 if num_classes > 20 else num_classes
    print(f"\n  Per-class detection metrics (showing top-{show_top} by #GT):")
    print(f"\n  {'Class':<25s} {'AP@.5':>6s} {'AP@.75':>6s} "
          f"{'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'#GT':>5s}")
    print("  " + "-" * 65)
    sorted_cls = sorted(range(num_classes),
                        key=lambda c: -det_metrics["per_class"][c]["n_gt"])[:show_top]
    for c in sorted_cls:
        m = det_metrics["per_class"][c]
        name = class_names[c] if c < len(class_names) else f"class_{c}"
        print(f"  {name[:25]:<25s} {m['ap50']:6.3f} {m['ap75']:6.3f} "
              f"{m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} {m['n_gt']:5d}")
    total_gt = sum(det_metrics['per_class'][c]['n_gt'] for c in range(num_classes))
    print("  " + "-" * 65)
    print(f"  {'AVERAGE':<25s} {det_metrics['mAP@0.5']:6.3f} {det_metrics['mAP@0.75']:6.3f} "
          f"{det_metrics['avg_precision']:6.3f} {det_metrics['avg_recall']:6.3f} "
          f"{det_metrics['weighted_f1']:6.3f} {total_gt:5d}")

    print(f"\n  Computing equivariance error ...")
    equiv_err = compute_equivariance_error(model, test_dl, device)
    test_metrics["equivariance_error"] = equiv_err
    print(f"  {'Layer':<25s} {'Equiv Error':>12s}")
    print("  " + "-" * 40)
    for name, err in sorted(equiv_err.items(), key=lambda x: -x[1]):
        print(f"  {name:<25s} {err:12.4f}")

    # ── Save final metrics and mark variant complete ─────────────
    # (history.json and wavelet_layers.json already written above; rewrite
    # history once more in case any post-loop changes accumulated.)
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    # done.flag tells future jobs "this variant is fully trained + evaluated"
    open(done_flag, "w").close()

    print(f"\n  Results saved to: {run_dir}/")
    print(f"  Variant '{variant}' COMPLETE.")
    return history, test_metrics


# =====================================================================
#  CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Train detection model (multi-dataset)")
    parser.add_argument("--variant", type=str, default="baseline",
                        choices=["baseline", "A", "B", "blurpool", "gaussian", "all"])
    parser.add_argument("--dataset", type=str, default="moth",
                        choices=list(cfg.DATASET_CONFIGS.keys()),
                        help="Which dataset to train on (default: moth)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cpu/cuda/mps). Auto-detected if omitted.")
    parser.add_argument("--eval-tta", action="store_true",
                        help="Skip training; load existing best.pt and run "
                             "Test-Time Augmentation evaluation only.  Saves "
                             "to test_metrics_tta.json (preserves original).")
    args = parser.parse_args()

    # auto-detect device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # validate dataset is prepared
    ds_cfg = cfg.load_dataset_cfg(args.dataset)
    ds_yaml = os.path.join(ds_cfg["split_dir"], "data.yaml")
    if not os.path.exists(ds_yaml):
        print(f"ERROR: data.yaml not found for dataset '{args.dataset}' at {ds_yaml}")
        print("Run the appropriate preprocess script first:")
        if args.dataset == "moth":
            print("  python preprocess_augment_split.py")
        elif args.dataset == "ip102":
            print("  python preprocess_ip102.py")
        return

    # "all" trains the three anti-aliasing comparison variants + baseline
    # (Variant B is excluded from "all" since it changes architecture, not just filter)
    ALL_VARIANTS = ["baseline", "A", "blurpool", "gaussian"]
    variants = ALL_VARIANTS if args.variant == "all" else [args.variant]

    # ── TTA-only mode: skip training entirely ────────────────────
    if args.eval_tta:
        print("=" * 60)
        print(f"  TTA-only mode — dataset: {args.dataset}")
        print(f"  Variants: {variants}")
        print("=" * 60)
        tta_results = {}
        for v in variants:
            run_dir = os.path.join(cfg.RUNS_DIR, f"{args.dataset}_{v.lower()}")
            if not os.path.exists(os.path.join(run_dir, "weights", "best.pt")):
                print(f"\n  [SKIP] No checkpoint for variant '{v}' at {run_dir}/weights/best.pt")
                continue
            m = evaluate_variant_tta(v, dataset=args.dataset, device=device)
            if m is not None:
                tta_results[v] = m

        # Side-by-side summary: TTA vs baseline (no-TTA)
        print("\n" + "=" * 80)
        print(f"  TTA vs Baseline — Test Set Results ({args.dataset})")
        print("=" * 80)
        print(f"  {'Variant':<10s} {'metric':<15s} {'no-TTA':>10s} {'TTA':>10s} {'Δ':>10s}")
        print("  " + "-" * 60)
        for v, m_tta in tta_results.items():
            tm_path = os.path.join(cfg.RUNS_DIR, f"{args.dataset}_{v.lower()}",
                                   "test_metrics.json")
            if not os.path.exists(tm_path):
                continue
            with open(tm_path) as f:
                m_orig = json.load(f)
            for k in ("mAP@0.5", "mAP@0.75", "weighted_f1"):
                a, b = m_orig.get(k, 0), m_tta.get(k, 0)
                print(f"  {v:<10s} {k:<15s} {a:>10.4f} {b:>10.4f} {b-a:>+10.4f}")
        print()
        return

    # Per-job session cap: train at most this many epochs in one SLURM job
    # so each invocation fits in <2 h wall time. Subsequent jobs resume from
    # the on-disk checkpoint until the variant's total epoch goal is reached.
    SESSION_EPOCHS = 25

    # Pick the first variant that isn't already complete and train it.
    trained_this_session = False
    for v in variants:
        run_dir = os.path.join(cfg.RUNS_DIR, f"{args.dataset}_{v.lower()}")
        done_flag = os.path.join(run_dir, "done.flag")
        if os.path.exists(done_flag):
            print(f"\n  Variant '{v}' already complete — skipping.")
            continue
        train_variant(v, dataset=args.dataset, device=device,
                      session_epochs=SESSION_EPOCHS)
        trained_this_session = True
        # One variant per job: exit so the user (or next dependency job)
        # can pick up the next variant in a fresh allocation.
        print(f"\n  Done for this session. `sbatch` again to continue.")
        break

    if not trained_this_session:
        # All variants done — print the comparison summary by reading
        # each saved test_metrics.json.
        print("\n  All variants complete.\n")
        print("=" * 60)
        print(f"  SUMMARY — Test Set Results  ({args.dataset})")
        print("=" * 60)
        print(f"  {'Variant':<12s}  {'Loss':<10s}  {'Cls':<8s}  {'Reg':<8s}  "
              f"{'mAP@.5':<8s}  {'mAP@.75':<8s}  {'wF1':<8s}  {'AvgP':<8s}  {'AvgR':<8s}")
        print("  " + "-" * 92)
        for v in variants:
            tm_path = os.path.join(cfg.RUNS_DIR, f"{args.dataset}_{v.lower()}",
                                   "test_metrics.json")
            if not os.path.exists(tm_path):
                print(f"  {v:<12s}  (test_metrics.json missing)")
                continue
            with open(tm_path) as f:
                m = json.load(f)
            print(f"  {v:<12s}  {m['loss']:<10.4f}  {m['cls_loss']:<8.4f}  {m['reg_loss']:<8.4f}  "
                  f"{m.get('mAP@0.5',0):<8.4f}  {m.get('mAP@0.75',0):<8.4f}  "
                  f"{m.get('weighted_f1',0):<8.4f}  {m.get('avg_precision',0):<8.4f}  "
                  f"{m.get('avg_recall',0):<8.4f}")
        print()


if __name__ == "__main__":
    main()
