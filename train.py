"""
Train the moth detection model — fully transparent, no black-box library.

All architecture code lives in model.py (every layer visible).
WaveletPool is applied SELECTIVELY based on the aliasing analysis results.

Variants:
  baseline  –  Standard YOLOv8-S (no WaveletPool)
  A         –  YOLOv8-S + WaveletPool on HIGH-aliasing layers only
  B         –  Bottom-heavy YOLOv8-S + WaveletPool (selective)

Run:
    python train.py --variant baseline
    python train.py --variant A
    python train.py --variant B
    python train.py --variant all
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
from torch.utils.data import Dataset, DataLoader
import cv2

import config as cfg
from model import YOLOv8S
from wavelet_pool import WaveletPool
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

    For each scale, we:
      1. Assign ground-truth boxes to grid cells (SimOTA-like simple assignment)
      2. Classification: BCE loss on positive cells
      3. Regression: CIoU-like L1 loss on box offsets

    This is a transparent, readable implementation — not a black-box call.
    """

    def __init__(self, num_classes, strides=(8, 16, 32)):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.cls_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.reg_loss = nn.SmoothL1Loss(reduction="mean")

    def forward(self, predictions, targets_list):
        """
        predictions: list of (cls_out, reg_out) per scale
        targets_list: list of (N_i, 5) tensors per image  [cls, x, y, w, h]
        """
        device = predictions[0][0].device
        B = predictions[0][0].shape[0]
        total_cls = torch.tensor(0.0, device=device)
        total_reg = torch.tensor(0.0, device=device)
        n_pos = 0

        for scale_idx, (cls_out, reg_out) in enumerate(predictions):
            stride = self.strides[scale_idx]
            _, _, fH, fW = cls_out.shape

            for b in range(B):
                tgt = targets_list[b].to(device)
                if tgt.shape[0] == 0:
                    continue

                # create target maps
                cls_target = torch.zeros(self.num_classes, fH, fW, device=device)
                reg_target = torch.zeros(4, fH, fW, device=device)
                mask = torch.zeros(1, fH, fW, dtype=torch.bool, device=device)

                for t in range(tgt.shape[0]):
                    cls_id = int(tgt[t, 0])
                    cx, cy, bw, bh = tgt[t, 1], tgt[t, 2], tgt[t, 3], tgt[t, 4]
                    # map to grid cell
                    gx = int((cx * cfg.INPUT_SIZE / stride).clamp(0, fW - 1))
                    gy = int((cy * cfg.INPUT_SIZE / stride).clamp(0, fH - 1))

                    cls_target[cls_id, gy, gx] = 1.0
                    reg_target[0, gy, gx] = cx
                    reg_target[1, gy, gx] = cy
                    reg_target[2, gy, gx] = bw
                    reg_target[3, gy, gx] = bh
                    mask[0, gy, gx] = True
                    n_pos += 1

                # classification loss (all cells, but positive cells weighted)
                total_cls += self.cls_loss(cls_out[b], cls_target)

                # regression loss (positive cells only)
                if mask.any():
                    # mask is (1, fH, fW), reg tensors are (4, fH, fW)
                    mask4 = mask.expand(4, fH, fW)  # (4, fH, fW)
                    pred_reg = reg_out[b][mask4].reshape(-1, 4)
                    tgt_reg = reg_target[mask4].reshape(-1, 4)
                    total_reg += self.reg_loss(pred_reg, tgt_reg)

        n_scales = len(predictions)
        loss = (total_cls / max(B * n_scales, 1)) + (total_reg / max(n_pos, 1))
        return loss, {
            "cls_loss": total_cls.item() / max(B * n_scales, 1),
            "reg_loss": total_reg.item() / max(n_pos, 1),
            "n_positives": n_pos,
        }


# =====================================================================
#  Selective WaveletPool Replacement
# =====================================================================

def apply_wavelet_selective(model: YOLOv8S, target_layers: list = None):
    """
    Replace specific down-sampling layers with WaveletPool.

    If target_layers is given, replace only those.
    Otherwise, run aliasing analysis and replace layers above threshold.

    Returns list of replaced layer names.
    """
    replaced = []

    if target_layers is None:
        # auto-detect from aliasing analysis
        try:
            from aliasing_analysis import run_analysis
            scores, target_layers = run_analysis()
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

        # build replacement: Conv(stride=1) → BN → SiLU → WaveletPool
        if hasattr(old_layer, "conv") and isinstance(old_layer.conv, nn.Conv2d):
            old_conv = old_layer.conv
            new_conv = nn.Conv2d(
                old_conv.in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size, stride=1,
                padding=old_conv.padding, groups=old_conv.groups, bias=False,
            )
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight)

            new_block = nn.Sequential(
                nn.Sequential(new_conv, old_layer.bn, old_layer.act),
                WaveletPool(wavelet=cfg.WAVELET_NAME),
            )
            setattr(parent, attr_name, new_block)
            replaced.append(name)
        else:
            new_block = nn.Sequential(old_layer, WaveletPool(wavelet=cfg.WAVELET_NAME))
            setattr(parent, attr_name, new_block)
            replaced.append(name)

    return replaced


def apply_bottom_heavy(model: YOLOv8S):
    """
    Bottom-heavy modification: keep more computation at high resolution.

    Concretely for YOLOv8-S:
      - Stage 2 (P3, 80x80): increase C2f bottlenecks from 2 → 4
      - Stage 4 (P5, 20x20): decrease C2f bottlenecks from 1 → 0
    This shifts compute toward early layers where spatial detail matters.
    """
    from model import C2f
    ch = model.backbone.out_channels  # [128, 256, 512]

    model.backbone.c2f_2 = C2f(ch[0], ch[0], n_bottlenecks=4)
    model.backbone.c2f_4 = C2f(ch[2], ch[2], n_bottlenecks=0)

    total, _ = model.count_parameters()
    print(f"  [Bottom-Heavy] Shifted compute to early layers. Params: {total:,}")
    return model


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
        imgs = imgs.to(device)
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
#  Training Loop
# =====================================================================

def train_variant(variant: str, device: str = "cpu"):
    print(f"\n{'='*60}")
    print(f"  Training variant: {variant.upper()}")
    print(f"{'='*60}")

    # ── Build model ──────────────────────────────────────────────
    model = YOLOv8S(num_classes=cfg.NUM_CLASSES)
    total_p, _ = model.count_parameters()
    print(f"  Base model parameters: {total_p:,}")

    # ── THIS BLOCK for using pretrained weights ───────────────────
    pretrained_path = "/Users/z7a/Desktop/UTK_PhD_material/Semester 6 Spring-26/Project/yolov8s.pt"  # or add PRETRAINED_WEIGHTS = "yolov8s.pt" to config.py
    loaded = load_pretrained_ultralytics(model, pretrained_path)
    if loaded > 0:
        print(f"  Transfer learning enabled: {loaded} layers loaded from COCO pretrained weights")
    # ─────────────────────────────────────────────────────────────

    # ── Freeze backbone for warm-up phase ────────────────────────
    # Must freeze BEFORE building the optimizer so frozen params are
    # excluded from the optimizer's parameter group entirely.
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("  Backbone frozen for warm-up phase")

    if variant.upper() in ("A", "B"):
        print("\n  Running aliasing analysis for selective WaveletPool application...")
        replaced = apply_wavelet_selective(model)
        print(f"  Replaced {len(replaced)} layer(s) with WaveletPool:")
        for r in replaced:
            print(f"    -> {r}")

    if variant.upper() == "B":
        model = apply_bottom_heavy(model)

    total_p, train_p = model.count_parameters()
    print(f"  Final parameters: {total_p:,} total, {train_p:,} trainable")

    model = model.to(device)

    # ── Datasets ─────────────────────────────────────────────────
    train_ds = MothDataset(
        os.path.join(cfg.SPLIT_DIR, "train", "images"),
        os.path.join(cfg.SPLIT_DIR, "train", "labels"),
        cfg.INPUT_SIZE,
    )
    val_ds = MothDataset(
        os.path.join(cfg.SPLIT_DIR, "val", "images"),
        os.path.join(cfg.SPLIT_DIR, "val", "labels"),
        cfg.INPUT_SIZE,
    )
    test_ds = MothDataset(
        os.path.join(cfg.SPLIT_DIR, "test", "images"),
        os.path.join(cfg.SPLIT_DIR, "test", "labels"),
        cfg.INPUT_SIZE,
    )

    train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                          collate_fn=_collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                        collate_fn=_collate_fn, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                         collate_fn=_collate_fn, num_workers=0)

    print(f"  Train: {len(train_ds)} images, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # ── Loss & Optimizer ─────────────────────────────────────────
    loss_fn = DetectionLoss(num_classes=cfg.NUM_CLASSES)

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
        progress = (epoch - cfg.WARMUP_EPOCHS) / max(cfg.EPOCHS - cfg.WARMUP_EPOCHS, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training ─────────────────────────────────────────────────
    run_dir = os.path.join(cfg.RUNS_DIR, f"moth_{variant.lower()}")
    os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    print(f"\n  Starting training: {cfg.EPOCHS} epochs, batch={cfg.BATCH_SIZE}, "
          f"lr={cfg.LEARNING_RATE}, device={device}")
    print(f"  Early stopping patience: {cfg.PATIENCE}")
    print()

    for epoch in range(cfg.EPOCHS):


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
            remaining = cfg.EPOCHS - epoch
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)
            print(f"\n  Epoch {epoch+1}: Backbone unfrozen — full fine-tuning begins (lr={cfg.LEARNING_RATE * 0.1})")
        # ──────────────────────────────────────────────────────────


        model.train()
        epoch_loss = 0.0
        epoch_cls = 0.0
        epoch_reg = 0.0
        t0 = time.time()

        for batch_idx, (imgs, targets) in enumerate(train_dl):
            imgs = imgs.to(device)
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
        print(f"  Epoch {epoch+1:3d}/{cfg.EPOCHS}  "
              f"train_loss={epoch_loss:.4f} (cls={epoch_cls:.4f} reg={epoch_reg:.4f})  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"lr={lr_now:.6f}  {elapsed:.1f}s")

        history.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "val_loss": val_metrics["loss"],
            "lr": lr_now,
        })

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

    # save last weights
    torch.save(model.state_dict(), os.path.join(run_dir, "weights", "last.pt"))

    # ── Test evaluation ──────────────────────────────────────────
    print(f"\n  Loading best weights and evaluating on TEST set ...")
    model.load_state_dict(torch.load(os.path.join(run_dir, "weights", "best.pt"),
                                     map_location=device))
    test_metrics = evaluate(model, test_dl, loss_fn, device)
    print(f"  TEST  loss={test_metrics['loss']:.4f}  "
          f"cls={test_metrics['cls_loss']:.4f}  reg={test_metrics['reg_loss']:.4f}")

    # ── Save history ─────────────────────────────────────────────
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\n  Results saved to: {run_dir}/")
    return history, test_metrics


# =====================================================================
#  CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Train moth detection model")
    parser.add_argument("--variant", type=str, default="baseline",
                        choices=["baseline", "A", "B", "all"])
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cpu/cuda/mps). Auto-detected if omitted.")
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

    if not os.path.exists(cfg.DATA_YAML):
        print("ERROR: data.yaml not found. Run these first:")
        print("  python generate_dataset.py")
        print("  python preprocess_augment_split.py")
        return

    variants = ["baseline", "A", "B"] if args.variant == "all" else [args.variant]

    all_results = {}
    for v in variants:
        _, test_metrics = train_variant(v, device=device)
        all_results[v] = test_metrics

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY — Test Set Results")
    print("=" * 60)
    print(f"  {'Variant':<12s}  {'Loss':<10s}  {'Cls Loss':<10s}  {'Reg Loss':<10s}")
    print("  " + "-" * 45)
    for v, m in all_results.items():
        print(f"  {v:<12s}  {m['loss']:<10.4f}  {m['cls_loss']:<10.4f}  {m['reg_loss']:<10.4f}")
    print()


if __name__ == "__main__":
    main()
