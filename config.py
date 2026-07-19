"""
Central configuration for the Moth Detection project.
All paths, hyperparameters, class definitions, and augmentation settings.
"""
import os

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# RAW_DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw")          # generated synthetic images + labels
ORIGINAL_DATASET_DIR = os.path.join(PROJECT_ROOT, "Original_dataset", "Detect")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed")      # after augmentation
SPLIT_DIR = os.path.join(PROJECT_ROOT, "dataset", "split")              # train/val/test
DATA_YAML = os.path.join(SPLIT_DIR, "data.yaml")                       # YOLO data config
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")                          # training outputs

# ──────────────────────────────────────────────────────────────────────
# Class Definitions
# ──────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "CD",                # 0  – Codling moth
    "EH",                # 1  – European honey moth
    "IM",                # 2  – Indian meal moth
    "TD",                # 3  – Tobacco/drug store moth
]
NUM_CLASSES = len(CLASS_NAMES)

# ──────────────────────────────────────────────────────────────────────
# Synthetic Dataset Generation
# ──────────────────────────────────────────────────────────────────────
# NUM_SYNTHETIC_IMAGES = 420          # ~419 as described in the proposal
# IMAGES_PER_CLASS = 105              # balanced dataset
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640
# OBJECTS_PER_IMAGE = (1, 5)          # random 1-5 objects per image
SEED = 42

# ──────────────────────────────────────────────────────────────────────
# Augmentation (Albumentations)
# ──────────────────────────────────────────────────────────────────────
AUG_PER_IMAGE = 4                   # augmented copies per original image
TARGET_AUGMENTED_TOTAL = 2000       # aimed effective training set size

# ──────────────────────────────────────────────────────────────────────
# Train / Val / Test Split
# ──────────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.18
TEST_RATIO = 0.12                   # ~52 images held out
STRATIFY = True

# ──────────────────────────────────────────────────────────────────────
# Model & Training Hyperparameters
# ──────────────────────────────────────────────────────────────────────
# Architecture is defined explicitly in model.py (no black-box library).
# YOLOv8-S channel width_mult = 0.5 (channels: 32, 64, 128, 256, 512).
WIDTH_MULT = 0.5
INPUT_SIZE = 640                    # can try 1280 if GPU allows
BATCH_SIZE = 16
EPOCHS = 100
PATIENCE = 100                       # early stopping patience
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 5
FREEZE_EPOCHS = 10                  # freeze backbone for first N epochs
OPTIMIZER = "AdamW"                 # AdamW as alternative to MuSGD
ALIASING_THRESHOLD = 0.10           # Fallback only; adaptive Otsu threshold used by default

# ──────────────────────────────────────────────────────────────────────
# Wavelet Pooling
# ──────────────────────────────────────────────────────────────────────
WAVELET_NAME = "coif3"              # Cohen(3,3) ≈ Coiflet-3 wavelet
WAVELET_MODE = "zero"               # padding mode for DWT

# ──────────────────────────────────────────────────────────────────────
# BlurPool Anti-Aliasing (Zhang 2019 – fixed binomial pre-filter)
# ──────────────────────────────────────────────────────────────────────
BLURPOOL_FILTER_SIZE = 3            # binomial kernel size (odd: 3, 5, 7)

# ──────────────────────────────────────────────────────────────────────
# Gaussian Pre-filter Anti-Aliasing
# ──────────────────────────────────────────────────────────────────────
# Bumped from 5/1.0 → 7/1.5 after IP102 results showed Gaussian was the
# best-performing variant — stronger low-pass filtering helps on noisy
# field photos.  Moth dataset (clean studio shots) also tolerates this.
GAUSSIAN_KERNEL_SIZE = 7            # Gaussian kernel size (odd)
GAUSSIAN_SIGMA = 1.5                # standard deviation

# ──────────────────────────────────────────────────────────────────────
# Small-filter ablation (DOTA cost-benefit mechanism validation)
# ──────────────────────────────────────────────────────────────────────
# If the "filter footprint ≈ object size" theory is right, a smaller AA
# filter should hurt DOTA less (or even match baseline) because the
# smoothing cost shrinks while the aliasing reduction stays comparable.
BLURPOOL_FILTER_SIZE_SMALL = 2      # 2-tap binomial = [0.5, 0.5]
GAUSSIAN_KERNEL_SIZE_SMALL = 3      # 3×3
GAUSSIAN_SIGMA_SMALL = 0.5          # narrow Gaussian (most weight on center)

# ──────────────────────────────────────────────────────────────────────
# Multi-Dataset Support
# ──────────────────────────────────────────────────────────────────────
# Per-dataset configuration. Use `load_dataset_cfg("moth")` or
# `load_dataset_cfg("ip102")` from train.py / plot_results.py / etc.
# Adding a new dataset means adding one entry here + a preprocess script.

IP102_ROOT = os.path.join(PROJECT_ROOT, "IP102_v1_1_dataset")
IP102_SPLIT_DIR = os.path.join(PROJECT_ROOT, "dataset", "ip102", "split")

DOTA_ROOT = os.path.join(PROJECT_ROOT, "DOTA_1_5")
DOTA_SPLIT_DIR = os.path.join(PROJECT_ROOT, "dataset", "dota", "split")

# DOTA v1.0 label set (15 classes, sorted alphabetically). v1.5 adds
# container-crane but only ships annotations for val, so we use v1.0
# throughout for train/test consistency.
DOTA_CLASS_NAMES = [
    "baseball-diamond", "basketball-court", "bridge", "ground-track-field",
    "harbor", "helicopter", "large-vehicle", "plane", "roundabout", "ship",
    "small-vehicle", "soccer-ball-field", "storage-tank", "swimming-pool",
    "tennis-court",
]

DIOR_ROOT = os.path.join(PROJECT_ROOT, "DIOR_dataset")
DIOR_SPLIT_DIR = os.path.join(PROJECT_ROOT, "dataset", "dior", "split")

# DIOR class set (20 classes). Sorted case-insensitively but original
# capitalization is preserved because it matches the XML <name> strings
# exactly — the loader compares case-sensitively when building label files.
DIOR_CLASS_NAMES = [
    "airplane", "airport", "baseballfield", "basketballcourt", "bridge",
    "chimney", "dam", "Expressway-Service-area", "Expressway-toll-station",
    "golffield", "groundtrackfield", "harbor", "overpass", "ship", "stadium",
    "storagetank", "tenniscourt", "trainstation", "vehicle", "windmill",
]

# Dataset-specific epoch counts (defaults; overridden via train.py --epochs)
EPOCHS_MOTH = EPOCHS
EPOCHS_IP102 = 50                   # IP102 has ~13K train images vs 471 moth
EPOCHS_DOTA = 50                    # DOTA: ~50K train patches at stride=320
EPOCHS_DIOR = 50                    # DIOR: 11,725 native 800x800 images

DATASET_CONFIGS = {
    "moth": {
        "split_dir":   SPLIT_DIR,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "epochs":      EPOCHS_MOTH,
    },
    "ip102": {
        "split_dir":   IP102_SPLIT_DIR,
        "num_classes": 102,
        "class_names": None,        # loaded at runtime from classes.txt
        "epochs":      EPOCHS_IP102,
    },
    "dota": {
        "split_dir":   DOTA_SPLIT_DIR,
        "num_classes": len(DOTA_CLASS_NAMES),
        "class_names": DOTA_CLASS_NAMES,
        "epochs":      EPOCHS_DOTA,
    },
    "dior": {
        "split_dir":   DIOR_SPLIT_DIR,
        "num_classes": len(DIOR_CLASS_NAMES),
        "class_names": DIOR_CLASS_NAMES,
        "epochs":      EPOCHS_DIOR,
    },
}


def load_dataset_cfg(name: str) -> dict:
    """
    Resolve a dataset config by name.  For IP102, lazily loads the 102 class
    names from `IP102_v1_1_dataset/classes.txt` (1-indexed in the file but
    stored 0-indexed internally to match YOLO convention).
    """
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_CONFIGS)}")
    ds = dict(DATASET_CONFIGS[name])  # copy

    if name == "ip102" and ds["class_names"] is None:
        classes_path = os.path.join(IP102_ROOT, "classes.txt")
        if os.path.exists(classes_path):
            names = []
            with open(classes_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # lines look like: "1  rice leaf roller"
                    parts = line.split(None, 1)
                    if len(parts) == 2 and parts[0].isdigit():
                        names.append(parts[1].strip())
                    else:
                        names.append(line)
            ds["class_names"] = names
        else:
            # fallback: numeric placeholders
            ds["class_names"] = [f"class_{i}" for i in range(ds["num_classes"])]
    return ds
