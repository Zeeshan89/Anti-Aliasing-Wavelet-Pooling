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
    "whole_moth",        # 0
    "head_thorax",       # 1
    "wing_fragment",     # 2
    "abdomen_fragment",  # 3
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
EPOCHS = 50
PATIENCE = 20                       # early stopping patience
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 5
FREEZE_EPOCHS = 10                  # freeze backbone for first N epochs
OPTIMIZER = "AdamW"                 # AdamW as alternative to MuSGD
ALIASING_THRESHOLD = 0.25           # layers above this score get WaveletPool

# ──────────────────────────────────────────────────────────────────────
# Wavelet Pooling
# ──────────────────────────────────────────────────────────────────────
WAVELET_NAME = "coif3"              # Cohen(3,3) ≈ Coiflet-3 wavelet
WAVELET_MODE = "zero"               # padding mode for DWT
