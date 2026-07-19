# Anti-Aliasing Wavelet Pooling for Object Detection

Code for the paper **"When Does Anti-Aliasing Help Object Detection? A Filter-Footprint vs Object-Size Framework."**

This repository studies when a zero-parameter anti-aliasing (AA) pre-filter placed before strided
downsampling helps or hurts CNN object detection, and provides a simple, training-free rule that
predicts the sign of the effect from a single dataset statistic.

## Overview

Strided convolutions violate the Nyquist sampling condition and introduce aliasing that can corrupt
the features of small, dense objects. A low-pass AA pre-filter reduces this aliasing, but if the
filter is too large it smears the objects themselves. We formalize this trade-off:

- **Filter footprint** `F = k * s`, the number of input-image pixels a `k x k` kernel covers at a
  layer with input stride `s`.
- **Object size** `q25`, the 25th-percentile of the object-size distribution `d = sqrt(w * h)`.

**Predictive rule.** AA improves mAP@0.5 when `F < q25` and degrades it when `F > q25`. The rule
depends only on `q25` (a dataset statistic) and `F` (a design choice), so it can be evaluated before
any training is run.

The pipeline has three parts: (1) an FFT + Otsu analysis that scores each downsampling layer by its
above-Nyquist energy and selects the most aliasing-prone layers, (2) insertion of a zero-parameter AA
pre-filter at those layers, and (3) the footprint-vs-`q25` rule that anticipates the sign of the change
in mAP.

## Datasets

| Dataset   | Domain                         | Classes | q25 (px @ 640) |
|-----------|--------------------------------|:-------:|:--------------:|
| DIOR      | Aerial, dense small objects    | 20      | 13             |
| DOTA v1.5 | Aerial patches (640x640)       | 16      | 26             |
| IP102     | Close-up insect pests          | 102     | 288            |

The three datasets span more than an order of magnitude in object size and cover both regimes where
AA fails (DIOR, DOTA) and where it succeeds (IP102).

## Repository structure

```
config.py, config_ip102.py        Central configuration (paths, hyperparameters, per-dataset settings)
model.py                          Pure-PyTorch YOLOv8-S (every layer visible for AA insertion)
wavelet_pool.py                   Zero-parameter AA pre-filters: Coiflet-3 wavelet, binomial BlurPool, Gaussian
aliasing_analysis.py              FFT + Otsu per-layer aliasing scoring and data-driven layer selection
aliasing_analysis_ip102.py        Same analysis for IP102
train.py, train_ip102.py          Training entry points (baseline + AA variants)
bootstrap_ci.py                   1000-sample bootstrap confidence intervals over test images
dota_task2_eval.py                DOTA Task-2 (axis-aligned) evaluation
preprocess_dior.py                DIOR preprocessing to YOLO format
preprocess_dota.py                DOTA slicing into 640x640 patches (stride 320)
preprocess_ip102.py               IP102 preprocessing and stratified split
unzip_dior.py, unzip_dota.py      Archive extraction helpers
plot_results.py, paper_figures.py Result plots and paper figures
system_model.py                   Standalone renderer for the system-overview figure
dataset/{dior,dota,ip102}/split/  Per-dataset data.yaml files
```

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.9+ and PyTorch 2.x. See `requirements.txt` for the full list.

## Usage

### 1. Prepare a dataset

Download the raw dataset, then run the corresponding preprocessing script (edit the paths in
`config.py` / `config_ip102.py` first):

```bash
python preprocess_dior.py     # DIOR  -> dataset/dior/split
python preprocess_dota.py     # DOTA  -> dataset/dota/split  (640x640 patches, stride 320)
python preprocess_ip102.py    # IP102 -> dataset/ip102/split
```

### 2. Select the layers to modify (FFT + Otsu)

```bash
python aliasing_analysis.py --dataset dior
```

This scores each stride-2 layer by its fraction of feature-map energy above the Nyquist bound and
writes the selected (aliasing-prone) layers used by the AA variants.

### 3. Train

```bash
# No-anti-aliasing baseline
python train.py --dataset dior --variant baseline

# Anti-aliasing variants (inserted only at the selected layers)
python train.py --dataset dior --variant A          # Coiflet-3 wavelet
python train.py --dataset dior --variant blurpool   # binomial BlurPool
python train.py --dataset dior --variant gaussian   # Gaussian pre-filter

# Filter-size ablation (smaller footprints)
python train.py --dataset dota --variant blurpool_small   # 2-tap
python train.py --dataset dota --variant gaussian_small   # 3x3, sigma=0.5

# Run every variant for a dataset
python train.py --dataset ip102 --variant all
```

Useful flags:

- `--augment` enables the online augmentation pipeline (routes output to `runs/{dataset}_aug_{variant}/`).
- `--seed N` fixes all RNGs for reproducible, independent repeats (routes to a suffixed run dir).
- `--eval-tta` reports test-time-augmentation results.
- `--epochs`, `--device` override the schedule and target device.

Available `--variant` values: `baseline`, `A`, `B`, `blurpool`, `gaussian`, `blurpool_small`,
`gaussian_small`, `ablation_small`, `all`. Available `--dataset` values: `dior`, `dota`, `ip102`.

## Notes

- AA filters are parameter-free, so all variants share identical parameter counts with the baseline.
- DOTA is evaluated on 640x640 patches with axis-aligned boxes (`dota_task2_eval.py`), which differs
  from the oriented-detection competition benchmark.
- The DOTA development kit is a separate third-party toolkit and must be installed on its own for
  Task-2 evaluation.
