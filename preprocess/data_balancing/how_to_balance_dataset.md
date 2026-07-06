# Dataset Balancing Pipeline Guide

This pipeline balances the training split of the dataset by addressing class imbalance:

- **Oversampling (Minority Classes):** Targeted image augmentation (flips, rotations, crops, color jitters, and blur) using Pillow to reach target sizes.
- **Undersampling (Majority Classes):** Diversity-preserving K-Means clustering on deep CNN features (using EfficientNet-B0 or other backbones) to prune visual redundancies while maintaining overall variance. Includes a **rank-based backfill mechanism** to guarantee target counts are met exactly even if empty clusters occur.
- **Passthrough & Val Splits:** Other classes and the entire validation split are copied completely unchanged to preserve validity.

---

## Directory Structure

```
preprocess/data_balancing/
├── config.json              # Main configuration file (targets & hyperparameters)
├── balance.py               # Orchestrator pipeline script (main entry point)
├── oversample.py            # Augmentation logic
├── undersample.py           # Feature extraction, clustering, and backfill logic
├── utils.py                 # File collection and device helpers
└── how_to_balance_dataset.md# This guide
```

---

## Installation & Setup

Ensure all dependencies, including PyTorch and Scikit-Learn, are installed in your virtual environment:

```bash
# Activate your virtual environment and install requirements
.\venv\Scripts\pip install -r preprocess/requirements.txt
.\venv\Scripts\pip install scikit-learn
```

---

## Configuration (`config.json`)

Configure your targets and experiment parameters directly in `config.json`:

```json
{
  "oversample": {
    "1_Electronic": 8000
  },
  "undersample": {
    "2_Organic": 8000
  },
  "augmentation_params": {
    "flip_h_prob": 0.5,
    "flip_v_prob": 0.25,
    "rotation_range": [-30, 30],
    "crop_range": [0.75, 1.0],
    "brightness_range": [0.8, 1.2],
    "contrast_range": [0.8, 1.2],
    "saturation_range": [0.75, 1.25],
    "blur_prob": 0.3,
    "blur_max_radius": 1.5
  },
  "undersample_params": {
    "extractor_model": "efficientnet_b0",
    "batch_size": 1024
  }
}
```

---

## Usage

### 1. Local Execution (Recommended with GPU)

Run a **dry run** to preview changes before writing any files:

```bash
.\venv\Scripts\python preprocess/data_balancing/balance.py --dry-run
```

Run the **full balancing pipeline**:

```bash
.\venv\Scripts\python preprocess/data_balancing/balance.py
```

## Options (`balance.py`)

| Argument       | Description                                                           | Default                                 |
| -------------- | --------------------------------------------------------------------- | --------------------------------------- |
| `--input-dir`  | Path to the directory containing split folders (`train/` and `val/`). | `TrainImagesSplit`                      |
| `--output-dir` | Target directory where the balanced splits will be saved.             | `TrainImagesBalanced`                   |
| `--config`     | Path to the JSON configuration file.                                  | `preprocess/data_balancing/config.json` |
| `--seed`       | Random seed for augmentations, clustering, and backfill.              | `729`                                   |
| `--device`     | Execution device for feature extraction (`cuda` or `cpu`).            | `cuda` (if available)                   |
| `--dry-run`    | Preview the execution plan without modifying any files.               | _Disabled_                              |
