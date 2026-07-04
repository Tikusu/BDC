# Post-Anomaly Audit: Training Data Distribution and Image Corruption Check

We have implemented a subfolder named `post_anomaly_audit` inside `preprocess/` containing basic utilities to audit the training dataset (`TrainImages`) before splitting it into train/val.

## Scripts

1. [inspect_dataset.py](./inspect_dataset.py) - Script to count files per class and plot the distribution.
2. [check_corrupted_images.py](./check_corrupted_images.py) - Script to check for corrupt/damaged files in parallel using Python's Pillow library.

---

## 1. Dataset Inspection

The dataset inspection script scans all subdirectories of `TrainImages` and counts files matching valid image extensions.

### Usage

```bash
python3 preprocess/post_anomaly_audit/inspect_dataset.py
  --data-dir <path_to_dataset>
  --output-plot <path_to_plot>
```

### Verification Results

Running the script on a valid set will output:

```
Inspecting dataset in: <path_to_dataset>

=============================================
Class Name                | Image Count
---------------------------------------------
0_Recyclable              | <count>
1_Electronic              | <count>
2_Organic                 | <count>
---------------------------------------------
Total                     | <count>
=============================================

Distribution plot successfully saved to: <path_to_plot>
```

### Generated Distribution Plot

Here is the distribution plot saved as `class_distribution.png`:

![Class Distribution](./class_distribution.png)

---

## 2. Corrupted Image Check

The corruption check script uses multi-threaded verificaetion (`ThreadPoolExecutor`) to inspect image file headers and attempt to load and decode their pixels.

### Usage

```bash
python3 preprocess/post_anomaly_audit/check_corrupted_images.py
  --data-dir <path_to_dataset>
  --output-report <path_to_report>
  --quarantine-dir <path_to_quarantine>
  --workers <number_of_threads>
```

### Verification Results

Running the script on a valid set will output:

```
Scanning for images in: <path_to_dataset>
Found <total_images> images to verify.
Verifying in parallel using 8 threads...
Verifying images: 100%|##########| <total_verified_images>/<total_images> [03:52<00:00, <rate>img/s]

=============================================
Verification Summary
---------------------------------------------
Total checked:   <total_verified_images>
Total healthy:   <total_healthy_images>
Total corrupted: <total_corrupted_images>
=============================================
Report saved to: <path_to_report>
```
