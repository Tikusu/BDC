# BDC

## Preprocessing

Before executing a script, activate venv first using :
<details>
<summary> Linux </summary>

```bash
cd path/to/project
source venv/bin/activate
```
or
```bash
cd path/to/project
source venv/Scripts/activate
```
</details>

<details>
<summary> Windows </summary>

```bash
cd path\to\project
venv\Scripts\activate
```
or
```bash
cd path\to\project
.\venv\Scripts\activate.ps1
```
</details>

### Augment images
Batch rotate/blur/adjust brightness-contrast-saturation to generate more training data.
```bash
python3 preprocess/augment_images.py <files-or-folders> -o <output_dir> [--rotate DEG] [--blur R] [--random] [--copies N]
```
Full guide: [preprocess/how_to_augment_images.md](preprocess/how_to_augment_images.md)

### Sort images
Classifies images with EfficientNet and flags ones that don't match their expected category (Recyclable / Electronic / Organic) as anomalies.
```bash
python3 preprocess/sort_images.py --input-dir <src> --category "<Category>" --output-dir <dest> --limit 500
```
Full guide: [preprocess/how_to_sort_images.md](preprocess/how_to_sort_images.md)

### Post-anomaly audit
Scripts in [`preprocess/post_anomaly_audit/`](preprocess/post_anomaly_audit) to inspect/clean `TrainImages` before splitting into train/val.
```bash
python3 preprocess/post_anomaly_audit/inspect_dataset.py --data-dir <dataset>
python3 preprocess/post_anomaly_audit/check_corrupted_images.py --data-dir <dataset> --quarantine-dir <dest>
python3 preprocess/post_anomaly_audit/find_duplicates.py --data-dir <dataset> --threshold 3
python3 preprocess/post_anomaly_audit/quarantine_duplicates.py --report duplicates_report.json --dry-run
```
Full guides: [how_to_inspect_dataset.md](preprocess/post_anomaly_audit/how_to_inspect_dataset.md), [how_to_find_duplicates.md](preprocess/post_anomaly_audit/how_to_find_duplicates.md)
