# Duplicate & Near-Duplicate Detection

Added two scripts in `preprocess/post_anomaly_audit/`:

| Script                                                 | Purpose                                                                              |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| [find_duplicates.py](./find_duplicates.py)             | Detect duplicates using MD5 + pHash DCT, save report in JSON                         |
| [quarantine_duplicates.py](./quarantine_duplicates.py) | Move detected duplicates to `TrainImages/duplicates/` based on generated JSON report |

---

## How To Use

### Step 1: Duplicate Detection

```powershell
.\venv\Scripts\python preprocess/post_anomaly_audit/find_duplicates.py
  --data-dir <path_to_dataset>
  --output-report <path_to_report>
  --threshold <hamming_distance>
  --cross-class <trigger_cross_class_detection>
  --workers <number_of_threads>
```

> NOTE:
> Valid hamming distance value is 0-64

---

### Step 2: Preview (dry-run, SAFE – no files will be moved)

```powershell
.\venv\Scripts\python preprocess/post_anomaly_audit/quarantine_duplicates.py
  --report <path_to_json_report>
  --data-dir <path_to_dataset>
  --quarantine-dir <path_to_quarantine_folder>
  --dry-run
```

### Final: Permanent Quarantine

Run this to actually move the detected dupliates

```powershell
.\venv\Scripts\python preprocess/post_anomaly_audit/quarantine_duplicates.py
  --report <path_to_json_report>
  --data-dir <path_to_dataset>
  --quarantine-dir <path_to_quarantine_folder>
```

> !IMPORTANT
> Before running quarantine without `--dry-run`, make sure to review the `duplicates_report.json` to validate whether the threshold meets the needs. Increase the threshold for more quarantines, decrease it for more selectivity.
