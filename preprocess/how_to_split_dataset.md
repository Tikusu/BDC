# Stratified Train/Val Dataset Split

Implemented `split_dataset.py` inside `preprocess/` to perform a stratified split of the training image dataset into train and validation sets per class, ensuring the class distribution remains consistent across both training and validation splits.

## Usage

```bash
python3 preprocess/split_dataset.py [OPTIONS]
```

### Options

- `--data-dir PATH`: Path to the source dataset directory containing class subfolders (default: `TrainImages`).
- `--output-dir PATH`: Path to the output directory where split datasets will be saved (default: `TrainImagesSplit`).
- `--val-ratio FLOAT`: Proportion of the dataset to allocate to the validation set (default: `0.15` or 15%).
- `--seed INT`: Random seed for reproducibility (default: `729`).
- `--action {copy,move}`: File operations strategy:
  - `copy` (default): Copies the images to the destination. Safe, does not modify original files, but uses extra disk space.
  - `move`: Moves the images to the destination. Fast, saves disk space, but modifies the source folder.
- `--dry-run`: Performs a trial run to preview the split distribution without writing, copying, or moving any files.

---

## Split Verification (Dry Run)

To perform a dry run and inspect class-wise splits, execute:

```bash
python3 preprocess/split_dataset.py --dry-run
```

### Expected Output

```
Source Directory: <path_to_dataset>
Output Directory: <path_to_split>
Validation Ratio: <ratio>
Random Seed:      <seed>
Action:           <action> (DRY RUN)
------------------------------------------------------------

================================================================================
Class Name      | Original     | Train Split             | Val Split
--------------------------------------------------------------------------------
0_Recyclable    |  <original>  | <count> (<percentage>%) | <count> (<percentage>%)
1_Electronic    |  <original>  | <count> (<percentage>%) | <count> (<percentage>%)
2_Organic       |  <original>  | <count> (<percentage>%) | <count> (<percentage>%)
--------------------------------------------------------------------------------
Total           |  <original>  | <count> (<percentage>%) | <count> (<percentage>%)
================================================================================

Dry run completed. No files were written, copied, or moved.
```
