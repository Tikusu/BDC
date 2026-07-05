import argparse
import random
import shutil
import sys
import os
from pathlib import Path

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IGNORE_DIRS = {"duplicates", "train", "val", "test"}

def main():
    # Resolve default paths relative to this script
    script_dir = Path(__file__).resolve().parent
    default_data_dir = script_dir.parent / "TrainImages"
    default_output_dir = script_dir.parent / "TrainImagesSplit"

    ap = argparse.ArgumentParser(
        description="Stratified train/val split for class-wise image datasets."
    )
    ap.add_argument(
        "--data-dir",
        default=str(default_data_dir),
        help="Path to the source dataset directory (default: ../TrainImages)"
    )
    ap.add_argument(
        "--output-dir",
        default=str(default_output_dir),
        help="Path to output split dataset (creates train/ and val/ subdirectories) (default: ../TrainImagesSplit)"
    )
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of the dataset to include in the validation split (default: 0.15)"
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=729,
        help="Random seed for reproducibility (default: 729)"
    )
    ap.add_argument(
        "--action",
        choices=["copy", "move", "hardlink"],
        default="copy",
        help="Action to perform on images: 'copy' or 'move' or 'hardlink' (default: copy)"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a trial run without writing, copying, or moving"
    )
    args = ap.parse_args()

    # Validate inputs
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if args.val_ratio <= 0.0 or args.val_ratio >= 1.0:
        print(f"Error: val-ratio must be between 0.0 and 1.0 (exclusive). Got {args.val_ratio}", file=sys.stderr)
        sys.exit(1)

    print(f"Source Directory: {data_dir.resolve()}")
    print(f"Output Directory: {output_dir.resolve()}")
    print(f"Validation Ratio: {args.val_ratio}")
    print(f"Random Seed:      {args.seed}")
    print(f"Action:           {args.action}" + (" (DRY RUN)" if args.dry_run else ""))
    print("-" * 60)

    # Find class subdirectories
    class_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.lower() not in IGNORE_DIRS and not d.name.startswith(".")
    ])

    if not class_dirs:
        print(f"No class subdirectories found in '{data_dir}'.", file=sys.stderr)
        sys.exit(1)

    # Initialize random seed for splitting
    random.seed(args.seed)

    split_plan = {}
    total_train = 0
    total_val = 0
    total_original = 0

    # Plan the split per class
    for class_dir in class_dirs:
        class_name = class_dir.name
        # Gather all valid images in this class
        images = sorted([
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_EXT
        ])

        original_count = len(images)
        if original_count == 0:
            print(f"Warning: Class '{class_name}' contains no valid image files. Skipping.")
            continue

        # Shuffle list of images deterministically
        random.shuffle(images)

        # Stratified partition index calculation
        val_count = int(round(original_count * args.val_ratio))
        # Boundary cases: ensure at least one sample is in train and val if we have >= 2 files
        if val_count == 0 and original_count >= 2:
            val_count = 1
        elif val_count == original_count and original_count >= 2:
            val_count = original_count - 1

        val_files = sorted(images[:val_count])
        train_files = sorted(images[val_count:])

        split_plan[class_name] = {
            "train": train_files,
            "val": val_files,
            "original_count": original_count
        }

        total_train += len(train_files)
        total_val += len(val_files)
        total_original += original_count

    if not split_plan:
        print("No images found to split.", file=sys.stderr)
        sys.exit(1)

    # Print split preview/summary
    print("\n" + "=" * 80)
    print(f"{'Class Name':<25} | {'Original':<10} | {'Train Split':<18} | {'Val Split':<18}")
    print("-" * 80)
    for class_name, info in split_plan.items():
        tr_len = len(info["train"])
        val_len = len(info["val"])
        orig_len = info["original_count"]
        tr_pct = (tr_len / orig_len) * 100
        val_pct = (val_len / orig_len) * 100
        
        tr_str = f"{tr_len} ({tr_pct:.1f}%)"
        val_str = f"{val_len} ({val_pct:.1f}%)"
        
        print(f"{class_name:<25} | {orig_len:<10} | {tr_str:<18} | {val_str:<18}")
    print("-" * 80)
    tot_tr_pct = (total_train / total_original) * 100
    tot_val_pct = (total_val / total_original) * 100
    tot_tr_str = f"{total_train} ({tot_tr_pct:.1f}%)"
    tot_val_str = f"{total_val} ({tot_val_pct:.1f}%)"
    print(f"{'Total':<25} | {total_original:<10} | {tot_tr_str:<18} | {tot_val_str:<18}")
    print("=" * 80 + "\n")

    if args.dry_run:
        print("Dry run completed. No files were written, copied, or moved.")
        sys.exit(0)

    # Perform the split
    print("Executing split...")
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    processed_count = 0
    
    # Define worker action helper
    def process_file(src_path, split_name, class_name):
        dest_dir = output_dir / split_name / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / src_path.name

        if args.action == "copy":
            shutil.copy2(src_path, dest_path)
        elif args.action == "move":
            shutil.move(str(src_path), str(dest_path))
        elif args.action == "hardlink":
            # On Windows, os.link requires appropriate permissions, but works on same drive.
            # Remove target if it already exists to overwrite
            if dest_path.exists():
                dest_path.unlink()
            os.link(src_path, dest_path)

    # Flatten plan for progress tracking
    flattened_tasks = []
    for class_name, info in split_plan.items():
        for f in info["train"]:
            flattened_tasks.append((f, "train", class_name))
        for f in info["val"]:
            flattened_tasks.append((f, "val", class_name))

    total_tasks = len(flattened_tasks)

    if has_tqdm:
        for src_path, split_name, class_name in tqdm(flattened_tasks, desc="Splitting dataset", unit="file"):
            process_file(src_path, split_name, class_name)
    else:
        for src_path, split_name, class_name in flattened_tasks:
            process_file(src_path, split_name, class_name)
            processed_count += 1
            if processed_count % 1000 == 0 or processed_count == total_tasks:
                print(f"Progress: {processed_count}/{total_tasks} files processed...")

    print(f"\nDataset successfully split and saved to: {output_dir.resolve()}")
    print(f"Created: \n  - {output_dir.resolve()}/train\n  - {output_dir.resolve()}/val")

if __name__ == "__main__":
    main()
