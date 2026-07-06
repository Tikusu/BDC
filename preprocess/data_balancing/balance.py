"""
balance.py
==========
Main pipeline orchestrator for balancing the training split dataset.
Reads `config.json` and runs oversampling, undersampling, or passthrough
for each training class, then copies the validation split completely unchanged.

"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

# Ensure the script directory is in sys.path for importing local modules
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.append(str(_SCRIPT_DIR))

from tqdm import tqdm
import utils as _util
from utils import collect_images, _cuda_available
from oversample import oversample_class
from undersample import undersample_class


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run dataset balancing pipeline (oversampling, undersampling, and passthrough)."
    )
    ap.add_argument(
        "--input-dir",
        default=str(_util._DEFAULT_INPUT),
        help=f"Path to input directory. Default: {_util._DEFAULT_INPUT}",
    )
    ap.add_argument(
        "--output-dir",
        default=str(_util._DEFAULT_OUTPUT),
        help=f"Path to output balanced directory. Default: {_util._DEFAULT_OUTPUT}",
    )
    ap.add_argument(
        "--config",
        default=str(_util._CONFIG_PATH),
        help=f"Path to json configuration file. Default: {_util._CONFIG_PATH}",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=729,
        help="Random seed for replication (default: 729)",
    )
    ap.add_argument(
        "--device",
        default="cuda" if _cuda_available() else "cpu",
        help="Device to use for PyTorch feature extraction (default: cuda if available, else cpu)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a trial run with plan preview without writing/copying files.",
    )

    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    config_path = Path(args.config)
    train_dir = input_dir / "train"
    val_dir = input_dir / "val"

    # Validate directories and config
    if not train_dir.exists():
        print(f"Error: train split directory not found at '{train_dir}'", file=sys.stderr)
        sys.exit(1)

    if not config_path.exists():
        print(f"Error: configuration file not found at '{config_path}'", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r") as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in '{config_path}': {e}", file=sys.stderr)
            sys.exit(1)

    oversample_targets = config.get("oversample", {})
    undersample_targets = config.get("undersample", {})
    aug_params = config.get("augmentation_params", {})
    under_params = config.get("undersample_params", {})

    extractor_model = under_params.get("extractor_model", "efficientnet_b0")
    batch_size = under_params.get("batch_size", 1024)

    # Detect class folders in train/
    class_dirs = sorted(d for d in train_dir.iterdir() if d.is_dir())
    if not class_dirs:
        print(f"Error: No class directories found in '{train_dir}'", file=sys.stderr)
        sys.exit(1)

    class_info = {}
    for cd in class_dirs:
        files = collect_images(cd)
        class_info[cd.name] = {"files": files, "count": len(files)}

    # Print Pipeline Configuration
    print("=" * 80)
    print("                      DATASET BALANCING PIPELINE                        ")
    print("=" * 80)
    print(f"Configuration   : {config_path.resolve()}")
    print(f"Input Directory : {input_dir.resolve()}")
    print(f"Output Directory: {output_dir.resolve()}")
    print(f"Random Seed     : {args.seed}")
    print(f"Device          : {args.device}")
    print(f"Dry Run         : " + ("YES" if args.dry_run else "NO"))
    print("-" * 80)

    # Build execution plan
    plan_rows = []
    total_in = 0
    total_out = 0
    for name, info in class_info.items():
        n = info["count"]
        total_in += n
        
        # Decide action and target size
        if name in oversample_targets:
            target = oversample_targets[name]
            action = "oversample (augmentation)"
        elif name in undersample_targets:
            target = undersample_targets[name]
            action = "undersample (K-Means)"
        else:
            target = n
            action = "passthrough (keep as-is)"
            
        total_out += target
        plan_rows.append((name, n, target, action))

    print(f"{'Class Name':<25} | {'Original':<10} | {'Target':<10} | {'Action':<30}")
    print("-" * 80)
    for name, cur, tgt, action in plan_rows:
        delta = tgt - cur
        sign = "+" if delta >= 0 else ""
        delta_str = f"({sign}{delta:,})" if delta != 0 else ""
        print(f"{name:<25} | {cur:<10,} | {tgt:<10,} | {action:<30} {delta_str}")
    print("-" * 80)
    print(f"{'Total (train)':<25} | {total_in:<10,} | {total_out:<10,} |")
    print("=" * 80)

    if args.dry_run:
        print("\nDry run completed. No files were written, modified, or moved.")
        sys.exit(0)

    # Execute
    random.seed(args.seed)
    out_train = output_dir / "train"
    out_val = output_dir / "val"

    # Run through train classes
    print("\n>>> Step 1/2: Balancing Train Split...")
    for name, cur, tgt, action in plan_rows:
        files = class_info[name]["files"]
        dest = out_train / name
        dest.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing [{name}] ({cur:,} -> {tgt:,}): {action}")
        
        if name in oversample_targets:
            if cur >= tgt:
                print(f"  Class already has {cur} images. Copying all as-is (target: {tgt}).")
                for f in tqdm(files, desc=f"  Copying {name}", unit="img", leave=False):
                    shutil.copy2(f, dest / f.name)
            else:
                written = oversample_class(
                    src_files=files,
                    dest_dir=dest,
                    target=tgt,
                    seed=args.seed,
                    aug_params=aug_params,
                    dry_run=False
                )
                print(f"  ✓ Produced {written:,} total images.")
                
        elif name in undersample_targets:
            if cur <= tgt:
                print(f"  Class only has {cur} images. Copying all as-is (target: {tgt}).")
                for f in tqdm(files, desc=f"  Copying {name}", unit="img", leave=False):
                    shutil.copy2(f, dest / f.name)
            else:
                written = undersample_class(
                    src_files=files,
                    dest_dir=dest,
                    target=tgt,
                    seed=args.seed,
                    device=args.device,
                    extractor_model=extractor_model,
                    batch_size=batch_size,
                    dry_run=False
                )
                print(f"  ✓ Produced {written:,} total images.")
                
        else:
            # Passthrough
            for f in tqdm(files, desc=f"  Copying {name}", unit="img", leave=False):
                shutil.copy2(f, dest / f.name)
            print(f"  ✓ Copied {cur:,} images as-is.")

    # Copy val split unchanged
    print("\n>>> Step 2/2: Copying Validation Split (As-Is)...")
    if val_dir.exists():
        val_total = 0
        for val_class_dir in sorted(val_dir.iterdir()):
            if not val_class_dir.is_dir():
                continue
            val_files = collect_images(val_class_dir)
            dest = out_val / val_class_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in tqdm(val_files, desc=f"  val/{val_class_dir.name}", unit="img", leave=False):
                shutil.copy2(f, dest / f.name)
            val_total += len(val_files)
            print(f"  Copied {val_class_dir.name}: {len(val_files):,} images")
        print(f"\n✓ Successfully copied {val_total:,} validation images.")
    else:
        print(f"Warning: val directory not found at '{val_dir}'. Skipping val split copying.")

    # Final Directory Summary verification
    print("\n" + "=" * 80)
    print("                          FINAL SUMMARY STATISTICS                            ")
    print("=" * 80)
    
    final_train_counts = {}
    for cd in sorted(out_train.iterdir()):
        if cd.is_dir():
            final_train_counts[cd.name] = len(collect_images(cd))
            
    final_val_counts = {}
    if out_val.exists():
        for cd in sorted(out_val.iterdir()):
            if cd.is_dir():
                final_val_counts[cd.name] = len(collect_images(cd))

    print(f"{'Class Name':<25} | {'Balanced Train':<15} | {'Validation (As-Is)':<20}")
    print("-" * 80)
    all_classes = sorted(set(final_train_counts.keys()) | set(final_val_counts.keys()))
    for name in all_classes:
        tr_cnt = final_train_counts.get(name, 0)
        val_cnt = final_val_counts.get(name, 0)
        print(f"{name:<25} | {tr_cnt:<15,} | {val_cnt:<20,}")
    print("-" * 80)
    print(f"{'Total':<25} | {sum(final_train_counts.values()):<15,} | {sum(final_val_counts.values()):<20,}")
    print("=" * 80)
    print(f"\nPipeline execution successfully finished! Output folder: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
