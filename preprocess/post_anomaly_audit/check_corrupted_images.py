import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def verify_single_image(img_path, data_dir):
    """
    Verifies a single image by checking structural integrity and loading pixels.
    Returns None if healthy, or a dict with corruption details if invalid.
    """
    try:
        # Try to open the file to verify the header/format
        with Image.open(img_path) as img:
            img.verify()
            
        # img.verify() closes the file. We need to reopen to load the image data (decode pixels)
        with Image.open(img_path) as img:
            img.load()
            
        return None
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        return {
            "file_path": str(img_path.resolve()),
            "relative_path": str(img_path.relative_to(data_dir)),
            "error": error_msg
        }

def main():
    script_dir = Path(__file__).resolve().parent
    default_data_dir = script_dir.parents[1] / "TrainImages"
    default_report_path = script_dir / "corrupted_report.json"

    ap = argparse.ArgumentParser(description="Check for corrupted or damaged images in a dataset using parallel execution.")
    ap.add_argument("--data-dir", default=str(default_data_dir), help="Path to the TrainImages directory")
    ap.add_argument("--output-report", default=str(default_report_path), help="Path to save the JSON report of corrupted images")
    ap.add_argument("--quarantine-dir", default=None, help="Optional path to move corrupted files to")
    ap.add_argument("--workers", type=int, default=None, help="Number of parallel worker threads (default: CPU cores * 4)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning for images in: {data_dir.resolve()}")

    # Find all image files recursively
    image_paths = []
    for ext in VALID_EXT:
        image_paths.extend(data_dir.rglob(f"*{ext}"))
        image_paths.extend(data_dir.rglob(f"*{ext.upper()}"))
    
    # Remove duplicates and sort
    image_paths = sorted(list(set(image_paths)))
    total_images = len(image_paths)
    print(f"Found {total_images} images to verify.")

    if total_images == 0:
        print("No images found with valid extensions.")
        sys.exit(0)

    # Determine workers count
    num_workers = args.workers if args.workers else min(32, (os.cpu_count() or 1) * 4)
    print(f"Verifying in parallel using {num_workers} threads...")

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
        print("Note: Install tqdm for a visual progress bar (pip install tqdm).")

    corrupted_files = []

    # Run multi-threaded verification
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {executor.submit(verify_single_image, path, data_dir): path for path in image_paths}
        
        if has_tqdm:
            with tqdm(total=total_images, desc="Verifying images", unit="img") as pbar:
                for future in as_completed(future_to_path):
                    res = future.result()
                    if res:
                        corrupted_files.append(res)
                        pbar.write(f"[CORRUPTED] {res['relative_path']} - {res['error']}")
                    pbar.update(1)
        else:
            completed_count = 0
            for future in as_completed(future_to_path):
                res = future.result()
                if res:
                    corrupted_files.append(res)
                    print(f"[CORRUPTED] {res['relative_path']} - {res['error']}")
                completed_count += 1
                if completed_count > 0 and completed_count % 1000 == 0:
                    print(f"Progress: {completed_count}/{total_images} verified...")

    # Print summary
    num_corrupted = len(corrupted_files)
    print("\n" + "=" * 45)
    print("Verification Summary")
    print("-" * 45)
    print(f"Total checked:   {total_images}")
    print(f"Total healthy:   {total_images - num_corrupted}")
    print(f"Total corrupted: {num_corrupted}")
    print("=" * 45)

    # Handle quarantine if requested
    if num_corrupted > 0 and args.quarantine_dir:
        quarantine_root = Path(args.quarantine_dir)
        quarantine_root.mkdir(parents=True, exist_ok=True)
        import shutil
        for item in corrupted_files:
            file_to_move = Path(item["file_path"])
            dest = quarantine_root / file_to_move.name
            try:
                shutil.move(str(file_to_move), str(dest))
                item["quarantined_to"] = str(dest.resolve())
                print(f"Quarantined {file_to_move.name} to {dest}")
            except Exception as e:
                print(f"Failed to move {file_to_move.name}: {e}", file=sys.stderr)

    # Save JSON report
    report_path = Path(args.output_report)
    report_data = {
        "summary": {
            "total_checked": total_images,
            "total_healthy": total_images - num_corrupted,
            "total_corrupted": num_corrupted
        },
        "corrupted_files": corrupted_files
    }
    
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)
        print(f"Report saved to: {report_path.resolve()}")
    except Exception as e:
        print(f"Failed to save report to {report_path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
