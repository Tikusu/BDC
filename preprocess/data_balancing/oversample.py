import argparse
import random
import shutil
import sys
from pathlib import Path
import json

from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

# import from utils.py
import utils as _util
from utils import collect_images


def _aug_transform(img: Image.Image, rng: random.Random, aug_params: dict) -> Image.Image:
    """Apply a randomised but realistic set of transforms to one image based on aug_params."""
    img = img.convert("RGB")

    # Horizontal flip
    flip_h_prob = aug_params.get("flip_h_prob", 0.5)
    if rng.random() < flip_h_prob:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Vertical flip
    flip_v_prob = aug_params.get("flip_v_prob", 0.25)
    if rng.random() < flip_v_prob:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # Random rotation
    rot_range = aug_params.get("rotation_range", [-30, 30])
    if rot_range and len(rot_range) == 2:
        angle = rng.uniform(rot_range[0], rot_range[1])
        img = img.rotate(angle, expand=True, resample=Image.BICUBIC)

    # Random crop then resize back to original dimensions
    crop_range = aug_params.get("crop_range", [0.75, 1.0])
    w, h = img.size
    if crop_range and len(crop_range) == 2:
        crop_ratio = rng.uniform(crop_range[0], crop_range[1])
        new_w = max(1, int(w * crop_ratio))
        new_h = max(1, int(h * crop_ratio))
        x0 = rng.randint(0, w - new_w)
        y0 = rng.randint(0, h - new_h)
        img = img.crop((x0, y0, x0 + new_w, y0 + new_h))
        img = img.resize((w, h), Image.BICUBIC)

    # Brightness jitter
    bright_range = aug_params.get("brightness_range", [0.80, 1.20])
    if bright_range and len(bright_range) == 2:
        factor = rng.uniform(bright_range[0], bright_range[1])
        img = ImageEnhance.Brightness(img).enhance(factor)

    # Contrast jitter
    contrast_range = aug_params.get("contrast_range", [0.80, 1.20])
    if contrast_range and len(contrast_range) == 2:
        factor = rng.uniform(contrast_range[0], contrast_range[1])
        img = ImageEnhance.Contrast(img).enhance(factor)

    # Colour saturation jitter
    sat_range = aug_params.get("saturation_range", [0.75, 1.25])
    if sat_range and len(sat_range) == 2:
        factor = rng.uniform(sat_range[0], sat_range[1])
        img = ImageEnhance.Color(img).enhance(factor)

    # Gaussian blur
    blur_prob = aug_params.get("blur_prob", 0.30)
    blur_max = aug_params.get("blur_max_radius", 1.5)
    if rng.random() < blur_prob:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0, blur_max)))

    return img


def oversample_class(
    src_files: list[Path],
    dest_dir: Path,
    target: int,
    seed: int,
    aug_params: dict,
    dry_run: bool,
) -> int:
    """
    Copy all original images to *dest_dir* then generate augmented copies until
    we reach *target* images.  Returns the number of files written / planned.
    """
    rng = random.Random(seed)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # First pass: copy originals
    for src in src_files:
        dst = dest_dir / src.name
        if not dry_run:
            shutil.copy2(src, dst)

    n_originals = len(src_files)
    needed = target - n_originals
    if needed <= 0:
        return n_originals

    # Second pass: augmentation
    # Decide how many augmented copies each source image must produce.
    # Cycle through source images so the generation load is spread evenly.
    aug_count = 0
    source_cycle = list(src_files)
    rng.shuffle(source_cycle)
    idx = 0
    aug_counter: dict[str, int] = {}  # track per-name counter to avoid collisions

    pbar = tqdm(total=needed, desc=f"  Augmenting → {dest_dir.name}", unit="img", leave=False)
    while aug_count < needed:
        src = source_cycle[idx % len(source_cycle)]
        idx += 1
        stem = src.stem
        aug_counter[stem] = aug_counter.get(stem, 0) + 1
        suffix = f"_aug{aug_counter[stem]}{src.suffix}"
        dst = dest_dir / (stem + suffix)

        if not dry_run:
            img = Image.open(src)
            aug_img = _aug_transform(img, rng, aug_params)
            aug_img.save(dst)
            img.close()

        aug_count += 1
        pbar.update(1)
    pbar.close()

    return n_originals + aug_count


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Oversample minority classes using targeted augmentation."
        )
    )

    ap.add_argument(
        "--input-dir",
        default=str(_util._DEFAULT_INPUT),
    )
    ap.add_argument(
        "--output-dir",
        default=str(_util._DEFAULT_OUTPUT),
    )
    ap.add_argument(
        "--config",
        default=str(_util._CONFIG_PATH),
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=729,
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    config_path = Path(args.config)
    train_dir = input_dir / "train"

    if not train_dir.exists():
        print(f"Error: train directory not found in '{train_dir}'", file=sys.stderr)
        sys.exit(1)

    if not config_path.exists():
        print(f"Error: config file not found in '{config_path}'", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r") as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in {config_path}", file=sys.stderr)
            sys.exit(1)
        
    oversample_targets = config.get("oversample", {})
    aug_params = config.get("augmentation_params", {})

    if not oversample_targets:
        print("WARNING: No \"oversample\" key or target class in the config.")
        sys.exit(0)

    print("=" * 70)
    print("Starting Oversampling (Image Augmentation)")
    print("=" * 70)
    print(f"Configuration   : {config_path.resolve()}")
    print(f"Input Dataset   : {input_dir.resolve()}")
    print(f"Output Dataset  : {output_dir.resolve()}")
    print(f"Seed            : {args.seed}")
    print("Dry run         : " + ("YES" if args.dry_run else "NO"))
    print("-" * 70)
    
    for class_name, target in oversample_targets.items():
        class_dir = train_dir / class_name
        dest = output_dir / "train" / class_name

        if not class_dir.exists():
            print(f"Class name {class_name} not found in input directory. Skipped.")
            continue

        files = collect_images(class_dir)
        n = len(files)

        print(f"\nClass {class_name} (original: {n} → target: {target})")

        if n >= target:
            print(f"Class {class_name} already has {n} images (target: {target}). Skipped.")
            if not args.dry_run:
                dest.mkdir(parents=True, exist_ok=True)
                for f in tqdm(files, desc=f"copying {class_name} as-is", unit="img", leave=False):
                    shutil.copy2(f, dest / f.name)
            continue

        if not args.dry_run:
            written = oversample_class(files, dest, target, args.seed, aug_params, dry_run=False)
            print(f"  ✓ Done oversampling: produced {written:,} total images.")
        else:
            print(f"  Dry run: would result in {target:,} total images.")


if __name__ == "__main__":
    main()