"""
balance_dataset.py
==================
Balances the training split produced by split_dataset.py using:

  - OVERSAMPLING  (minority classes): targeted augmentation via Pillow transforms
                                       to generate new, visually diverse images.
  - UNDERSAMPLING (majority classes): diversity-preserving K-Means clustering on
                                       deep feature embeddings from a pre-trained
                                       EfficientNet, retaining one centroid-nearest
                                       image per cluster.

Only the *train* split is re-balanced; the *val* split is copied as-is so
validation statistics remain representative of the original distribution.

Usage
-----
    python preprocess/balance_dataset.py [options]

Run with --dry-run first to preview the plan without touching any files.
"""

import argparse
import math
import os
import random
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_INPUT  = _SCRIPT_DIR.parent / "TrainImagesSplit"
_DEFAULT_OUTPUT = _SCRIPT_DIR.parent / "TrainImagesBalanced"

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _aug_transform(img: Image.Image, rng: random.Random) -> Image.Image:
    """Apply a randomised but realistic set of transforms to one image."""
    img = img.convert("RGB")

    # 1. Horizontal flip (50 %)
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 2. Vertical flip (25 %) – safe for e-waste objects that have no canonical
    #    "up" orientation (cables, circuit boards, keyboards…)
    if rng.random() < 0.25:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # 3. Random rotation within ±30 °
    angle = rng.uniform(-30, 30)
    img = img.rotate(angle, expand=True, resample=Image.BICUBIC)

    # 4. Random crop then resize back to original dimensions
    #    Crop between 75 % and 100 % of each side.
    w, h = img.size
    crop_ratio = rng.uniform(0.75, 1.0)
    new_w = max(1, int(w * crop_ratio))
    new_h = max(1, int(h * crop_ratio))
    x0 = rng.randint(0, w - new_w)
    y0 = rng.randint(0, h - new_h)
    img = img.crop((x0, y0, x0 + new_w, y0 + new_h))
    img = img.resize((w, h), Image.BICUBIC)

    # 5. Brightness jitter ±20 %
    factor = rng.uniform(0.80, 1.20)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # 6. Contrast jitter ±20 %
    factor = rng.uniform(0.80, 1.20)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # 7. Colour saturation jitter ±25 %
    factor = rng.uniform(0.75, 1.25)
    img = ImageEnhance.Color(img).enhance(factor)

    # 8. Gaussian blur with small radius (30 % chance, radius 0–1.5 px)
    if rng.random() < 0.30:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.5)))

    return img


def oversample_class(
    src_files: list[Path],
    dest_dir: Path,
    target: int,
    seed: int,
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
            aug_img = _aug_transform(img, rng)
            aug_img.save(dst)
            img.close()

        aug_count += 1
        pbar.update(1)
    pbar.close()

    return n_originals + aug_count



def _extract_features(files: list[Path], device: str) -> "np.ndarray":
    """Extract L2-normalised embeddings using EfficientNet-B0 (global avg pool)."""
    import numpy as np
    import torch
    import timm
    from torchvision import transforms

    print("  Loading EfficientNet-B0 for feature extraction…")
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
    model.eval()
    model.to(device)

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    embeddings = []
    pbar = tqdm(files, desc="  Extracting features", unit="img", leave=False)
    with torch.no_grad():
        for f in pbar:
            img = Image.open(f).convert("RGB")
            tensor = tfm(img).unsqueeze(0).to(device)
            feat = model(tensor).squeeze(0).cpu().numpy()
            # L2 normalise so cosine and Euclidean agree
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            embeddings.append(feat)

    return np.stack(embeddings)


def undersample_class(
    src_files: list[Path],
    dest_dir: Path,
    target: int,
    seed: int,
    device: str,
    dry_run: bool,
) -> int:
    """
    Select *target* images from *src_files* using K-Means clustering on deep
    embeddings, then copy them to *dest_dir*.
    """
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    dest_dir.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        print(f"  Running MiniBatchKMeans (k={target}) on {len(src_files)} images…")
        embeddings = _extract_features(src_files, device)

        kmeans = MiniBatchKMeans(
            n_clusters=target,
            random_state=seed,
            batch_size=min(1024, len(src_files)),
            n_init=3,
            max_iter=300,
        )
        labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_

        # For each cluster, pick the image closest to its centroid
        selected_indices = []
        for k in range(target):
            mask = labels == k
            if not mask.any():
                continue
            cluster_idxs = np.where(mask)[0]
            cluster_embs = embeddings[cluster_idxs]
            dists = np.linalg.norm(cluster_embs - centroids[k], axis=1)
            selected_indices.append(int(cluster_idxs[np.argmin(dists)]))

        selected_files = [src_files[i] for i in selected_indices]

        pbar = tqdm(selected_files, desc=f"  Copying → {dest_dir.name}", unit="img", leave=False)
        for src in pbar:
            shutil.copy2(src, dest_dir / src.name)

        return len(selected_files)
    else:
        # Dry-run: just return the intended count
        return target



def collect_images(class_dir: Path) -> list[Path]:
    return sorted(
        f for f in class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXT
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Balance the training split by oversampling minority classes "
            "(augmentation) and undersampling majority classes "
            "(diversity-preserving K-Means)."
        )
    )
    ap.add_argument(
        "--input-dir",
        default=str(_DEFAULT_INPUT),
    )
    ap.add_argument(
        "--output-dir",
        default=str(_DEFAULT_OUTPUT),
    )
    ap.add_argument(
        "--target-count",
        type=int,
        default=None,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=729,
    )
    ap.add_argument(
        "--device",
        default="cuda" if _cuda_available() else "cpu",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
    )
    args = ap.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    train_dir  = input_dir / "train"
    val_dir    = input_dir / "val"

    if not train_dir.exists():
        print(f"Error: train split not found at '{train_dir}'", file=sys.stderr)
        sys.exit(1)

    class_dirs = sorted(d for d in train_dir.iterdir() if d.is_dir())
    if not class_dirs:
        print(f"Error: No class subdirectories found in '{train_dir}'", file=sys.stderr)
        sys.exit(1)


    class_info: dict[str, dict] = {}
    for cd in class_dirs:
        files = collect_images(cd)
        class_info[cd.name] = {"files": files, "count": len(files)}

    counts = [v["count"] for v in class_info.values()]
    if args.target_count is not None:
        target = args.target_count
    else:
        # Default: match the size of 0_Recyclable
        # If 0_Recyclable doesn't exist, fall back to the max count.
        if "0_Recyclable" in class_info:
            target = class_info["0_Recyclable"]["count"]
        else:
            target = max(counts)


    print(f"Input  directory : {input_dir.resolve()}")
    print(f"Output directory : {output_dir.resolve()}")
    print(f"Target per class : {target:,}")
    print(f"Device           : {args.device}")
    print(f"Seed             : {args.seed}")
    print("Dry run          : " + ("YES" if args.dry_run else "no"))
    print("-" * 70)

    total_in = total_out = 0
    plan_rows = []
    for name, info in class_info.items():
        n = info["count"]
        action = (
            "oversample (augmentation)"   if n < target else
            "undersample (K-Means)"       if n > target else
            "keep as-is"
        )
        plan_rows.append((name, n, target, action))
        total_in  += n
        total_out += target

    print(f"\n{'Class':<25} {'Current':>8} {'Target':>8}  {'Action'}")
    print("-" * 70)
    for name, cur, tgt, action in plan_rows:
        delta = tgt - cur
        sign  = "+" if delta >= 0 else ""
        print(f"{name:<25} {cur:>8,} {tgt:>8,}  {action}  ({sign}{delta:,})")
    print("-" * 70)
    print(f"{'Total (train)':<25} {total_in:>8,} {total_out:>8,}")
    print()

    if args.dry_run:
        print("Dry run complete – no files were written.")
        sys.exit(0)


    random.seed(args.seed)
    out_train = output_dir / "train"
    out_val   = output_dir / "val"

    print("=" * 70)
    print("Step 1 / 2 – Balancing train split")
    print("=" * 70)
    for name, info in class_info.items():
        files  = info["files"]
        n      = info["count"]
        dest   = out_train / name
        dest.mkdir(parents=True, exist_ok=True)
        print(f"\n[{name}]  {n:,} → {target:,}")

        if n < target:
            written = oversample_class(files, dest, target, args.seed, dry_run=False)
            print(f"  ✓ Oversampled to {written:,} images")
        elif n > target:
            written = undersample_class(
                files, dest, target, args.seed, args.device, dry_run=False
            )
            print(f"  ✓ Undersampled to {written:,} images")
        else:
            for f in tqdm(files, desc=f"  Copying → {name}", unit="img", leave=False):
                shutil.copy2(f, dest / f.name)
            print(f"  ✓ Copied {n:,} images (no change needed)")

    print()
    print("=" * 70)
    print("Step 2 / 2 – Copying val split (unchanged)")
    print("=" * 70)
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
            print(f"  {val_class_dir.name}: {len(val_files):,} images")
        print(f"\n  Total val images copied: {val_total:,}")
    else:
        print(f"  Warning: val directory not found at '{val_dir}' – skipped.")


    print()
    print("=" * 70)
    actual_counts = {}
    for cd in sorted(out_train.iterdir()):
        if cd.is_dir():
            actual_counts[cd.name] = len(collect_images(cd))

    print(f"{'Class':<25} {'Images':>8}")
    print("-" * 35)
    for name, cnt in actual_counts.items():
        print(f"{name:<25} {cnt:>8,}")
    print("-" * 35)
    print(f"{'Total (train)':<25} {sum(actual_counts.values()):>8,}")
    print()
    print(f"Balanced dataset saved to: {output_dir.resolve()}")
    print("Done ✓")


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
