import argparse
import shutil
import sys
from pathlib import Path
import json

from tqdm import tqdm
from PIL import Image

# import from utils.py
import utils as _util
from utils import collect_images
from utils import _cuda_available

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

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            ""
        )
    )

    ap.add_argument(
        "--input-dir",
        default=_util._DEFAULT_INPUT,
    )
    ap.add_argument(
        "--output-dir",
        default=_util._DEFAULT_OUTPUT,
    )
    ap.add_argument(
        "--config",
        default=_util._CONFIG_PATH,
    )
    ap.add_argument(
        "--seed",
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
    
    undersample_targets = config.get("undersample", {})
    if not undersample_targets:
        print("WARNING: No \"undersample\" key or target class in the config.")
        sys.exit(0)
    
    print("=" * 70)
    print("Starting Undersampling (K-Means Clustering)")
    print("=" * 70)
    print(f"Configuration   : {config_path.resolve()}")
    print(f"Input Dataset   : {input_dir.resolve()}")
    print(f"Output Dataset  : {output_dir.resolve()}")
    print(f"Seed            : {args.seed}")
    print("Dry run         : " + ("YES" if args.dry_run else "NO"))
    print("-" * 70)

    for class_name, target in undersample_targets.items():

        class_dir = train_dir / class_name
        dest = output_dir / "train" / class_name

        if not class_dir.exists():
            print(f"Class name {class_name} not found in input directory. Skipped.")
            continue
            
        files = collect_images(class_dir)
        n = len(files)

        print(f"\nClass {class_name} (original: {n} → target: {target})")

        
        if n <= target:
            print(f"Class {class_name} already has {n} images (target: {target}). Skipped.")
            
            if not args.dry_run:
                dest.mkdir(parents=True, exist_ok=True)
                for f in tqdm(files, desc=f"copying {class_name} as-is", unit="img", leave=False):
                    shutil.copy2(f, dest / f.name)
            continue

        if not args.dry_run:
            written = undersample_class(files, dest, target, args.seed, args.device, args.dry_run)
            print(f"  ✓ Done undersampling: produced {written:,} total images.")
        else:
            print(f"  Dry run: would result in {target:,} total images.")

        print("-" * 70)
        print("\Process done.")

if __name__ == "__main__":
    main()