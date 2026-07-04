import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------

def compute_md5(path: Path) -> str:
    """Read the file in blocks and return the MD5 hex-digest."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _dct_matrix(n: int) -> np.ndarray:
    """
    Build a normalized DCT-II matrix of size n×n using pure numpy.
    Equivalent to scipy.fft.dct(type=2, norm='ortho') without scipy dependencies.
    """
    k = np.arange(n, dtype=np.float64)
    cols = np.arange(n, dtype=np.float64)
    C = np.cos(np.pi * np.outer(k, 2.0 * cols + 1.0) / (2.0 * n))
    C[0] *= 1.0 / np.sqrt(n)
    C[1:] *= np.sqrt(2.0 / n)
    return C


# Pre-compute DCT matrix for default size (hash_size=8 → img_size=32)
_DEFAULT_HASH_SIZE = 8
_DCT_MAT = _dct_matrix(_DEFAULT_HASH_SIZE * 4)


def compute_phash(path: Path, hash_size: int = 8) -> np.ndarray:
    """
    Calculate the perceptual hash (DCT pHash) of an image.
    Returns a numpy bool array with a length of hash_size**2.
    Pure-numpy implementation without scipy.
    """
    img_size = hash_size * 4
    img = Image.open(path).convert("L").resize((img_size, img_size), Image.LANCZOS)
    pixels = np.array(img, dtype=np.float64)

    # Use the pre-computed DCT matrix if the size matches, create a new one if it doesn't.
    C = _DCT_MAT if img_size == _DEFAULT_HASH_SIZE * 4 else _dct_matrix(img_size)

    # 2D DCT via separable matrix multiplication: C @ pixels @ C^T
    dct_pixels = C @ pixels @ C.T
    dct_low = dct_pixels[:hash_size, :hash_size]
    avg = (dct_low.sum() - dct_low[0, 0]) / (hash_size * hash_size - 1)
    return dct_low.flatten() > avg


def hamming_distance(h1: np.ndarray, h2: np.ndarray) -> int:
    return int(np.count_nonzero(h1 != h2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_images(data_dir: Path) -> list[Path]:
    paths = []
    for ext in VALID_EXT:
        paths.extend(data_dir.rglob(f"*{ext}"))
        paths.extend(data_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(paths))


def choose_keeper(paths: list[Path]) -> Path:
    """
    Choose the file to keep from a group of duplicates.
    Priority: highest resolution (width*height), then the smallest alphabetical name.
    """
    def sort_key(p: Path):
        try:
            with Image.open(p) as img:
                w, h = img.size
            return (-w * h, p.name)
        except Exception:
            return (0, p.name)

    return sorted(paths, key=sort_key)[0]


def main():
    script_dir = Path(__file__).resolve().parent
    default_data_dir = script_dir.parents[1] / "TrainImages"
    default_report = script_dir / "duplicates_report.json"

    ap = argparse.ArgumentParser(
        description="Duplicate (MD5) and near-duplicate (pHash) image detection in Dataset."
    )
    ap.add_argument("--data-dir", default=str(default_data_dir),
                    help="Dataset directory (default: TrainImages)")
    ap.add_argument("--output-report", default=str(default_report),
                    help="JSON report output Path")
    ap.add_argument("--threshold", type=int, default=3,
                    help="Hamming distance max for near-duplicate (default: 3. lower -> more strict, higher -> more loose)")
    ap.add_argument("--cross-class", action="store_true",
                    help="Activate cross-class near-duplicate detection (default: Inactive)")
    ap.add_argument("--workers", type=int, default=None,
                    help="Number of paralel worker thread (default: CPU cores * 4)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Directory '{data_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    num_workers = args.workers or min(32, ((__import__("os").cpu_count() or 1) * 4))

    print(f"Scanning images in: {data_dir.resolve()}")
    all_images = collect_images(data_dir)
    
    # Exclude 'duplicates' folder from checking
    all_images = [p for p in all_images if "duplicates" not in p.parts]
    
    total = len(all_images)
    print(f"Found {total} images to process.\n")

    if total == 0:
        print("No images found.")
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Exact Duplicate via MD5
    # -----------------------------------------------------------------------
    print("=" * 55)
    print("Running Step 1: Exact Duplicate Detection (MD5 Hash)")
    print("=" * 55)

    md5_map: dict[str, list[Path]] = {}

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(compute_md5, p): p for p in all_images}
        for fut in tqdm(as_completed(futures), total=total, desc="Hashing MD5", unit="img"):
            p = futures[fut]
            try:
                digest = fut.result()
                md5_map.setdefault(digest, []).append(p)
            except Exception as e:
                print(f"  [WARN] Failed to hash {p.name}: {e}", file=sys.stderr)

    exact_groups: list[dict] = []
    exact_dup_paths: set[Path] = set()   # all files that are not "keeper"

    for digest, paths in md5_map.items():
        if len(paths) < 2:
            continue
        keeper = choose_keeper(paths)
        to_move = [p for p in paths if p != keeper]
        exact_groups.append({
            "type": "exact",
            "md5": digest,
            "keeper": str(keeper),
            "to_move": [str(p) for p in to_move],
        })
        exact_dup_paths.update(to_move)

    print(f"  Exact duplicate group found : {len(exact_groups)}")
    print(f"  Files to move               : {len(exact_dup_paths)}\n")

    # -----------------------------------------------------------------------
    # Near-Duplicate via pHash
    # -----------------------------------------------------------------------
    print("=" * 55)
    print(f"Runnig Step 2: Near-Duplicate Detection (pHash, threshold={args.threshold})")
    print("=" * 55)

    # Only process files that are not exact duplicates (not to_move).
    non_exact = [p for p in all_images if p not in exact_dup_paths]
    print(f"  Calculating pHash for {len(non_exact)} images (non-exact-duplicate)...")

    phash_map: dict[Path, np.ndarray] = {}

    def _safe_phash(p):
        return p, compute_phash(p)

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures2 = {ex.submit(_safe_phash, p): p for p in non_exact}
        for fut in tqdm(as_completed(futures2), total=len(non_exact), desc="Hashing pHash", unit="img"):
            try:
                p, ph = fut.result()
                phash_map[p] = ph
            except Exception as e:
                print(f"  [WARN] Failed to phash {futures2[fut].name}: {e}", file=sys.stderr)

    # Group by class (parent directory directly under data_dir)
    class_map: dict[str, list[Path]] = {}
    for p in phash_map:
        # Kelas = first subfolder after data_dir
        try:
            rel = p.relative_to(data_dir)
            kelas = rel.parts[0] if len(rel.parts) > 1 else "__root__"
        except ValueError:
            kelas = "__root__"
        class_map.setdefault(kelas, []).append(p)

    print(f"  Comparing pHash pairs in {'all classes' if args.cross_class else 'the same class'}...")

    near_pairs: list[dict] = []
    near_dup_paths: set[Path] = set()

    def compare_within_group_vectorized(group: list[Path]) -> list[dict]:
        """
        Calculate the Hamming distance for all pairs at once using numpy.
         Much faster than Python's O(n²) loop for large groups.
        """
        if len(group) < 2:
            return []

        # Arrange all pHashes into a uint8 matrix.: shape (n, hash_bits)
        hash_matrix = np.array([phash_map[p].astype(np.uint8) for p in group])  # (n, 64)
        n = len(group)

        pairs = []
        # Process in chunks to avoid excessive memory allocation
        # for n=12000 per class: matrix 12000×12000 = 144M uint8 = ~144MB (still OK)
        # XOR: (n, 1, 64) XOR (1, n, 64) -> (n, n, 64), then sum axis=-1
        # For memory efficiency, we will continue with a single operation if n < 15000
        CHUNK = 2000  # process 2000 linee at once
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            # (chunk, n, 64) XOR → sum → Hamming distance matrix (chunk, n)
            chunk_hashes = hash_matrix[start:end]                         # (chunk, 64)
            dist_chunk = np.sum(
                chunk_hashes[:, np.newaxis, :] != hash_matrix[np.newaxis, :, :],
                axis=-1
            )  # shape: (chunk, n)

            # Only take pairs where i < j (upper triangle according to offset start)
            for local_i, global_i in enumerate(range(start, end)):
                # Only compare with j > global_i
                row = dist_chunk[local_i, global_i + 1:]
                match_indices = np.where(row <= args.threshold)[0]
                for rel_j in match_indices:
                    global_j = global_i + 1 + rel_j
                    dist = int(dist_chunk[local_i, global_j])
                    keeper = choose_keeper([group[global_i], group[global_j]])
                    to_move = group[global_j] if keeper == group[global_i] else group[global_i]
                    pairs.append({
                        "type": "near",
                        "hamming_distance": dist,
                        "keeper": str(keeper),
                        "to_move": str(to_move),
                        "file_a": str(group[global_i]),
                        "file_b": str(group[global_j]),
                    })
        return pairs

    if args.cross_class:
        all_paths_list = list(phash_map.keys())
        near_pairs = compare_within_group_vectorized(all_paths_list)
    else:
        for kelas, paths_in_class in class_map.items():
            pairs = compare_within_group_vectorized(paths_in_class)
            near_pairs.extend(pairs)

    for pair in near_pairs:
        near_dup_paths.add(Path(pair["to_move"]))

    # Remove overlaps with exact_dup_paths (do not count twice)
    near_dup_unique = near_dup_paths - exact_dup_paths

    print(f"  Pasangan near-duplicate ditemukan : {len(near_pairs)}")
    print(f"  File unik yang akan dipindah       : {len(near_dup_unique)}\n")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_to_move = len(exact_dup_paths) + len(near_dup_unique)
    print("=" * 55)
    print("Duplicates Detection Summary")
    print("-" * 55)
    print(f"Total images checked     : {total}")
    print(f"Exact duplicate (move) : {len(exact_dup_paths)}")
    print(f"Near-duplicate (move)  : {len(near_dup_unique)}")
    print(f"Total images to move     : {total_to_move}")
    print(f"Total images remaining   : {total - total_to_move}")
    print("=" * 55)

    # -----------------------------------------------------------------------
    # Save JSON report
    # -----------------------------------------------------------------------
    report = {
        "config": {
            "data_dir": str(data_dir.resolve()),
            "threshold": args.threshold,
            "cross_class": args.cross_class,
        },
        "summary": {
            "total_checked": total,
            "exact_duplicate_files_to_move": len(exact_dup_paths),
            "near_duplicate_files_to_move": len(near_dup_unique),
            "total_to_move": total_to_move,
            "total_remaining": total - total_to_move,
        },
        "exact_duplicate_groups": exact_groups,
        "near_duplicate_pairs": near_pairs,
    }

    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print(f"\nReport saved to: {report_path.resolve()}")
    print("Run quarantine_duplicates.py to move duplicate files.")


if __name__ == "__main__":
    main()
