import argparse
import shutil
from pathlib import Path
from classifier import Classifier

# nanti tambah laeng, tpi harusnya ini sih
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="folder of images for one category")
    ap.add_argument("--category", required=True, help="expected category name, must match a value in category_mapping.json")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mapping-file", default="category_mapping.json")
    ap.add_argument("--model", default="efficientnet-b0")
    ap.add_argument("--limit", type=int, default=None, help="max number of images to process")
    ap.add_argument("--min-confidence", type=float, default=0.0, help="below this, force anomaly regardless of category match")
    ap.add_argument("--copy", action="store_true", help="copy instead of move")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_root = Path(args.output_dir)
    
    # untuk sementara bgini dp direktori
    non_anomaly_dir = out_root / "non_anomaly" / args.category
    anomaly_dir = out_root / "anomaly" / args.category
    non_anomaly_dir.mkdir(parents=True, exist_ok=True)
    anomaly_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in VALID_EXT)
    if args.limit:
        files = files[: args.limit]

    clf = Classifier(model_name=args.model, mapping_path=args.mapping_file)
    transfer = shutil.copy2 if args.copy else shutil.move

    results = {"non_anomaly": 0, "anomaly": 0}
    for path in files:
        pred = clf.predict(path)
        is_match = pred["category"] == args.category and pred["confidence"] >= args.min_confidence
        dest_dir = non_anomaly_dir if is_match else anomaly_dir
        transfer(str(path), str(dest_dir / path.name))
        results["non_anomaly" if is_match else "anomaly"] += 1
        print(f"{path.name}: {pred['class_name']} -> {pred['category']} ({pred['confidence']:.2f}) [{'OK' if is_match else 'ANOMALY'}]")

    print(f"\nDone. {results['non_anomaly']} non_anomaly, {results['anomaly']} anomaly, {len(files)} total processed. :wilted_rose:")


if __name__ == "__main__":
    main()