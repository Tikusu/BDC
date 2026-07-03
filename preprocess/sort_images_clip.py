import argparse
import shutil
from pathlib import Path
from classifier_clip import ClipClassifier

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--prompts-file", default="prompts.json")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--copy", action="store_true")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_root = Path(args.output_dir)
    non_anomaly_dir = out_root / "non_anomaly" / args.category
    anomaly_dir = out_root / "anomaly" / args.category
    non_anomaly_dir.mkdir(parents=True, exist_ok=True)
    anomaly_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in VALID_EXT)
    if args.limit:
        files = files[: args.limit]

    clf = ClipClassifier(model_name=args.model, pretrained=args.pretrained, prompts_file=args.prompts_file)
    transfer = shutil.copy2 if args.copy else shutil.move

    results = {"non_anomaly": 0, "anomaly": 0}
    for path in files:
        pred = clf.predict(path)
        is_match = pred["category"] == args.category and pred["confidence"] >= args.min_confidence
        dest_dir = non_anomaly_dir if is_match else anomaly_dir
        transfer(str(path), str(dest_dir / path.name))
        results["non_anomaly" if is_match else "anomaly"] += 1
        print(f"{path.name}: {pred['category']} ({pred['confidence']:.2f}) [{'OK' if is_match else 'ANOMALY'}]")

    print(f"\nDone. {results['non_anomaly']} non_anomaly, {results['anomaly']} anomaly, {len(files)} total processed.")


if __name__ == "__main__":
    main()