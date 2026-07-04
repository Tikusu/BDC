import argparse, os, glob, random
from PIL import Image, ImageFilter, ImageEnhanceimport os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "..", "image_output", "augmented_images")
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def collect_inputs(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for ext in VALID_EXT:
                files += glob.glob(os.path.join(p, f"*{ext}"))
        elif os.path.isfile(p):
            files.append(p)
    return sorted(set(files))

def process_image(img, args):
    if args.rotate is not None:
        angle = random.uniform(-args.rotate, args.rotate) if args.random else args.rotate
        img = img.rotate(angle, expand=True)

    if args.blur:
        radius = random.uniform(0, args.blur) if args.random else args.blur
        img = img.filter(ImageFilter.GaussianBlur(radius))

    if args.brightness is not None:
        factor = random.uniform(1 - args.brightness, 1 + args.brightness) if args.random else args.brightness
        img = ImageEnhance.Brightness(img).enhance(factor)

    if args.contrast is not None:
        factor = random.uniform(1 - args.contrast, 1 + args.contrast) if args.random else args.contrast
        img = ImageEnhance.Contrast(img).enhance(factor)

    if args.saturation is not None:
        factor = random.uniform(1 - args.saturation, 1 + args.saturation) if args.random else args.saturation
        img = ImageEnhance.Color(img).enhance(factor)

    if args.max_dim and max(img.size) > args.max_dim:
        ratio = args.max_dim / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)))

    return img

def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="image files and/or folders")
    p.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    p.add_argument("--rotate", type=float, default=None, help="degrees; max angle if --random")
    p.add_argument("--blur", type=float, default=None, help="gaussian radius; max if --random")
    p.add_argument("--brightness", type=float, default=None, help="factor (1.0=none); +/- range if --random")
    p.add_argument("--contrast", type=float, default=None)
    p.add_argument("--saturation", type=float, default=None)
    p.add_argument("--random", action="store_true", help="randomize each param within given range")
    p.add_argument("--copies", type=int, default=1, help="augmented copies per source image")
    p.add_argument("--limit", type=int, default=None, help="max number of source files to process")
    p.add_argument("--max-dim", type=int, default=None, help="cap longest side (px) of output image")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    files = collect_inputs(args.inputs)
    if not files:
        print("No valid images found.")
        return
    if args.limit:
        files = files[:args.limit]

    for f in files:
        img = Image.open(f).convert("RGB")
        name, ext = os.path.splitext(os.path.basename(f))
        for i in range(args.copies):
            out_img = process_image(img.copy(), args)
            suffix = f"_aug{i+1}" if args.copies > 1 else "_aug"
            out_path = os.path.join(args.output, f"{name}{suffix}{ext}")
            out_img.save(out_path)
            print(f"Saved {out_path}")

if __name__ == "__main__":
    main()