# Image Sorter

## Setup
```
pip install -r requirements.txt
```

## How it works
`classifier_clip.py` uses CLIP (open_clip) for zero-shot classification — no training or class-mapping file needed. Each category in `prompts.json` has a list of text prompts describing it; the image is matched against whichever category's prompts it's most similar to. Edit `prompts.json` to add/adjust prompts per category or add new categories.

## Usage
Process one source category folder at a time:
```
python3 sort_images_clip.py \
  --input-dir /path/to/RecyclableObjects \
  --category "Recyclable Objects" \
  --output-dir /path/to/sorted_result \
  --limit 500
```
Repeat per category (`Electronic Objects`, `Organic Objects`) pointing `--input-dir` at each source folder. Results land in:
```
sorted_result/
  non_anomaly/
    Recyclable Objects/
    Electronic Objects/
    Organic Objects/
  anomaly/
    Recyclable Objects/
    Electronic Objects/
    Organic Objects/
```
Each `anomaly/<category>/` holds images from that source folder whose predicted class didn't map to the expected category.

## Flags
- `--limit N` — cap images processed per run (safety net)
- `--copy` — copy instead of move (default is move; use `--copy` while testing)
- `--min-confidence 0.X` — additionally flag low-confidence predictions as anomaly even if the category matches
- `--prompts-file` — path to prompts JSON (default: `prompts.json`)
- `--model` — CLIP model architecture (default: `ViT-B-32`)
- `--pretrained` — CLIP pretrained weights tag (default: `laion2b_s34b_b79k`)

## Reusing for other categories
1. Add a new category key + prompt list to `prompts.json`.
2. Call `sort_images_clip.py --category "<NewCategory>"` pointing at the matching folder.