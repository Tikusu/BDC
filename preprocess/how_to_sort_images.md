# Image Category Anomaly Sorter

## Setup
```
pip install -r requirements.txt
```

## How it works
`build_mapping.py` already generated `category_mapping.json`, mapping each of the 1000 ImageNet classes EfficientNet can predict to one of: `Recyclable Objects`, `Electronic Objects`, `Organic Objects`, or `null` (unrelated/unmapped class). Edit the `RULES` dict in `build_mapping.py` and re-run it to adjust keyword coverage, add categories, or reuse for a totally different category set.

## Usage
Process one source category folder at a time:
```
python3 sort_images.py \
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
- `--model efficientnet-b4` — swap model size (b0 default, faster; b4+ more accurate, slower on CPU)

## Reusing for other categories
1. Add new category + keyword list to `RULES` in `build_mapping.py`, re-run it.
2. Call `sort_images.py --category "<NewCategory>"` pointing at the matching folder.

## Other Info
- nope