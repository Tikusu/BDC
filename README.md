# BDC

## Preprocessing

Two scripts in [`preprocess/`](preprocess) to preprocess

### Augment images
Batch rotate/blur/adjust brightness-contrast-saturation to generate more training data.
```
python3 preprocess/augment_images.py <files-or-folders> -o <output_dir> [--rotate DEG] [--blur R] [--random] [--copies N]
```
Full guide: [preprocess/how_to_augment_images.md](preprocess/how_to_augment_images.md)

### Sort images
Classifies images with EfficientNet and flags ones that don't match their expected category (Recyclable / Electronic / Organic) as anomalies.
```
python3 preprocess/sort_images.py --input-dir <src> --category "<Category>" --output-dir <dest> --limit 500
```
Full guide: [preprocess/how_to_sort_images.md](preprocess/how_to_sort_images.md)