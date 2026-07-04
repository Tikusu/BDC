# Image Augmentation Script

Batch-processes images (rotate, blur, brightness/contrast/saturation) for ML training data.

## Requirements
```
pip install -r preprocess/requirements.txt
```

## Usage
```
python augment_images.py <files-or-folders> -o <output_dir> [options]
```

## Options
| Flag | Description |
|---|---|
| `-o, --output` | Output folder (default: `preprocess/image_output/augmented_images`) |
| `--rotate DEG` | Rotation angle in degrees |
| `--blur RADIUS` | Gaussian blur radius |
| `--brightness F` | Brightness factor (1.0 = unchanged) |
| `--contrast F` | Contrast factor (1.0 = unchanged) |
| `--saturation F` | Saturation factor (1.0 = unchanged) |
| `--random` | Randomize each param within the given value as a range instead of applying it as fixed |
| `--copies N` | Number of augmented copies per source image (default: 1) |
| `--limit N` | Max number of source files to process |
| `--max-dim N` | Cap the longest side of output images to N pixels |

## Examples
Single image, fixed transforms:
```
python3 preprocess/augment_images/augment_images.py photo.jpg --rotate 15 --blur 2
```

Folder, randomized augmentation, 3 variants each, capped resolution:
```
python3 preprocess/augment_images/augment_images.py /path/to/folder --brightness 0.2 --contrast 0.3 --saturation 0.3 --random --copies 3 --max-dim 1024
```

Process only the first 50 files in a folder:
```
python3 preprocess/augment_images/augment_images.py /path/to/folder --rotate 10 --limit 50
```

## Notes
- Supported input formats: jpg, jpeg, png, bmp, webp
- Outputs keep the original file extension
- `--random` ranges: `--rotate 15` → random angle between -15 and 15; `--brightness 0.2` → random factor between 0.8 and 1.2