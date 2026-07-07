from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_INPUT  = _SCRIPT_DIR.parent.parent / "TrainImagesSplit"
_DEFAULT_OUTPUT = _SCRIPT_DIR.parent.parent / "TrainImagesBalanced"
_CONFIG_PATH = _SCRIPT_DIR / "config.json"

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def collect_images(class_dir: Path) -> list[Path]:
    return sorted(
        f for f in class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXT
    )

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False