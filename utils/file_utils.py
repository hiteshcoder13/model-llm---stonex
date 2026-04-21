# utils/file_utils.py
from pathlib import Path
from config.settings import IMAGE_EXTS


def get_all_images(parent_dir: str) -> list[tuple[Path, str]]:
    """
    Return list of (image_path, family_name) for all images under parent_dir.
    family_name = immediate subfolder name (stone family).
    """
    parent = Path(parent_dir)
    results = []
    for subfolder in sorted(parent.iterdir()):
        if subfolder.is_dir():
            for img in subfolder.iterdir():
                if img.suffix.lower() in IMAGE_EXTS:
                    results.append((img, subfolder.name))
    return results


def load_image_pil(path: str):
    """Load image as PIL Image (RGB), return None on failure."""
    try:
        from PIL import Image
        return Image.open(path).convert("RGB")
    except Exception:
        return None