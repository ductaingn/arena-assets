from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_TEXTURE_PX = 2048
SUFFIXES = (".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tif", ".tiff")

Image.MAX_IMAGE_PIXELS = None


def _sixteen_bit_png(path: Path) -> bool:
    with open(path, "rb") as f:
        header = f.read(25)
    return header[:8] == b"\x89PNG\r\n\x1a\n" and header[24] == 16


def limit_image(path: Path, max_px: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Fit an image file into max_px with 8 bits per channel in place, its sizes before and after, None if left as it is."""
    try:
        opened = Image.open(path)
    except UnidentifiedImageError:
        return None
    with opened as image:
        image.load()
        size = image.size
        deep = image.mode.startswith("I") or _sixteen_bit_png(path)
        if max(size) <= max_px and not deep:
            return None
        image_format = image.format
        if image.mode.startswith("I"):
            image = Image.fromarray((np.asarray(image, dtype=np.uint32) >> 8).astype(np.uint8))
        if max(size) > max_px:
            image.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
    image.save(path, format=image_format, **({"quality": 95} if image_format == "JPEG" else {}))
    return size, image.size


def limit_images(directory: Path, max_px: int) -> dict[Path, tuple[tuple[int, int], tuple[int, int]]]:
    """Fit every image file under directory into max_px, the sizes before and after of the ones that changed."""
    changed = {}
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() in SUFFIXES and path.is_file():
            sizes = limit_image(path, max_px)
            if sizes is not None:
                changed[path] = sizes
    return changed
