from __future__ import annotations

import dataclasses
import io

import numpy as np
from PIL import Image

JPEG_QUALITY = 90


@dataclasses.dataclass
class TextureLimit:
    name: str
    width: int
    height: int
    new_width: int
    new_height: int


def limited(data: bytes, max_px: int) -> tuple[bytes, str, tuple[int, int], tuple[int, int]]:
    """An image fit into max_px on its longer side with 8 bits per channel, as JPEG unless it uses its alpha and as PNG then, with its format and its size before and after."""
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        size = image.size
        if image.mode.startswith("I"):
            image = Image.fromarray((np.asarray(image, dtype=np.uint32) >> 8).astype(np.uint8))
        elif image.mode not in ("RGB", "RGBA", "L", "LA"):
            image = image.convert("RGBA")
        if image.mode in ("RGBA", "LA") and image.getchannel("A").getextrema()[0] == 255:
            image = image.convert(image.mode[:-1])
        if max(size) > max_px:
            image.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
        image_format = "PNG" if image.mode in ("RGBA", "LA") else "JPEG"
        out = io.BytesIO()
        image.save(out, format=image_format, **({"quality": JPEG_QUALITY} if image_format == "JPEG" else {}))
        return out.getvalue(), image_format, size, image.size
