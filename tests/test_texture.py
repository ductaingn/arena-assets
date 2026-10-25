import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from arena_assets.impl import ANNOTATION_NAME
from arena_assets.utils.texture import MAX_TEXTURE_PX, limit_image, limit_images

BOMB_SIDE = 13400


def _chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def _sixteen_bit_rgb(path: Path, size: tuple[int, int]) -> None:
    row = b"\x00" + b"\x12\x34" * 3 * size[0]
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", size[0], size[1], 16, 2, 0, 0, 0)) + _chunk(b"IDAT", zlib.compress(row * size[1])) + _chunk(b"IEND", b""))


def test_an_oversized_image_fits_the_longer_side_in_its_own_format(tmp_path):
    jpeg = tmp_path / "wall.jpg"
    Image.new("RGB", (4096, 1024), (200, 100, 50)).save(jpeg, format="JPEG")
    assert limit_image(jpeg, 2048) == ((4096, 1024), (2048, 512))
    with Image.open(jpeg) as image:
        assert (image.format, image.size) == ("JPEG", (2048, 512))


def test_an_image_past_the_decompression_bomb_limit_is_fit_too(tmp_path):
    png = tmp_path / "huge.png"
    Image.new("1", (BOMB_SIDE, BOMB_SIDE)).save(png)
    assert limit_image(png, MAX_TEXTURE_PX) == ((BOMB_SIDE, BOMB_SIDE), (MAX_TEXTURE_PX, MAX_TEXTURE_PX))


def test_a_small_8_bit_image_is_left_as_it_is(tmp_path):
    png = tmp_path / "small.png"
    Image.new("RGBA", (512, 512), (1, 2, 3, 4)).save(png)
    before = png.read_bytes()
    assert limit_image(png, MAX_TEXTURE_PX) is None
    assert png.read_bytes() == before


@pytest.mark.parametrize("deep", ["rgb", "gray"])
def test_a_16_bit_png_is_stored_with_8_bits_per_channel(tmp_path, deep):
    png = tmp_path / "deep.png"
    if deep == "rgb":
        _sixteen_bit_rgb(png, (8, 4))
    else:
        Image.fromarray(np.full((4, 8), 0x1234, dtype=np.uint16)).save(png)
    assert png.read_bytes()[24] == 16
    assert limit_image(png, MAX_TEXTURE_PX) == ((8, 4), (8, 4))
    assert png.read_bytes()[24] == 8
    with Image.open(png) as image:
        assert image.getpixel((0, 0)) == ((0x12, 0x12, 0x12) if deep == "rgb" else 0x12)


def test_limit_images_walks_a_directory_and_leaves_other_and_unreadable_files(tmp_path):
    (tmp_path / "maps").mkdir()
    Image.new("RGB", (3000, 3000)).save(tmp_path / "maps" / "color.png")
    Image.new("L", (100, 100)).save(tmp_path / "maps" / "mask.png")
    (tmp_path / "maps" / "broken.jpg").write_bytes(b"not a jpeg")
    (tmp_path / "plank.mdl").write_text("mdl 1.6;\n")
    assert limit_images(tmp_path, 1024) == {tmp_path / "maps" / "color.png": ((3000, 3000), (1024, 1024))}
    assert (tmp_path / "maps" / "broken.jpg").read_bytes() == b"not a jpeg"


@pytest.mark.parametrize(("option", "side"), [(None, 2048), ("512", 512), ("0", 4096)])
def test_a_material_build_fits_its_maps_unless_the_option_is_0(tmp_path, option, side):
    from arena_assets.impl.build.MaterialDatabaseBuilder import MaterialDatabaseBuilder

    source = tmp_path / "input" / "Material" / "Plank"
    source.mkdir(parents=True)
    Image.new("RGB", (4096, 4096), (90, 60, 30)).save(source / "Plank_BaseColor.png")
    (source / ANNOTATION_NAME).write_text(yaml.safe_dump({"desc": "plank"}))
    builder = MaterialDatabaseBuilder(tmp_path / "input", tmp_path / "output")
    if option is not None:
        builder.enable("max-texture", option)
    dest = tmp_path / "output" / "Material" / "Plank"
    dest.mkdir(parents=True)
    assert builder.process_entity(builder.read_annotation_file(source), dest) is not None
    with Image.open(dest / "Plank_BaseColor.png") as image:
        assert image.size == (side, side)
    with Image.open(source / "Plank_BaseColor.png") as image:
        assert image.size == (4096, 4096)
