from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import attrs

from arena_assets.impl import ANNOTATION_NAME, Annotation, AssetType
from arena_assets.utils.logging import get_logger
from arena_assets.utils.texture import MAX_TEXTURE_PX, limit_images

from . import DatabaseBuilder, OptionRegistry

logger = get_logger("build.material")


@attrs.define
class MaterialAnnotation(Annotation):
    pass


class MaterialDatabaseBuilder(DatabaseBuilder[MaterialAnnotation]):
    _annotation_cls = MaterialAnnotation
    _DISCOVER_PATH = "Material"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        def store_db(annotation: MaterialAnnotation) -> None:
            self._db.store(AssetType.MATERIAL.value, annotation)

        self._pipeline.append(store_db)
        self._max_texture = MAX_TEXTURE_PX

    def process_entity(self, annotation, dest):
        try:
            shutil.copytree(
                annotation.path,
                dest,
                ignore=shutil.ignore_patterns(ANNOTATION_NAME),
                dirs_exist_ok=True,
            )
            if self._max_texture > 0:
                for path, (size, new_size) in limit_images(Path(dest), self._max_texture).items():
                    logger.info("Texture %s of %s: %dx%d to %dx%d", path.name, annotation.path, *size, *new_size)

            return annotation

        except Exception as e:
            logger.error(
                "Unexpected error processing entity %s: %s", annotation.path, e
            )
            return None

    enable = OptionRegistry()

    @enable.register("max-texture")
    def max_texture(self, pixels: str = str(MAX_TEXTURE_PX)):
        self._max_texture = int(pixels)
        logger.info("Fitting textures into %d px, 0 keeps them as they are.", self._max_texture)

    @enable.register("procthor")
    def procthor(self):
        logger.info("Enabling Procthor export option.")
        self._procthor = {}
        self._pipeline.append(
            lambda annotation: self._procthor.update(
                {annotation.path.replace(os.sep, "_"): {"todo": None}}
            )
        )

        def export():
            with open(self.output_path / "arena-materials-database.json", "w") as f:
                json.dump(self._procthor, f, indent=4)

        self._post.append(export)
