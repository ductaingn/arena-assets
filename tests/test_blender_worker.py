import os

import numpy as np
import pytest
import yaml

from arena_assets.impl import ANNOTATION_NAME, AssetType
from arena_assets.impl.build import DatabaseBuilder, OverwriteMode
from arena_assets.utils import ModelConverter
from arena_assets.utils.geom import BoundingBox
from arena_assets.utils.ModelConverter import COMMAND, COMMAND_ENV, BlenderError, BlenderUnavailable, BlenderWorker


def _environment(**values: str | None):
    previous = {key: os.environ.get(key) for key in values}
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    return previous


def test_command_splits_the_configured_override():
    previous = _environment(**{COMMAND_ENV: "/opt/blender-venv/bin/python -m arena_assets_blender"})
    try:
        assert ModelConverter.command() == ["/opt/blender-venv/bin/python", "-m", "arena_assets_blender"]
    finally:
        _environment(**previous)


def test_missing_worker_names_the_package_and_the_override(tmp_path):
    previous = _environment(**{COMMAND_ENV: None, "PATH": str(tmp_path)})
    try:
        with pytest.raises(BlenderUnavailable) as error:
            ModelConverter.command()
    finally:
        _environment(**previous)
    assert "arena_assets_blender" in str(error.value)
    assert COMMAND in str(error.value)
    assert COMMAND_ENV in str(error.value)


def test_worker_environment_drops_the_parent_interpreter_paths():
    previous = _environment(PYTHONPATH="/opt/ros/lib/python3.12/site-packages", PYTHONHOME="/usr")
    try:
        environment = ModelConverter.environment()
    finally:
        _environment(**previous)
    assert "PYTHONPATH" not in environment and "PYTHONHOME" not in environment
    assert environment["PATH"] == os.environ["PATH"]


def test_configured_worker_that_does_not_exist_is_unavailable(tmp_path):
    previous = _environment(**{COMMAND_ENV: str(tmp_path / "arena-assets-blender")})
    ModelConverter.exts.cache_clear()
    try:
        with pytest.raises(BlenderUnavailable, match="cannot run the Blender worker"):
            ModelConverter.exts()
        with pytest.raises(BlenderUnavailable, match="cannot run the Blender worker"):
            BlenderWorker().start()
    finally:
        _environment(**previous)


def test_worker_lists_the_formats_the_builder_exports(blender):
    assert {"usdz", "sdf", "fbx", "obj", "dae", "glb"} <= set(ModelConverter.exts())


def test_run_rectifies_measures_and_saves_in_one_scene(blender, cube, tmp_path):
    worker = BlenderWorker()
    try:
        result = worker.run([("load", str(cube)), ("rectify",), ("bounding_box",), ("save", str(tmp_path / "cube.usdz")), ("save", str(tmp_path / "cube.sdf"))])
    finally:
        worker.close()
    assert result.values[0] is None
    assert np.asarray(result.values[1]).shape == (4, 4)
    box = BoundingBox(result.values[2]).round(4)
    assert (box.max_x - box.min_x, box.max_y - box.min_y, box.max_z - box.min_z) == pytest.approx((1.0, 1.0, 1.0))
    assert box.min_z == pytest.approx(0.0)
    assert (tmp_path / "cube.usdz").stat().st_size > 0
    assert (tmp_path / "cube.sdf" / "cube.sdf").read_text().count("<uri>cube.dae</uri>") == 2
    assert (tmp_path / "cube.sdf" / "cube.dae").stat().st_size > 0


def test_run_renders_both_previews(blender, cube, tmp_path):
    worker = BlenderWorker()
    try:
        worker.run(
            [
                ("load", str(cube)),
                ("rectify",),
                ("render_perspective", str(tmp_path / "thumb.png"), {"resolution": (64, 64), "theta": 0.5}),
                ("render_topdown", str(tmp_path / "topdown.png"), {"resolution": (64, 64)}),
            ]
        )
    finally:
        worker.close()
    for name in ("thumb.png", "topdown.png"):
        assert (tmp_path / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_failed_step_raises_and_the_worker_keeps_serving(blender, cube, tmp_path):
    worker = BlenderWorker()
    try:
        with pytest.raises(BlenderError, match="Unsupported file extension"):
            worker.run([("load", str(tmp_path / "missing.xyz"))])
        with pytest.raises(BlenderError, match="unknown operation"):
            worker.run([("quit_blender",)])
        assert worker.run([("load", str(cube)), ("bounding_box",)]).values[1] is not None
    finally:
        worker.close()


def _dataset(tmp_path, asset_type: str, name: str):
    source = tmp_path / "dataset" / asset_type / name
    source.mkdir(parents=True)
    (source / ANNOTATION_NAME).write_text(yaml.safe_dump({"desc": name, "tags": []}))
    return source


def test_human_build_renders_its_preview_through_the_worker(blender, cube, tmp_path):
    source = _dataset(tmp_path, "Human", "walker")
    (source / "meshes").mkdir()
    worker = BlenderWorker()
    try:
        worker.run([("load", str(cube)), ("save", str(source / "meshes" / "walker.dae"))])
    finally:
        worker.close()
    builder = DatabaseBuilder.Builder(AssetType.HUMAN)(input_path=tmp_path / "dataset", output_path=tmp_path / "out", overwrite=OverwriteMode.OVERWRITE)
    builder.enable("previews")
    builder.build()
    assert (tmp_path / "out" / "Human" / "walker" / "walker_thumb.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (tmp_path / "out" / "Human" / "walker" / ANNOTATION_NAME).is_file()


def test_annotations_only_build_starts_no_worker(tmp_path):
    _dataset(tmp_path, "Object", "crate")
    previous = _environment(**{COMMAND_ENV: str(tmp_path / "arena-assets-blender")})
    try:
        builder = DatabaseBuilder.Builder(AssetType.OBJECT)(input_path=tmp_path / "dataset", output_path=tmp_path / "out", overwrite=OverwriteMode.ANNOTATIONS_ONLY)
        builder.enable("formats", "usdz,sdf")
        builder.build()
    finally:
        _environment(**previous)
    assert yaml.safe_load((tmp_path / "out" / "Object" / "crate" / ANNOTATION_NAME).read_text())["path"] == "Object/crate"


def test_converting_build_without_a_worker_fails_before_any_entity(tmp_path):
    _dataset(tmp_path, "Object", "crate")
    previous = _environment(**{COMMAND_ENV: str(tmp_path / "arena-assets-blender")})
    try:
        builder = DatabaseBuilder.Builder(AssetType.OBJECT)(input_path=tmp_path / "dataset", output_path=tmp_path / "out", overwrite=OverwriteMode.OVERWRITE)
        with pytest.raises(BlenderUnavailable):
            builder.build()
    finally:
        _environment(**previous)
    assert not (tmp_path / "out" / "Object" / "crate").exists()
