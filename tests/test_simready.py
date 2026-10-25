import io
import json
import urllib.parse
import zipfile
from pathlib import Path

import pytest
import yaml
from PIL import Image

from arena_assets.impl import ANNOTATION_NAME
from arena_assets.importers import ATTRIBUTION_FILE, IMPORTERS, read_attribution
from arena_assets.importers.providers import HFHub
from arena_assets.importers.simready import KEPT, SimReady, SimReadyDSX, description, kind
from http_host import Host

ITEM = "bin_waste_a01"
ROOT = f"simready_usd/sm_{ITEM}_01.usd"
SMALL_ITEM = "bandage_medical_a01"


def _png(size: tuple[int, int]) -> bytes:
    data = io.BytesIO()
    Image.new("RGB", size, (120, 120, 120)).save(data, format="PNG")
    return data.getvalue()


def _stage(edge_m: float) -> str:
    points = ", ".join(f"({x}, {y}, {z})" for x in (0, edge_m) for y in (0, edge_m) for z in (0, 0.3))
    return f"""#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Root"
{{
    def Mesh "Body" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3]
        point3f[] points = [{points}]
        rel material:binding = </Root/Looks/Plastic>
    }}

    def Scope "Looks"
    {{
        def Material "Plastic"
        {{
            token outputs:surface.connect = </Root/Looks/Plastic/Surface.outputs:surface>

            def Shader "Surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </Root/Looks/Plastic/Albedo.outputs:rgb>
                float inputs:roughness.connect = </Root/Looks/Plastic/Orm.outputs:g>
                token outputs:surface
            }}

            def Shader "Albedo"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @./Textures/t_plastic_alb.png@
                float3 outputs:rgb
            }}

            def Shader "Orm"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @./source_assets/t_plastic_orm.png@
                float outputs:g
            }}
        }}
    }}
}}
"""


def package(host: Host, item: str = ITEM, repo: str = SimReady.REPO, revision: str = SimReady.REVISION, license: str = "CC-BY-4.0", edge_m: float = 0.25) -> dict[str, bytes]:
    """Serve one SimReady package the way the Hugging Face tree and resolve endpoints do."""
    files = {
        "com.nvidia.simready.packaging.json": json.dumps({"format_version": "1.0", "license": license}).encode(),
        ".metadata/com.nvidia.simready.root_usds.json": json.dumps({"format_version": "1.0", "entries": [ROOT.replace(ITEM, item)]}).encode(),
        ".metadata/com.nvidia.simready.conformance.Package@1.0.0.json": json.dumps({"profile": "Package"}).encode(),
        ".metadata/LICENSE.txt": b"Attribution 4.0 International\n",
        ROOT.replace(ITEM, item): _stage(edge_m).encode(),
        "simready_usd/payloads/Physics/physics.usda": b'#usda 1.0\n\nover "Root"\n{\n    float physics:density = 900\n}\n',
        "simready_usd/Textures/t_plastic_alb.png": _png((4100, 4)),
        "simready_usd/source_assets/t_plastic_orm.png": _png((16, 16)),
        "simready_usd/source_assets/t_plastic_unused.png": _png((16, 16)),
        "simready_usd/.thumbs/256x256/preview.png": _png((4, 4)),
    }
    tree = f"/api/datasets/{repo}/tree/{revision}"
    host.pages[f"{tree}?recursive=false"] = json.dumps([{"type": "file", "path": ".gitattributes"}, {"type": "directory", "path": item}]).encode()
    host.pages[f"{tree}/{item}?recursive=true"] = json.dumps([{"type": "file", "path": f"{item}/{path}"} for path in files]).encode()
    for path, body in files.items():
        host.pages[f"/datasets/{repo}/resolve/{revision}/{urllib.parse.quote(f'{item}/{path}')}"] = body
    return files


def importer(host: Host, tmp_path: Path) -> SimReady:
    return SimReady(cache=tmp_path / "cache", provider=HFHub(SimReady.REPO, revision=SimReady.REVISION, host=host.base))


def served(host: Host) -> set[str]:
    return {urllib.parse.unquote(path).split(f"{ITEM}/", 1)[-1] for path in host.served if "/resolve/" in path}


def test_registry_resolves_both_sources_to_their_pinned_repositories():
    assert IMPORTERS.load("object", "simready") is SimReady
    assert IMPORTERS.load("object", "simready_dsx") is SimReadyDSX
    assert (SimReady.REPO, SimReadyDSX.REPO) == ("nvidia/simready-assets", "nvidia/simready-dsx")
    assert IMPORTERS.meta("object", "simready").listing_ttl_s is None


@pytest.mark.parametrize(
    ("item", "words", "category"), [("bed_hospital_standard_a01", "bed hospital standard", "bed"), ("IV_pole_a01", "IV pole", "iv"), ("appliance_handDryer_automatic_b01", "appliance hand dryer automatic", "appliance"), ("handTool_wrench_a03", "hand tool wrench", "hand_tool"), ("GB300", "GB300", "gb300")]
)
def test_description_and_kind_come_from_the_folder_name(item, words, category):
    assert (description(item), kind(item)) == (words, category)


def test_items_are_the_asset_folders(host, tmp_path):
    package(host)
    assert list(importer(host, tmp_path).items()) == [ITEM]


def test_fetch_takes_the_layers_and_only_the_files_the_stage_refers_to(host, tmp_path):
    package(host)
    dest = importer(host, tmp_path).fetch(ITEM)
    assert served(host) == {
        "com.nvidia.simready.packaging.json",
        ".metadata/com.nvidia.simready.root_usds.json",
        ".metadata/com.nvidia.simready.conformance.Package@1.0.0.json",
        ".metadata/LICENSE.txt",
        ROOT,
        "simready_usd/payloads/Physics/physics.usda",
        "simready_usd/Textures/t_plastic_alb.png",
        "simready_usd/source_assets/t_plastic_orm.png",
    }
    assert (dest / "simready_usd" / "source_assets" / "t_plastic_orm.png").is_file()


def test_fetch_refuses_a_package_under_another_license_before_its_textures(host, tmp_path):
    package(host, license="CC-BY-NC-4.0")
    with pytest.raises(ValueError, match="bin_waste_a01 is under CC-BY-NC-4.0, only CC-BY-4.0 SimReady assets are imported"):
        importer(host, tmp_path).fetch(ITEM)
    assert not any(path.endswith(".png") for path in served(host))


def test_fetch_of_an_unknown_folder_names_the_listing_command(host, tmp_path):
    package(host)
    host.pages[f"/api/datasets/{SimReady.REPO}/tree/{SimReady.REVISION}/nothing_a01?recursive=true"] = b"[]"
    with pytest.raises(KeyError, match="no SimReady asset 'nothing_a01'.*import object simready"):
        importer(host, tmp_path).fetch("nothing_a01")


def test_import_refuses_a_footprint_the_builder_would_rescale_and_drops_the_download(host, tmp_path):
    package(host, edge_m=6.0)
    imp = importer(host, tmp_path)
    with pytest.raises(ValueError, match=r"bin_waste_a01 is 6.00 x 6.00 m, the object builder rescales a footprint over 5 m"):
        imp.import_item(ITEM, tmp_path / "out")
    assert not imp.package(ITEM).exists()


def test_import_builds_the_object_and_keeps_the_simready_metadata_and_physics(blender, host, tmp_path):
    package(host)
    imp = importer(host, tmp_path)
    target = imp.import_item(ITEM, tmp_path / "out")
    assert target == tmp_path / "out" / "Common" / "Object" / ITEM
    assert {f"{ITEM}.usdz", f"{ITEM}.fbx", f"{ITEM}.sdf"} <= {path.name for path in target.iterdir()}
    annotation = yaml.safe_load((target / ANNOTATION_NAME).read_text())
    assert "origin::simready" in annotation["tags"]
    assert annotation["desc"] == "bin waste"
    kept = target / KEPT
    assert sorted(path.relative_to(kept).as_posix() for path in kept.rglob("*") if path.is_file()) == ["com.nvidia.simready.packaging.json", "metadata/LICENSE.txt", "metadata/com.nvidia.simready.conformance.Package@1.0.0.json", "metadata/com.nvidia.simready.root_usds.json", "payloads/Physics/physics.usda"]
    attribution = read_attribution(target / ATTRIBUTION_FILE)
    assert (attribution.license, attribution.authors) == ("CC-BY-4.0", "NVIDIA Corporation")
    assert attribution.url == f"https://huggingface.co/datasets/{SimReady.REPO}/tree/{SimReady.REVISION}/{ITEM}"
    with zipfile.ZipFile(target / f"{ITEM}.usdz") as usdz:
        sizes = [Image.open(io.BytesIO(usdz.read(name))).size for name in usdz.namelist() if name.endswith((".png", ".jpg"))]
    assert sizes
    assert max(max(size) for size in sizes) == 2048
    assert not imp.package(ITEM).exists()


@pytest.mark.network
def test_live_sources_list_their_assets(tmp_path):
    assets = list(SimReady(cache=tmp_path / "assets").items())
    assert len(assets) > 2500
    assert {"bed_hospital_standard_a01", "IV_pole_a01", ITEM} <= set(assets)
    assert list(SimReadyDSX(cache=tmp_path / "dsx").items()) == ["GB300"]


@pytest.mark.network
def test_live_package_fetches_its_license_root_and_textures(tmp_path):
    imp = SimReady(cache=tmp_path / "cache")
    dest = imp.fetch(SMALL_ITEM)
    assert json.loads((dest / "com.nvidia.simready.packaging.json").read_text())["license"] == "CC-BY-4.0"
    assert imp.root(SMALL_ITEM).is_file()
    assert any((dest / "simready_usd" / "Textures").iterdir())
