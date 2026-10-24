import io
import json
import urllib.parse
import zipfile
from pathlib import Path

import pytest
import yaml

from arena_assets.impl import ANNOTATION_NAME, DATABASE_NAME
from arena_assets.importers import ATTRIBUTION_FILE, IMPORTERS, read_attribution
from arena_assets.importers.ambientcg import AmbientCG
from arena_assets.importers.material import Maps, MaterialImporter, mdl, stage
from arena_assets.importers.polyhaven import PolyHaven
from arena_assets.importers.providers import AmbientCG as AmbientCGHost
from arena_assets.importers.providers import PolyHaven as PolyHavenHost
from arena_assets.semantic import CURRENT_ASA
from arena_assets.utils.Database import Database
from http_host import Host

PLANKS = "worn_planks"
PLANKS_INFO = {"name": "Worn Planks", "description": "Worn oak floor planks with scratches.", "categories": ["wood", "floor", "indoor"], "authors": {"Rob Tuytel": "All", "Dario Barresi": "Processing"}}
METAL = "Metal032"
METAL_RECORD = {"assetId": METAL, "dataType": "Material", "displayName": "Metal 032", "displayCategory": "Metal", "description": "", "tags": ["32", "grey", "metal", "smooth"], "shortLink": f"https://ambientcg.com/a/{METAL}"}
LIVE_PLANKS = "aerial_asphalt_01"


def files(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def polyhaven(host: Host, tmp_path: Path, maps: tuple[str, ...] = ("Diffuse", "nor_gl", "arm"), sizes: tuple[str, ...] = ("1k", "2k")) -> PolyHaven:
    listing = {key: {size: {"jpg": {"url": f"{host.base}/dl/{size}/{PLANKS}_{key}.jpg"}} for size in sizes} for key in maps}
    host.pages.update({f"/dl/{size}/{PLANKS}_{key}.jpg": f"{key} {size}".encode() for key in maps for size in sizes})
    host.pages[f"/files/{PLANKS}"] = json.dumps({**listing, "blend": {"2k": {"blend": {"url": f"{host.base}/dl/{PLANKS}.blend"}}}}).encode()
    host.pages[f"/info/{PLANKS}"] = json.dumps(PLANKS_INFO).encode()
    host.pages["/assets?t=textures"] = json.dumps({PLANKS: PLANKS_INFO, "brick_wall": {}}).encode()
    return PolyHaven(cache=tmp_path / "cache", provider=PolyHavenHost(api=host.base, files=host.base))


def ambientcg(host: Host, tmp_path: Path, maps: tuple[str, ...] = ("Color", "NormalGL", "NormalDX", "Roughness", "Metalness", "Displacement"), attributes: tuple[str, ...] = ("1K-JPG", "2K-JPG")) -> AmbientCG:
    downloads = [{"attribute": attribute, "fileName": f"{METAL}_{attribute}.zip"} for attribute in attributes]
    record = {**METAL_RECORD, "downloadFolders": {"default": {"downloadFiletypeCategories": {"zip": {"downloads": downloads}}}}}
    for attribute in attributes:
        body = io.BytesIO()
        with zipfile.ZipFile(body, "w") as archive:
            archive.writestr(f"{METAL}_{attribute}.usdc", "usd")
            for name in maps:
                archive.writestr(f"{METAL}_{attribute}_{name}.jpg", f"{name} {attribute}")
        host.pages[f"/get?file={METAL}_{attribute}.zip"] = body.getvalue()
    for asked in (METAL, METAL.lower()):
        host.pages[f"/api/v2/full_json?{urllib.parse.urlencode({'id': asked, 'include': 'downloadData,tagData,displayData'})}"] = json.dumps({"foundAssets": [record]}).encode()
    return AmbientCG(cache=tmp_path / "cache", provider=AmbientCGHost(host=host.base))


def test_registry_resolves_both_sources_with_a_default_provider():
    assert IMPORTERS.load("material", "polyhaven") is PolyHaven
    assert IMPORTERS.load("material", "ambientcg") is AmbientCG
    assert issubclass(PolyHaven, MaterialImporter) and issubclass(AmbientCG, MaterialImporter)
    assert PolyHaven.default_provider().url("info/x y") == "https://api.polyhaven.com/info/x%20y"
    assert PolyHaven.default_provider().url("https://dl.polyhaven.org/file/a b.jpg") == "https://dl.polyhaven.org/file/a b.jpg"
    assert AmbientCG.default_provider().url("Metal032_2K-JPG.zip") == "https://ambientcg.com/get?file=Metal032_2K-JPG.zip"


def test_mdl_binds_a_packed_set_with_the_normal_map_unflipped():
    text = mdl("Oak", {"color": "./Oak/Oak_BaseColor.jpg", "orm": "./Oak/Oak_ORM.jpg", "normal": "./Oak/Oak_N.jpg"})
    assert text.splitlines() == [
        "mdl 1.4;",
        "",
        "import ::OmniPBR::OmniPBR;",
        "import ::tex::gamma_mode;",
        "",
        "export material Oak(*)",
        " = ::OmniPBR::OmniPBR(",
        '    diffuse_texture: texture_2d("./Oak/Oak_BaseColor.jpg", ::tex::gamma_srgb),',
        "    reflection_roughness_texture_influence: 1.f,",
        "    metallic_texture_influence: 1.f,",
        "    enable_ORM_texture: true,",
        '    ORM_texture: texture_2d("./Oak/Oak_ORM.jpg", ::tex::gamma_linear),',
        '    normalmap_texture: texture_2d("./Oak/Oak_N.jpg", ::tex::gamma_linear),',
        "    flip_tangent_v: false);",
    ]


def test_mdl_binds_separate_maps_and_leaves_absent_ones_at_the_constants():
    text = mdl("Oak", {"color": "./Oak/Oak_BaseColor.png", "roughness": "./Oak/Oak_R.png", "occlusion": "./Oak/Oak_AO.png"})
    assert text.splitlines()[7:] == [
        '    diffuse_texture: texture_2d("./Oak/Oak_BaseColor.png", ::tex::gamma_srgb),',
        "    reflection_roughness_texture_influence: 1.f,",
        '    reflectionroughness_texture: texture_2d("./Oak/Oak_R.png", ::tex::gamma_linear),',
        '    ao_texture: texture_2d("./Oak/Oak_AO.png", ::tex::gamma_linear));',
    ]
    assert mdl("Oak", {"color": "./Oak/Oak_BaseColor.png"}).splitlines()[7:] == ['    diffuse_texture: texture_2d("./Oak/Oak_BaseColor.png", ::tex::gamma_srgb));']


def test_stage_lays_the_maps_out_under_the_material_name(tmp_path):
    color, normal = tmp_path / "c.JPG", tmp_path / "n.png"
    color.write_bytes(b"color")
    normal.write_bytes(b"normal")
    source = stage("Oak_Floor", Maps(color=color, normal=normal), "oak floor", ["origin::test", "wood", "floor"], tmp_path / "dataset")
    assert source == tmp_path / "dataset" / "Common" / "Material" / "Oak_Floor"
    assert files(source) == ["Oak_Floor.mdl", "Oak_Floor/Oak_Floor_BaseColor.jpg", "Oak_Floor/Oak_Floor_N.png", ANNOTATION_NAME]
    assert (source / "Oak_Floor" / "Oak_Floor_BaseColor.jpg").read_bytes() == b"color"
    assert '"./Oak_Floor/Oak_Floor_N.png"' in (source / "Oak_Floor.mdl").read_text()
    raw = yaml.safe_load((source / ANNOTATION_NAME).read_text())
    assert (raw["name"], raw["desc"], raw["asa"]) == ("Oak_Floor", "oak floor", CURRENT_ASA)
    assert raw["tags"] == ["domain::common", "kind::wood", "origin::test", "floor"]


def test_stage_takes_the_kind_from_a_plural_tag_before_a_later_kind(tmp_path):
    color = tmp_path / "c.jpg"
    color.write_bytes(b"color")
    source = stage("Checker", Maps(color=color), "", ["origin::test", "tiles", "stone"], tmp_path / "dataset")
    assert yaml.safe_load((source / ANNOTATION_NAME).read_text())["tags"] == ["domain::common", "kind::tile", "origin::test", "stone"]


@pytest.mark.parametrize("name", ["2x4_planks", "oak-floor", "oak floor", ""])
def test_stage_refuses_a_name_that_is_no_mdl_identifier(tmp_path, name):
    color = tmp_path / "c.jpg"
    color.write_bytes(b"color")
    with pytest.raises(ValueError, match="is no MDL identifier"):
        stage(name, Maps(color=color), "", [], tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()


def test_polyhaven_lists_the_textures(host, tmp_path):
    assert list(polyhaven(host, tmp_path).items()) == ["brick_wall", PLANKS]


def test_polyhaven_imports_the_2k_packed_set(host, tmp_path):
    out = tmp_path / "out"
    target = polyhaven(host, tmp_path).import_item(PLANKS, out)
    assert target == out / "Common" / "Material" / PLANKS
    assert files(target) == [ATTRIBUTION_FILE, ANNOTATION_NAME, f"{PLANKS}.mdl", f"{PLANKS}/{PLANKS}_BaseColor.jpg", f"{PLANKS}/{PLANKS}_N.jpg", f"{PLANKS}/{PLANKS}_ORM.jpg"]
    assert (target / PLANKS / f"{PLANKS}_BaseColor.jpg").read_bytes() == b"Diffuse 2k"
    assert (target / PLANKS / f"{PLANKS}_N.jpg").read_bytes() == b"nor_gl 2k"
    assert (target / PLANKS / f"{PLANKS}_ORM.jpg").read_bytes() == b"arm 2k"
    text = (target / f"{PLANKS}.mdl").read_text()
    assert f"export material {PLANKS}(*)" in text
    assert f'ORM_texture: texture_2d("./{PLANKS}/{PLANKS}_ORM.jpg", ::tex::gamma_linear)' in text
    raw = yaml.safe_load((target / ANNOTATION_NAME).read_text())
    assert (raw["name"], raw["path"], raw["desc"]) == (PLANKS, f"Material/{PLANKS}", "Worn oak floor planks with scratches.")
    assert raw["tags"] == ["domain::common", "kind::wood", "origin::polyhaven", "floor", "indoor"]
    attribution = read_attribution(target / ATTRIBUTION_FILE)
    assert (attribution.source, attribution.license, attribution.authors, attribution.url) == ("polyhaven", "CC0-1.0", "Rob Tuytel, Dario Barresi", f"https://polyhaven.com/a/{PLANKS}")
    assert Database(out / "Common" / DATABASE_NAME).list_all("material")["ids"] == [f"Material/{PLANKS}"]
    assert not (tmp_path / "cache" / "stage" / PLANKS).exists()
    assert sorted(path for path in host.served if path.startswith("/dl/")) == [f"/dl/2k/{PLANKS}_Diffuse.jpg", f"/dl/2k/{PLANKS}_arm.jpg", f"/dl/2k/{PLANKS}_nor_gl.jpg"]


def test_polyhaven_reimport_downloads_nothing_and_replaces_the_item(host, tmp_path):
    importer = polyhaven(host, tmp_path)
    out = tmp_path / "out"
    stale = importer.import_item(PLANKS, out) / "stale.txt"
    stale.write_text("x")
    served = len(host.served)
    importer.import_item(PLANKS, out)
    assert len(host.served) == served
    assert not stale.exists()


def test_polyhaven_without_a_packed_map_binds_the_separate_ones(host, tmp_path):
    target = polyhaven(host, tmp_path, maps=("Diffuse", "nor_gl", "Rough", "AO", "Displacement")).import_item(PLANKS, tmp_path / "out")
    assert files(target / PLANKS) == [f"{PLANKS}_AO.jpg", f"{PLANKS}_BaseColor.jpg", f"{PLANKS}_N.jpg", f"{PLANKS}_R.jpg"]
    text = (target / f"{PLANKS}.mdl").read_text()
    assert "enable_ORM_texture" not in text and "metallic_texture" not in text
    assert f'reflectionroughness_texture: texture_2d("./{PLANKS}/{PLANKS}_R.jpg", ::tex::gamma_linear)' in text


def test_polyhaven_refuses_a_texture_without_a_color_map(host, tmp_path):
    with pytest.raises(ValueError, match=f"{PLANKS} has no Diffuse map, it has arm, blend, nor_gl"):
        polyhaven(host, tmp_path, maps=("nor_gl", "arm")).import_item(PLANKS, tmp_path / "out")
    assert not any(path.startswith("/dl/") for path in host.served)


def test_polyhaven_refuses_a_texture_without_the_2k_size(host, tmp_path):
    with pytest.raises(ValueError, match=f"{PLANKS} has no 2k jpg Diffuse map, it comes in 1k"):
        polyhaven(host, tmp_path, sizes=("1k",)).import_item(PLANKS, tmp_path / "out")


def test_polyhaven_unknown_texture_points_at_the_listing(host, tmp_path):
    with pytest.raises(KeyError, match="no texture 'nope' on Poly Haven, list the items with: import material polyhaven"):
        polyhaven(host, tmp_path).import_item("nope", tmp_path / "out")


def test_ambientcg_imports_the_2k_opengl_set(host, tmp_path):
    out = tmp_path / "out"
    target = ambientcg(host, tmp_path).import_item(METAL, out)
    assert target == out / "Common" / "Material" / METAL
    assert files(target) == [ATTRIBUTION_FILE, f"{METAL}.mdl", f"{METAL}/{METAL}_BaseColor.jpg", f"{METAL}/{METAL}_M.jpg", f"{METAL}/{METAL}_N.jpg", f"{METAL}/{METAL}_R.jpg", ANNOTATION_NAME]
    assert (target / METAL / f"{METAL}_N.jpg").read_bytes() == b"NormalGL 2K-JPG"
    assert (target / METAL / f"{METAL}_M.jpg").read_bytes() == b"Metalness 2K-JPG"
    text = (target / f"{METAL}.mdl").read_text()
    assert f'metallic_texture: texture_2d("./{METAL}/{METAL}_M.jpg", ::tex::gamma_linear)' in text
    assert "flip_tangent_v: false" in text and "ao_texture" not in text
    raw = yaml.safe_load((target / ANNOTATION_NAME).read_text())
    assert (raw["name"], raw["desc"]) == (METAL, "Metal 032")
    assert raw["tags"] == ["domain::common", "kind::metal", "origin::ambientcg", "grey", "smooth"]
    attribution = read_attribution(target / ATTRIBUTION_FILE)
    assert (attribution.license, attribution.authors, attribution.url) == ("CC0-1.0", "ambientCG", f"https://ambientcg.com/a/{METAL}")
    assert [path for path in host.served if path.startswith("/get")] == [f"/get?file={METAL}_2K-JPG.zip"]


def test_ambientcg_names_the_material_as_the_host_spells_it(host, tmp_path):
    target = ambientcg(host, tmp_path).import_item(METAL.lower(), tmp_path / "out")
    assert target == tmp_path / "out" / "Common" / "Material" / METAL


def test_ambientcg_refuses_a_material_without_the_2k_download(host, tmp_path):
    with pytest.raises(ValueError, match=f"{METAL} has no 2K-JPG download, it comes as 1K-JPG"):
        ambientcg(host, tmp_path, attributes=("1K-JPG",)).import_item(METAL, tmp_path / "out")


def test_ambientcg_refuses_an_archive_without_a_color_map(host, tmp_path):
    with pytest.raises(ValueError, match=f"{METAL} has no Color map, its archive holds"):
        ambientcg(host, tmp_path, maps=("NormalGL", "Roughness")).import_item(METAL, tmp_path / "out")


def test_ambientcg_unknown_material_points_at_the_listing(host, tmp_path):
    host.pages[f"/api/v2/full_json?{urllib.parse.urlencode({'id': 'nope', 'include': 'downloadData,tagData,displayData'})}"] = json.dumps({"foundAssets": []}).encode()
    with pytest.raises(KeyError, match="no material 'nope' on ambientCG, list the items with: import material ambientcg"):
        ambientcg(host, tmp_path).import_item("nope", tmp_path / "out")


def test_ambientcg_lists_every_page(host, tmp_path):
    importer = ambientcg(host, tmp_path)

    def page(offset: int) -> str:
        return f"/api/v2/full_json?{urllib.parse.urlencode({'type': 'Material', 'sort': 'Alphabet', 'limit': 500, 'offset': offset})}"

    host.pages[page(0)] = json.dumps({"foundAssets": [{"assetId": "Bricks001"}, {"assetId": METAL}]}).encode()
    host.pages[page(2)] = json.dumps({"foundAssets": [{"assetId": "Wood001"}]}).encode()
    host.pages[page(3)] = json.dumps({"foundAssets": []}).encode()
    assert list(importer.items()) == ["Bricks001", METAL, "Wood001"]


@pytest.mark.network
def test_polyhaven_live_listing_and_import(tmp_path):
    importer = PolyHaven(cache=tmp_path / "cache")
    items = list(importer.items())
    assert len(items) >= 800 and LIVE_PLANKS in items
    target = importer.import_item(LIVE_PLANKS, tmp_path / "out")
    maps = {path.name: path.stat().st_size for path in (target / LIVE_PLANKS).iterdir()}
    assert sorted(maps) == [f"{LIVE_PLANKS}_BaseColor.jpg", f"{LIVE_PLANKS}_N.jpg", f"{LIVE_PLANKS}_ORM.jpg"]
    assert all(size > 100_000 for size in maps.values())
    assert all((target / LIVE_PLANKS / name).read_bytes()[:2] == b"\xff\xd8" for name in maps)
    assert yaml.safe_load((target / ANNOTATION_NAME).read_text())["tags"][:3] == ["domain::common", "kind::asphalt", "origin::polyhaven"]
    assert read_attribution(target / ATTRIBUTION_FILE).authors == "Rob Tuytel"


@pytest.mark.network
def test_ambientcg_live_listing_and_import(tmp_path):
    importer = AmbientCG(cache=tmp_path / "cache")
    items = list(importer.items())
    assert len(items) >= 2000 and METAL in items
    target = importer.import_item(METAL, tmp_path / "out")
    maps = {path.name: path.read_bytes()[:2] for path in (target / METAL).iterdir()}
    assert maps == dict.fromkeys([f"{METAL}_BaseColor.jpg", f"{METAL}_M.jpg", f"{METAL}_N.jpg", f"{METAL}_R.jpg"], b"\xff\xd8")
    assert yaml.safe_load((target / ANNOTATION_NAME).read_text())["tags"][:3] == ["domain::common", "kind::metal", "origin::ambientcg"]
