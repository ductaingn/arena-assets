import json

import pytest

from arena_assets.importers.providers import Fuel, Provider

MODELS = "https://fuel.gazebosim.org/1.0/GoogleResearch/models"
FIXTURE = "Jawbone_UP24_Wireless_Activity_Tracker_Pink_Coral_L"


def test_fuel_name_and_base():
    assert Fuel.name == "fuel"
    assert issubclass(Fuel, Provider)
    assert Fuel("GoogleResearch")._models == MODELS


def test_fuel_zip_and_metadata_urls():
    fuel = Fuel("GoogleResearch")
    assert fuel.url(f"{FIXTURE}/tip/{FIXTURE}.zip") == f"{MODELS}/{FIXTURE}/tip/{FIXTURE}.zip"
    assert fuel.url(f"/{FIXTURE}") == f"{MODELS}/{FIXTURE}"


def test_fuel_quotes_owner_and_path():
    assert Fuel("Open Robotics").url("a b/1/a b.zip") == "https://fuel.gazebosim.org/1.0/Open%20Robotics/models/a%20b/1/a%20b.zip"


def test_fuel_page_url():
    assert Fuel("GoogleResearch")._page_url(3) == f"{MODELS}?page=3&per_page=100"


@pytest.mark.network
def test_fuel_first_page_is_full_and_links_to_the_next():
    entries, next_url = Fuel("GoogleResearch").page(1)
    assert len(entries) == 100
    assert {entry["owner"] for entry in entries} == {"GoogleResearch"}
    assert {entry["license_name"] for entry in entries} == {"Creative Commons Attribution 4.0 International"}
    assert next_url == f"{MODELS}?page=2&per_page=100"


@pytest.mark.network
def test_fuel_gets_metadata_idempotently(tmp_path):
    fuel = Fuel("GoogleResearch")
    dest = fuel.get(FIXTURE, tmp_path / f"{FIXTURE}.json")
    with open(dest) as f:
        metadata = json.load(f)
    assert metadata["name"] == FIXTURE
    assert metadata["license_url"] == "http://creativecommons.org/licenses/by/4.0/"
    assert metadata["filesize"] == 1463783
    mtime = dest.stat().st_mtime_ns
    assert fuel.get(FIXTURE, dest) == dest
    assert dest.stat().st_mtime_ns == mtime
