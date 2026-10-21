import importlib.util
import json
import stat
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from arena_assets.__main__ import app
from arena_assets.impl import converter
from arena_assets.impl.build.ObjectDatabaseBuilder import ObjectAnnotation
from arena_assets.semantic import seats
from arena_assets.utils.geom import BoundingBox

STORED = json.loads((Path(__file__).parent / "data" / "seats_answers_sofa_b.json").read_text())
SOFA_BBOX = BoundingBox(STORED["bbox"])
FRAME = seats.RenderFrame.of(SOFA_BBOX)

runner = CliRunner()
needs_geometry = pytest.mark.skipif(importlib.util.find_spec("arena_simulation_setup") is None, reason="arena_simulation_setup comes with the Arena workspace")


def write_annotation(root: Path, asset: str, data: dict) -> Path:
    path = root / asset / "annotation.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(data))
    return path


def test_render_frame_puts_plus_y_at_the_top():
    (x_min, x_max), (y_min, y_max) = FRAME.to_footprint([0, 0, 100, 1000])
    assert (x_min, x_max) == pytest.approx((-1.4646, 1.4645), abs=1e-3)
    assert y_max == pytest.approx(FRAME.center_y + FRAME.span / 2)
    assert y_min == pytest.approx(FRAME.center_y + 0.4 * FRAME.span)


def test_render_frame_roundtrips_boxes():
    footprint = ((-0.46, 0.48), (0.03, 0.72))
    roundtrip = FRAME.to_footprint(FRAME.to_box_2d(footprint))
    assert [bound for side in roundtrip for bound in side] == pytest.approx([-0.46, 0.48, 0.03, 0.72], abs=FRAME.span / 1000)


@needs_geometry
def test_example_answer_is_the_first_stored_answer():
    assert seats.example_answer(seats.load_example(), SOFA_BBOX) == STORED["answers"][0]


@needs_geometry
@pytest.mark.parametrize(("yaw", "word"), [("+x", "right"), ("xy", "up-right"), ("-x", "left"), ("-y", "down"), (-2.3562, "down-left"), (3.1, "left")])
def test_facing_word(yaw, word):
    assert seats.facing_word(yaw) == word


def test_parse_seats_clips_to_the_object_and_drops_bad_slots():
    response = {
        "slots": [
            {"box_2d": [0, 0, 1000, 1000], "facing": ["down", "down"]},
            {"box_2d": [500, 500, 501, 501], "facing": ["down"]},
            {"box_2d": [0, 0, 100], "facing": ["down"]},
            {"box_2d": [0, 0, 500, 500], "facing": ["sideways"]},
            {"facing": ["up"]},
        ]
    }
    slots = seats.parse_seats(response, FRAME, SOFA_BBOX)
    assert len(slots) == 1
    assert [bound for side in slots[0].footprint for bound in side] == pytest.approx([SOFA_BBOX.min_x, SOFA_BBOX.max_x, SOFA_BBOX.min_y, SOFA_BBOX.max_y])
    assert slots[0].facing == ("down",)


@needs_geometry
def test_consensus_of_stored_answers_recovers_the_hand_annotation():
    answers = [seats.parse_seats(answer, FRAME, SOFA_BBOX) for answer in STORED["answers"]]
    merged = [slot.as_annotation() for slot in seats.consensus(answers)]
    example = seats.load_example()["sit"]
    assert len(merged) == len(example)
    for proposed, hand in zip(merged, example, strict=True):
        assert [bound for side in proposed["bbox"] for bound in side] == pytest.approx([bound for side in hand["bbox"] for bound in side], abs=0.005)
        assert proposed["yaw"] == hand["yaw"]


def test_consensus_drops_facings_named_by_a_minority():
    slot = seats.ProposedSlot(((0.0, 0.5), (0.0, 0.5)), ("down",))
    turned = seats.ProposedSlot(((0.0, 0.5), (0.0, 0.5)), ("down", "left"))
    assert seats.consensus([[turned], [slot], [slot]])[0].facing == ("down",)


def test_consensus_without_answers_is_empty():
    assert seats.consensus([]) == []


def test_consensus_drops_a_slot_matched_in_one_of_three_samples():
    seat = seats.ProposedSlot(((0.0, 0.5), (0.0, 0.5)), ("down",))
    stray = seats.ProposedSlot(((1.0, 1.5), (0.0, 0.5)), ("left",))
    assert seats.consensus([[seat, stray], [seat], [seat]]) == [seat]


def test_consensus_of_two_disagreeing_samples_keeps_only_shared_slots_in_either_order():
    seat = seats.ProposedSlot(((0.0, 0.5), (0.0, 0.5)), ("down",))
    left = seats.ProposedSlot(((-1.5, -1.0), (0.0, 0.5)), ("right",))
    right = seats.ProposedSlot(((1.0, 1.5), (0.0, 0.5)), ("left",))
    assert seats.consensus([[seat, left], [seat, right]]) == [seat]
    assert seats.consensus([[seat, right], [seat, left]]) == [seat]
    assert seats.consensus([[left], [right]]) == []


def test_parse_sittable_keeps_known_asset_paths_answered_true():
    response = {"objects": [{"path": "Office/Object/Chair", "sit": True}, {"path": "Office/Object/Shelf", "sit": False}, {"path": "Chair", "sit": True}]}
    assert seats.parse_sittable(response, ["Office/Object/Chair", "Hospital/Object/Chair", "Office/Object/Shelf"]) == {"Office/Object/Chair"}


def test_object_annotations_lists_object_assets_only(tmp_path):
    write_annotation(tmp_path, "Residential/Object/Sofa_B", {"desc": "Sofa"})
    write_annotation(tmp_path, "Residential/Material/Oak", {"desc": "Oak"})
    assert list(seats.object_annotations(tmp_path)) == ["Residential/Object/Sofa_B"]


@needs_geometry
def test_apply_writes_slots_and_keeps_other_keys(tmp_path):
    path = write_annotation(tmp_path, "Residential/Object/Sofa_B", {"desc": "Sofa", "face": "-y", "tags": ["residential"]})
    proposals = {"Residential/Object/Sofa_B": [{"bbox": [[-0.46, 0.48], [0.03, 0.72]], "yaw": "-y"}]}
    assert seats.apply_proposals(proposals, tmp_path) == (["Residential/Object/Sofa_B"], [])
    data = yaml.safe_load(path.read_text())
    assert data["tags"] == ["residential"]
    annotation = converter.structure({**data, "name": "Sofa_B", "path": "Residential/Object/Sofa_B"}, ObjectAnnotation)
    assert annotation.query_affordance("sit").facing.serialize() == "-y"


@needs_geometry
def test_apply_keeps_existing_slots_unless_overwriting(tmp_path):
    existing = [{"bbox": [[0.0, 0.5], [0.0, 0.5]], "yaw": "+x"}]
    path = write_annotation(tmp_path, "Office/Object/Chair", {"affordances": {"sit": existing}})
    proposals = {"Office/Object/Chair": [{"bbox": [[0.1, 0.4], [0.1, 0.4]], "yaw": "-y"}], "Office/Object/Ghost": []}
    assert seats.apply_proposals(proposals, tmp_path) == ([], ["Office/Object/Chair", "Office/Object/Ghost"])
    assert yaml.safe_load(path.read_text())["affordances"]["sit"] == existing
    seats.apply_proposals(proposals, tmp_path, overwrite=True)
    assert yaml.safe_load(path.read_text())["affordances"]["sit"][0]["yaw"] == "-y"


def test_apply_removes_slots_proposed_empty(tmp_path):
    path = write_annotation(tmp_path, "Office/Object/Shelf", {"desc": "Shelf", "affordances": {"sit": [{}]}})
    seats.apply_proposals({"Office/Object/Shelf": []}, tmp_path, overwrite=True)
    assert yaml.safe_load(path.read_text()) == {"desc": "Shelf"}


def test_apply_of_an_empty_proposal_without_sit_slots_leaves_the_file_alone(tmp_path):
    path = write_annotation(tmp_path, "Office/Object/Shelf", {})
    path.write_text("desc: Shelf  # hand note\naffordances:\n")
    assert seats.apply_proposals({"Office/Object/Shelf": []}, tmp_path, overwrite=True) == ([], [])
    assert path.read_text() == "desc: Shelf  # hand note\naffordances:\n"


@needs_geometry
def test_apply_fills_null_affordances_keeping_key_order_and_unicode(tmp_path):
    path = write_annotation(tmp_path, "Office/Object/Bench", {})
    path.write_text("tags: [office]\ndesc: Bank f\u00fcr zwei\naffordances:\nface: -y\n")
    proposals = {"Office/Object/Bench": [{"bbox": [[0.0, 0.5], [0.0, 0.5]], "yaw": "-y"}]}
    assert seats.apply_proposals(proposals, tmp_path) == (["Office/Object/Bench"], [])
    text = path.read_text()
    assert "Bank f\u00fcr zwei" in text
    data = yaml.safe_load(text)
    assert list(data) == ["tags", "desc", "affordances", "face"]
    assert data["affordances"] == {"sit": proposals["Office/Object/Bench"]}


@needs_geometry
def test_apply_rejects_an_invalid_slot_before_writing_any_file(tmp_path):
    chair = write_annotation(tmp_path, "Office/Object/Chair", {"desc": "Chair"})
    stool = write_annotation(tmp_path, "Office/Object/Stool", {"desc": "Stool"})
    proposals = {"Office/Object/Chair": [{"bbox": [[0.0, 0.5], [0.0, 0.5]], "yaw": "-y"}], "Office/Object/Stool": [{"bbox": [[0.0, 0.5], [0.0, 0.5]], "yaw": "-y"}, {"bbox": [[0.5, 0.0], [0.0, 0.5]], "yaw": "-y"}]}
    with pytest.raises(ValueError, match="Office/Object/Stool: sit slot 1"):
        seats.apply_proposals(proposals, tmp_path)
    assert yaml.safe_load(chair.read_text()) == {"desc": "Chair"}
    assert yaml.safe_load(stool.read_text()) == {"desc": "Stool"}


def test_write_atomic_replaces_the_file_keeping_its_mode_and_no_temporary(tmp_path):
    path = tmp_path / "annotation.yaml"
    path.write_text("old\n")
    path.chmod(0o644)
    seats.write_atomic(path, "d\u00e9sk\n")
    assert path.read_text(encoding="utf-8") == "d\u00e9sk\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert [entry.name for entry in tmp_path.iterdir()] == ["annotation.yaml"]


def test_cli_seats_without_verb_lists_verbs():
    result = runner.invoke(app, ["semantic", "seats"])
    assert "propose" in result.output
    assert "apply" in result.output


def test_cli_propose_and_apply_without_arguments_show_their_help():
    propose = runner.invoke(app, ["semantic", "seats", "propose"])
    assert "--renders" in propose.output
    assert "--thinking" in propose.output
    apply = runner.invoke(app, ["semantic", "seats", "apply"])
    assert "--overwrite" in apply.output


def test_cli_propose_help_lists_the_thinking_levels():
    output = runner.invoke(app, ["semantic", "seats", "propose", "--help"]).output
    for level in ("minimal", "low", "medium", "high"):
        assert level in output


@needs_geometry
def test_cli_apply_reports_what_it_wrote(tmp_path):
    write_annotation(tmp_path / "source", "Office/Object/Chair", {"desc": "Chair"})
    proposals = tmp_path / "seats.yaml"
    proposals.write_text(yaml.safe_dump({"Office/Object/Chair": [{"bbox": [[0.0, 0.5], [0.0, 0.5]], "yaw": "-y"}]}))
    result = runner.invoke(app, ["semantic", "seats", "apply", str(proposals), str(tmp_path / "source")])
    assert result.exit_code == 0
    assert "wrote sit slots into 1 annotations, skipped 0" in result.output


def test_cli_propose_rejects_unknown_assets_before_calling_gemini(tmp_path):
    write_annotation(tmp_path, "Office/Object/Chair", {"desc": "Chair"})
    result = runner.invoke(app, ["semantic", "seats", "propose", str(tmp_path), "--renders", str(tmp_path), "--out", str(tmp_path / "out.yaml"), "--only", "Office/Object/Sofa"])
    assert result.exit_code == 1
    assert "no source annotation for Office/Object/Sofa" in result.output
    assert "Office/Object/Chair" in result.output


def test_cli_propose_rejects_an_out_file_that_is_not_a_proposals_mapping(tmp_path):
    write_annotation(tmp_path, "Office/Object/Chair", {"desc": "Chair"})
    out = tmp_path / "out.yaml"
    out.write_text("- Office/Object/Chair\n")
    result = runner.invoke(app, ["semantic", "seats", "propose", str(tmp_path), "--renders", str(tmp_path), "--out", str(out), "--only", "Office/Object/Chair"])
    assert result.exit_code == 1
    assert "does not map asset paths to sit slots" in result.output
    assert out.read_text() == "- Office/Object/Chair\n"


@needs_geometry
def test_cli_apply_names_the_invalid_slot_and_writes_nothing(tmp_path):
    path = write_annotation(tmp_path / "source", "Office/Object/Chair", {"desc": "Chair"})
    proposals = tmp_path / "seats.yaml"
    proposals.write_text(yaml.safe_dump({"Office/Object/Chair": [{"bbox": [[0.0, 0.5], [0.0, 0.5]], "yaw": "sideways"}]}))
    result = runner.invoke(app, ["semantic", "seats", "apply", str(proposals), str(tmp_path / "source")])
    assert result.exit_code == 1
    assert "Office/Object/Chair: sit slot 0 is invalid" in result.output
    assert yaml.safe_load(path.read_text()) == {"desc": "Chair"}
