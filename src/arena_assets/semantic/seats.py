"""Sit slots proposed by a vision language model from top-down renders, and their merge into source annotations."""

from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import tempfile
import typing
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from importlib import resources
from pathlib import Path

import attrs
import yaml

from arena_assets.impl import ANNOTATION_NAME
from arena_assets.utils.geom import BoundingBox
from arena_assets.utils.logging import get_logger

if typing.TYPE_CHECKING:
    from arena_simulation_setup.utils.affordance import Footprint

logger = get_logger("semantic.seats")

FACINGS = {word: index * math.pi / 4 for index, word in enumerate(("right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"))}

MIN_SLOT_AREA = 0.01
MATCH_IOU = 0.3

NAMES_PROMPT = """For each object below, given as asset path and description, say whether a person can sit on it.
Chairs, armchairs, sofas, benches, stools, beds and toilets are for sitting.
Tables (operating and exam tables too), gurneys, wheelchairs, shelves, cabinets, racks and accessories such as toilet paper or bedside tables are not.
Answer with every asset path exactly as given."""

SEATS_PROMPT = """You annotate where people can sit on one piece of furniture, seen from straight above in an orthographic render.
Return one slot per place where one person can sit.
A slot's box_2d is [ymin, xmin, ymax, xmax] in 0 to 1000 image coordinates. It covers only the seat surface a person's bottom rests on: never backrests, armrests, cushions lying against the back, or the floor.
A seated person's legs hang over an open edge of the seat. Every facing points from the slot toward an open edge where legs can go down: up, down, left, right, or a diagonal such as down-left.
Give several facings when the legs can go over several open edges, as at the free end of a chaise.
Slots may overlap when one place can be used in different ways.
Return no slots when nobody can sit on the object."""

NAMES_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "objects": {
            "type": "ARRAY",
            "items": {"type": "OBJECT", "properties": {"path": {"type": "STRING"}, "sit": {"type": "BOOLEAN"}}, "required": ["path", "sit"]},
        }
    },
    "required": ["objects"],
}

SEATS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "slots": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "box_2d": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                    "facing": {"type": "ARRAY", "items": {"type": "STRING", "enum": list(FACINGS)}},
                },
                "required": ["box_2d", "facing"],
            },
        }
    },
    "required": ["slots"],
}


@attrs.frozen
class RenderFrame:
    """Frame of a top-down render: a square image spanning the larger bbox side, centered on the bbox, +y up."""

    center_x: float
    center_y: float
    span: float

    @classmethod
    def of(cls, bbox: BoundingBox) -> RenderFrame:
        return cls((bbox.min_x + bbox.max_x) / 2, (bbox.min_y + bbox.max_y) / 2, max(bbox.max_x - bbox.min_x, bbox.max_y - bbox.min_y))

    def to_footprint(self, box_2d: Sequence[float]) -> Footprint:
        """Object-frame footprint of a [ymin, xmin, ymax, xmax] box in 0 to 1000 image coordinates."""
        v_min, u_min, v_max, u_max = (float(value) / 1000 for value in box_2d)
        return (
            (self.center_x + (min(u_min, u_max) - 0.5) * self.span, self.center_x + (max(u_min, u_max) - 0.5) * self.span),
            (self.center_y + (0.5 - max(v_min, v_max)) * self.span, self.center_y + (0.5 - min(v_min, v_max)) * self.span),
        )

    def to_box_2d(self, footprint: Footprint) -> list[int]:
        (x_min, x_max), (y_min, y_max) = footprint
        return [
            round((0.5 - (y_max - self.center_y) / self.span) * 1000),
            round(((x_min - self.center_x) / self.span + 0.5) * 1000),
            round((0.5 - (y_min - self.center_y) / self.span) * 1000),
            round(((x_max - self.center_x) / self.span + 0.5) * 1000),
        ]


@attrs.frozen
class ProposedSlot:
    footprint: Footprint
    facing: tuple[str, ...]

    @property
    def area(self) -> float:
        (x_min, x_max), (y_min, y_max) = self.footprint
        return max(0.0, x_max - x_min) * max(0.0, y_max - y_min)

    def iou(self, other: ProposedSlot) -> float:
        (x_min, x_max), (y_min, y_max) = self.footprint
        (o_x_min, o_x_max), (o_y_min, o_y_max) = other.footprint
        inter = max(0.0, min(x_max, o_x_max) - max(x_min, o_x_min)) * max(0.0, min(y_max, o_y_max) - max(y_min, o_y_min))
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def as_annotation(self) -> dict[str, object]:
        from arena_simulation_setup.utils.geometry import Facing

        (x_min, x_max), (y_min, y_max) = ((round(low, 3) + 0.0, round(high, 3) + 0.0) for low, high in self.footprint)
        serialized = Facing.parse([FACINGS[word] for word in self.facing]).serialize()
        yaws = [yaw if isinstance(yaw, str) else round(yaw, 4) for yaw in (serialized if isinstance(serialized, list) else [serialized])]
        return {"bbox": [[x_min, x_max], [y_min, y_max]], "yaw": yaws[0] if len(yaws) == 1 else yaws}


def facing_word(yaw: str | float) -> str:
    """Image direction nearest to a yaw or axis name."""
    from arena_simulation_setup.utils.geometry import Facing

    angle = Facing.parse(yaw).yaws[0]
    return min(FACINGS, key=lambda word: abs(math.remainder(angle - FACINGS[word], 2 * math.pi)))


def parse_seats(response: Mapping[str, typing.Any], frame: RenderFrame, bbox: BoundingBox) -> list[ProposedSlot]:
    """Slots of one model answer in the object frame, clipped to the object bbox, malformed or tiny ones dropped."""
    slots = []
    for entry in response.get("slots") or ():
        try:
            box_2d = [float(value) for value in entry["box_2d"]]
            facing = tuple(dict.fromkeys(str(word) for word in entry["facing"]))
            if len(box_2d) != 4 or not facing or any(word not in FACINGS for word in facing):
                raise ValueError(entry)
        except (KeyError, TypeError, ValueError):
            logger.warning("Dropping malformed slot %r", entry)
            continue
        (x_min, x_max), (y_min, y_max) = frame.to_footprint(box_2d)
        slot = ProposedSlot(((max(x_min, bbox.min_x), min(x_max, bbox.max_x)), (max(y_min, bbox.min_y), min(y_max, bbox.max_y))), facing)
        if slot.area < MIN_SLOT_AREA:
            logger.warning("Dropping slot %r, %.4f m2 inside the object bbox", entry, slot.area)
            continue
        slots.append(slot)
    return slots


def _merge_from(reference: int, samples: Sequence[Sequence[ProposedSlot]]) -> list[ProposedSlot]:
    unused = [list(sample) for index, sample in enumerate(samples) if index != reference]
    merged = []
    for slot in samples[reference]:
        matched = [slot]
        for candidates in unused:
            best = max(candidates, key=slot.iou, default=None)
            if best is not None and slot.iou(best) >= MATCH_IOU:
                candidates.remove(best)
                matched.append(best)
        if len(matched) * 2 <= len(samples):
            continue
        (x_min, x_max), (y_min, y_max) = ((statistics.median(match.footprint[axis][0] for match in matched), statistics.median(match.footprint[axis][1] for match in matched)) for axis in range(2))
        votes = Counter(word for match in matched for word in match.facing)
        facing = tuple(word for word in FACINGS if votes[word] * 2 > len(matched)) or slot.facing
        merged.append(ProposedSlot(((x_min, x_max), (y_min, y_max)), facing))
    return merged


def consensus(samples: Sequence[Sequence[ProposedSlot]]) -> list[ProposedSlot]:
    """Slots matched in a strict majority of the samples, read off the sample keeping the most (the first on a tie), with median boxes and majority facings."""
    return max((_merge_from(reference, samples) for reference in range(len(samples))), key=len, default=[])


def parse_sittable(response: Mapping[str, typing.Any], assets: Iterable[str]) -> set[str]:
    known = set(assets)
    sittable = set()
    for entry in response.get("objects") or ():
        if entry.get("sit") is True and entry.get("path") in known:
            sittable.add(entry["path"])
    return sittable


def object_annotations(source: Path) -> dict[str, Path]:
    """Source annotation files of objects by asset path relative to the source root, such as Residential/Object/Sofa_B."""
    return {str(path.parent.relative_to(source)): path for path in sorted(source.rglob(ANNOTATION_NAME)) if path.parent.parent.name == "Object"}


def load_example() -> dict:
    return yaml.safe_load(resources.files("arena_assets.semantic").joinpath("seats_example.yaml").read_text())


def example_answer(example: Mapping[str, typing.Any], bbox: BoundingBox) -> dict:
    from arena_simulation_setup.utils.affordance import slot_box
    from arena_simulation_setup.utils.geometry import Facing

    frame = RenderFrame.of(bbox)
    slots = []
    for slot in example["sit"]:
        slots.append({"box_2d": frame.to_box_2d(slot_box(slot["bbox"])[:2]), "facing": [facing_word(yaw) for yaw in Facing.parse(slot["yaw"]).yaws]})
    return {"slots": slots}


def write_atomic(path: Path, text: str) -> None:
    """Replace path with text through a temporary file in the same directory, keeping the file mode."""
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(text)
        if path.exists():
            shutil.copymode(path, temporary)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def dump_yaml(data: object) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def _validated(asset: str, slots: object) -> list[dict[str, typing.Any]]:
    if not isinstance(slots, list | tuple):
        raise ValueError(f"{asset}: sit slots must be a list, got {slots!r}")
    if not slots:
        return []
    from arena_simulation_setup.utils.affordance import Slot, structure_affordances

    for index, slot in enumerate(slots):
        try:
            Slot.parse(slot).unstructure()
        except (TypeError, ValueError) as e:
            raise ValueError(f"{asset}: sit slot {index} is invalid, {e}") from e
    try:
        structure_affordances({"sit": slots})
    except ValueError as e:
        raise ValueError(f"{asset}: {e}") from e
    return [dict(slot) for slot in slots]


def apply_proposals(proposals: Mapping[str, Sequence[Mapping[str, typing.Any]]], source: Path, *, overwrite: bool = False) -> tuple[list[str], list[str]]:
    """Write proposed sit slots into the source annotations, returning the written and the skipped assets, all validated before any write."""
    skipped = []
    files = object_annotations(source)
    changes: list[tuple[str, Path, dict]] = []
    for asset, slots in proposals.items():
        validated = _validated(asset, slots)
        path = files.get(asset)
        if path is None:
            logger.warning("No source annotation for %s under %s", asset, source)
            skipped.append(asset)
            continue
        data = yaml.safe_load(path.read_text()) or {}
        affordances = data.get("affordances") or {}
        if "sit" in affordances and not overwrite:
            skipped.append(asset)
            continue
        if not validated and "sit" not in affordances:
            continue
        if validated:
            affordances["sit"] = validated
            data["affordances"] = affordances
        else:
            affordances.pop("sit")
            if not affordances:
                data.pop("affordances")
        changes.append((asset, path, data))
    for _, path, data in changes:
        write_atomic(path, dump_yaml(data))
    return [asset for asset, _, _ in changes], skipped


class Gemini:
    """Gemini JSON answers through google-genai, the key from GEMINI_API_KEY or GOOGLE_API_KEY."""

    def __init__(self, model: str, thinking: str) -> None:
        try:
            from google import genai
        except ImportError as e:
            raise RuntimeError("google-genai is missing, install arena_assets[vlm]") from e
        self._genai = genai
        self._client = genai.Client()
        self._model = model
        self._thinking = thinking
        self.usage: Counter[str] = Counter()

    def image(self, path: Path) -> object:
        return self._genai.types.Part.from_bytes(data=path.read_bytes(), mime_type="image/png")

    def ask(self, contents: list[typing.Any], schema: Mapping[str, typing.Any]) -> dict:
        config: typing.Any = {"response_mime_type": "application/json", "response_schema": schema, "thinking_config": {"thinking_level": self._thinking}}
        response = self._client.models.generate_content(model=self._model, contents=contents, config=config)
        if (metadata := response.usage_metadata) is not None:
            self.usage.update(requests=1, input=metadata.prompt_token_count or 0, output=metadata.candidates_token_count or 0, thinking=metadata.thoughts_token_count or 0)
        if response.text is None:
            raise ValueError(f"{self._model} answered without text")
        return json.loads(response.text)
