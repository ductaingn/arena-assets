"""Body measurements read off a skinned human model (COLLADA) in its bind pose, clearances from the thigh vertices behind the hip joints."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import attrs
import numpy as np

PELVIS_JOINT = "Hips"  # cmu_mb rig, the skeleton's root joint
THIGH_JOINTS = ("LeftUpLeg", "RightUpLeg")  # cmu_mb rig, the hip joints the thighs hang from
LEG_CHAINS = (("LeftUpLeg", "LeftLeg", "LeftFoot"), ("RightUpLeg", "RightLeg", "RightFoot"))  # cmu_mb rig, hip, knee, ankle


@attrs.define(frozen=True)
class PelvisClearance:
    """Height of the pelvis joint above the surface a body rests on, in m: seated on its thighs, lying on its back."""

    sit: float = attrs.field(converter=float)
    lie: float = attrs.field(converter=float)


@attrs.define(frozen=True)
class Legs:
    """Leg geometry of a standing body in m: pelvis joint above the hip joints, thigh, shin, ankle joint above the soles."""

    hip: float = attrs.field(converter=float)
    thigh: float = attrs.field(converter=float)
    shin: float = attrs.field(converter=float)
    ankle: float = attrs.field(converter=float)


@attrs.define(frozen=True)
class Skin:
    """Bind pose of a skinned mesh: joint positions by name, vertex positions and the joint that moves each vertex most."""

    joints: dict[str, np.ndarray]
    points: np.ndarray
    dominant: list[str]


def pelvis_clearance(skin: Skin, forward: np.ndarray) -> PelvisClearance:
    """Clearances of a body facing `forward` (a horizontal unit vector) in its bind pose, see the module docstring."""
    pelvis = skin.joints[PELVIS_JOINT]
    dominant = np.asarray(skin.dominant)
    sit, lie = [], []
    for name in THIGH_JOINTS:
        hip = skin.joints[name]
        thigh = skin.points[dominant == name]
        if len(thigh) == 0:
            raise ValueError(f"no vertex follows {name}")
        behind = float(((hip[:2] - thigh[:, :2]) @ forward).max())
        sit.append(pelvis[2] - hip[2] + behind)
        lie.append(float((pelvis[:2] - hip[:2]) @ forward) + behind)
    return PelvisClearance(sit=round(max(sit), 3), lie=round(max(lie), 3))


def legs(skin: Skin) -> Legs:
    """Leg geometry of a body in its bind pose, both legs averaged, the soles at its lowest vertex."""
    floor = float(skin.points[:, 2].min())
    hip, knee, ankle = (np.mean([skin.joints[chain[i]] for chain in LEG_CHAINS], axis=0) for i in range(3))
    return Legs(
        hip=round(float(skin.joints[PELVIS_JOINT][2] - hip[2]), 3),
        thigh=round(float(np.linalg.norm(hip - knee)), 3),
        shin=round(float(np.linalg.norm(knee - ankle)), 3),
        ankle=round(float(ankle[2]) - floor, 3),
    )


def read_skin(dae: Path) -> Skin:
    """The bind pose of every skin controller in a COLLADA file, merged."""
    root = ET.parse(dae).getroot()
    ns = root.tag[: root.tag.index("}") + 1]
    by_id = {element.get("id"): element for element in root.iter() if element.get("id")}

    def source(ref: str | None) -> ET.Element:
        if ref is None or (element := by_id.get(ref.lstrip("#"))) is None:
            raise ValueError(f"{dae}: missing COLLADA source {ref}")
        return element

    def floats(ref: str | None) -> np.ndarray:
        element = source(ref)
        accessor = element.find(f"{ns}technique_common/{ns}accessor")
        stride = int(accessor.get("stride", "1")) if accessor is not None else 1
        return np.array((element.findtext(f"{ns}float_array") or "").split(), dtype=float).reshape(-1, stride)

    joints: dict[str, np.ndarray] = {}
    points: list[np.ndarray] = []
    dominant: list[str] = []
    for skin in root.iter(f"{ns}skin"):
        bind_shape = np.array((skin.findtext(f"{ns}bind_shape_matrix") or "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1").split(), dtype=float).reshape(4, 4)
        inputs = {element.get("semantic"): element.get("source") for element in skin.iterfind(f"{ns}joints/{ns}input")}
        names = (source(inputs.get("JOINT")).findtext(f"{ns}Name_array") or "").split()
        inverse_binds = floats(inputs.get("INV_BIND_MATRIX")).reshape(-1, 4, 4)
        for name, inverse_bind in zip(names, inverse_binds, strict=True):
            joints.setdefault(name, np.linalg.inv(inverse_bind)[:3, 3])

        weights_element = skin.find(f"{ns}vertex_weights")
        if weights_element is None:
            raise ValueError(f"{dae}: skin without vertex_weights")
        offsets = {element.get("semantic"): (int(element.get("offset", "0")), element.get("source")) for element in weights_element.iterfind(f"{ns}input")}
        joint_offset, weight_offset, weight_source = offsets["JOINT"][0], offsets["WEIGHT"][0], offsets["WEIGHT"][1]
        weights = floats(weight_source).reshape(-1)
        counts = np.array((weights_element.findtext(f"{ns}vcount") or "").split(), dtype=int)
        pairs = np.array((weights_element.findtext(f"{ns}v") or "").split(), dtype=int).reshape(-1, max(joint_offset, weight_offset) + 1)
        start = 0
        for count in counts:
            influences = pairs[start : start + count]
            dominant.append(names[influences[np.argmax(weights[influences[:, weight_offset]]), joint_offset]] if count else "")
            start += count

        mesh = source(skin.get("source"))
        position = mesh.find(f"{ns}mesh/{ns}vertices/{ns}input[@semantic='POSITION']")
        vertices = floats(position.get("source") if position is not None else None)
        points.append((np.c_[vertices, np.ones(len(vertices))] @ bind_shape.T)[:, :3])
    if not points:
        raise ValueError(f"{dae}: no skin controller")
    merged = np.concatenate(points)
    if len(merged) != len(dominant):
        raise ValueError(f"{dae}: {len(dominant)} skin weights for {len(merged)} vertices")
    return Skin(joints=joints, points=merged, dominant=dominant)
