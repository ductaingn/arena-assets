from __future__ import annotations

import dataclasses

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


@dataclasses.dataclass(frozen=True, kw_only=True)
class SourceLight:
    """Light found in a source model scene, in world coordinates of the rectified model."""

    name: str
    type: str
    shape: str = ""
    spot_size: float = 0.0
    power: float
    color: Vector3
    location: Vector3
    rotation: Quaternion = (1.0, 0.0, 0.0, 0.0)
    """Unit quaternion (w, x, y, z) of the light, which emits along its local -z axis."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class SourceMaterial:
    """Emissive material found in a source model scene, with the world-space center of its faces."""

    name: str
    center: Vector3
