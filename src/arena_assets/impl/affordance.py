"""Affordances of object annotations, structured by arena_simulation_setup.utils.affordance only when an annotation has any."""

from __future__ import annotations

import typing
from collections.abc import Iterable, Mapping

if typing.TYPE_CHECKING:
    from arena_simulation_setup.utils.affordance import Affordances
else:
    Affordances = dict[str, list[typing.Any]]


def structure_affordances(value: Mapping[str, Iterable[typing.Any]] | None) -> Affordances:
    if not value:
        return {}
    from arena_simulation_setup.utils.affordance import structure_affordances as structure

    return structure(value)


def unstructure_affordances(affordances: Affordances) -> dict[str, list[dict[str, object]]]:
    if not affordances:
        return {}
    from arena_simulation_setup.utils.affordance import unstructure_affordances as unstructure

    return unstructure(affordances)
