"""Client for the arena_assets_blender worker, the Blender side of model conversion in its own interpreter."""

from __future__ import annotations

import functools
import json
import os
import shlex
import shutil
import subprocess
from collections.abc import Sequence

import attrs

COMMAND_ENV = "ARENA_ASSETS_BLENDER"
COMMAND = "arena-assets-blender"
PARENT_ONLY_ENV = ("PYTHONPATH", "PYTHONHOME")


def _mesh_element(tag: str, name: str, uri: str, material: str = "") -> str:
    return f"""      <{tag} name="{name}">
        <geometry>
          <mesh>
            <uri>{uri}</uri>
          </mesh>
        </geometry>{material}
      </{tag}>"""


def sdf_model(stem: str, glow: Sequence[tuple[str, str, str]] = (), *, plain: bool = True) -> str:
    """Static SDF model whose visual and collision are the sibling `{stem}.dae`, plus a visual and a collision per glow mesh (name, uri, material)."""
    parts = [_mesh_element("visual", "visual", f"{stem}.dae"), _mesh_element("collision", "collision", f"{stem}.dae")] if plain else []
    for index, (name, uri, material) in enumerate(glow):
        parts += [_mesh_element("visual", name, uri, material), _mesh_element("collision", f"collision_glow_{index}", uri)]
    link = "\n".join(parts)
    return f"""<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="{stem}">
    <static>true</static>
    <link name="link">
{link}
    </link>
  </model>
</sdf>"""


class BlenderUnavailable(RuntimeError):
    pass


class BlenderError(RuntimeError):
    pass


def command() -> list[str]:
    configured = os.environ.get(COMMAND_ENV)
    if configured:
        return shlex.split(configured)
    found = shutil.which(COMMAND)
    if found is None:
        raise BlenderUnavailable(f"model conversion needs the Blender worker: install the arena_assets_blender package (blender/ in the arena-assets repo, Python 3.11) so {COMMAND} is on PATH, or set {COMMAND_ENV} to its command")
    return [found]


def environment() -> dict[str, str]:
    """The worker's environment: the parent's without the variables that bind it to the parent's interpreter."""
    return {key: value for key, value in os.environ.items() if key not in PARENT_ONLY_ENV}


@functools.cache
def exts() -> tuple[str, ...]:
    """Model file extensions the worker loads and saves."""
    try:
        proc = subprocess.run([*command(), "--exts"], capture_output=True, text=True, check=False, env=environment())
    except OSError as e:
        raise BlenderUnavailable(f"cannot run the Blender worker {command()}: {e}") from e
    if proc.returncode:
        raise BlenderUnavailable(f"{COMMAND} --exts failed with {proc.returncode}: {proc.stderr.strip()[-2000:]}")
    return tuple(json.loads(proc.stdout.strip().splitlines()[-1]))


@attrs.frozen
class Result:
    values: list
    stdout: str
    stderr: str


class BlenderWorker:
    """One worker process per build, each run is a fresh scene."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self._process is None:
            try:
                self._process = subprocess.Popen([*command(), "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=environment())
            except OSError as e:
                raise BlenderUnavailable(f"cannot run the Blender worker {command()}: {e}") from e

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        process.wait()

    def run(self, steps: list[tuple], keep: bool = False) -> Result:
        """Run converter operations in order: (operation, *args) with an optional trailing kwargs dict, keep leaves the scene open for the next run."""
        self.start()
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise BlenderUnavailable(f"{COMMAND} did not start")
        process.stdin.write(json.dumps({"steps": [list(step) for step in steps], "keep": keep}) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            code = process.wait()
            self._process = None
            raise BlenderError(f"{COMMAND} exited with {code} during {[step[0] for step in steps]}")
        reply = json.loads(line)
        if not reply["ok"]:
            raise BlenderError(f"{reply['error']}\n{reply['stderr']}".strip())
        return Result(values=reply["values"], stdout=reply["stdout"], stderr=reply["stderr"])
