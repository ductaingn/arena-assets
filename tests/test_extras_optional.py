import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

SCRIPT = """
import sys

for name in ("chromadb", "bpy", "mathutils", "numpy", "scipy"):
    sys.modules[name] = None

import arena_assets
import arena_assets.importers
import arena_assets.importers.gso
import arena_assets.importers.openrobotics
import arena_assets.impl
import arena_assets.impl.build
import arena_assets.impl.build.HumanDatabaseBuilder
import arena_assets.impl.build.ObjectDatabaseBuilder
import arena_assets.impl.build.SoundDatabaseBuilder
import arena_assets.impl.listall
import arena_assets.impl.query
import arena_assets.utils.Database
import arena_assets.utils.ModelConverter
import arena_assets.cli
import arena_assets.__main__

print(sorted(arena_assets.importers.IMPORTERS.meta("object", name).name for name in ("gso",)))
"""


def test_package_imports_without_chromadb_and_blender():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), *filter(None, [env.get("PYTHONPATH")])])
    proc = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, env=env, check=False)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "['gso']"
