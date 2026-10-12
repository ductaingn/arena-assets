"""Collider command: attach a collision mesh to an object bundle, or remove it."""

from pathlib import Path

import typer

from .utils import safe_echo

HELP = """Give an object bundle a collider, or take it away.

BUNDLE is an object bundle directory with an annotation.yaml, as 'arena asset pull object NAME' or 'arena-assets splat' leave it. MESH is a triangle mesh, .obj or .ply (ASCII or binary, polygons are split into triangles), in the bundle frame: the frame of the bundle's model files, meters, Z up. --matrix maps a mesh from another frame into it.

The mesh replaces the collision of every model in the bundle: NAME.usdz (or .usd, .usda, .usdc) gets it as an invisible static triangle-mesh collider under its default prim, NAME.sdf/NAME.sdf as its only <collision>. It is also kept as NAME.collider.obj, which annotation.yaml names under 'collider:'. Robots collide with it, cameras, lidar and depth cameras keep seeing the visual model. Running the command again replaces the collider. --remove takes it out again, and the models collide as they were built.

Next: check the printed models and triangle count, then publish with 'arena asset push object NAME'."""

EPILOG = """Examples: arena-assets collider ./Object/SM_Chair_01a chair_hull.obj
arena-assets collider ./Object/SM_Chair_01a --remove"""


def _fail(message: str) -> typer.Exit:
    typer.echo(f"arena-assets collider: {message}, nothing was changed", err=True)
    return typer.Exit(1)


def _names(bundle: Path, paths: list[Path]) -> str:
    return ", ".join(str(path.relative_to(bundle)) for path in paths) or "no model"


def collider_command(
    ctx: typer.Context,
    bundle: Path = typer.Argument(..., help="Object bundle directory holding annotation.yaml, for example ./Object/SM_Chair_01a"),
    mesh: Path | None = typer.Argument(None, help="Collision mesh, an .obj or .ply file in the bundle frame. Omit it together with --remove"),
    matrix: str | None = typer.Option(None, "--matrix", metavar="'16 NUMBERS'", help="Row-major 4x4 that maps MESH coordinates into the bundle frame in meters (p_bundle = M @ p_mesh, last row 0 0 0 1), as one quoted argument. Default: MESH is already in the bundle frame"),
    remove: bool = typer.Option(False, "--remove", help="Take the collider out of the bundle's models, delete NAME.collider.obj and the 'collider:' annotation entry instead of writing one"),
):
    """Attach a collision mesh to an object bundle, or remove it."""
    from arena_assets.impl.collider import ColliderError, attach_mesh, remove_collider
    from arena_assets.impl.splat import SplatError, parse_matrix

    if remove and (mesh is not None or matrix is not None):
        raise _fail("--remove takes only BUNDLE, drop MESH and --matrix")
    if not remove and mesh is None:
        raise _fail("MESH is missing, pass an .obj or .ply collision mesh in the bundle frame, or --remove to take the collider away")
    if not bundle.is_dir():
        raise _fail(f"BUNDLE {bundle} is not a directory, pass an object bundle directory holding annotation.yaml")

    try:
        if remove:
            stripped = remove_collider(bundle)
            safe_echo(f"removed the collider from {_names(bundle, stripped)}" if stripped is not None else "the bundle had no collider", ctx)
        else:
            transform = parse_matrix(matrix, "the mesh frame to the bundle frame") if matrix is not None else None
            baked, triangles = attach_mesh(bundle, mesh, transform)
            safe_echo(f"baked {triangles} collider triangles into {_names(bundle, baked)}", ctx)
    except (ColliderError, SplatError) as error:
        raise _fail(str(error)) from None
    typer.echo(str(bundle))


def add_to_cmd(cmd: typer.Typer):
    cmd.command("collider", help=HELP, epilog=EPILOG, no_args_is_help=True)(collider_command)
