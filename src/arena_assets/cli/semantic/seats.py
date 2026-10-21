"""Sit slot proposals from Gemini and their merge into source annotations."""

import enum
import json
from pathlib import Path

import typer
import yaml

from arena_assets.semantic import seats
from arena_assets.utils.geom import BoundingBox

from ..utils import safe_echo

NAMES_PER_REQUEST = 50


class Thinking(enum.StrEnum):
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def _fail(message: str) -> typer.Exit:
    typer.echo(f"arena-assets semantic seats: {message}", err=True)
    return typer.Exit(1)


def _usage(gemini: seats.Gemini) -> str:
    usage = gemini.usage
    return f"{usage['requests']} requests, {usage['input']} input, {usage['output']} output and {usage['thinking']} thinking tokens (thinking is billed as output)"


def _built(renders: Path, asset: str) -> tuple[BoundingBox, Path, Path] | None:
    name = Path(asset).name
    annotation, topdown, thumb = renders / asset / "annotation.yaml", renders / asset / f"{name}_topdown.png", renders / asset / f"{name}_thumb.png"
    if not (annotation.is_file() and topdown.is_file() and thumb.is_file()):
        return None
    return BoundingBox(yaml.safe_load(annotation.read_text())["bounding_box"]), topdown, thumb


def _describe(asset: str, path: Path) -> str:
    data = yaml.safe_load(path.read_text()) or {}
    return f"{asset}: {data.get('desc', '')}"


def propose_command(
    ctx: typer.Context,
    source: Path = typer.Argument(..., help="Source dataset root holding <Domain>/Object/<Name>/annotation.yaml", exists=True, file_okay=False),
    renders: Path = typer.Option(..., "--renders", "-r", help="Built database root holding <asset>/annotation.yaml and the <name>_topdown.png and <name>_thumb.png previews of a build with -o previews", exists=True, file_okay=False),
    out: Path = typer.Option(..., "--out", help="YAML file mapping asset paths to sit slots, read by 'apply'. Rewritten after every asset, an existing file is merged into: its other assets are kept, re-proposed ones replaced", dir_okay=False),
    only: list[str] = typer.Option([], "--only", help="Asset path such as Residential/Object/Sofa_B to propose slots for, skipping the name pass. Repeat for several"),
    model: str = typer.Option("gemini-3.8-flash", "--model", help="Gemini model id"),
    samples: int = typer.Option(3, "--samples", min=1, help="Answers per object, merged by consensus"),
    thinking: Thinking = typer.Option(Thinking.LOW, "--thinking", help="Gemini thinking level, as the model supports"),
):
    """Ask Gemini which objects are for sitting and where their sit slots are, writing proposals for review. Needs GEMINI_API_KEY and arena_assets[vlm]."""
    files = seats.object_annotations(source)
    if unknown := [asset for asset in only if asset not in files]:
        raise _fail(f"no source annotation for {', '.join(unknown)} under {source}, asset paths look like {next(iter(files), 'Residential/Object/Sofa_B')}")
    if not out.parent.is_dir():
        raise _fail(f"--out {out} is in a missing directory, create {out.parent} first")
    try:
        proposals = (yaml.safe_load(out.read_text()) or {}) if out.is_file() else {}
    except yaml.YAMLError as e:
        raise _fail(f"--out {out} exists but is not YAML ({e}), pass a new file or a proposals file 'propose' wrote") from None
    if not isinstance(proposals, dict):
        raise _fail(f"--out {out} exists but does not map asset paths to sit slots, pass a new file or a proposals file 'propose' wrote")
    try:
        gemini = seats.Gemini(model, thinking.value)
    except RuntimeError as e:
        raise _fail(str(e)) from None

    if only:
        candidates = list(only)
    else:
        sittable: set[str] = set()
        assets = list(files)
        for start in range(0, len(assets), NAMES_PER_REQUEST):
            batch = assets[start : start + NAMES_PER_REQUEST]
            listing = "\n".join(_describe(asset, files[asset]) for asset in batch)
            try:
                answer = gemini.ask([seats.NAMES_PROMPT, listing], seats.NAMES_SCHEMA)
            except Exception as e:
                raise _fail(f"name pass request for objects {start + 1} to {start + len(batch)} of {len(assets)} failed, {e!r}. Check GEMINI_API_KEY, --model and the quota, or pass --only to skip the name pass") from None
            sittable |= seats.parse_sittable(answer, batch)
        candidates = sorted(sittable)
        safe_echo(f"name pass: {len(candidates)} of {len(files)} objects are for sitting", ctx)

    example = seats.load_example()
    example_built = _built(renders, example["asset"])
    if example_built is None:
        safe_echo(f"worked example {example['asset']} has no previews under {renders}, asking without it", ctx)

    proposed: list[str] = []
    missing: list[str] = []
    for asset in candidates:
        built = _built(renders, asset)
        if built is None:
            missing.append(asset)
            continue
        bbox, topdown, thumb = built
        frame = seats.RenderFrame.of(bbox)
        contents: list[object] = [seats.SEATS_PROMPT]
        if example_built is not None and asset != example["asset"]:
            example_bbox, example_topdown, _ = example_built
            contents += ["Example top-down render and its answer:", gemini.image(example_topdown), json.dumps(seats.example_answer(example, example_bbox))]
        width, depth = bbox.max_x - bbox.min_x, bbox.max_y - bbox.min_y
        contents += [f"Object {Path(asset).name}, {width:.2f} m wide and {depth:.2f} m deep. Top-down render, then a perspective view:", gemini.image(topdown), gemini.image(thumb)]
        answers = []
        error: Exception | None = None
        for _ in range(samples):
            try:
                answers.append(seats.parse_seats(gemini.ask(contents, seats.SEATS_SCHEMA), frame, bbox))
            except Exception as e:
                error = e
                safe_echo(f"{asset}: dropped an answer, {e!r}", ctx)
        if not answers:
            safe_echo(_usage(gemini), ctx)
            raise _fail(f"{asset}: all {samples} answers failed, last {error!r}. {len(proposed)} proposals of this run are saved in {out}. Check GEMINI_API_KEY, --model and the quota, then rerun with --only for the remaining assets")
        proposals[asset] = [slot.as_annotation() for slot in seats.consensus(answers)]
        seats.write_atomic(out, seats.dump_yaml(proposals))
        proposed.append(asset)
        safe_echo(f"{asset}: {len(proposals[asset])} slots", ctx)

    safe_echo(_usage(gemini), ctx)
    if missing:
        safe_echo(f"{len(missing)} candidates have no previews under {renders}, build them with -o previews: {', '.join(missing)}", ctx)
    safe_echo(f"wrote {len(proposed)} proposals to {out}, review them, then run: arena-assets semantic seats apply {out} {source}", ctx)


def apply_command(
    ctx: typer.Context,
    proposals: Path = typer.Argument(..., help="Proposals YAML written by 'propose'", exists=True, dir_okay=False),
    source: Path = typer.Argument(..., help="Source dataset root the proposals were made for", exists=True, file_okay=False),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace sit slots an annotation already has, otherwise it is skipped"),
):
    """Write reviewed sit slot proposals into the source annotation.yaml files."""
    data = yaml.safe_load(proposals.read_text()) or {}
    if not isinstance(data, dict):
        raise _fail(f"{proposals} must map asset paths to lists of slots, as 'propose' writes it")
    try:
        written, skipped = seats.apply_proposals(data, source, overwrite=overwrite)
    except ValueError as e:
        raise _fail(f"{e}. Nothing was written, fix {proposals} and rerun") from None
    safe_echo(f"wrote sit slots into {len(written)} annotations, skipped {len(skipped)}" + (f" (existing slots kept, pass --overwrite to replace them, or no source annotation): {', '.join(skipped)}" if skipped else ""), ctx)


def add_to_cmd(semantic_cmd: typer.Typer):
    seats_app = typer.Typer(
        name="seats",
        help="Sit slots of objects proposed by Gemini from top-down renders: 'propose' writes a review file, 'apply' merges it into the source annotations.",
        no_args_is_help=True,
    )
    seats_app.command("propose", no_args_is_help=True)(propose_command)
    seats_app.command("apply", no_args_is_help=True)(apply_command)
    semantic_cmd.add_typer(seats_app, name="seats")
