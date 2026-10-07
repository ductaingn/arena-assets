"""Import command for arena_assets CLI."""

import typer

from .run import add_to_cmd as add_run_cmd


def add_to_cmd(cmd: typer.Typer):
    """Add the import command to the main CLI."""
    add_run_cmd(cmd)
