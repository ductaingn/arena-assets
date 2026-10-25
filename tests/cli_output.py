import re

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    """CLI output without color codes and panel borders, whitespace collapsed to single spaces."""
    return " ".join(_ANSI.sub("", text).replace("\u2502", " ").split())
