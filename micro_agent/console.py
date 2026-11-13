"""Progress output. Everything goes to stderr so stdout only carries the final answer."""

import sys


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def truncate(text: str, limit: int = 300) -> str:
    text = str(text).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"
