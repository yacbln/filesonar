"""Decide which paths may be indexed.

Glob patterns follow the familiar gitignore-like rules:

* ``*`` matches anything except ``/``
* ``?`` matches one character except ``/``
* ``**/`` matches zero or more whole folders
* a trailing ``/**`` matches everything inside a folder
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=512)
def _compile(pattern: str) -> re.Pattern[str]:
    out = []
    i = 0
    n = len(pattern)
    while i < n:
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("/**", i) and i + 3 == n:
            out.append("/.*")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    body = "".join(out)
    # Relative patterns (the usual case) may match at any depth.
    if not pattern.startswith("/") and not pattern.startswith("**"):
        body = "(?:.*/)?" + body
    return re.compile(rf"^{body}$", re.IGNORECASE)


def matches_any(path: Path | str, patterns: Iterable[str]) -> bool:
    text = Path(path).as_posix()
    return any(_compile(p).match(text) for p in patterns)


def is_hidden(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    return any(part.startswith(".") for part in rel.parts)


def within(path: Path, roots: Iterable[Path]) -> Path | None:
    """Return the root that contains ``path`` (both already resolved), if any."""
    for root in roots:
        if path == root or path.is_relative_to(root):
            return root
    return None
