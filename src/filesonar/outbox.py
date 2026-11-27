"""Copies of indexed files, staged for an agent to attach to a chat message.

Chat agents such as OpenClaw may only read their own folders when viewing or sending a
file. Instead of the agent running ``cp`` (a shell command that could write anywhere),
filesonar makes the copy itself, under strict rules:

* only files that are in the index and still allowed by the configuration are copied;
* the source is read once, without following symlinks;
* copies go only into ``outbox_dir``, one sub-folder per file content, and are private
  (0700 folders, 0600 files);
* nothing outside ``outbox_dir`` is ever written or deleted, and copies older than
  ``KEEP_HOURS`` are removed.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

KEEP_HOURS = 24.0
_ENTRY = re.compile(r"[0-9a-f]{16}")
_UNSAFE_NAME = re.compile(r"[\x00-\x1f\x7f/\\]")


class OutboxError(Exception):
    pass


@dataclass(frozen=True)
class Staged:
    path: Path
    reused: bool


def safe_name(name: str) -> str:
    """The file name to use for the copy: no folders, control characters or dot names."""
    cleaned = _UNSAFE_NAME.sub("_", name).strip()
    if cleaned in ("", ".", ".."):
        cleaned = "file"
    # Keep the extension (chat apps use it to pick a preview); trim the stem instead.
    if len(cleaned.encode()) > 200:
        stem, dot, ext = cleaned.rpartition(".")
        if not dot or len(ext) > 10:
            stem, ext = cleaned, ""
        stem = stem.encode()[: 200 - len(ext) - 1].decode(errors="ignore")
        cleaned = f"{stem}.{ext}" if ext else stem
    return cleaned


def _private_dir(path: Path) -> None:
    """Create ``path`` (mode 0700) and make sure it is a real folder, not a symlink."""
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode):
        raise OutboxError(f"{path} is not a folder")
    if st.st_uid != os.getuid():
        raise OutboxError(f"{path} belongs to another user")


def stage(outbox: Path, name: str, sha256: str, data: bytes) -> Staged:
    """Write ``data`` as ``outbox/<sha256[:16]>/<name>`` and return where it is."""
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise OutboxError("invalid content hash")
    _private_dir(outbox)
    entry = outbox / sha256[:16]
    _private_dir(entry)
    target = entry / safe_name(name)
    try:
        st = os.lstat(target)
    except FileNotFoundError:
        st = None
    if st is not None and stat.S_ISREG(st.st_mode) and st.st_size == len(data):
        os.utime(entry)  # restart the expiry clock
        return Staged(target, reused=True)
    fd, tmp = tempfile.mkstemp(dir=entry, prefix=".part-")
    try:
        with os.fdopen(fd, "wb") as fh:
            os.fchmod(fh.fileno(), 0o600)
            fh.write(data)
        os.replace(tmp, target)  # replaces a name, never writes through a symlink
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    return Staged(target, reused=False)


def clean(outbox: Path, keep_hours: float = KEEP_HOURS, now: float | None = None) -> int:
    """Delete staged copies older than ``keep_hours``. Only entries filesonar created
    (folders named by a content hash) are touched; symlinks are never followed."""
    now = time.time() if now is None else now
    try:
        entries = list(os.scandir(outbox))
    except FileNotFoundError:
        return 0
    removed = 0
    for entry in entries:
        if not _ENTRY.fullmatch(entry.name) or not entry.is_dir(follow_symlinks=False):
            continue
        if now - entry.stat(follow_symlinks=False).st_mtime < keep_hours * 3600:
            continue
        shutil.rmtree(entry.path, ignore_errors=True)  # rmtree does not follow symlinks
        removed += 1
    return removed
