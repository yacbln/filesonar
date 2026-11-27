"""Walk the configured folders and keep the index in step with what is on disk."""

from __future__ import annotations

import hashlib
import logging
import os
import stat
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .chunking import chunk_text
from .config import Config
from .db import Database
from .extract import (
    Extraction,
    ExtractionError,
    FileTooLarge,
    Segment,
    extract,
    kind_for,
    read_regular_file,
)
from .models import Captioner, Embedder
from .pathrules import is_hidden, within

log = logging.getLogger(__name__)

_HASH_BLOCK = 1 << 20


@dataclass
class IndexReport:
    scanned: int = 0
    added: int = 0
    updated: int = 0
    moved: int = 0
    unchanged: int = 0
    removed: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)
    seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"{self.scanned} files checked in {self.seconds:.1f}s: {self.added} added, "
            f"{self.updated} updated, {self.moved} moved, {self.removed} removed, "
            f"{self.unchanged} unchanged, {self.skipped} skipped, {self.failed} failed"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(_HASH_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def _fail(report: IndexReport, key: str, error: str) -> None:
    report.failed += 1
    report.errors.append((key, error))


_DEPENDENCY_MARKERS = (".dist-info", ".egg-info")


@lru_cache(maxsize=8192)
def is_dependency_dir(folder: str) -> bool:
    """A folder holding ``*.dist-info`` entries is where Python packages were installed
    (``pip install --target``, bundled Lambda code, ...): library files, not the user's."""
    try:
        with os.scandir(folder) as entries:
            return any(
                e.name.endswith(_DEPENDENCY_MARKERS) and e.is_dir(follow_symlinks=False)
                for e in entries
            )
    except OSError:
        return False


class Indexer:
    def __init__(
        self,
        cfg: Config,
        db: Database,
        embedder: Embedder,
        captioner: Captioner | None = None,
        progress: Callable[[str], None] | None = None,
    ):
        self.cfg = cfg
        self.db = db
        self.embedder = embedder
        self.captioner = captioner if cfg.models.caption_images else None
        self.progress = progress or (lambda _msg: None)
        self._vector_ready = False

    # -- eligibility -----------------------------------------------------

    def eligible(self, path: Path) -> bool:
        """True if ``path`` is a regular file this index is allowed to read.

        ``path`` must already be absolute. Symlinks are never followed, so a link
        inside an indexed folder cannot expose a file outside it.
        """
        root = within(path, self.cfg.roots)
        if root is None:
            return False
        if self.cfg.excluded(path, root):
            return False
        if not self.cfg.index.include_hidden and is_hidden(path, root):
            return False
        if kind_for(path) is None:
            return False
        if self._in_dependency_dir(path, root):
            return False
        try:
            st = os.lstat(path)
        except OSError:
            return False
        if not stat.S_ISREG(st.st_mode):
            return False
        # Reject paths whose real location escapes the roots (e.g. a symlinked parent).
        return within(Path(os.path.realpath(path)), self.cfg.roots) is not None

    def _in_dependency_dir(self, path: Path, root: Path) -> bool:
        if not self.cfg.index.skip_dependency_folders:
            return False
        folder = path.parent
        while folder != root and folder.is_relative_to(root):
            if is_dependency_dir(str(folder)):
                return True
            folder = folder.parent
        return False

    def iter_candidates(self, start: Path | None = None) -> Iterator[Path]:
        for top in (start,) if start is not None else self.cfg.roots:
            root = within(top, self.cfg.roots)
            if root is None or not top.is_dir():
                log.warning("indexed folder does not exist: %s", top)
                continue
            for dirpath, dirnames, filenames in os.walk(top, followlinks=False):
                here = Path(dirpath)
                if (
                    self.cfg.index.skip_dependency_folders
                    and here != root
                    and is_dependency_dir(dirpath)
                ):
                    dirnames[:] = []
                    continue
                kept = []
                for d in dirnames:
                    sub = here / d
                    if not self.cfg.index.include_hidden and d.startswith("."):
                        continue
                    # Test the folder itself and a notional child, so "x/**" prunes "x".
                    if self.cfg.excluded(sub, root) or self.cfg.excluded(sub / "_", root):
                        continue
                    kept.append(d)
                dirnames[:] = sorted(kept)
                for name in sorted(filenames):
                    yield here / name

    # -- indexing --------------------------------------------------------

    def _needs_caption(self, row) -> bool:
        return (
            row is not None
            and row["kind"] == "image"
            and not row["captioned"]
            and self.captioner is not None
            and self.captioner.available()
        )

    def _find_moved(self, sha: str, path: str):
        """A record with the same content whose file no longer exists is a move."""
        for row in self.db.conn.execute(
            "SELECT * FROM files WHERE sha256 = ? AND path != ? AND origin = 'local'",
            (sha, path),
        ):
            if not os.path.lexists(row["path"]):
                return row
        return None

    def _copy_from_twin(self, path: Path, key: str, sha: str, md5: str, st, row, report) -> bool:
        """Reuse the index entry of an identical file (e.g. "report (1).pdf", or the same
        photo in Google Drive) instead of reading, captioning and embedding it again."""
        twin = self.db.conn.execute(
            "SELECT * FROM files WHERE (sha256 = ? OR md5 = ?) AND path != ? "
            "AND status != 'error' ORDER BY captioned DESC, origin = 'local' DESC LIMIT 1",
            (sha, md5, key),
        ).fetchone()
        if twin is None or self._needs_caption(twin) or not self._vector_ready_for_model():
            return False
        chunks, vectors = self.db.stored_chunks(twin["id"])
        if not chunks:
            return False
        self.db.replace_file(
            path=key,
            name=path.name,
            kind=twin["kind"],
            size=st.st_size,
            mtime=st.st_mtime,
            sha256=sha,
            captioned=bool(twin["captioned"]),
            status=twin["status"],
            error=None,
            details=twin["details"],
            indexed_at=time.time(),
            chunks=chunks,
            vectors=vectors,
            md5=md5,
        )
        self.progress(f"copied  {key} (same content as {twin['path']})")
        if row is None:
            report.added += 1
        else:
            report.updated += 1
        return True

    def _vector_ready_for_model(self) -> bool:
        """Stored vectors may be reused only if they came from the configured model."""
        return self.db.has_vectors() and self.db.get_meta("embedding_model") == self.embedder.name

    def _ensure_vectors(self, sample: list[float]) -> None:
        if not self._vector_ready:
            self.db.ensure_vector_table(self.embedder.name, len(sample))
            self._vector_ready = True

    def _chunks_for(self, path: Path, extraction: Extraction) -> list[tuple[int | None, str, str]]:
        ix = self.cfg.index
        out: list[tuple[int | None, str, str]] = []
        meta: list[tuple[int | None, str, str]] = []
        for seg in extraction.segments:
            target = meta if seg.source == "metadata" else out
            for piece in chunk_text(seg.text, ix.chunk_chars, ix.chunk_overlap):
                target.append((seg.page, seg.source, piece))
        if len(out) > ix.max_chunks_per_file:
            # Very long files (books, data exports, logs) would otherwise crowd out
            # everything else in search results; their beginning is kept.
            log.info(
                "%s: indexed the first %d of %d chunks", path, ix.max_chunks_per_file, len(out)
            )
            out = out[: ix.max_chunks_per_file]
        out.extend(meta)
        if not out:
            # Keep the file findable by its name even when it holds no readable text.
            out.append((None, "metadata", path.stem.replace("_", " ").replace("-", " ")))
        return out

    def index_file(self, path: Path, report: IndexReport) -> None:
        report.scanned += 1
        key = str(path)
        max_bytes = int(self.cfg.index.max_file_size_mb * 1024 * 1024)
        try:
            st = os.lstat(path)
        except OSError as exc:
            _fail(report, key, str(exc))
            return
        if not stat.S_ISREG(st.st_mode) or st.st_size > max_bytes:
            report.skipped += 1
            return

        row = self.db.get_file(key)
        if (
            row is not None
            and row["size"] == st.st_size
            and row["mtime"] == st.st_mtime
            and not self._needs_caption(row)
            and row["md5"] is not None
        ):
            report.unchanged += 1
            return

        # From here on only these bytes are used: hashed, parsed and indexed.
        try:
            data, st = read_regular_file(path, max_bytes)
        except FileTooLarge:
            report.skipped += 1
            return
        except (OSError, ExtractionError) as exc:
            _fail(report, key, str(exc))
            return
        # The parent folders may have changed since the walk; check where we really are.
        if within(Path(os.path.realpath(path)), self.cfg.roots) is None:
            report.skipped += 1
            return
        sha = hashlib.sha256(data).hexdigest()
        # Drive reports MD5 checksums; storing ours lets the same file be recognised
        # in both places.
        md5 = hashlib.md5(data, usedforsecurity=False).hexdigest()

        if row is not None and row["sha256"] == sha and not self._needs_caption(row):
            self.db.touch_file(row["id"], key, path.name, st.st_mtime)
            if row["md5"] != md5:
                self.db.set_md5(row["id"], md5)
            report.unchanged += 1
            return

        if row is None:
            moved = self._find_moved(sha, key)
            if moved is not None and not (
                moved["kind"] == "image" and not moved["captioned"] and self._needs_caption(moved)
            ):
                self.db.touch_file(moved["id"], key, path.name, st.st_mtime)
                if moved["md5"] != md5:
                    self.db.set_md5(moved["id"], md5)
                report.moved += 1
                self.progress(f"moved   {key}")
                return

        started = time.monotonic()
        if self._copy_from_twin(path, key, sha, md5, st, row, report):
            return
        status, error = "ok", None
        try:
            extraction = extract(path, self.cfg.index, self.captioner, data=data)
        except ExtractionError as exc:
            status, error = "error", str(exc)[:500]
            extraction = Extraction(kind=kind_for(path) or "text")
            _fail(report, key, error)
        if status == "ok" and not any(
            s.text.strip() for s in extraction.segments if s.source != "metadata"
        ):
            status = "empty"

        chunks = self._chunks_for(path, extraction)
        # Give the embedding model the file name as context for every chunk.
        texts = [f"{path.name}\n{text}" for _page, _src, text in chunks]
        vectors = self.embedder.embed_documents(texts)
        self._ensure_vectors(vectors[0])
        self.db.replace_file(
            path=key,
            name=path.name,
            kind=extraction.kind,
            size=st.st_size,
            mtime=st.st_mtime,
            sha256=sha,
            captioned=extraction.captioned,
            status=status,
            error=error,
            details=extraction.details,
            indexed_at=time.time(),
            chunks=chunks,
            vectors=vectors,
            md5=md5,
        )
        self.progress(f"indexed {key} ({time.monotonic() - started:.1f}s)")
        if status != "error":
            if row is None:
                report.added += 1
            else:
                report.updated += 1

    def remove_missing(self, seen: set[str], report: IndexReport) -> None:
        for key in self.db.all_paths():
            if key in seen:
                continue
            path = Path(key)
            if not os.path.lexists(path) or not self.eligible(path):
                self.db.delete_path(key)
                report.removed += 1

    def run(self) -> IndexReport:
        """Bring the whole index up to date."""
        started = time.monotonic()
        report = IndexReport()
        seen: set[str] = set()
        is_dependency_dir.cache_clear()
        for path in self.iter_candidates():
            if not self.eligible(path):
                continue
            seen.add(str(path))
            self.index_file(path, report)
        self.remove_missing(seen, report)
        report.seconds = time.monotonic() - started
        return report

    def update_paths(self, paths: Iterable[Path]) -> IndexReport:
        """Re-check specific paths (used by the folder watcher)."""
        started = time.monotonic()
        report = IndexReport()
        present: list[Path] = []
        is_dependency_dir.cache_clear()
        gone: list[Path] = []
        for raw in sorted({Path(os.path.abspath(p)) for p in paths}):
            if os.path.isdir(raw) and not os.path.islink(raw):
                present.extend(self._walk_dir(raw))
            elif os.path.lexists(raw):
                present.append(raw)
            else:
                gone.append(raw)
        # Index what exists first, so a moved file is recognised by its old record
        # before that record is cleaned up.
        for path in dict.fromkeys(present):
            if self.eligible(path):
                self.index_file(path, report)
            elif self.db.delete_path(str(path)):
                report.removed += 1  # e.g. a file that is now excluded
        for path in gone:
            key = str(path)
            if self.db.delete_path(key):
                report.removed += 1
                continue
            # A folder was removed or renamed: drop everything that lived inside it.
            prefix = key.rstrip("/") + "/"
            for stored in self.db.all_paths():
                if stored.startswith(prefix) and not os.path.lexists(stored):
                    self.db.delete_path(stored)
                    report.removed += 1
        report.seconds = time.monotonic() - started
        return report

    def _walk_dir(self, folder: Path) -> list[Path]:
        if within(folder, self.cfg.roots) is None:
            return []
        out = []
        for candidate in self.iter_candidates(start=folder):
            out.append(candidate)
        return out


__all__ = ["IndexReport", "Indexer", "Segment", "sha256_file"]
