"""SQLite storage: file records, text chunks, a full-text index and a vector index."""

from __future__ import annotations

import os
import sqlite3
import struct
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import sqlite_vec

SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    id          INTEGER PRIMARY KEY,
    path        TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,           -- document | pdf | image | text
    size        INTEGER NOT NULL,
    mtime       REAL NOT NULL,
    sha256      TEXT NOT NULL,
    captioned   INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL,           -- ok | empty | error
    error       TEXT,
    details     TEXT,                    -- short human-readable metadata summary
    indexed_at  REAL NOT NULL,
    origin      TEXT NOT NULL DEFAULT 'local',  -- local | drive
    remote_id   TEXT,                    -- Drive file id
    link        TEXT,                    -- Drive web link (private to the owner)
    location    TEXT,                    -- Drive folder, e.g. "My Drive/Taxes"
    md5         TEXT,                    -- what Drive reports; compared with local files
    ocr         INTEGER NOT NULL DEFAULT 1  -- 0: a scan or photo read while OCR was missing
);
CREATE INDEX IF NOT EXISTS files_sha ON files(sha256);
CREATE INDEX IF NOT EXISTS files_kind ON files(kind);

CREATE TABLE IF NOT EXISTS chunks (
    id       INTEGER PRIMARY KEY,
    file_id  INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    ordinal  INTEGER NOT NULL,
    page     INTEGER,
    source   TEXT NOT NULL,              -- text | ocr | caption | metadata
    text     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_file ON chunks(file_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, name,
    tokenize='unicode61 remove_diacritics 2'
);
"""


# Added in schema 2 (Google Drive). Columns are appended so older indexes keep their data.
_V2_COLUMNS = (
    ("origin", "TEXT NOT NULL DEFAULT 'local'"),
    ("remote_id", "TEXT"),
    ("link", "TEXT"),
    ("location", "TEXT"),
    ("md5", "TEXT"),
)
# Version 3: remember files read without OCR, to read them again once Tesseract is
# installed. On upgrade, only scans and photos that yielded no text at all are marked:
# OCR is all they can gain, and there are few of them.
_V3_COLUMNS = (("ocr", "INTEGER NOT NULL DEFAULT 1"),)
_V2_INDEXES = """
CREATE INDEX IF NOT EXISTS files_md5 ON files(md5);
CREATE INDEX IF NOT EXISTS files_origin ON files(origin);
CREATE UNIQUE INDEX IF NOT EXISTS files_remote ON files(remote_id) WHERE remote_id IS NOT NULL;
"""

FILE_COLUMNS = (
    "path",
    "name",
    "kind",
    "size",
    "mtime",
    "sha256",
    "captioned",
    "status",
    "error",
    "details",
    "indexed_at",
    "origin",
    "remote_id",
    "link",
    "location",
    "md5",
    "ocr",
)


def _serialize(vector: Sequence[float] | bytes) -> bytes:
    if isinstance(vector, bytes):
        return vector  # already in sqlite-vec's float32 format
    return struct.pack(f"{len(vector)}f", *vector)


class Database:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
        new = not path.exists()
        if new:
            # Create the file with private permissions before SQLite opens it.
            fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
            os.close(fd)
        self.conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.executescript(_SCHEMA)
        for suffix in ("-wal", "-shm"):
            side = Path(str(path) + suffix)
            if side.exists():
                os.chmod(side, 0o600)
        current = self.get_meta("schema_version")
        if current == "1":
            self._add_columns(_V2_COLUMNS, 2)
            current = "2"
        if current == "2":
            self._add_columns(_V3_COLUMNS, 3)
            with self.conn:
                self.conn.execute(
                    "UPDATE files SET ocr = 0 WHERE kind IN ('image', 'pdf') AND status = 'empty'"
                )
            current = "3"
        self.conn.executescript(_V2_INDEXES)
        if current is None:
            self.set_meta("schema_version", str(SCHEMA_VERSION))
        elif int(current) != SCHEMA_VERSION:
            raise RuntimeError(
                f"index at {path} uses schema {current}, this version expects "
                f"{SCHEMA_VERSION}; delete it and run `filesonar index` again"
            )

    def _add_columns(self, columns: tuple[tuple[str, str], ...], version: int) -> None:
        have = {r["name"] for r in self.conn.execute("PRAGMA table_info(files)")}
        with self.conn:
            for name, decl in columns:
                if name not in have:
                    # Names and declarations are the constants above, never input.
                    self.conn.execute(f"ALTER TABLE files ADD COLUMN {name} {decl}")
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(version),),
            )

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.conn:
            yield self.conn

    # -- meta -------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # -- vectors ----------------------------------------------------------

    def ensure_vector_table(self, model: str, dim: int) -> None:
        """Create the vector table, or refuse if the embedding model has changed."""
        stored_model = self.get_meta("embedding_model")
        stored_dim = self.get_meta("embedding_dim")
        if stored_model is not None and (stored_model != model or int(stored_dim) != dim):
            raise RuntimeError(
                f"the index was built with {stored_model} ({stored_dim} dims) but the "
                f"config now uses {model} ({dim} dims). Run `filesonar index --rebuild`."
            )
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0("
            f"embedding float[{int(dim)}] distance_metric=cosine)"
        )
        self.set_meta("embedding_model", model)
        self.set_meta("embedding_dim", str(dim))

    def has_vectors(self) -> bool:
        row = self.conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'chunks_vec'").fetchone()
        return row is not None

    def reset(self) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM chunks_fts")
            if self.has_vectors():
                self.conn.execute("DROP TABLE chunks_vec")
            self.conn.execute("DELETE FROM chunks")
            self.conn.execute("DELETE FROM files")
            self.conn.execute("DELETE FROM meta WHERE key IN ('embedding_model','embedding_dim')")

    # -- files ------------------------------------------------------------

    def get_file(self, path: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()

    def all_paths(self, origin: str = "local") -> list[str]:
        return [
            r["path"]
            for r in self.conn.execute("SELECT path FROM files WHERE origin = ?", (origin,))
        ]

    def get_remote(self, remote_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM files WHERE remote_id = ?", (remote_id,)).fetchone()

    def find_by_md5(self, md5: str, origin: str | None = None) -> list[sqlite3.Row]:
        if origin is None:
            return self.conn.execute(
                "SELECT * FROM files WHERE md5 = ? ORDER BY origin = 'local' DESC, id", (md5,)
            ).fetchall()
        return self.conn.execute(
            "SELECT * FROM files WHERE md5 = ? AND origin = ? ORDER BY id", (md5, origin)
        ).fetchall()

    def set_md5(self, file_id: int, md5: str) -> None:
        with self.conn:
            self.conn.execute("UPDATE files SET md5 = ? WHERE id = ?", (md5, file_id))

    def update_remote_meta(
        self, file_id: int, *, name: str, location: str | None, link: str | None, mtime: float
    ) -> None:
        """A Drive file was renamed or moved, or its link changed: no re-read needed."""
        with self.conn:
            self.conn.execute(
                "UPDATE files SET name = ?, location = ?, link = ?, mtime = ? WHERE id = ?",
                (name, location, link, mtime, file_id),
            )
            self.conn.execute(
                "UPDATE chunks_fts SET name = ? WHERE rowid IN "
                "(SELECT id FROM chunks WHERE file_id = ?)",
                (name, file_id),
            )

    def delete_file(self, conn: sqlite3.Connection, file_id: int) -> None:
        ids = [r[0] for r in conn.execute("SELECT id FROM chunks WHERE file_id = ?", (file_id,))]
        for cid in ids:
            conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (cid,))
            if self.has_vectors():
                conn.execute("DELETE FROM chunks_vec WHERE rowid = ?", (cid,))
        conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        conn.execute("DELETE FROM files WHERE id = ?", (file_id,))

    def delete_path(self, path: str) -> bool:
        row = self.get_file(path)
        if row is None:
            return False
        with self.transaction() as conn:
            self.delete_file(conn, row["id"])
        return True

    def touch_file(self, file_id: int, path: str, name: str, mtime: float) -> None:
        """Record a move or a metadata-only change without re-reading the file."""
        with self.conn:
            self.conn.execute(
                "UPDATE files SET path = ?, name = ?, mtime = ? WHERE id = ?",
                (path, name, mtime, file_id),
            )
            self.conn.execute(
                "UPDATE chunks_fts SET name = ? WHERE rowid IN "
                "(SELECT id FROM chunks WHERE file_id = ?)",
                (name, file_id),
            )

    def replace_file(
        self,
        *,
        path: str,
        name: str,
        kind: str,
        size: int,
        mtime: float,
        sha256: str,
        captioned: bool,
        status: str,
        error: str | None,
        details: str | None,
        indexed_at: float,
        chunks: Sequence[tuple[int | None, str, str]],
        vectors: Sequence[Sequence[float] | bytes],
        origin: str = "local",
        remote_id: str | None = None,
        link: str | None = None,
        location: str | None = None,
        md5: str | None = None,
        ocr: bool = True,
    ) -> int:
        """Atomically replace everything stored for one file."""
        if len(chunks) != len(vectors):
            raise ValueError("every chunk needs exactly one vector")
        with self.transaction() as conn:
            existing = conn.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()
            if existing is not None:
                self.delete_file(conn, existing["id"])
            values = (
                path, name, kind, size, mtime, sha256, int(captioned), status, error,
                details, indexed_at, origin, remote_id, link, location, md5, int(ocr),
            )  # fmt: skip
            # Column names come from the FILE_COLUMNS constant; values are bound.
            cur = conn.execute(
                f"INSERT INTO files({', '.join(FILE_COLUMNS)}) "  # noqa: S608
                f"VALUES ({','.join('?' * len(FILE_COLUMNS))})",
                values,
            )
            file_id = cur.lastrowid
            for ordinal, ((page, source, text), vector) in enumerate(
                zip(chunks, vectors, strict=True)
            ):
                cur = conn.execute(
                    "INSERT INTO chunks(file_id, ordinal, page, source, text) VALUES (?,?,?,?,?)",
                    (file_id, ordinal, page, source, text),
                )
                cid = cur.lastrowid
                conn.execute(
                    "INSERT INTO chunks_fts(rowid, text, name) VALUES (?, ?, ?)",
                    (cid, text, name),
                )
                conn.execute(
                    "INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
                    (cid, _serialize(vector)),
                )
        return file_id

    # -- search primitives ------------------------------------------------

    def fts_search(self, match: str, limit: int) -> list[tuple[int, float]]:
        rows = self.conn.execute(
            "SELECT rowid, bm25(chunks_fts, 1.0, 2.0) AS score FROM chunks_fts "
            "WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
            (match, limit),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def vector_search(self, vector: Sequence[float], limit: int) -> list[tuple[int, float]]:
        if not self.has_vectors():
            return []
        rows = self.conn.execute(
            "SELECT rowid, distance FROM chunks_vec WHERE embedding MATCH ? AND k = ? "
            "ORDER BY distance",
            (_serialize(vector), limit),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def chunks_with_files(self, chunk_ids: Sequence[int]) -> list[sqlite3.Row]:
        if not chunk_ids:
            return []
        marks = ",".join("?" * len(chunk_ids))
        # Only "?" placeholders are formatted in; the values are bound parameters.
        return self.conn.execute(
            f"SELECT c.id AS chunk_id, c.page, c.source, c.text, f.* FROM chunks c "  # noqa: S608
            f"JOIN files f ON f.id = c.file_id WHERE c.id IN ({marks})",
            list(chunk_ids),
        ).fetchall()

    def stored_chunks(self, file_id: int) -> tuple[list[tuple[int | None, str, str]], list[bytes]]:
        """Chunks and their raw vectors for one file, in order (to reuse for a copy)."""
        rows = self.conn.execute(
            "SELECT c.page, c.source, c.text, v.embedding FROM chunks c "
            "JOIN chunks_vec v ON v.rowid = c.id WHERE c.file_id = ? ORDER BY c.ordinal",
            (file_id,),
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows], [r[3] for r in rows]

    def copies_of(
        self, sha256: str, path: str, md5: str | None = None, limit: int = 5
    ) -> list[str]:
        """Other places the same content is stored: local paths, or "Google Drive: ..."."""
        rows = self.conn.execute(
            "SELECT path, name, origin, location FROM files "
            "WHERE (sha256 = ? OR (md5 IS NOT NULL AND md5 = ?)) AND path != ? "
            "ORDER BY origin = 'local' DESC, path LIMIT ?",
            (sha256, md5, path, limit),
        )
        return [
            f"Google Drive: {r['location']}/{r['name']}" if r["origin"] == "drive" else r["path"]
            for r in rows
        ]

    def stats(self) -> dict:
        drive = self.conn.execute("SELECT COUNT(*) FROM files WHERE origin = 'drive'").fetchone()[0]
        row = self.conn.execute(
            "SELECT COUNT(*) AS files, "
            "SUM(status = 'ok') AS ok, SUM(status = 'empty') AS empty, "
            "SUM(status = 'error') AS errors, "
            "SUM(kind = 'image') AS images, SUM(kind = 'image' AND captioned) AS captioned, "
            "MAX(indexed_at) AS last_indexed FROM files"
        ).fetchone()
        chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        by_kind = {
            r["kind"]: r["n"]
            for r in self.conn.execute("SELECT kind, COUNT(*) AS n FROM files GROUP BY kind")
        }
        return {
            "files": row["files"] or 0,
            "ok": row["ok"] or 0,
            "empty": row["empty"] or 0,
            "errors": row["errors"] or 0,
            "images": row["images"] or 0,
            "images_captioned": row["captioned"] or 0,
            "chunks": chunks,
            "by_kind": by_kind,
            "drive_files": drive,
            "last_indexed": row["last_indexed"],
            "embedding_model": self.get_meta("embedding_model"),
        }
