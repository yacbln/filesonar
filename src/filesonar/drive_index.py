"""Index Google Drive files and keep them in step using Drive's change feed.

The first sync lists every file once. After that only the Changes API is asked "what
changed since last time?", which costs one request when nothing happened. Files whose
MD5 matches something already indexed (a photo that also lives on this computer, a PDF
uploaded twice) reuse that entry and are never downloaded.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path, PurePosixPath

from .config import ALWAYS_EXCLUDED, DEFAULT_EXCLUDE, Config
from .db import Database
from .drive import FOLDER_MIME, DriveClient, DriveError
from .extract import Extraction, ExtractionError, Segment, extract, kind_for
from .indexer import Indexer, IndexReport, _fail
from .models import Captioner, Embedder
from .pathrules import matches_any

log = logging.getLogger(__name__)

KEY_PREFIX = "gdrive:"
MY_DRIVE = "My Drive"
SHARED = "Shared with me"
PREVIEW_SIDE = 1600

# Google's own formats have no file to download; they are exported.
GOOGLE_EXPORTS = {
    "application/vnd.google-apps.document": ("text/plain", "document"),
    "application/vnd.google-apps.presentation": ("text/plain", "document"),
    "application/vnd.google-apps.spreadsheet": ("text/csv", "text"),
}
# What to send when the user asks for a Google Docs file itself.
GOOGLE_SEND_AS = {
    "application/vnd.google-apps.document": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.presentation": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
}
# Drive export is capped by Google at 10 MB.
_EXPORT_LIMIT = 10 * 1024 * 1024


def drive_key(file_id: str) -> str:
    return KEY_PREFIX + file_id


def _parse_time(value: str | None) -> float:
    if not value:
        return 0.0
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


class Folders:
    """Folder ids to readable paths ("My Drive/Taxes/2024")."""

    def __init__(self, root_id: str, folders: Iterable[dict]):
        self.root_id = root_id
        self.by_id = {f["id"]: f for f in folders}
        self._cache: dict[str, str] = {}

    def path_of(self, folder_id: str | None, _depth: int = 0) -> str:
        if folder_id is None:
            return SHARED
        if folder_id == self.root_id:
            return MY_DRIVE
        if folder_id in self._cache:
            return self._cache[folder_id]
        folder = self.by_id.get(folder_id)
        if folder is None or _depth > 64:  # not ours (shared), or a cycle
            return SHARED
        parents = folder.get("parents") or [None]
        path = f"{self.path_of(parents[0], _depth + 1)}/{_clean(folder['name'])}"
        self._cache[folder_id] = path
        return path

    def location_of(self, file: dict) -> str:
        parents = file.get("parents") or [None]
        return self.path_of(parents[0])


def _clean(name: str) -> str:
    return name.replace("/", "_").strip() or "untitled"


class DriveIndexer:
    def __init__(
        self,
        cfg: Config,
        db: Database,
        embedder: Embedder,
        client: DriveClient,
        captioner: Captioner | None = None,
        progress: Callable[[str], None] | None = None,
    ):
        self.cfg = cfg
        self.db = db
        self.client = client
        # Shares chunking, embedding and caption rules with local indexing.
        self.local = Indexer(cfg, db, embedder, captioner, progress)
        self.progress = self.local.progress
        self._folders: Folders | None = None

    # -- scope -----------------------------------------------------------

    def _query(self) -> str:
        query = f"trashed = false and mimeType != '{FOLDER_MIME}'"
        if not self.cfg.drive.include_shared:
            query += " and 'me' in owners"
        return query

    def _load_folders(self) -> Folders:
        root = self.client.get_file("root")["id"]
        query = f"mimeType = '{FOLDER_MIME}' and trashed = false"
        folders = list(self.client.list_files(query, fields="id,name,parents"))
        self._folders = Folders(root, folders)
        return self._folders

    def wanted(self, file: dict, location: str) -> str | None:
        """The kind to index ``file`` as, or None if it is out of scope."""
        d, ix = self.cfg.drive, self.cfg.index
        if file.get("trashed") or file.get("mimeType") == FOLDER_MIME:
            return None
        if not d.include_shared and file.get("ownedByMe") is False:
            return None
        name = file.get("name", "")
        rel = f"{location}/{_clean(name)}"
        inside = rel.removeprefix(MY_DRIVE + "/") if rel.startswith(MY_DRIVE + "/") else rel
        if d.folders and not any(
            inside == f.strip("/") or inside.startswith(f.strip("/") + "/") for f in d.folders
        ):
            return None
        patterns = ALWAYS_EXCLUDED + d.exclude + (DEFAULT_EXCLUDE if ix.default_excludes else ())
        if matches_any("/" + inside, patterns):
            return None
        if not ix.include_hidden and any(p.startswith(".") for p in inside.split("/")):
            return None
        mime = file.get("mimeType", "")
        if mime in GOOGLE_EXPORTS:
            return GOOGLE_EXPORTS[mime][1]
        kind = kind_for(Path(name))
        if kind is None:
            return None
        size = int(file.get("size") or 0)
        if size > ix.max_file_size_mb * 1024 * 1024:
            return None
        return kind

    # -- sync ------------------------------------------------------------

    def sync(self, full: bool = False) -> IndexReport:
        token = self.db.get_meta("drive_page_token")
        if full or token is None:
            return self.full_scan()
        started = time.monotonic()
        report = IndexReport()
        changes, new_token = self.client.changes(token)
        if any(self._may_be_folder_change(c) for c in changes):
            # A folder was renamed, moved or deleted: locations of many files may have
            # changed. Listing is cheap (no downloads for unchanged files).
            return self.full_scan()
        folders = self._folders or self._load_folders()
        for change in changes:
            file = change.get("file")
            file_id = change.get("fileId")
            if change.get("removed") or file is None:
                self._remove(file_id, report)
                continue
            report.scanned += 1
            if self.wanted(file, folders.location_of(file)) is None:
                self._remove(file["id"], report)
                continue
            self.index_file(file, folders, report, counted=True)
        self.db.set_meta("drive_page_token", new_token)
        report.seconds = time.monotonic() - started
        return report

    def full_scan(self) -> IndexReport:
        started = time.monotonic()
        report = IndexReport()
        # Taken before listing, so nothing that changes during the listing is missed.
        token = self.client.start_page_token()
        folders = self._load_folders()
        seen: set[str] = set()
        for file in self.client.list_files(self._query()):
            if self.wanted(file, folders.location_of(file)) is None:
                continue
            seen.add(drive_key(file["id"]))
            self.index_file(file, folders, report)
        for key in self.db.all_paths("drive"):
            if key not in seen:
                self.db.delete_path(key)
                report.removed += 1
        self.db.set_meta("drive_page_token", token)
        self.db.set_meta("drive_last_full", str(time.time()))
        report.seconds = time.monotonic() - started
        return report

    def _may_be_folder_change(self, change: dict) -> bool:
        file = change.get("file")
        if file is not None:
            return file.get("mimeType") == FOLDER_MIME
        # A removed entry carries no details; anything we did not index might be a folder.
        file_id = change.get("fileId")
        return bool(file_id) and self.db.get_remote(file_id) is None

    def _remove(self, file_id: str | None, report: IndexReport) -> None:
        if file_id and self.db.delete_path(drive_key(file_id)):
            report.removed += 1

    # -- one file --------------------------------------------------------

    def index_file(
        self, file: dict, folders: Folders, report: IndexReport, counted: bool = False
    ) -> None:
        if not counted:
            report.scanned += 1
        location = folders.location_of(file)
        kind = self.wanted(file, location)
        if kind is None:
            report.skipped += 1
            return
        file_id, name = file["id"], file["name"]
        key = drive_key(file_id)
        md5 = file.get("md5Checksum")
        mtime = _parse_time(file.get("modifiedTime"))
        link = file.get("webViewLink")
        row = self.db.get_file(key)

        if row is not None and not self.local._needs_caption(row):
            same = row["md5"] == md5 if md5 else row["mtime"] == mtime
            if same and row["status"] != "error":
                if (row["name"], row["location"], row["link"], row["mtime"]) != (
                    name,
                    location,
                    link,
                    mtime,
                ):
                    self.db.update_remote_meta(
                        row["id"], name=name, location=location, link=link, mtime=mtime
                    )
                    report.moved += 1
                else:
                    report.unchanged += 1
                return

        meta = dict(
            path=key,
            name=name,
            size=int(file.get("size") or 0),
            mtime=mtime,
            origin="drive",
            remote_id=file_id,
            link=link,
            location=location,
            md5=md5,
        )
        if md5 and self._copy_from_twin(md5, key, meta, row, report):
            return

        started = time.monotonic()
        try:
            data, extraction = self._fetch_and_extract(file, kind)
        except DriveError as exc:
            # Not stored: the next full sync tries again.
            _fail(report, f"{location}/{name}", str(exc))
            return
        status, error = "ok", None
        if extraction is None:
            status, error = "error", "could not be read"
            extraction = Extraction(kind=kind)
        elif not any(s.text.strip() for s in extraction.segments if s.source != "metadata"):
            status = "empty"

        chunks = self.local._chunks_for(Path(name), extraction)
        texts = [f"{name}\n{text}" for _page, _src, text in chunks]
        vectors = self.local.embedder.embed_documents(texts)
        self.local._ensure_vectors(vectors[0])
        self.db.replace_file(
            **{**meta, "size": meta["size"] or len(data)},
            kind=kind,
            sha256=hashlib.sha256(data).hexdigest(),
            captioned=extraction.captioned,
            status=status,
            error=error,
            details=extraction.details,
            indexed_at=time.time(),
            chunks=chunks,
            vectors=vectors,
        )
        self.progress(f"indexed {location}/{name} ({time.monotonic() - started:.1f}s)")
        if status == "error":
            _fail(report, f"{location}/{name}", error or "error")
        elif row is None:
            report.added += 1
        else:
            report.updated += 1

    def _copy_from_twin(self, md5: str, key: str, meta: dict, row, report) -> bool:
        twin = next(
            (
                r
                for r in self.db.find_by_md5(md5)
                if r["path"] != key and r["status"] != "error" and not self.local._needs_caption(r)
            ),
            None,
        )
        if twin is None or not self.local._vector_ready_for_model():
            return False
        chunks, vectors = self.db.stored_chunks(twin["id"])
        if not chunks:
            return False
        self.db.replace_file(
            **meta,
            kind=twin["kind"],
            sha256=twin["sha256"],
            captioned=bool(twin["captioned"]),
            status=twin["status"],
            error=None,
            details=twin["details"],
            indexed_at=time.time(),
            chunks=chunks,
            vectors=vectors,
        )
        self.progress(f"copied  {meta['location']}/{meta['name']} (same as {twin['path']})")
        if row is None:
            report.added += 1
        else:
            report.updated += 1
        return True

    def _fetch_and_extract(self, file: dict, kind: str) -> tuple[bytes, Extraction | None]:
        ix = self.cfg.index
        limit = int(ix.max_file_size_mb * 1024 * 1024)
        name, mime = file["name"], file.get("mimeType", "")
        if mime in GOOGLE_EXPORTS:
            data = self.client.export(file["id"], GOOGLE_EXPORTS[mime][0], _EXPORT_LIMIT)
            stand_in = Path(PurePosixPath(_clean(name)).name + ".txt")
            extraction = self._extract(stand_in, data)
            if extraction is not None:
                extraction.kind = kind
            return data, extraction

        data = None
        if kind == "image" and self.cfg.drive.image_previews and file.get("thumbnailLink"):
            try:
                data = self.client.thumbnail(file["thumbnailLink"], PREVIEW_SIDE, limit)
            except DriveError as exc:
                log.info("%s: no preview (%s), downloading the file", name, exc)
        if data is None:
            data = self.client.download(file["id"], limit)
        extraction = self._extract(Path(_clean(name)), data)
        if extraction is not None and kind == "image":
            _add_drive_photo_details(extraction, file)
        return data, extraction

    def _extract(self, path: Path, data: bytes) -> Extraction | None:
        try:
            return extract(path, self.cfg.index, self.local.captioner, data=data)
        except ExtractionError as exc:
            log.info("%s: %s", path, exc)
            return None


def _add_drive_photo_details(extraction: Extraction, file: dict) -> None:
    """Drive's preview images carry no EXIF; Drive reports date and camera itself."""
    photo = file.get("imageMediaMetadata") or {}
    parts = []
    taken = photo.get("time")
    if isinstance(taken, str) and len(taken) >= 10:
        parts.append(f"taken: {taken[:10].replace(':', '-')}")
    camera = " ".join(str(photo[k]).strip() for k in ("cameraMake", "cameraModel") if photo.get(k))
    if camera:
        parts.append(f"camera: {camera}")
    if not parts:
        return
    extra = "; ".join(parts)
    extraction.details = f"{extraction.details}; {extra}" if extraction.details else extra
    extraction.segments = [s for s in extraction.segments if s.source != "metadata"]
    extraction.segments.append(Segment(extraction.details, source="metadata"))
