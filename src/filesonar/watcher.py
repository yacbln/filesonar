"""Keep the index current by watching the configured folders for changes."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .indexer import Indexer

if TYPE_CHECKING:
    from .drive_index import DriveIndexer

log = logging.getLogger(__name__)


class _Collector(FileSystemEventHandler):
    def __init__(self) -> None:
        self.pending: set[Path] = set()
        self.last_event = 0.0
        self.lock = threading.Lock()

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.event_type in ("opened", "closed_no_write"):
            return
        # A folder is "modified" whenever a file in it changes; that file has its own
        # event, so re-checking the whole folder would only waste time.
        if event.is_directory and event.event_type == "modified":
            return
        with self.lock:
            for attr in ("src_path", "dest_path"):
                raw = getattr(event, attr, "")
                if raw:
                    self.pending.add(Path(raw if isinstance(raw, str) else raw.decode()))
            self.last_event = time.monotonic()

    def drain(self, quiet_seconds: float) -> set[Path]:
        """Return pending paths once no new event has arrived for ``quiet_seconds``."""
        with self.lock:
            if not self.pending or time.monotonic() - self.last_event < quiet_seconds:
                return set()
            out, self.pending = self.pending, set()
            return out


def watch(
    indexer: Indexer,
    quiet_seconds: float = 3.0,
    full_scan_hours: float = 24.0,
    stop: threading.Event | None = None,
    drive: DriveIndexer | None = None,
    drive_poll_minutes: float = 5.0,
) -> None:
    """Run until ``stop`` is set (or forever). Changes are batched after a quiet period
    so that copying a folder of 500 photos results in one pass, not 500. With ``drive``,
    Google Drive is asked for changes every ``drive_poll_minutes``."""
    stop = stop or threading.Event()
    collector = _Collector()
    observer = Observer()
    for root in indexer.cfg.roots:
        if root.is_dir():
            observer.schedule(collector, str(root), recursive=True)
        else:
            log.warning("not watching missing folder %s", root)
    observer.start()
    log.info("watching %d folder(s)", len(indexer.cfg.roots))
    next_full = time.monotonic() + full_scan_hours * 3600
    next_drive = time.monotonic()  # sync Drive right away
    try:
        while not stop.wait(0.5):
            changed = collector.drain(quiet_seconds)
            if changed:
                report = indexer.update_paths(changed)
                if report.added or report.updated or report.moved or report.removed:
                    log.info("%s", report.summary())
            if time.monotonic() >= next_full:
                # Safety net for anything inotify missed (e.g. changes while asleep).
                log.info("periodic full scan: %s", indexer.run().summary())
                if drive is not None:
                    _sync_drive(drive, full=True)
                next_full = time.monotonic() + full_scan_hours * 3600
                next_drive = time.monotonic() + drive_poll_minutes * 60
            if drive is not None and time.monotonic() >= next_drive:
                _sync_drive(drive, full=False)
                next_drive = time.monotonic() + drive_poll_minutes * 60
    finally:
        observer.stop()
        observer.join(timeout=5)


def _sync_drive(drive: DriveIndexer, full: bool) -> None:
    # Network trouble must not stop the local watcher; the next poll tries again.
    try:
        report = drive.sync(full=full)
    except Exception as exc:
        log.warning("Google Drive sync failed: %s", exc)
        return
    if report.added or report.updated or report.moved or report.removed or full:
        log.info("Google Drive: %s", report.summary())
