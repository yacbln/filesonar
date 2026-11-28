"""Command-line interface: ``filesonar init | doctor | index | search | status | watch |
serve | drive``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .config import (
    EXAMPLE_CONFIG,
    Config,
    ConfigError,
    default_config_path,
    load_config,
    write_private,
)
from .db import Database
from .extract import tesseract_available
from .indexer import Indexer
from .models import ModelError, OllamaCaptioner, OllamaClient, OllamaEmbedder
from .search import KINDS, SOURCES, Searcher


def _open(cfg: Config) -> tuple[Database, OllamaClient]:
    return Database(cfg.db_path), OllamaClient(cfg.models)


def _transcriber(cfg: Config):
    """Speech recognition for audio and video, if turned on (loaded only when used)."""
    if not (cfg.media.enabled and cfg.media.transcribe):
        return None
    from .media import WhisperTranscriber

    return WhisperTranscriber(cfg.media.speech_model, cfg.models_dir)


def _drive_indexer(cfg: Config, db: Database, client: OllamaClient, progress=None):
    """A Drive indexer if Drive is enabled and signed in, else None (with a warning)."""
    if not cfg.drive.enabled:
        return None
    from .drive import DriveClient, NotSignedIn
    from .drive_index import DriveIndexer

    try:
        drive = DriveClient.from_data_dir(cfg.data_dir)
    except NotSignedIn as exc:
        print(f"warning: Google Drive skipped: {exc}", file=sys.stderr)
        return None
    return DriveIndexer(
        cfg, db, OllamaEmbedder(client), drive, OllamaCaptioner(client), progress, _transcriber(cfg)
    )


def _print_errors(report) -> None:
    for path, error in report.errors[:20]:
        print(f"  failed: {path}: {error}", file=sys.stderr)
    if len(report.errors) > 20:
        print(f"  ... and {len(report.errors) - 20} more", file=sys.stderr)


def cmd_init(args: argparse.Namespace) -> int:
    path: Path = args.config or default_config_path()
    if path.exists() and not args.force:
        print(f"{path} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    write_private(path, EXAMPLE_CONFIG)
    print(f"wrote {path}\nedit the folder list, then run: filesonar index")
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    from .wizard import Options, Setup

    opts = Options(
        config_path=args.config,
        assume_yes=args.yes,
        folders=args.folders,
        vision=args.vision,
        drive_client_secret=args.drive_client_secret,
        openclaw=args.openclaw,
        claude_code=False if args.no_claude_code else None,
        service=False if args.no_service else None,
        index_now=False if args.no_index else None,
    )
    try:
        return Setup(opts).main()
    except KeyboardInterrupt:
        print("\nsetup stopped; run `filesonar setup` again to continue")
        return 130


def cmd_doctor(cfg: Config) -> int:
    ok = True
    client = OllamaClient(cfg.models)
    installed = client.installed_models()
    print(f"config:      {cfg.source}")
    print(f"index:       {cfg.db_path}")
    for root in cfg.roots:
        print(f"folder:      {root}" + ("" if root.is_dir() else "  (missing!)"))
    if not installed:
        print(f"ollama:      not reachable at {cfg.models.host}")
        ok = False
    else:
        print(f"ollama:      reachable at {cfg.models.host}")
    for label, name, needed in (
        ("embeddings", cfg.models.embedding_model, True),
        ("vision", cfg.models.vision_model, cfg.models.caption_images),
    ):
        if not needed:
            print(f"{label + ':':<12} disabled")
            continue
        present = name in installed
        state = "ok" if present else f"(missing: ollama pull {name})"
        print(f"{label + ':':<12} {name} {state}")
        ok = ok and (present or label == "vision")
    for name, (vram, size) in sorted(client.loaded_models().items()):
        if name.endswith(":latest") or not size or vram >= size:
            continue
        # Ollama splits a model between GPU and CPU when video memory was short at load
        # time; it stays that way (and runs several times slower) until reloaded.
        print(
            f"warning:     {name} is only {vram * 100 // size}% in GPU memory and will be "
            f"slow; free GPU memory, then run: ollama stop {name}"
        )
    print(f"tesseract:   {'ok' if tesseract_available() else 'missing (no OCR)'}")
    from . import media

    if not cfg.media.enabled:
        print("media:       disabled")
    else:
        ffmpeg = "ok" if media.tools_available() else "missing (audio and video by name only)"
        print(f"ffmpeg:      {ffmpeg}")
        if cfg.media.transcribe:
            speech = media.WhisperTranscriber(cfg.media.speech_model, cfg.models_dir)
            state = (
                f"ok ({cfg.media.speech_model} model)"
                if speech.ready()
                else "not set up (run `filesonar setup`)"
            )
            print(f"speech:      {state}")
        if cfg.media.sounds:
            sounds = media.SoundTagger(cfg.models_dir)
            print(
                f"sounds:      {'ok' if sounds.ready() else 'not set up (run `filesonar setup`)'}"
            )
    if cfg.drive.enabled:
        from .drive import DriveError, load_token, token_path

        try:
            account = load_token(token_path(cfg.data_dir)).account
            print(f"drive:       signed in as {account}")
        except DriveError as exc:
            print(f"drive:       {exc}")
            ok = False
    else:
        print("drive:       disabled")
    return 0 if ok else 1


def _progress_printer(verbose: bool):
    """Every file in verbose mode; otherwise a short line every 50 files."""
    count = 0

    def show(message: str) -> None:
        nonlocal count
        count += 1
        if verbose:
            print(message, file=sys.stderr)
        elif count % 50 == 0:
            print(f"... {count} files read", file=sys.stderr, flush=True)

    return show


def cmd_index(cfg: Config, args: argparse.Namespace) -> int:
    db, client = _open(cfg)
    try:
        if args.rebuild:
            db.reset()
        progress = _progress_printer(args.verbose)
        indexer = Indexer(
            cfg, db, OllamaEmbedder(client), OllamaCaptioner(client), progress, _transcriber(cfg)
        )
        if args.reembed and not args.rebuild:
            print(
                f"embedding the indexed text with {cfg.models.embedding_model} ...", file=sys.stderr
            )
            print(f"re-embedded {indexer.reembed()} pieces of text")
        report = indexer.run()
        print(report.summary())
        _print_errors(report)
        drive = _drive_indexer(cfg, db, client, progress)
        if drive is not None:
            drive_report = drive.sync(full=args.rebuild)
            print("Google Drive: " + drive_report.summary())
            _print_errors(drive_report)
    finally:
        db.close()
    return 0


def cmd_search(cfg: Config, args: argparse.Namespace) -> int:
    db, client = _open(cfg)
    try:
        searcher = Searcher(db, OllamaEmbedder(client), cfg.search)
        hits = searcher.search(
            " ".join(args.query),
            limit=args.limit,
            kind=args.kind,
            folder=args.folder,
            modified_after=args.after,
            modified_before=args.before,
            source=args.source,
        )
    finally:
        db.close()
    if searcher.last_warning:
        print(f"warning: {searcher.last_warning}", file=sys.stderr)
    if args.json:
        print(json.dumps([h.to_dict() for h in hits], indent=2, ensure_ascii=False))
        return 0
    if not hits:
        print("no matches")
        return 0
    if searcher.last_confidence == "low":
        print("nothing clearly matches; the closest files are:\n")
    for n, hit in enumerate(hits, 1):
        page = f" p.{hit.page}" if hit.page else ""
        where = f"Google Drive: {hit.location}/{hit.name}" if hit.source == "drive" else hit.path
        print(f"{n}. {where}{page}\n   {hit.kind}, {hit.modified}, matched in {hit.matched_in}")
        print(f"   {hit.snippet}\n")
    return 0


def cmd_status(cfg: Config) -> int:
    db = Database(cfg.db_path)
    try:
        print(json.dumps(db.stats(), indent=2))
    finally:
        db.close()
    return 0


def cmd_watch(cfg: Config, args: argparse.Namespace) -> int:
    from .watcher import watch

    db, client = _open(cfg)
    indexer = Indexer(
        cfg, db, OllamaEmbedder(client), OllamaCaptioner(client), transcriber=_transcriber(cfg)
    )
    drive = _drive_indexer(cfg, db, client)
    try:
        if not args.no_initial_scan:
            logging.info("initial scan: %s", indexer.run().summary())
        watch(
            indexer,
            full_scan_hours=args.full_scan_hours,
            drive=drive,
            drive_poll_minutes=cfg.drive.poll_minutes,
        )
    except KeyboardInterrupt:
        pass
    finally:
        db.close()
    return 0


def _drive_factory(cfg: Config):
    def make():
        from .drive import DriveClient

        return DriveClient.from_data_dir(cfg.data_dir)

    return make if cfg.drive.enabled else None


def cmd_serve(cfg: Config) -> int:
    from .server import build_server

    db, client = _open(cfg)
    try:
        build_server(cfg, db, OllamaEmbedder(client), drive=_drive_factory(cfg)).run("stdio")
    finally:
        db.close()
    return 0


def cmd_drive(cfg: Config, args: argparse.Namespace) -> int:
    import httpx

    from .drive import (
        DriveClient,
        DriveError,
        NotSignedIn,
        sign_in,
        token_path,
    )

    path = token_path(cfg.data_dir)
    if args.drive_command == "login":
        token = sign_in(cfg.data_dir, args.client_secret, open_browser=not args.no_browser)
        print(f"signed in as {token.account}; token saved to {path}")
        if not cfg.drive.enabled:
            print(f"now set `enabled = true` under [drive] in {cfg.source}")
        db = Database(cfg.db_path)
        try:
            first_time = not db.stats().get("drive_files")
        finally:
            db.close()
        if first_time:
            print("then run: filesonar index")
        return 0

    if args.drive_command == "logout":
        try:
            client = DriveClient.from_data_dir(cfg.data_dir)
        except NotSignedIn:
            print("not signed in")
            return 0
        try:
            client.revoke()
        except (httpx.HTTPError, DriveError) as exc:
            print(
                f"warning: could not reach Google to revoke access ({exc}); remove "
                "filesonar at https://myaccount.google.com/permissions",
                file=sys.stderr,
            )
        finally:
            client.close()
        path.unlink(missing_ok=True)
        db = Database(cfg.db_path)
        try:
            removed = 0
            for key in db.all_paths("drive"):
                removed += db.delete_path(key)
            for key in ("drive_page_token", "drive_last_full", "drive_share_folder"):
                db.conn.execute("DELETE FROM meta WHERE key = ?", (key,))
            db.conn.commit()
        finally:
            db.close()
        print(f"signed out; access revoked and {removed} Drive entries removed from the index")
        return 0

    if args.drive_command == "status":
        try:
            client = DriveClient.from_data_dir(cfg.data_dir)
            account = client.token.account or "(unknown account)"
            client.close()
        except NotSignedIn:
            account = None
        db = Database(cfg.db_path)
        try:
            count = len(db.all_paths("drive"))
            last = db.get_meta("drive_last_full")
        finally:
            db.close()
        print(f"enabled:     {cfg.drive.enabled}")
        print(f"signed in:   {account or 'no (run: filesonar drive login)'}")
        print(f"indexed:     {count} files")
        if last:
            from datetime import datetime

            print(
                f"full sync:   {datetime.fromtimestamp(float(last)).isoformat(timespec='minutes')}"
            )
        return 0

    # sync
    if not cfg.drive.enabled:
        print(f"Google Drive is disabled; set `enabled = true` under [drive] in {cfg.source}")
        return 1
    db, client = _open(cfg)
    try:
        drive = _drive_indexer(cfg, db, client, _progress_printer(args.verbose))
        if drive is None:
            return 1
        report = drive.sync(full=args.full)
    finally:
        db.close()
    print("Google Drive: " + report.summary())
    _print_errors(report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="filesonar", description="Search your files by what is inside them."
    )
    p.add_argument("--version", action="version", version=f"filesonar {__version__}")
    p.add_argument(
        "-c", "--config", type=Path, help="config file (default: %(default)s)", default=None
    )
    p.add_argument("-v", "--verbose", action="store_true", help="show more detail")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init", help="write a starter config file")
    s.add_argument("--force", action="store_true", help="overwrite an existing config")

    s = sub.add_parser(
        "setup",
        help="set everything up: models, config, agents, background indexing",
        description="Check and set up everything filesonar needs. Safe to run again: "
        "only what is missing is changed.",
    )
    s.add_argument("-y", "--yes", action="store_true", help="accept every default answer")
    s.add_argument(
        "--folder", dest="folders", action="append", type=Path, default=[], metavar="PATH",
        help="a folder to search (repeat for more); default: Documents, Pictures, ...",
    )  # fmt: skip
    s.add_argument(
        "--vision", action=argparse.BooleanOptionalAction, default=None,
        help="download the vision model that describes photos (about 6 GB)",
    )  # fmt: skip
    s.add_argument(
        "--drive-client-secret",
        type=Path,
        metavar="JSON",
        help="connect Google Drive with this file",
    )
    s.add_argument("--openclaw", metavar="PATH", help="path to the openclaw command")
    s.add_argument("--no-claude-code", action="store_true", help="do not add to Claude Code")
    s.add_argument("--no-service", action="store_true", help="no background indexing service")
    s.add_argument("--no-index", action="store_true", help="do not index right away")

    s = sub.add_parser(
        "uninstall",
        help="undo setup: stop the service, unregister from agents, sign out of Drive",
        description="Remove what `filesonar setup` added. Your own files are never touched; "
        "the index and settings are kept unless you ask. Then: pipx uninstall filesonar",
    )
    s.add_argument("-y", "--yes", action="store_true", help="no questions; keeps the index")
    s.add_argument("--delete-data", action="store_true", help="also delete index and settings")
    s.add_argument("--openclaw", metavar="PATH", help="path to the openclaw command")

    sub.add_parser("doctor", help="check models, OCR and folders")

    s = sub.add_parser("index", help="index new and changed files")
    s.add_argument("--rebuild", action="store_true", help="start from an empty index")
    s.add_argument(
        "--reembed",
        action="store_true",
        help="after changing embedding_model: compute the vectors again from the text "
        "already indexed (much faster than --rebuild; nothing is read again)",
    )

    s = sub.add_parser("search", help="search the index")
    s.add_argument("query", nargs="+")
    s.add_argument("-n", "--limit", type=int, default=8)
    s.add_argument("-k", "--kind", choices=KINDS)
    s.add_argument("--folder", help="only files inside this folder")
    s.add_argument("--after", help="modified on or after YYYY-MM-DD")
    s.add_argument("--before", help="modified on or before YYYY-MM-DD")
    s.add_argument("--source", choices=SOURCES, help="only local files or only Google Drive")
    s.add_argument("--json", action="store_true", help="machine-readable output")

    sub.add_parser("status", help="show index statistics")

    s = sub.add_parser("watch", help="keep the index up to date as files change")
    s.add_argument("--no-initial-scan", action="store_true")
    s.add_argument("--full-scan-hours", type=float, default=24.0)

    sub.add_parser("serve", help="run the MCP server on stdio (for AI agents)")

    s = sub.add_parser("drive", help="Google Drive: sign in, sync, status, sign out")
    drive = s.add_subparsers(dest="drive_command", required=True)
    d = drive.add_parser("login", help="sign in with your own Google OAuth client")
    d.add_argument(
        "--client-secret",
        type=Path,
        help="the client_secret_*.json downloaded from Google Cloud Console (needed the "
        "first time; signing in again reuses the saved client)",
    )
    d.add_argument("--no-browser", action="store_true", help="only print the sign-in link")
    d = drive.add_parser("sync", help="fetch changes from Drive now")
    d.add_argument("--full", action="store_true", help="re-list every file")
    drive.add_parser("status", help="show the Drive connection and what is indexed")
    drive.add_parser("logout", help="revoke access and remove Drive files from the index")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # stdout belongs to the MCP protocol in `serve`, so logs always go to stderr.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )
    for noisy in ("httpx", "httpcore", "PIL", "watchdog", "pytesseract", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if args.command == "init":
        return cmd_init(args)
    if args.command == "setup":
        return cmd_setup(args)
    if args.command == "uninstall":
        from .wizard import Options, Uninstall

        opts = Options(config_path=args.config, assume_yes=args.yes, openclaw=args.openclaw)
        return Uninstall(opts, purge=True if args.delete_data else None).main()
    try:
        cfg = load_config(args.config)
        handlers = {
            "doctor": lambda: cmd_doctor(cfg),
            "index": lambda: cmd_index(cfg, args),
            "search": lambda: cmd_search(cfg, args),
            "status": lambda: cmd_status(cfg),
            "watch": lambda: cmd_watch(cfg, args),
            "serve": lambda: cmd_serve(cfg),
            "drive": lambda: cmd_drive(cfg, args),
        }
        return handlers[args.command]()
    except (ConfigError, ModelError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
