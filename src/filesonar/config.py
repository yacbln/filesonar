"""Configuration loading and validation.

Configuration lives in a TOML file (by default ``$XDG_CONFIG_HOME/filesonar/config.toml``).
Every value has a safe default, so an empty file is a valid configuration once at least
one folder to index is listed.
"""

from __future__ import annotations

import ipaddress
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from .pathrules import matches_any, within

APP_NAME = "filesonar"

# Paths that are never indexed, whatever the configuration says. These hold keys,
# passwords, tokens and browser data: content an agent should never be able to see.
ALWAYS_EXCLUDED = (
    "**/.ssh/**",
    "**/.gnupg/**",
    "**/.password-store/**",
    "**/.aws/**",
    "**/.azure/**",
    "**/.kube/**",
    "**/.docker/**",
    "**/.config/gcloud/**",
    "**/.local/share/keyrings/**",
    "**/.mozilla/**",
    "**/.config/google-chrome/**",
    "**/.config/chromium/**",
    "**/.config/BraveSoftware/**",
    "**/.thunderbird/**",
    "**/.electrum/**",
    "**/.bitcoin/**",
    "**/.ethereum/**",
    "**/.env",
    "**/.env.*",
    "**/.netrc",
    "**/.pgpass",
    "**/.git-credentials",
    "**/id_rsa*",
    "**/id_ed25519*",
    "**/id_ecdsa*",
    "**/*.pem",
    "**/*.key",
    "**/*.p12",
    "**/*.pfx",
    "**/*.kdbx",
    "**/*.gpg",
    "**/*.asc",
    "**/wallet.dat",
)

# Excluded by default because they are noise (tooling, dependencies, build output),
# not because they are sensitive. Turn off with `index.default_excludes = false`.
DEFAULT_EXCLUDE = (
    "**/.git/**",
    "**/.hg/**",
    "**/.svn/**",
    "**/node_modules/**",
    "**/bower_components/**",
    "**/.venv/**",
    "**/venv/**",
    "**/site-packages/**",
    "**/dist-packages/**",
    "**/__pycache__/**",
    "**/.tox/**",
    "**/.mypy_cache/**",
    "**/.pytest_cache/**",
    "**/.ruff_cache/**",
    "**/.terraform/**",
    "**/.gradle/**",
    "**/.next/**",
    "**/.nuxt/**",
    "**/vendor/**",
    "**/Pods/**",
    "**/target/debug/**",
    "**/target/release/**",
    "**/.cache/**",
    "**/.Trash*/**",
    "**/.local/share/Trash/**",
)


class ConfigError(ValueError):
    """Raised when the configuration file is invalid."""


def xdg_dir(env: str, fallback: str) -> Path:
    value = os.environ.get(env)
    # The XDG spec says relative paths must be ignored.
    if value and os.path.isabs(value):
        return Path(value)
    return Path.home() / fallback


def default_config_path() -> Path:
    return xdg_dir("XDG_CONFIG_HOME", ".config") / APP_NAME / "config.toml"


def default_data_dir() -> Path:
    return xdg_dir("XDG_DATA_HOME", ".local/share") / APP_NAME


def default_outbox_dir() -> Path:
    """Where ``prepare_file`` puts copies for sending: OpenClaw's own media folder,
    which its view and send tools may read without extra permissions."""
    state = os.environ.get("OPENCLAW_STATE_DIR")
    base = Path(state) if state and os.path.isabs(state) else Path.home() / ".openclaw"
    return (base / "media" / APP_NAME).resolve()


@dataclass(frozen=True)
class ModelConfig:
    host: str = "http://127.0.0.1:11434"
    embedding_model: str = "nomic-embed-text"
    vision_model: str = "qwen2.5vl:7b"
    caption_images: bool = True
    allow_remote_host: bool = False
    timeout_seconds: float = 120.0


@dataclass(frozen=True)
class IndexConfig:
    max_file_size_mb: float = 50.0
    max_pdf_pages: int = 300
    max_chunks_per_file: int = 300
    ocr: bool = True
    ocr_languages: str = "eng"
    include_hidden: bool = False
    default_excludes: bool = True
    skip_dependency_folders: bool = True
    chunk_chars: int = 1200
    chunk_overlap: int = 200


@dataclass(frozen=True)
class SearchConfig:
    # Cosine distance above which a meaning-only match is dropped as unrelated.
    # Calibrated for nomic-embed-text; other models may need a different value.
    max_distance: float = 0.44
    # Meaning-only matches must also be this close to the best one.
    relative_margin: float = 0.12


@dataclass(frozen=True)
class DriveConfig:
    enabled: bool = False
    # Drive folders to index, as paths inside My Drive ("Taxes", "Photos/2024").
    # Empty means all of My Drive.
    folders: tuple[str, ...] = ()
    # Glob patterns to skip, matched against "/<path inside My Drive>".
    exclude: tuple[str, ...] = ()
    # Also index files other people shared with you (they are never uploaded to).
    include_shared: bool = False
    # Folder in My Drive that share_file uploads to (created on first use).
    share_folder: str = "filesonar"
    # How often `filesonar watch` asks Drive for changes.
    poll_minutes: float = 5.0
    # Describe Drive photos from Drive's own preview image instead of downloading
    # the full file: much less traffic, same search quality.
    image_previews: bool = True


@dataclass(frozen=True)
class Config:
    roots: tuple[Path, ...]
    exclude: tuple[str, ...] = ()
    data_dir: Path = field(default_factory=default_data_dir)
    outbox_dir: Path = field(default_factory=default_outbox_dir)
    models: ModelConfig = field(default_factory=ModelConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    drive: DriveConfig = field(default_factory=DriveConfig)
    source: Path | None = None

    @property
    def db_path(self) -> Path:
        return self.data_dir / "index.sqlite3"

    def excluded(self, path: Path, root: Path) -> bool:
        """Built-in rules apply to the full path; user rules to the path inside ``root``."""
        if matches_any(path, ALWAYS_EXCLUDED) or within(path, (self.outbox_dir,)):
            return True
        patterns = self.exclude + (DEFAULT_EXCLUDE if self.index.default_excludes else ())
        if not patterns:
            return False
        rel = "/" + path.relative_to(root).as_posix() if path != root else "/"
        return matches_any(rel, patterns)


def _expect(table: dict, key: str, kind: type | tuple[type, ...], where: str):
    value = table[key]
    # bool is a subclass of int; do not let `true` pass as a number.
    if isinstance(value, bool) and bool not in (kind if isinstance(kind, tuple) else (kind,)):
        raise ConfigError(f"{where}.{key} must be {kind}, got a boolean")
    if not isinstance(value, kind):
        raise ConfigError(f"{where}.{key} has the wrong type ({type(value).__name__})")
    return value


def _build(cls, table: dict, where: str):
    known = {f.name: f for f in cls.__dataclass_fields__.values()}
    unknown = set(table) - set(known)
    if unknown:
        raise ConfigError(f"unknown option(s) in [{where}]: {', '.join(sorted(unknown))}")
    kwargs = {}
    for key in table:
        default = getattr(cls(), key)
        if isinstance(default, tuple):
            value = table[key]
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ConfigError(f"{where}.{key} must be a list of strings")
            kwargs[key] = tuple(value)
            continue
        kind: type | tuple[type, ...] = type(default)
        if kind is float:
            kind = (int, float)
        kwargs[key] = _expect(table, key, kind, where)
        if kind == (int, float):
            kwargs[key] = float(kwargs[key])
    return cls(**kwargs)


def is_loopback_url(url: str) -> bool:
    host = urlparse(url).hostname
    if host is None:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate(cfg: Config) -> None:
    m, ix = cfg.models, cfg.index
    if urlparse(m.host).scheme not in ("http", "https"):
        raise ConfigError("models.host must be an http(s) URL")
    if not m.allow_remote_host and not is_loopback_url(m.host):
        raise ConfigError(
            "models.host points to another machine. File contents would leave this "
            "computer. Set models.allow_remote_host = true if that is intended."
        )
    if m.timeout_seconds <= 0:
        raise ConfigError("models.timeout_seconds must be positive")
    if ix.max_file_size_mb <= 0 or ix.max_pdf_pages <= 0 or ix.max_chunks_per_file <= 0:
        raise ConfigError("index limits must be positive")
    if ix.chunk_chars < 200:
        raise ConfigError("index.chunk_chars must be at least 200")
    if not 0 <= ix.chunk_overlap < ix.chunk_chars // 2:
        raise ConfigError("index.chunk_overlap must be between 0 and half of chunk_chars")
    s = cfg.search
    if not 0 < s.max_distance <= 2 or not 0 <= s.relative_margin <= 2:
        raise ConfigError("search.max_distance and search.relative_margin must be in (0, 2]")
    d = cfg.drive
    if d.poll_minutes < 1:
        raise ConfigError("drive.poll_minutes must be at least 1")
    if not d.share_folder.strip() or "/" in d.share_folder or d.share_folder in (".", ".."):
        raise ConfigError("drive.share_folder must be a plain folder name")
    if not cfg.roots and not d.enabled:
        raise ConfigError("no folders to index: add them under `roots = [...]`")
    for root in cfg.roots:
        # An outbox inside an indexed folder is fine (it is skipped); one that contains
        # an indexed folder would put the user's files among the disposable copies.
        if within(root, (cfg.outbox_dir,)):
            raise ConfigError(f"outbox_dir ({cfg.outbox_dir}) must not contain {root}")


def _resolve_root(raw: str, base: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(raw)))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_config(path: Path | None = None) -> Config:
    path = path or default_config_path()
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path} (run `filesonar init`)") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from None
    return config_from_dict(raw, base=path.parent, source=path)


def config_from_dict(raw: dict, base: Path, source: Path | None = None) -> Config:
    allowed = {"roots", "exclude", "data_dir", "outbox_dir", "models", "index", "search", "drive"}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"unknown top-level option(s): {', '.join(sorted(unknown))}")

    roots_raw = raw.get("roots", [])
    if not isinstance(roots_raw, list) or not all(isinstance(r, str) for r in roots_raw):
        raise ConfigError("roots must be a list of folder paths")
    roots = tuple(dict.fromkeys(_resolve_root(r, base) for r in roots_raw))

    exclude_raw = raw.get("exclude", [])
    if not isinstance(exclude_raw, list) or not all(isinstance(e, str) for e in exclude_raw):
        raise ConfigError("exclude must be a list of glob patterns")

    data_dir = default_data_dir()
    if "data_dir" in raw:
        if not isinstance(raw["data_dir"], str):
            raise ConfigError("data_dir must be a path")
        data_dir = _resolve_root(raw["data_dir"], base)

    outbox_dir = default_outbox_dir()
    if "outbox_dir" in raw:
        if not isinstance(raw["outbox_dir"], str):
            raise ConfigError("outbox_dir must be a path")
        outbox_dir = _resolve_root(raw["outbox_dir"], base)

    for section in ("models", "index", "search", "drive"):
        if section in raw and not isinstance(raw[section], dict):
            raise ConfigError(f"[{section}] must be a table")

    cfg = Config(
        roots=roots,
        exclude=tuple(exclude_raw),
        data_dir=data_dir,
        outbox_dir=outbox_dir,
        models=_build(ModelConfig, raw.get("models", {}), "models"),
        index=_build(IndexConfig, raw.get("index", {}), "index"),
        search=_build(SearchConfig, raw.get("search", {}), "search"),
        drive=_build(DriveConfig, raw.get("drive", {}), "drive"),
        source=source,
    )
    _validate(cfg)
    return cfg


EXAMPLE_CONFIG = """\
# filesonar configuration. See the README for every option.

# Folders to index. Subfolders are included.
roots = [
  "~/Documents",
  "~/Pictures",
  "~/Downloads",
]

# Extra glob patterns to skip, matched against the path inside each folder
# (a leading "/" anchors a pattern to the folder itself), for example
#   exclude = ["**/old-backups/**", "*.log"]
# These add to the defaults (version control, dependencies, build output,
# caches, trash). On top of everything, sensitive locations (SSH keys,
# password stores, browser profiles, .env files, ...) are always skipped.
exclude = []

# When the agent is asked to send you a file, filesonar copies it here first
# (OpenClaw's media folder) so it can be attached without any shell command.
# Copies are deleted after 24 hours.
# outbox_dir = "~/.openclaw/media/filesonar"

[models]
# Ollama server. Only addresses on this machine are accepted unless
# allow_remote_host is set, because file contents are sent to it.
host = "http://127.0.0.1:11434"
embedding_model = "nomic-embed-text"
# Describes photos and screenshots so you can search for what they show.
vision_model = "qwen2.5vl:7b"
caption_images = true

[index]
max_file_size_mb = 50
max_pdf_pages = 300
max_chunks_per_file = 300   # long books and data exports: only the start is indexed
ocr = true
ocr_languages = "eng"   # e.g. "eng+fra+ara" if those Tesseract packs are installed
include_hidden = false
default_excludes = true          # skip .git, node_modules, build output, caches, ...
skip_dependency_folders = true   # skip folders where software packages were installed

[search]
# How close (cosine distance) a match by meaning must be. Lower is stricter.
max_distance = 0.44

[drive]
# Google Drive. Sign in first with `filesonar drive login` (see the README).
enabled = false
folders = []            # e.g. ["Taxes", "Photos/2024"]; empty = all of My Drive
exclude = []
include_shared = false  # also index files shared with you
share_folder = "filesonar"
poll_minutes = 5
"""
