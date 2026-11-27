"""``filesonar setup``: everything needed to go from ``pipx install`` to searching.

Each step checks first and only changes what is missing, so running setup again is
safe (and is how you pick up a new version: it refreshes the skill and restarts the
background service). Nothing here uses sudo; when a system package is missing,
setup prints the command to install it.
"""

from __future__ import annotations

import contextlib
import getpass
import importlib
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from . import media
from .config import (
    Config,
    ConfigError,
    ModelConfig,
    add_roots,
    default_config_path,
    load_config,
    render_config,
    set_toml_value,
    write_private,
)
from .extract import tesseract_available
from .media import WhisperTranscriber
from .models import ModelError, OllamaClient

SERVICE_NAME = "filesonar-watch.service"
DRIVE_SETUP_URL = "https://github.com/yacbln/filesonar#one-time-setup-about-5-minutes"
CANDIDATE_FOLDERS = ("Documents", "Pictures", "Downloads", "Desktop", "Videos", "Music")
MEDIA_FOLDERS = ("Videos", "Music")
VISION_DOWNLOAD = "about 6 GB"

Runner = Callable[..., subprocess.CompletedProcess]


def run_quiet(cmd: Sequence[str], timeout: float = 120) -> subprocess.CompletedProcess:
    env = dict(os.environ, NODE_NO_WARNINGS="1")  # OpenClaw warns on every call otherwise
    try:
        return subprocess.run(
            list(cmd), capture_output=True, text=True, timeout=timeout, env=env, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(list(cmd), 127, "", str(exc))


class Prompter:
    """Questions with a default. With ``assume_yes`` (or no terminal), take the default."""

    def __init__(self, assume_yes: bool, ask: Callable[[str], str] = input):
        self.assume_yes = assume_yes
        self._ask = ask

    def yes(self, question: str, default: bool = True) -> bool:
        hint = "Y/n" if default else "y/N"
        if self.assume_yes:
            print(f"  {question} [{hint}] {'yes' if default else 'no'}")
            return default
        while True:
            answer = self._ask(f"  {question} [{hint}] ").strip().lower()
            if not answer:
                return default
            if answer in ("y", "yes"):
                return True
            if answer in ("n", "no"):
                return False

    def text(self, question: str, default: str = "") -> str:
        if self.assume_yes:
            return default
        shown = f" [{default}]" if default else ""
        return self._ask(f"  {question}{shown}: ").strip() or default


@dataclass
class Options:
    config_path: Path | None = None
    assume_yes: bool = False
    folders: list[Path] = field(default_factory=list)
    vision: bool | None = None  # None: ask (default yes with a GPU)
    speech: bool | None = None  # speech recognition for audio and video; None: ask
    drive_client_secret: Path | None = None
    openclaw: str | None = None
    claude_code: bool | None = None
    service: bool | None = None
    index_now: bool | None = None


class Setup:
    def __init__(
        self,
        opts: Options,
        *,
        prompter: Prompter | None = None,
        run: Runner = run_quiet,
        which: Callable[[str], str | None] = shutil.which,
        ollama: Callable[[ModelConfig], OllamaClient] = OllamaClient,
        executable: str | None = None,
        home: Path | None = None,
    ):
        self.opts = opts
        self.ask = prompter or Prompter(opts.assume_yes or not sys.stdin.isatty())
        self.run = run
        self.which = which
        self.make_ollama = ollama
        self.home = home or Path.home()
        self.config_path = opts.config_path or default_config_path()
        self.exe = executable or self._find_executable()
        self.problems: list[str] = []

    # -- output -------------------------------------------------------------

    @staticmethod
    def ok(message: str) -> None:
        print(f"  ✓ {message}")

    def warn(self, message: str) -> None:
        print(f"  ! {message}")
        self.problems.append(message)

    @staticmethod
    def step(title: str) -> None:
        print(f"\n{title}")

    # -- steps --------------------------------------------------------------

    def main(self) -> int:
        print("filesonar setup: each step checks first and only changes what is missing.")
        existing = self._load_existing()
        if existing is False:
            return 2
        models_cfg = existing.models if existing else ModelConfig()

        self.step("1. Local models (Ollama)")
        client = self.make_ollama(models_cfg)
        vision = self.models(client, models_cfg, existing)
        if vision is None:
            return 1

        self.step("2. Text in scans and screenshots (Tesseract)")
        self.tesseract()

        self.step("3. Audio and video")
        self.media_step(existing)

        self.step("4. Folders to search")
        cfg = self.config(existing, vision)
        if cfg is None:
            return 2

        self.step("5. Google Drive (optional)")
        cfg = self.drive(cfg, first_run=existing is None)

        self.step("6. AI agents")
        self.openclaw()
        self.claude_code(first_run=existing is None)

        self.step("7. Keep the index up to date")
        background = self.service()

        self.step("8. First index")
        self.first_index(cfg, background)

        self.summary()
        return 0

    def _load_existing(self) -> Config | bool | None:
        if not self.config_path.exists():
            return None
        try:
            return load_config(self.config_path)
        except ConfigError as exc:
            print(f"error: {exc}\nfix the config file, or move it away and run setup again")
            return False

    def models(
        self, client: OllamaClient, cfg: ModelConfig, existing: Config | None
    ) -> bool | None:
        """Make sure Ollama runs and has the models. Returns whether to describe photos,
        or None if setup cannot go on."""
        version = client.version()
        if version is None:
            if self.which("ollama"):
                print(
                    f"  ✗ Ollama is installed but not answering at {cfg.host}. Start it "
                    "(`systemctl start ollama`, or `ollama serve` in another terminal) and "
                    "run `filesonar setup` again."
                )
            else:
                print(
                    "  ✗ Ollama is not installed. It runs the models on this computer:\n"
                    "      curl -fsSL https://ollama.com/install.sh | sh\n"
                    "    (or see https://ollama.com/download), then run `filesonar setup` again."
                )
            return None
        self.ok(f"Ollama {version} at {cfg.host}")
        installed = client.installed_models()

        if cfg.embedding_model in installed:
            self.ok(f"{cfg.embedding_model} (understands text)")
        elif not self._pull(client, cfg.embedding_model, "274 MB, required"):
            return None

        wants = self.opts.vision
        if existing is not None and wants is None:
            wants = existing.models.caption_images  # keep what the config says
        if cfg.vision_model in installed:
            if wants is False:
                self.ok(f"{cfg.vision_model} installed; describing photos stays off")
                return False
            self.ok(f"{cfg.vision_model} (describes photos and screenshots)")
            return True
        if wants is None:
            gpu = self._has_gpu()
            print(
                f"  The vision model {cfg.vision_model} lets you search photos by what they "
                f"show.\n  It downloads {VISION_DOWNLOAD}"
                + (" and works best with a GPU." if gpu else "; no GPU found, so it would be slow.")
            )
            wants = self.ask.yes("Download it?", default=gpu)
        if not wants:
            self.ok("photos: indexed by their text and details only (no descriptions)")
            return False
        if not self._pull(client, cfg.vision_model, VISION_DOWNLOAD):
            self.warn(f"photos will not be described until `ollama pull {cfg.vision_model}`")
            return False
        return True

    def _pull(self, client: OllamaClient, name: str, size: str) -> bool:
        print(f"  downloading {name} ({size}) ...")
        last = [-1]

        def show(done: int, total: int) -> None:
            percent = done * 100 // total // 10 * 10
            if percent > last[0]:  # every 10%, never backwards
                last[0] = percent
                print(f"    {percent}%", flush=True)

        try:
            client.pull(name, show)
        except ModelError as exc:
            print(f"  ✗ {exc}")
            return False
        self.ok(f"{name} downloaded")
        return True

    def _has_gpu(self) -> bool:
        return bool(self.which("nvidia-smi")) or Path("/dev/kfd").exists()

    def _install_command(self, package: str) -> str:
        return next(
            (
                cmd
                for tool, cmd in (
                    ("apt", f"sudo apt install {package}"),
                    ("dnf", f"sudo dnf install {package}"),
                    ("pacman", f"sudo pacman -S {package}"),
                    ("brew", f"brew install {package}"),
                )
                if self.which(tool)
            ),
            f"install {package} with your package manager",
        )

    def media_step(self, existing: Config | None) -> None:
        from .config import MediaConfig, default_data_dir

        mcfg = existing.media if existing else MediaConfig()
        if not mcfg.enabled:
            self.ok("audio and video are turned off in the config")
            return
        if media.tools_available():
            self.ok("ffmpeg found")
        else:
            self.warn(
                "ffmpeg is missing, so audio and video are found by name only: "
                + self._install_command("ffmpeg")
            )
        if not mcfg.transcribe:
            self.ok("speech recognition is turned off in the config")
            return
        data_dir = existing.data_dir if existing else default_data_dir()
        speech = WhisperTranscriber(mcfg.speech_model, data_dir / "models")
        if speech.ready():
            self.ok(f"speech recognition ({mcfg.speech_model} model)")
            return
        wants = self.opts.speech
        if wants is None:
            size = media.MODEL_SIZES.get(mcfg.speech_model, "")
            print(
                "  Speech recognition finds recordings by what is said in them, on this computer."
            )
            print(f"  It installs about 150 MB of software and a {size} speech model, once.")
            wants = self.ask.yes("Set it up?")
        if not wants:
            self.ok("skipped; recordings are found by what they show and their details")
            return
        if not speech.installed() and not self._install_speech():
            return
        print(f"  downloading the {mcfg.speech_model} speech model ...")
        try:
            speech.download()
        except media.MediaError as exc:
            self.warn(f"{exc}; run `filesonar setup` again later")
            return
        self.ok("speech recognition ready")

    def _install_speech(self) -> bool:
        """Install the optional faster-whisper package into filesonar's own environment."""
        prefix = Path(sys.prefix)
        pipx = self.which("pipx")
        if pipx and prefix.parent.name == "venvs" and "pipx" in prefix.parts:
            cmd = [pipx, "inject", "filesonar", "faster-whisper>=1.1"]  # survives upgrades
        elif "uv" in prefix.parts and "tools" in prefix.parts:
            self.warn(
                "installed with uv: run `uv tool install --force --with faster-whisper "
                "filesonar`, then `filesonar setup` again"
            )
            return False
        else:
            cmd = [sys.executable, "-m", "pip", "install", "faster-whisper>=1.1"]
        print("  installing speech recognition (a few minutes) ...")
        result = self.run(cmd, timeout=1800)
        importlib.invalidate_caches()
        if result.returncode != 0 or not WhisperTranscriber.installed():
            detail = (result.stderr or result.stdout).strip().splitlines()[-1:]
            self.warn(
                f"could not install speech recognition ({' '.join(detail)[:120]}); "
                f"try: {shlex.join(cmd)}"
            )
            return False
        self.ok("speech recognition installed")
        return True

    def tesseract(self) -> None:
        if tesseract_available():
            self.ok("Tesseract found")
            return
        install = next(
            (
                cmd
                for tool, cmd in (
                    ("apt", "sudo apt install tesseract-ocr"),
                    ("dnf", "sudo dnf install tesseract"),
                    ("pacman", "sudo pacman -S tesseract tesseract-data-eng"),
                    ("brew", "brew install tesseract"),
                )
                if self.which(tool)
            ),
            "install Tesseract OCR with your package manager",
        )
        self.warn(f"Tesseract is missing, so text in scans and photos is not read: {install}")

    def config(self, existing: Config | None, vision: bool) -> Config | None:
        path = self.config_path
        if existing is not None:
            text = path.read_text()
            if existing.models.caption_images != vision:
                text = set_toml_value(
                    text, "models", "caption_images", "true" if vision else "false"
                )
                write_private(path, text)
            missing = [
                self.home / name
                for name in MEDIA_FOLDERS
                if existing.media.enabled
                and (self.home / name).is_dir()
                and not any((self.home / name).is_relative_to(r) for r in existing.roots)
            ]
            if missing and self.ask.yes(
                f"Also search {' and '.join(f'~/{m.name}' for m in missing)} "
                "(audio and video can be found now)?"
            ):
                text = add_roots(text, missing)
                write_private(path, text)
                existing = load_config(path)
            for root in existing.roots:
                self.ok(f"{root}" + ("" if root.is_dir() else "  (missing!)"))
            self.ok(f"config kept: {path} (edit it to change folders)")
            return load_config(path)

        roots = [p.expanduser().resolve() for p in self.opts.folders]
        if not roots:
            roots = [self.home / name for name in CANDIDATE_FOLDERS if (self.home / name).is_dir()]
            if roots:
                print("  filesonar will search inside:")
                for root in roots:
                    print(f"    {root}")
                if not self.ask.yes("Use these folders?"):
                    roots = []
            if not roots:
                answer = self.ask.text("Folders to search, separated by commas")
                roots = [
                    Path(p.strip()).expanduser().resolve() for p in answer.split(",") if p.strip()
                ]
        missing = [r for r in roots if not r.is_dir()]
        if missing:
            print(f"  ✗ not a folder: {', '.join(map(str, missing))}")
            return None
        if not roots:
            print("  ✗ no folders chosen; run `filesonar setup --folder PATH` to pick some")
            return None
        write_private(path, render_config(roots, caption_images=vision))
        self.ok(f"wrote {path} (private to you)")
        return load_config(path)

    def drive(self, cfg: Config, first_run: bool) -> Config:
        from .drive import DriveError, NotSignedIn, load_token, sign_in, token_path

        try:
            token = load_token(token_path(cfg.data_dir))
        except NotSignedIn:
            token = None
        except DriveError as exc:
            self.warn(str(exc))
            return cfg
        secret = self.opts.drive_client_secret
        if token is None and secret is None:
            if not first_run:  # asked during the first setup already
                print(f"  not connected; to add it see {DRIVE_SETUP_URL}")
                return cfg
            if self.opts.assume_yes or not self.ask.yes(
                "Search your Google Drive too? (needs a one-time 5-minute Google setup)",
                default=False,
            ):
                print(f"  skipped; add it any time, see {DRIVE_SETUP_URL}")
                return cfg
            print(f"  Follow the steps at {DRIVE_SETUP_URL}")
            answer = self.ask.text("Path to the client_secret_*.json you downloaded")
            if not answer:
                print("  skipped")
                return cfg
            secret = Path(answer).expanduser()
        if token is None:
            try:
                token = sign_in(cfg.data_dir, secret)
            except DriveError as exc:
                self.warn(f"Google Drive not connected: {exc}")
                return cfg
        self.ok(f"Google Drive: signed in as {token.account}")
        if not cfg.drive.enabled:
            text = set_toml_value(self.config_path.read_text(), "drive", "enabled", "true")
            write_private(self.config_path, text)
            self.ok("Google Drive turned on in the config")
            cfg = load_config(self.config_path)
        return cfg

    def _serve_args(self) -> list[str]:
        if self.opts.config_path is None:
            return ["serve"]
        return ["--config", str(self.config_path.resolve()), "serve"]

    def openclaw(self) -> None:
        openclaw = self.opts.openclaw or os.environ.get("OPENCLAW") or self.which("openclaw")
        if not openclaw:
            print("  OpenClaw not found. For any MCP client, the server command is:")
            print(f"    {shlex.join([self.exe, *self._serve_args()])}")
            return
        shown = self.run([openclaw, "mcp", "show", "filesonar"])
        if shown.returncode == 0 and self.exe in shown.stdout:
            self.ok("OpenClaw: filesonar is registered")
        elif shown.returncode == 0:
            # Registered by hand with another command: that may be on purpose.
            self.warn(
                "OpenClaw: filesonar is registered with another command, left as is "
                "(to use this one: openclaw mcp unset filesonar, then run setup again)"
            )
        else:
            cmd = [openclaw, "mcp", "add", "filesonar", "--command", self.exe]
            for arg in self._serve_args():
                cmd += ["--arg", arg]
            added = self.run([*cmd, "--timeout", "60"], timeout=180)
            if added.returncode != 0:
                detail = (added.stderr or added.stdout).strip().splitlines()[-1:]
                self.warn(f"OpenClaw: could not register filesonar: {' '.join(detail)}")
                return
            self.ok("OpenClaw: filesonar registered")
            self.run([openclaw, "mcp", "reload"])
        self._install_skill(openclaw)

    def _install_skill(self, openclaw: str) -> None:
        got = self.run([openclaw, "config", "get", "agents.defaults.workspace"])
        lines = got.stdout.strip().splitlines() if got.returncode == 0 else []
        workspace = Path(lines[-1]).expanduser() if lines else self.home / ".openclaw/workspace"
        skill = resources.files("filesonar").joinpath("skill/SKILL.md").read_text()
        target = workspace / "skills" / "filesonar" / "SKILL.md"
        if target.exists() and target.read_text() == skill:
            self.ok(f"OpenClaw: skill is up to date ({target})")
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(skill)
        self.ok(f"OpenClaw: skill installed ({target})")

    def claude_code(self, first_run: bool) -> None:
        claude = self.which("claude")
        if not claude:
            return
        if self.run([claude, "mcp", "get", "filesonar"]).returncode == 0:
            self.ok("Claude Code: filesonar is registered")
            return
        wanted = self.opts.claude_code
        if wanted is None and not first_run:
            return  # asked during the first setup already
        if wanted is None:
            wanted = self.ask.yes("Claude Code is installed. Add filesonar to it too?")
        if not wanted:
            return
        cmd = [claude, "mcp", "add", "--scope", "user", "filesonar", "--", self.exe]
        if self.run([*cmd, *self._serve_args()]).returncode == 0:
            self.ok("Claude Code: filesonar registered")
        else:
            self.warn("Claude Code: could not register filesonar")

    def service_unit(self) -> str:
        args = [self.exe]
        if self.opts.config_path is not None:
            args += ["--config", str(self.config_path.resolve())]
        exec_start = " ".join(f'"{a}"' if any(c.isspace() for c in a) else a for a in args)
        return (
            "# Written by `filesonar setup`; running setup again rewrites it.\n"
            "[Unit]\n"
            "Description=filesonar: keep the file content index up to date\n\n"
            "[Service]\n"
            "Type=simple\n"
            f"ExecStart={exec_start} watch\n"
            "Restart=on-failure\n"
            "RestartSec=30\n"
            "# Indexing is background work: stay out of the way of interactive programs.\n"
            "Nice=10\n"
            "IOSchedulingClass=idle\n"
            "NoNewPrivileges=yes\n"
            "UMask=0077\n\n"
            "[Install]\n"
            "WantedBy=default.target\n"
        )

    def service(self) -> bool:
        """Install and start the background watcher. Returns whether it runs."""
        systemctl = self.which("systemctl")
        if not systemctl or self.run([systemctl, "--user", "show-environment"]).returncode:
            print("  No systemd user session here. To keep the index current, leave this")
            print(f"  running somewhere: {shlex.join([self.exe, 'watch'])}")
            return False
        unit = self.home / ".config/systemd/user" / SERVICE_NAME
        wanted = self.opts.service
        if wanted is None and unit.exists():
            wanted = True  # installed before: keep it, refreshed for this version
        if wanted is None:
            wanted = self.ask.yes("Re-index files in the background as they change?")
        if not wanted:
            print("  skipped; run `filesonar index` now and then")
            return False
        text = self.service_unit()
        if not unit.exists() or unit.read_text() != text:
            unit.parent.mkdir(parents=True, exist_ok=True)
            unit.write_text(text)
            self.run([systemctl, "--user", "daemon-reload"])
        active = self.run([systemctl, "--user", "is-active", SERVICE_NAME]).returncode == 0
        action = ["restart"] if active else ["enable", "--now"]
        if active:  # also enable, in case it was started by hand
            self.run([systemctl, "--user", "enable", SERVICE_NAME])
        result = self.run([systemctl, "--user", *action, SERVICE_NAME])
        if result.returncode != 0:
            self.warn(f"could not start {SERVICE_NAME}: {result.stderr.strip()}")
            return False
        self.ok(f"{SERVICE_NAME} {'restarted' if active else 'started'} (starts with your session)")
        self._linger_note()
        return True

    def _linger_note(self) -> None:
        """On a server (no desktop), user services stop at logout unless lingering is on."""
        loginctl = self.which("loginctl")
        if not loginctl or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            return
        user = getpass.getuser()
        got = self.run([loginctl, "show-user", user, "--property=Linger", "--value"])
        if got.returncode == 0 and got.stdout.strip() == "no":
            print("    It stops when you log out. On a server, keep it running with:")
            print(f"      sudo loginctl enable-linger {user}")

    def first_index(self, cfg: Config, background: bool) -> None:
        from .db import Database

        db = Database(cfg.db_path)
        try:
            files = db.stats().get("files", 0)
        finally:
            db.close()
        if files:
            later = (
                "changes are picked up in the background"
                if background
                else ("run `filesonar index` to pick up changes")
            )
            self.ok(f"{files} file{'' if files == 1 else 's'} already indexed; {later}")
            return
        if background:
            self.ok("indexing has started in the background")
            print("    progress: `filesonar status`")
            print("    details:  journalctl --user -u filesonar-watch -f")
            return
        wanted = self.opts.index_now
        if wanted is None:
            wanted = self.ask.yes("Index your files now? (photos take about a second each)")
        if not wanted:
            print("  run `filesonar index` when you are ready")
            return
        from .cli import cmd_index

        cmd_index(cfg, _IndexArgs())

    def summary(self) -> None:
        print("\nDone." if not self.problems else "\nDone, with things to look at:")
        for problem in self.problems:
            print(f"  ! {problem}")
        print('\nTry it:  filesonar search "the lease for my apartment"')
        print('Or ask your agent:  "find my lease and send it to me"')

    def _find_executable(self) -> str:
        found = self.which("filesonar")
        if found:
            return str(Path(found).absolute())
        return str(Path(sys.argv[0]).absolute())


@dataclass
class _IndexArgs:
    rebuild: bool = False
    verbose: bool = False


SKILL_MARKER = "name: filesonar"
UNIT_MARKER = "filesonar: keep the file content index up to date"


class Uninstall(Setup):
    """Undo what setup did, so that `pipx uninstall filesonar` leaves nothing running.

    The index and config are kept unless asked (``purge``): they took time to build,
    and a reinstall picks them up again."""

    def __init__(self, opts: Options, *, purge: bool | None = None, **kwargs):
        super().__init__(opts, **kwargs)
        self.purge = purge

    def main(self) -> int:
        print("filesonar uninstall: removes what `filesonar setup` added.")
        cfg = self._load_existing()
        cfg = cfg if isinstance(cfg, Config) else None

        self.step("Background service")
        self.remove_service()
        self.step("AI agents")
        self.remove_openclaw()
        self.remove_claude_code()
        self.step("Google Drive")
        self.sign_out_of_drive(cfg)
        self.step("Index and settings")
        self.remove_data(cfg)

        print("\nDone." if not self.problems else "\nDone, with things to look at:")
        for problem in self.problems:
            print(f"  ! {problem}")
        print("\nTo remove the program itself:  pipx uninstall filesonar")
        return 0

    def remove_service(self) -> None:
        unit = self.home / ".config/systemd/user" / SERVICE_NAME
        systemctl = self.which("systemctl")
        if not unit.exists():
            self.ok("no background service installed")
            return
        if UNIT_MARKER not in unit.read_text():
            self.warn(f"{unit} was not written by filesonar; left as is")
            return
        if systemctl:
            self.run([systemctl, "--user", "disable", "--now", SERVICE_NAME])
        unit.unlink()
        if systemctl:
            self.run([systemctl, "--user", "daemon-reload"])
        self.ok(f"{SERVICE_NAME} stopped and removed")

    def remove_openclaw(self) -> None:
        openclaw = self.opts.openclaw or os.environ.get("OPENCLAW") or self.which("openclaw")
        if not openclaw:
            return
        shown = self.run([openclaw, "mcp", "show", "filesonar"])
        if shown.returncode == 0:
            if self.run([openclaw, "mcp", "unset", "filesonar"]).returncode == 0:
                self.ok("OpenClaw: filesonar unregistered")
                self.run([openclaw, "mcp", "reload"])
            else:
                self.warn("OpenClaw: could not unregister; run: openclaw mcp unset filesonar")
        got = self.run([openclaw, "config", "get", "agents.defaults.workspace"])
        lines = got.stdout.strip().splitlines() if got.returncode == 0 else []
        workspace = Path(lines[-1]).expanduser() if lines else self.home / ".openclaw/workspace"
        skill = workspace / "skills" / "filesonar" / "SKILL.md"
        if skill.is_file() and not skill.is_symlink() and SKILL_MARKER in skill.read_text():
            skill.unlink()
            with contextlib.suppress(OSError):
                skill.parent.rmdir()  # only if nothing else is in it
            self.ok(f"OpenClaw: skill removed ({skill})")

    def remove_claude_code(self) -> None:
        claude = self.which("claude")
        if not claude or self.run([claude, "mcp", "get", "filesonar"]).returncode != 0:
            return
        if self.run([claude, "mcp", "remove", "--scope", "user", "filesonar"]).returncode == 0:
            self.ok("Claude Code: filesonar removed")
        else:
            self.warn("Claude Code: could not remove; run: claude mcp remove filesonar")

    def sign_out_of_drive(self, cfg: Config | None) -> None:
        import httpx

        from .config import default_data_dir
        from .drive import DriveClient, DriveError, NotSignedIn, token_path

        data_dir = cfg.data_dir if cfg else default_data_dir()
        path = token_path(data_dir)
        try:
            client = DriveClient.from_data_dir(data_dir)
        except NotSignedIn:
            self.ok("not signed in")
            return
        except DriveError as exc:
            self.warn(f"{exc}")
            return
        try:
            client.revoke()
            self.ok("Google access revoked")
        except (httpx.HTTPError, DriveError):
            self.warn(
                "could not reach Google to revoke access; remove filesonar at "
                "https://myaccount.google.com/permissions"
            )
        finally:
            client.close()
        path.unlink(missing_ok=True)
        self.ok("Drive sign-in deleted from this computer")

    def remove_data(self, cfg: Config | None) -> None:
        from . import outbox
        from .config import default_data_dir, default_outbox_dir

        out = cfg.outbox_dir if cfg else default_outbox_dir()
        if out.is_dir():
            removed = outbox.clean(out, keep_hours=0)
            with contextlib.suppress(OSError):
                out.rmdir()
            self.ok(f"removed {removed} file(s) prepared for sending")
        data_dir = cfg.data_dir if cfg else default_data_dir()
        # Only files filesonar creates, by name: data_dir may be a folder holding
        # other things, so it is never deleted wholesale.
        ours = [
            p
            for p in [*sorted(data_dir.glob("index*.sqlite3*")), self.config_path]
            if p.is_file() and not p.is_symlink()
        ]
        if not ours and not (data_dir / "models").is_dir():
            return
        purge = self.purge
        if purge is None:
            print("  The index (what filesonar learned about your files) and your settings:")
            for p in ours:
                print(f"    {p}")
            purge = self.ask.yes("Delete them too? Your own files are never touched", False)
        if not purge:
            self.ok("index and settings kept; `filesonar setup` would use them again")
            return
        for p in ours:
            p.unlink()
        self.ok(f"deleted the index and settings ({len(ours)} files)")
        # The speech model setup downloaded (Hugging Face cache layout).
        models = data_dir / "models"
        if models.is_dir() and not models.is_symlink():
            import shutil

            for entry in models.iterdir():
                ours = entry.name.startswith("models--") or entry.name == ".locks"
                if ours and entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry)
            with contextlib.suppress(OSError):
                models.rmdir()
            self.ok("deleted the speech model")
        for folder in (data_dir, self.config_path.parent):
            with contextlib.suppress(OSError):
                folder.rmdir()  # only if now empty
