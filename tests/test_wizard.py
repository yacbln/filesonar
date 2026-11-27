"""`filesonar setup`, run against fake Ollama, OpenClaw, Claude Code and systemd."""

import stat
import subprocess
from pathlib import Path

import pytest

from filesonar import wizard
from filesonar.config import load_config, render_config, set_toml_value, tomllib
from filesonar.drive import Token, save_token, token_path
from filesonar.wizard import Options, Prompter, Setup

EXE = "/opt/bin/filesonar"


class FakeOllama:
    def __init__(self, installed=(), up=True):
        self.installed = set(installed)
        self.up = up
        self.pulled = []

    def __call__(self, cfg):  # used as the client factory
        return self

    def version(self):
        return "0.12.0" if self.up else None

    def installed_models(self):
        return set(self.installed)

    def pull(self, name, progress=None):
        self.pulled.append(name)
        if progress:
            progress(50, 100)
            progress(100, 100)
        self.installed.add(name)


class FakeSystem:
    """Records commands; answers like OpenClaw, Claude Code and systemd would."""

    def __init__(self, workspace: Path, systemd=True):
        self.calls = []
        self.workspace = workspace
        self.systemd = systemd
        self.openclaw_servers = {}
        self.claude_servers = set()
        self.active = False
        self.enabled = False

    def __call__(self, cmd, timeout=120):
        self.calls.append(list(cmd))
        name, *rest = cmd

        def done(code=0, out=""):
            return subprocess.CompletedProcess(cmd, code, out, "" if code == 0 else "failed")

        if name == "openclaw":
            if rest[:2] == ["mcp", "show"]:
                server = self.openclaw_servers.get(rest[2])
                return done(0, f'{{"command": "{server}"}}') if server else done(1)
            if rest[:2] == ["mcp", "add"]:
                self.openclaw_servers[rest[2]] = rest[rest.index("--command") + 1]
                return done()
            if rest[:2] == ["config", "get"]:
                return done(0, f"{self.workspace}\n")
            return done()
        if name == "claude":
            if rest[:2] == ["mcp", "get"]:
                return done(0 if rest[2] in self.claude_servers else 1)
            if rest[:2] == ["mcp", "add"]:
                self.claude_servers.add("filesonar")
                return done()
        if name == "systemctl":
            if not self.systemd:
                return done(1)
            if "is-active" in rest:
                return done(0 if self.active else 3)
            if "enable" in rest:
                self.enabled = True
            if "--now" in rest or "restart" in rest:
                self.active = True
            return done()
        return done(127)

    def ran(self, *prefix):
        return [c for c in self.calls if c[: len(prefix)] == list(prefix)]


class Answers:
    """Scripted answers to questions; fails the test on an unexpected question."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.questions = []

    def __call__(self, question):
        self.questions.append(question)
        assert self.answers, f"unexpected question: {question}"
        return self.answers.pop(0)


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    for name in ("Documents", "Pictures"):
        (home / name).mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("OPENCLAW", raising=False)
    monkeypatch.setattr(wizard, "tesseract_available", lambda: True)
    system = FakeSystem(home / ".openclaw" / "workspace")
    tools = {"openclaw": "openclaw", "claude": "claude", "systemctl": "systemctl"}

    def make(opts=None, answers=None, ollama=None, which=None, assume_yes=None):
        opts = opts or Options()
        prompter = Prompter(
            opts.assume_yes if assume_yes is None else assume_yes,
            answers if answers is not None else Answers(),
        )
        lookup = tools if which is None else which
        return Setup(
            opts,
            prompter=prompter,
            run=system,
            which=lookup.get,
            ollama=ollama or FakeOllama(["nomic-embed-text", "qwen2.5vl:7b"]),
            executable=EXE,
            home=home,
        )

    make.home = home
    make.system = system
    make.config = home / ".config" / "filesonar" / "config.toml"
    make.unit = home / ".config/systemd/user/filesonar-watch.service"
    make.skill = system.workspace / "skills/filesonar/SKILL.md"
    return make


def test_fresh_machine_with_defaults(env, capsys):
    ollama = FakeOllama(up=True)
    setup = env(Options(assume_yes=True), ollama=ollama, which={
        "openclaw": "openclaw", "claude": "claude", "systemctl": "systemctl",
        "nvidia-smi": "/usr/bin/nvidia-smi",
    })  # fmt: skip
    assert setup.main() == 0
    out = capsys.readouterr().out

    # Models: the required one, and the vision one because there is a GPU.
    assert ollama.pulled == ["nomic-embed-text", "qwen2.5vl:7b"]
    # Config: existing home folders only (no Downloads/Desktop here), private.
    cfg = load_config(env.config)
    assert cfg.roots == (env.home / "Documents", env.home / "Pictures")
    assert stat.S_IMODE(env.config.stat().st_mode) == 0o600
    assert stat.S_IMODE(env.config.parent.stat().st_mode) == 0o700
    assert cfg.models.caption_images and not cfg.drive.enabled  # Drive is never assumed
    # Agents.
    assert env.system.openclaw_servers == {"filesonar": EXE}
    add = env.system.ran("openclaw", "mcp", "add")[0]
    assert add[add.index("--arg") + 1] == "serve"
    assert env.system.ran("openclaw", "mcp", "reload")
    assert "Use filesonar" in env.skill.read_text() or env.skill.read_text().startswith("---")
    assert env.system.claude_servers == {"filesonar"}
    claude_add = env.system.ran("claude", "mcp", "add")[0]
    assert claude_add[-3:] == ["--", EXE, "serve"] and "user" in claude_add
    # Background service, which also does the first index.
    assert f"ExecStart={EXE} watch" in env.unit.read_text()
    assert env.system.active and env.system.enabled
    assert "indexing has started in the background" in out
    assert "Done." in out


def test_running_again_changes_nothing(env, capsys):
    assert env(Options(assume_yes=True)).main() == 0
    config_before = env.config.read_text()
    unit_mtime = env.unit.stat().st_mtime_ns
    env.system.calls.clear()
    capsys.readouterr()

    assert env(Options(), answers=Answers()).main() == 0  # no questions asked at all
    out = capsys.readouterr().out
    assert env.config.read_text() == config_before
    assert env.unit.stat().st_mtime_ns == unit_mtime
    assert not env.system.ran("openclaw", "mcp", "add")
    assert not env.system.ran("claude", "mcp", "add")
    assert not env.system.ran("systemctl", "--user", "daemon-reload")
    # The watcher restarts so a new version of filesonar takes effect.
    assert env.system.ran("systemctl", "--user", "restart", "filesonar-watch.service")
    assert "skill is up to date" in out and "config kept" in out


def test_a_newer_skill_replaces_the_installed_one(env, capsys):
    env.skill.parent.mkdir(parents=True)
    env.skill.write_text("old skill")
    env(Options(assume_yes=True)).main()
    assert env.skill.read_text() != "old skill"
    assert "skill installed" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("which", "message"),
    [({"ollama": "/usr/bin/ollama"}, "not answering"), ({}, "ollama.com/install.sh")],
)
def test_without_ollama_setup_stops_before_writing_anything(env, capsys, which, message):
    setup = env(Options(assume_yes=True), ollama=FakeOllama(up=False), which=which)
    assert setup.main() == 1
    assert message in capsys.readouterr().out
    assert not env.config.exists() and not env.system.calls


def test_saying_no_everywhere(env, capsys):
    ollama = FakeOllama(["nomic-embed-text"])
    answers = Answers(
        "n",  # download the vision model? (no GPU, so the default is already no)
        "n",  # use these folders?
        f"{env.home / 'Pictures'}",  # folders to search
        "",  # Google Drive? default no
        "n",  # Claude Code?
        "n",  # background service?
        "n",  # index now?
    )
    assert env(answers=answers, ollama=ollama).main() == 0
    assert not answers.answers, "every scripted answer was used"
    cfg = load_config(env.config)
    assert cfg.roots == (env.home / "Pictures",)
    assert not cfg.models.caption_images and ollama.pulled == []
    assert env.system.openclaw_servers  # OpenClaw is registered without asking
    assert not env.system.claude_servers and not env.unit.exists()
    assert "run `filesonar index` when you are ready" in capsys.readouterr().out


def test_no_systemd_offers_to_index_now(env, monkeypatch, capsys):
    env.system.systemd = False
    indexed = []
    monkeypatch.setattr("filesonar.cli.cmd_index", lambda cfg, args: indexed.append(cfg))
    assert env(Options(assume_yes=True)).main() == 0
    assert "No systemd user session" in capsys.readouterr().out
    assert len(indexed) == 1 and not env.unit.exists()


def test_without_openclaw_prints_the_server_command(env, capsys):
    setup = env(Options(assume_yes=True), which={"systemctl": "systemctl"})
    assert setup.main() == 0
    out = capsys.readouterr().out
    assert "OpenClaw not found" in out and f"{EXE} serve" in out


def test_openclaw_registration_by_hand_is_left_alone(env, capsys):
    env.system.openclaw_servers["filesonar"] = "/home/me/dev/filesonar"
    assert env(Options(assume_yes=True)).main() == 0
    assert env.system.openclaw_servers["filesonar"] == "/home/me/dev/filesonar"
    assert not env.system.ran("openclaw", "mcp", "add")
    assert "registered with another command" in capsys.readouterr().out


def test_failed_openclaw_registration_is_reported(env, capsys):
    def failing(cmd, timeout=120):
        if cmd[:3] == ["openclaw", "mcp", "add"]:
            return subprocess.CompletedProcess(cmd, 1, "", "probe failed: timeout")
        return env.system(cmd, timeout)

    setup = env(Options(assume_yes=True))
    setup.run = failing
    assert setup.main() == 0
    out = capsys.readouterr().out
    assert "could not register filesonar: probe failed: timeout" in out
    assert "Done, with things to look at" in out


def test_custom_config_path_reaches_the_agents_and_the_service(env, tmp_path):
    custom = tmp_path / "elsewhere" / "fs.toml"
    assert env(Options(config_path=custom, assume_yes=True)).main() == 0
    assert custom.exists() and not env.config.exists()
    add = env.system.ran("openclaw", "mcp", "add")[0]
    assert add[add.index("--command") + 2 :][1::2][:3] == ["--config", str(custom), "serve"]
    assert f"ExecStart={EXE} --config {custom} watch" in env.unit.read_text()


def test_existing_config_is_kept_and_only_the_vision_switch_changes(env, capsys):
    env.config.parent.mkdir(parents=True)
    original = render_config([env.home / "Documents"]).replace(
        "exclude = []", 'exclude = ["*.log"]  # my own comment'
    )
    env.config.write_text(original)
    assert env(Options(assume_yes=True, vision=False)).main() == 0
    text = env.config.read_text()
    assert "# my own comment" in text
    assert text == original.replace("caption_images = true", "caption_images = false")


def test_existing_vision_choice_is_respected(env):
    env.config.parent.mkdir(parents=True)
    env.config.write_text(render_config([env.home / "Documents"], caption_images=False))
    ollama = FakeOllama(["nomic-embed-text"])
    assert env(Options(assume_yes=True), ollama=ollama).main() == 0
    assert ollama.pulled == [] and not load_config(env.config).models.caption_images


def test_broken_config_stops_setup(env, capsys):
    env.config.parent.mkdir(parents=True)
    env.config.write_text("roots = [")
    assert env(Options(assume_yes=True)).main() == 2
    assert "fix the config file" in capsys.readouterr().out


def test_missing_folder_is_refused(env, capsys):
    opts = Options(assume_yes=True, folders=[env.home / "nope"])
    assert env(opts).main() == 2
    assert "not a folder" in capsys.readouterr().out and not env.config.exists()


def test_drive_sign_in_turns_drive_on(env, monkeypatch, tmp_path, capsys):
    secret = tmp_path / "client_secret.json"
    signed = []

    def fake_sign_in(data_dir, client_secret, **kw):
        signed.append(client_secret)
        token = Token("cid", "csecret", "refresh", "me@example.com")
        save_token(token_path(data_dir), token)
        return token

    monkeypatch.setattr("filesonar.drive.sign_in", fake_sign_in)
    assert env(Options(assume_yes=True, drive_client_secret=secret)).main() == 0
    assert signed == [secret]
    assert load_config(env.config).drive.enabled
    assert "signed in as me@example.com" in capsys.readouterr().out
    # Next time it is already connected: no new sign-in.
    assert env(Options(assume_yes=True)).main() == 0
    assert len(signed) == 1


def test_missing_tesseract_is_a_warning_with_the_install_command(env, monkeypatch, capsys):
    monkeypatch.setattr(wizard, "tesseract_available", lambda: False)
    which = {"systemctl": "systemctl", "apt": "/usr/bin/apt"}
    assert env(Options(assume_yes=True), which=which).main() == 0
    out = capsys.readouterr().out
    assert "sudo apt install tesseract-ocr" in out and "things to look at" in out


def test_unit_quotes_paths_with_spaces(env):
    setup = env(Options(config_path=Path("/home/me/My Config/fs.toml")))
    assert 'ExecStart=/opt/bin/filesonar --config "/home/me/My Config/fs.toml" watch' in (
        setup.service_unit()
    )


# -- config helpers -----------------------------------------------------------


def test_set_toml_value_keeps_comments_and_only_touches_its_section():
    text = (
        "roots = []\n[models]\nenabled = true\n\n[drive]\n# a note\n"
        "enabled = false  # flip me\nfolders = []\n"
    )
    out = set_toml_value(text, "drive", "enabled", "true")
    assert out == text.replace("enabled = false  # flip me", "enabled = true  # flip me")
    assert tomllib.loads(out)["models"]["enabled"] is True


def test_set_toml_value_adds_missing_keys_and_sections():
    out = set_toml_value(
        'roots = ["~/a"]\n[models]\nhost = "x"\n', "models", "caption_images", "false"
    )
    assert tomllib.loads(out)["models"] == {"host": "x", "caption_images": False}
    out = set_toml_value('roots = ["~/a"]', "drive", "enabled", "true")
    assert tomllib.loads(out)["drive"] == {"enabled": True}


def test_set_toml_value_refuses_to_produce_broken_toml():
    with pytest.raises(tomllib.TOMLDecodeError):
        set_toml_value("[drive]\nenabled = false\n", "drive", "enabled", "tru e")


def test_render_config_handles_unusual_folder_names(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    odd = tmp_path / 'Café "quotes" \\ back'
    outside = Path("/srv/shared files")
    raw = tomllib.loads(render_config([odd, outside]))
    assert raw["roots"] == ['~/Café "quotes" \\ back', "/srv/shared files"]


def test_prompter_reprompts_until_it_understands():
    answers = Answers("maybe", "Y")
    assert Prompter(False, answers).yes("ok?", default=False) is True
    assert len(answers.questions) == 2
