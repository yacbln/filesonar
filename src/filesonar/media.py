"""Audio and video: what was said, what is shown, and when.

Recordings are never loaded into memory. The indexer opens each file once (without
following symlinks) and hands the open descriptor to ``ffmpeg`` / ``ffprobe`` as
``/dev/fd/N``, so the file that was checked is the file that is read.

- ``probe``: duration, streams, recording date, title/artist from the container.
- ``decode_audio``: 16 kHz mono samples for speech recognition, capped in length.
- ``contact_sheet``: a few frames spread over a video, as one picture for the vision
  model (one description per video instead of one per frame).
- ``WhisperTranscriber``: speech to text on this computer with faster-whisper (the
  optional ``filesonar[media]`` extra). Its model is downloaded once by
  ``filesonar setup``; indexing never downloads anything.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from PIL import Image

log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {
    ".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".oga", ".opus", ".wma", ".amr",
}  # fmt: skip
VIDEO_EXTENSIONS = {
    ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".3gp", ".mpg", ".mpeg", ".wmv",
}  # fmt: skip
SPEECH_MODELS = ("tiny", "base", "small", "medium")
MODEL_SIZES = {"tiny": "75 MB", "base": "140 MB", "small": "460 MB", "medium": "1.5 GB"}
SAMPLE_RATE = 16_000
_PROBE_TIMEOUT = 60
# Below these, a "transcript" is most likely music or noise read as words.
_MIN_LANGUAGE_PROBABILITY = 0.5
_MAX_NO_SPEECH = 0.6
_MIN_AVG_LOGPROB = -1.0


class MediaError(RuntimeError):
    pass


def tools_available() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _fd_path(fd: int) -> str:
    return f"/dev/fd/{fd}"


def _run(cmd: list[str], fd: int, timeout: float, max_output: int | None = None) -> bytes:
    """Run ffmpeg/ffprobe on an open file descriptor, without a shell."""
    try:
        proc = subprocess.run(  # a fixed program; file names never reach a shell
            cmd,
            pass_fds=(fd,),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"{Path(cmd[0]).name} took longer than {timeout:.0f}s") from exc
    except OSError as exc:
        raise MediaError(f"cannot run {cmd[0]}: {exc}") from exc
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip().splitlines()[-1:]
        raise MediaError(f"{Path(cmd[0]).name} failed: {' '.join(detail)[:200]}")
    if max_output is not None and len(proc.stdout) > max_output:
        return proc.stdout[:max_output]
    return proc.stdout


# -- probing -----------------------------------------------------------------------


@dataclass
class Probe:
    duration: float = 0.0
    has_audio: bool = False
    has_video: bool = False
    width: int = 0
    height: int = 0
    created: str | None = None  # YYYY-MM-DD
    tags: dict[str, str] = field(default_factory=dict)


def probe(fd: int) -> Probe:
    raw = _run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
         _fd_path(fd)],
        fd,
        _PROBE_TIMEOUT,
    )  # fmt: skip
    try:
        info = json.loads(raw or b"{}")
    except ValueError as exc:
        raise MediaError("ffprobe returned unreadable output") from exc
    out = Probe()
    fmt = info.get("format") or {}
    try:
        out.duration = float(fmt.get("duration") or 0)
    except ValueError:
        out.duration = 0.0
    tags = {str(k).lower(): str(v) for k, v in (fmt.get("tags") or {}).items()}
    for stream in info.get("streams") or []:
        kind = stream.get("codec_type")
        if kind == "audio":
            out.has_audio = True
        elif kind == "video" and not (stream.get("disposition") or {}).get("attached_pic"):
            # Album art inside an MP3 shows up as a "video" stream; it is not a video.
            out.has_video = True
            out.width = out.width or int(stream.get("width") or 0)
            out.height = out.height or int(stream.get("height") or 0)
        for k, v in (stream.get("tags") or {}).items():
            tags.setdefault(str(k).lower(), str(v))
    out.tags = tags
    created = tags.get("creation_time") or tags.get("date") or ""
    if len(created) >= 10 and created[4] == "-" and not created.startswith("1970"):
        out.created = created[:10]
    return out


def duration_text(seconds: float) -> str:
    seconds = round(seconds)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# -- audio ---------------------------------------------------------------------------


def decode_audio(fd: int, max_seconds: float):
    """Mono 16 kHz float32 samples of at most ``max_seconds`` (numpy array)."""
    import numpy as np  # comes with the media extra

    limit = int(max_seconds * SAMPLE_RATE) * 4
    pcm = _run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", _fd_path(fd), "-t", str(int(max_seconds)),
         "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
        fd,
        timeout=max(120.0, max_seconds),  # decoding is far faster than real time
        max_output=limit,
    )  # fmt: skip
    usable = len(pcm) - len(pcm) % 4
    return np.frombuffer(pcm[:usable], dtype=np.float32)


@dataclass
class Transcript:
    language: str | None
    parts: list[tuple[float, str]]  # (start in seconds, text)


class Transcriber(Protocol):
    name: str

    def available(self) -> bool: ...

    def transcribe(self, samples) -> Transcript: ...


class WhisperTranscriber:
    """faster-whisper on the CPU (int8). The model must already be downloaded."""

    def __init__(self, model: str, models_dir: Path, threads: int | None = None):
        self.name = f"whisper-{model}"
        self.model_name = model
        self.models_dir = models_dir
        self.threads = threads or max(1, (os.cpu_count() or 2) // 2)
        self._model = None
        self._available: bool | None = None

    @staticmethod
    def installed() -> bool:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False
        return True

    def ready(self) -> bool:
        """Installed and its model downloaded (no warning if not)."""
        return self.installed() and self._load(download=False) is not None

    def available(self) -> bool:
        if self._available is None:
            self._available = self.installed() and self._load(download=False) is not None
            if not self._available:
                log.warning(
                    "speech recognition is not set up, so audio and video are indexed "
                    "without what is said in them; run `filesonar setup` to add it"
                )
        return self._available

    def download(self) -> None:
        """Fetch the model (used by setup, the only place that downloads it)."""
        if self._load(download=True) is None:
            raise MediaError(f"could not download the speech model {self.model_name!r}")
        self._available = True

    def _load(self, download: bool):
        if self._model is not None:
            return self._model
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_name,
                device="cpu",
                compute_type="int8",
                download_root=str(self.models_dir),
                local_files_only=not download,
                cpu_threads=self.threads,
            )
        except Exception as exc:  # missing files, broken install, no network
            log.debug("speech model %s not loaded: %s", self.model_name, exc)
            return None
        return self._model

    def transcribe(self, samples) -> Transcript:
        model = self._load(download=False)
        if model is None or len(samples) < SAMPLE_RATE // 2:
            return Transcript(None, [])
        # The language is detected from the first 30 seconds before anything else is
        # transcribed (``segments`` is lazy). Low confidence means music or noise: the
        # "words" would only mislead search, so the rest is not even transcribed.
        segments, info = model.transcribe(
            samples, vad_filter=True, beam_size=1, condition_on_previous_text=False
        )
        if info.language_probability < _MIN_LANGUAGE_PROBABILITY:
            return Transcript(None, [])
        parts = [
            (seg.start, seg.text.strip())
            for seg in segments
            if seg.text.strip()
            and seg.no_speech_prob < _MAX_NO_SPEECH
            and seg.avg_logprob > _MIN_AVG_LOGPROB
        ]
        return Transcript(info.language if parts else None, parts)


def transcript_text(transcript: Transcript, every: float = 60.0) -> str:
    """Spoken text with a time mark at least every ``every`` seconds: "[1:05] ..."."""
    out: list[str] = []
    next_mark = 0.0
    for start, text in transcript.parts:
        if start >= next_mark:
            out.append(f"\n[{duration_text(start)}] ")
            next_mark = math.floor(start / every) * every + every
        out.append(text + " ")
    return "".join(out).strip()


# -- video frames --------------------------------------------------------------------


def frame_times(duration: float, count: int) -> list[float]:
    """Moments spread over the video, avoiding the very start and end (often black)."""
    if count <= 0:
        return []
    if duration <= 0:
        return [0.0]
    return [duration * (i + 0.5) / count for i in range(count)]


def contact_sheet(fd: int, times: Sequence[float], side: int = 512) -> bytes | None:
    """Frames at ``times`` side by side in one PNG (2 columns), or None if none read."""
    frames: list[Image.Image] = []
    for t in times:
        try:
            png = _run(
                ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{t:.2f}", "-i", _fd_path(fd),
                 "-frames:v", "1", "-vf", f"scale={side}:{side}:force_original_aspect_ratio="
                 "decrease", "-f", "image2pipe", "-vcodec", "png", "pipe:1"],
                fd,
                timeout=60,
                max_output=20 * 1024 * 1024,
            )  # fmt: skip
            if png:
                frames.append(Image.open(io.BytesIO(png)).convert("RGB"))
        except (MediaError, OSError, ValueError) as exc:
            log.debug("frame at %.1fs not read: %s", t, exc)
    if not frames:
        return None
    cols = 1 if len(frames) == 1 else 2
    rows = math.ceil(len(frames) / cols)
    w = max(f.width for f in frames)
    h = max(f.height for f in frames)
    sheet = Image.new("RGB", (cols * w, rows * h), "black")
    for i, frame in enumerate(frames):
        sheet.paste(frame, ((i % cols) * w, (i // cols) * h))
    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    return buf.getvalue()
