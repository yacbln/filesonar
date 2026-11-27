"""Audio and video. Test recordings are made with ffmpeg (a test pattern and a tone);
speech recognition is replaced by a stand-in, since the real model is an optional
extra. tests/test_integration.py runs the real one when it is installed."""

import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import FakeCaptioner, FakeEmbedder

from filesonar import media
from filesonar.db import Database
from filesonar.extract import ExtractionError, extract_media, open_regular_file
from filesonar.indexer import Indexer
from filesonar.media import Transcript

needs_ffmpeg = pytest.mark.skipif(not media.tools_available(), reason="ffmpeg is not installed")


def make(path: Path, *, video=True, audio=True, seconds=2, tags=None, art=False) -> Path:
    """A small recording: colour bars and/or a 440 Hz tone."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    if video:
        cmd += ["-f", "lavfi", "-i", f"testsrc=size=320x240:rate=10:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    if art:
        cover = path.with_suffix(".png")
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
             "color=c=red:size=64x64", "-frames:v", "1", str(cover)],
            check=True,
        )  # fmt: skip
        cmd += ["-i", str(cover), "-map", "0:a", "-map", "1:v", "-disposition:v", "attached_pic"]
    for key, value in (tags or {}).items():
        cmd += ["-metadata", f"{key}={value}"]
    if path.suffix == ".mp4":
        cmd += ["-pix_fmt", "yuv420p"] if video else []
    cmd += ["-shortest", str(path)]
    subprocess.run(cmd, check=True)
    return path


class FakeTranscriber:
    name = "fake-speech"

    def __init__(self, parts=((0.0, "hello from the birthday party"),), available=True):
        self.parts = list(parts)
        self._available = available
        self.calls = 0

    def available(self):
        return self._available

    def transcribe(self, samples):
        self.calls += 1
        assert len(samples) > 0  # real decoded sound reached the model
        return Transcript("en" if self.parts else None, self.parts)


@pytest.fixture
def media_indexer(cfg, db, embedder):
    def build(transcriber=None, captioner=None, **media_cfg):
        config = cfg
        if media_cfg:
            from dataclasses import replace

            config = replace(cfg, media=replace(cfg.media, **media_cfg))
        return Indexer(config, db, embedder, captioner or FakeCaptioner(), None, transcriber)

    return build


def chunks(db, name):
    return [
        (r["source"], r["text"])
        for r in db.conn.execute(
            "SELECT c.source, c.text FROM chunks c JOIN files f ON f.id = c.file_id "
            "WHERE f.name = ? ORDER BY c.ordinal",
            (name,),
        )
    ]


def row(db, name):
    return db.conn.execute("SELECT * FROM files WHERE name = ?", (name,)).fetchone()


# -- probing and helpers -------------------------------------------------------------


@needs_ffmpeg
def test_probe_reads_streams_size_and_tags(tmp_path):
    p = make(
        tmp_path / "v.mp4", tags={"title": "Birthday", "creation_time": "2024-06-01T10:00:00Z"}
    )
    fd, _ = open_regular_file(p, 10**9)
    try:
        info = media.probe(fd)
    finally:
        import os

        os.close(fd)
    assert info.has_video and info.has_audio and (info.width, info.height) == (320, 240)
    assert 1.5 < info.duration < 2.5
    assert info.tags["title"] == "Birthday" and info.created == "2024-06-01"


@needs_ffmpeg
def test_album_art_is_not_a_video(tmp_path):
    p = make(tmp_path / "song.mp3", video=False, art=True, tags={"artist": "Fairuz"})
    fd, _ = open_regular_file(p, 10**9)
    info = media.probe(fd)
    assert info.has_audio and not info.has_video


def test_transcript_has_time_marks():
    t = Transcript("en", [(0.0, "hello"), (30.0, "still the first minute"), (65.0, "later")])
    assert media.transcript_text(t) == "[0:00] hello still the first minute \n[1:05] later"


def test_frame_times_avoid_start_and_end():
    assert media.frame_times(100, 4) == [12.5, 37.5, 62.5, 87.5]
    assert media.frame_times(0, 4) == [0.0] and media.frame_times(50, 0) == []


@pytest.mark.parametrize(("seconds", "text"), [(5, "0:05"), (65, "1:05"), (3725, "1:02:05")])
def test_duration_text(seconds, text):
    assert media.duration_text(seconds) == text


# -- indexing ----------------------------------------------------------------------------


@needs_ffmpeg
def test_video_with_speech_and_frames(home, db, media_indexer):
    make(home / "party.mp4")
    speech, vision = FakeTranscriber(), FakeCaptioner("children around a birthday cake")
    report = media_indexer(speech, vision).run()
    assert report.added == 1 and report.failed == 0
    got = chunks(db, "party.mp4")
    assert ("speech", "[0:00] hello from the birthday party") in got
    assert ("caption", "children around a birthday cake") in got
    r = row(db, "party.mp4")
    assert r["kind"] == "video" and r["captioned"] and r["ocr"]
    assert r["details"].startswith("duration: 0:02") and "size: 320x240" in r["details"]
    assert "language: en" in r["details"]
    assert media_indexer(speech, vision).run().unchanged == 1  # settled


@needs_ffmpeg
def test_silent_video_is_described_and_says_so(home, db, media_indexer):
    make(home / "kittens.webm", audio=False)
    speech = FakeTranscriber()
    media_indexer(speech).run()
    assert speech.calls == 0
    assert "no sound" in row(db, "kittens.webm")["details"]


@needs_ffmpeg
def test_music_is_found_by_its_tags_not_by_made_up_words(home, db, media_indexer):
    make(home / "track.mp3", video=False, tags={"title": "Ya Rayah", "artist": "Dahmane"})
    media_indexer(FakeTranscriber(parts=[])).run()  # the model heard no speech
    r = row(db, "track.mp3")
    assert r["kind"] == "audio" and r["status"] == "empty"
    assert "title: Ya Rayah" in r["details"] and "artist: Dahmane" in r["details"]


@needs_ffmpeg
def test_sound_only_mp4_is_not_read_again_and_again(home, db, media_indexer):
    make(home / "voice-note.mp4", video=False)
    vision = FakeCaptioner()
    first = media_indexer(FakeTranscriber(), vision).run()
    assert first.added == 1 and vision.calls == 0
    assert media_indexer(FakeTranscriber(), vision).run().unchanged == 1


def test_without_ffmpeg_recordings_are_findable_by_name_then_read_later(
    home, db, media_indexer, monkeypatch
):
    if not media.tools_available():
        pytest.skip("needs ffmpeg for the second half")
    make(home / "wedding speech.mp4")
    monkeypatch.setattr(media, "tools_available", lambda: False)
    assert media_indexer(FakeTranscriber()).run().added == 1
    r = row(db, "wedding speech.mp4")
    assert r["ocr"] == 0 and chunks(db, "wedding speech.mp4") == [("metadata", "wedding speech")]
    assert media_indexer(FakeTranscriber()).run().unchanged == 1  # nothing to do yet

    monkeypatch.undo()  # ffmpeg is installed now
    assert media_indexer(FakeTranscriber()).run().updated == 1
    assert ("speech", "[0:00] hello from the birthday party") in chunks(db, "wedding speech.mp4")
    assert media_indexer(FakeTranscriber()).run().unchanged == 1


@needs_ffmpeg
def test_speech_set_up_later_reads_recordings_again(home, db, media_indexer):
    make(home / "memo.m4a", video=False)
    assert media_indexer(FakeTranscriber(available=False)).run().added == 1
    assert row(db, "memo.m4a")["ocr"] == 0
    assert media_indexer(FakeTranscriber()).run().updated == 1
    assert row(db, "memo.m4a")["ocr"] == 1


@needs_ffmpeg
def test_vision_model_installed_later_describes_videos(home, db, media_indexer):
    make(home / "clip.mkv")
    media_indexer(FakeTranscriber(), FakeCaptioner(available=False)).run()
    assert row(db, "clip.mkv")["captioned"] == 0
    assert media_indexer(FakeTranscriber(), FakeCaptioner()).run().updated == 1
    assert row(db, "clip.mkv")["captioned"] == 1


@needs_ffmpeg
def test_crashing_speech_model_costs_only_the_transcript(home, db, media_indexer):
    class Broken(FakeTranscriber):
        def transcribe(self, samples):
            raise RuntimeError("CUDA out of memory")

    make(home / "a.mp4")
    report = media_indexer(Broken()).run()
    assert report.added == 1 and report.failed == 0
    assert ("caption", "a photo of a red bicycle parked by a lake") in chunks(db, "a.mp4")


@needs_ffmpeg
def test_damaged_recording_is_an_error_not_a_crash(home, db, media_indexer):
    (home / "broken.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42 not really a video")
    report = media_indexer(FakeTranscriber()).run()
    assert report.failed == 1 and row(db, "broken.mp4")["status"] == "error"


def test_recordings_over_the_size_limit_are_skipped(home, db, media_indexer):
    (home / "huge.mp4").write_bytes(b"\x00" * 2_000_000)
    report = media_indexer(FakeTranscriber(), max_size_mb=1).run()
    assert report.skipped == 1 and row(db, "huge.mp4") is None


def test_media_can_be_turned_off(home, db, media_indexer):
    (home / "a.mp3").write_bytes(b"ID3 fake")
    assert media_indexer(FakeTranscriber(), enabled=False).run().scanned == 0


@needs_ffmpeg
def test_extract_media_reads_the_open_file_not_the_name(tmp_path):
    """The descriptor is what ffmpeg reads: replacing the file afterwards changes nothing."""
    p = make(tmp_path / "a.mp4", seconds=1)
    fd, _ = open_regular_file(p, 10**9)
    shutil.move(p, tmp_path / "moved.mp4")
    p.write_bytes(b"something else entirely")
    from filesonar.config import MediaConfig

    out = extract_media(p, fd, MediaConfig(), FakeCaptioner(), FakeTranscriber())
    assert "size: 320x240" in out.details


def test_extract_media_reports_unreadable_files(tmp_path):
    if not media.tools_available():
        pytest.skip("ffmpeg is not installed")
    from filesonar.config import MediaConfig

    p = tmp_path / "x.mp4"
    p.write_bytes(b"junk")
    fd, _ = open_regular_file(p, 10**9)
    with pytest.raises(ExtractionError):
        extract_media(p, fd, MediaConfig(), None, None)


# -- the real speech model's filters, with a stand-in model ------------------------------


class _Seg:
    def __init__(self, start, text, no_speech=0.1, logprob=-0.3):
        self.start, self.text, self.no_speech_prob, self.avg_logprob = (
            start,
            text,
            no_speech,
            logprob,
        )


class _Info:
    def __init__(self, language="en", probability=0.95):
        self.language, self.language_probability = language, probability


def whisper_with(monkeypatch, segments, info):
    import numpy as np

    t = media.WhisperTranscriber("base", Path("/nonexistent"))

    class Model:
        def transcribe(self, samples, **kw):
            return iter(segments), info

    monkeypatch.setattr(t, "_load", lambda download: Model())
    return t, np.zeros(media.SAMPLE_RATE * 5, dtype="float32")


def test_whisper_drops_unsure_segments(monkeypatch):
    t, samples = whisper_with(
        monkeypatch,
        [_Seg(0, " hello "), _Seg(3, "noise", no_speech=0.9), _Seg(6, "mumble", logprob=-2.0)],
        _Info(),
    )
    assert t.transcribe(samples).parts == [(0, "hello")]


def test_whisper_skips_music_without_transcribing_it(monkeypatch):
    consumed = []

    def lazy():
        consumed.append(True)
        yield _Seg(0, "la la la")

    t, samples = whisper_with(monkeypatch, lazy(), _Info("ar", 0.31))
    out = t.transcribe(samples)
    assert out.parts == [] and out.language is None and not consumed


def test_whisper_never_downloads_while_indexing(monkeypatch, tmp_path):
    seen = {}

    class Fake:
        def __init__(self, *a, **kw):
            seen.update(kw)
            raise OSError("not downloaded")

    import sys
    import types

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=Fake))
    t = media.WhisperTranscriber("base", tmp_path)
    assert t.available() is False
    assert seen["local_files_only"] is True and seen["download_root"] == str(tmp_path)


# -- Google Drive ----------------------------------------------------------------------------


def test_drive_videos_are_described_from_their_preview_without_downloading(cfg, embedder, tmp_path):
    from dataclasses import replace

    from fakedrive import FakeDrive
    from samples import photo

    from filesonar.drive_index import DriveIndexer

    dcfg = replace(cfg, drive=replace(cfg.drive, enabled=True))
    db = Database(dcfg.db_path)
    fake = FakeDrive()
    still = photo(tmp_path / "still.jpg").read_bytes()
    f = fake.add(
        "VID_2024.mp4",
        b"\x00" * 64,
        size=str(800 * 1024 * 1024),
        thumbnail=still,
        video={"durationMillis": "95000", "width": 1920, "height": 1080},
    )
    report = DriveIndexer(
        dcfg, db, embedder, fake.client(), FakeCaptioner("a beach at sunset")
    ).sync()
    assert report.added == 1 and f.id not in fake.downloads
    r = db.conn.execute("SELECT * FROM files").fetchone()
    assert r["kind"] == "video" and r["details"] == "duration: 1:35; size: 1920x1080"
    assert r["sha256"].startswith("md5:")
    text = [x[0] for x in db.conn.execute("SELECT text FROM chunks WHERE source = 'caption'")]
    assert text == ["a beach at sunset"]
    db.close()


@needs_ffmpeg
def test_drive_copy_of_a_local_video_gets_its_transcript(cfg, home, embedder):
    from dataclasses import replace

    from fakedrive import FakeDrive

    from filesonar.drive_index import DriveIndexer

    dcfg = replace(cfg, drive=replace(cfg.drive, enabled=True))
    db = Database(dcfg.db_path)
    video = make(home / "toast.mp4")
    Indexer(dcfg, db, embedder, FakeCaptioner(), None, FakeTranscriber()).run()
    fake = FakeDrive()
    f = fake.add("toast.mp4", video.read_bytes())
    DriveIndexer(dcfg, db, FakeEmbedder(), fake.client(), FakeCaptioner()).sync()
    drive_text = [
        r[0]
        for r in db.conn.execute(
            "SELECT c.text FROM chunks c JOIN files f ON f.id = c.file_id "
            "WHERE f.origin = 'drive' AND c.source = 'speech'"
        )
    ]
    assert drive_text == ["[0:00] hello from the birthday party"] and f.id not in fake.downloads
    db.close()
