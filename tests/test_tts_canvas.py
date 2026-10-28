"""Narration canvas duration regression; real FFmpeg checks opt in on Kaggle.

Run with DUBBINGSTORY_FFMPEG_TESTS=1 python -m pytest -q tests/test_tts_canvas.py.
No test calls a TTS provider, LLM, or downloader.
"""
import array
import math
import os
import shutil
import sys
import wave
from types import SimpleNamespace

import pytest

from dubbingstory.tts import voice_manager as voices


RATE = 24000


def write_tone(path, frames, frequency=440):
    samples = array.array("h", (round(6000 * math.sin(2 * math.pi * frequency * index / RATE))
                               for index in range(frames)))
    if sys.byteorder != "little":
        samples.byteswap()
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        audio.writeframes(samples.tobytes())


def test_pcm_duration_counts_samples_without_ffprobe(tmp_path, monkeypatch):
    path = tmp_path / "narration.wav"
    frames = round(22.167 * RATE)
    write_tone(path, frames)
    monkeypatch.setattr(voices.subprocess, "run", lambda *args, **kwargs: pytest.fail("PCM must not need FFprobe"))
    assert voices._get_audio_duration(str(path)) == frames / RATE


@pytest.mark.parametrize("stdout,returncode,expected", [
    ("2.125\n", 0, 2.125), ("2.125\n", 1, 0.0), ("N/A\n", 0, 0.0),
    ("nan\n", 0, 0.0), ("inf\n", 0, 0.0), ("-2\n", 0, 0.0), ("0\n", 0, 0.0),
])
def test_non_pcm_probe_rejects_missing_or_invalid_duration(tmp_path, monkeypatch, stdout, returncode, expected):
    path = tmp_path / "narration.mp3"
    path.write_bytes(b"fake non-PCM input")
    calls = []

    def probe(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)

    monkeypatch.setattr(voices.subprocess, "run", probe)
    assert voices._get_audio_duration(str(path)) == expected
    assert len(calls) == 1
    assert calls[0][0] == "ffprobe"
    assert calls[0][-1] == str(path)


@pytest.mark.parametrize("duration", [0.0, -1.0, float("nan"), float("inf")])
def test_canvas_rejects_invalid_target_before_starting_ffmpeg(tmp_path, monkeypatch, duration):
    monkeypatch.setattr(voices, "_run_subprocess", lambda *args: pytest.fail("Invalid canvas must not launch FFmpeg"))
    with pytest.raises(ValueError):
        voices._render_timeline_canvas(placements=[], output_path=str(tmp_path / "out.wav"),
                                       total_duration=duration, normalize_loudness=True)


MEDIA_TESTS = pytest.mark.skipif(
    os.environ.get("DUBBINGSTORY_FFMPEG_TESTS") != "1"
    or not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="Opt in on Kaggle with DUBBINGSTORY_FFMPEG_TESTS=1 and FFmpeg/FFprobe",
)


def rms(samples, start, end):
    window = samples[round(start * RATE):round(end * RATE)]
    assert window
    return math.sqrt(sum((sample / 32768) ** 2 for sample in window) / len(window))


@MEDIA_TESTS
@pytest.mark.parametrize("duration", [22.167, 667 / 30], ids=["reported-rednote", "fractional-frame-timeline"])
@pytest.mark.parametrize("normalize", [False, True], ids=["plain", "loudnorm"])
def test_canvas_exact_length_preserves_gaps_and_speech_at_video_end(tmp_path, duration, normalize):
    first, last, output = (tmp_path / name for name in ("first.wav", "last.wav", "canvas.wav"))
    write_tone(first, RATE // 2, frequency=440)
    write_tone(last, RATE // 2, frequency=880)
    # Millisecond-aligned start is within 1 ms of ending at the video boundary.
    # A duration-only padding fix must not pass after dropping the final tone.
    late_start = math.floor((duration - .5) * 1000) / 1000
    voices._render_timeline_canvas(
        placements=[{"path": str(first), "actual_start": .375},
                    {"path": str(last), "actual_start": late_start}],
        output_path=str(output), total_duration=duration, normalize_loudness=normalize,
    )
    with wave.open(str(output), "rb") as audio:
        assert audio.getframerate() == RATE
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getnframes() == round(duration * RATE)
        samples = array.array("h", audio.readframes(audio.getnframes()))
    if sys.byteorder != "little":
        samples.byteswap()
    assert voices._get_audio_duration(str(output)) == pytest.approx(duration, abs=1 / RATE)
    assert rms(samples, .05, .25) < .0001
    assert rms(samples, .45, .65) > .005
    assert rms(samples, duration / 2, duration / 2 + .25) < .0001
    assert rms(samples, duration - .075, duration - .025) > .005
