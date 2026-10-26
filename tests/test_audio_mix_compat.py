"""Stereo AAC + mono Edge WAV regression from the Kaggle render failure.

Media checks are opt-in: DUBBINGSTORY_FFMPEG_TESTS=1 python -m pytest -q
tests/test_audio_mix_compat.py. No external providers or models are required.
"""
import array
import json
import math
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from dubbingstory.render.audio_mix import build_audio_filter
from dubbingstory.render.video_render import render_final


def test_ducking_pins_layout_rate_and_sample_format_on_both_compressor_inputs():
    graph = build_audio_filter(duration=3, has_original_audio=True)
    compressor = "aformat=sample_fmts=dbl:sample_rates=48000:channel_layouts=stereo"
    mixing = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
    assert f"[orig]{compressor}[compress_main]" in graph
    assert f"[side_in]{compressor}[compress_side]" in graph
    assert "[compress_main][compress_side]sidechaincompress=" in graph
    assert f"[ducked]{mixing}[duckmix]" in graph
    assert "[duckmix][narrmix]amix=inputs=2:duration=first:normalize=0[out]" in graph
    assert graph.count("[0:a:0]") == graph.count("[1:a:0]") == 1


@pytest.mark.parametrize("has_audio,volume,strategy", [
    (False, .15, "dynamic_duck"), (True, 0, "dynamic_duck"),
    (True, .15, "replace"), (True, .15, "mute_original"),
])
def test_narration_only_does_not_add_a_compressor_or_force_stereo(has_audio, volume, strategy):
    graph = build_audio_filter(duration=3, has_original_audio=has_audio,
                               original_volume=volume, strategy=strategy)
    assert "[0:a:0]" not in graph
    assert "sidechaincompress=" not in graph
    assert "channel_layouts=stereo" not in graph
    assert "apad,atrim=duration=3.000000" in graph


def test_static_duck_has_matching_formats_but_no_compressor():
    graph = build_audio_filter(duration=3, has_original_audio=True, strategy="duck")
    assert graph.count("aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo") == 2
    assert "sidechaincompress=" not in graph
    assert "amix=inputs=2:duration=first:normalize=0" in graph


MEDIA_TESTS = pytest.mark.skipif(
    os.environ.get("DUBBINGSTORY_FFMPEG_TESTS") != "1"
    or not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="Opt in on Kaggle with DUBBINGSTORY_FFMPEG_TESTS=1 and FFmpeg/FFprobe",
)


def run_media(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    return result.stdout


def probe(path):
    return json.loads(run_media(["ffprobe", "-v", "error", "-count_frames",
                                 "-show_streams", "-of", "json", str(path)]))["streams"]


def tone_level(samples, start, frequency=200, rate=48000):
    """Measure only the source tone, independently of the narrator tone."""
    window = samples[round(start * rate):round((start + .2) * rate)]
    real = sum(value * math.cos(2 * math.pi * frequency * index / rate)
               for index, value in enumerate(window))
    imag = sum(value * math.sin(2 * math.pi * frequency * index / rate)
               for index, value in enumerate(window))
    return 2 * math.hypot(real, imag) / len(window)


@MEDIA_TESTS
@pytest.mark.parametrize("ratio", [None, "9:16"], ids=["original-16x9", "layered-9x16"])
def test_kaggle_stereo_aac_mono_wav_ducks_and_renders_both_ratios(tmp_path, ratio):
    source, voice, output = (tmp_path / name for name in ("summary_source.mp4", "summary_audio_id.wav", "final.mp4"))
    # Match the failing notebook: stereo AAC 44.1 kHz + mono PCM 24 kHz.
    run_media(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=12:duration=3",
               "-f", "lavfi", "-i", "sine=frequency=200:sample_rate=44100:duration=3",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "2", str(source)])
    run_media(["ffmpeg", "-y", "-f", "lavfi", "-i",
               "aevalsrc=if(between(t\\,0.8\\,1.6)\\,0.5*sin(2*PI*1000*t)\\,0):s=24000:d=3",
               "-ac", "1", "-c:a", "pcm_s16le", str(voice)])
    original_streams, voice_streams = probe(source), probe(voice)
    original_audio = next(item for item in original_streams if item["codec_type"] == "audio")
    assert (original_audio["sample_rate"], original_audio["channels"]) == ("44100", 2)
    assert (voice_streams[0]["sample_rate"], voice_streams[0]["channels"]) == ("24000", 1)
    cfg = SimpleNamespace(render_vertical_layout="layered", render_background_blur=5,
                          render_original_volume=.15, tts_narration_fit="legacy", subtitle_style="classic")
    render_final(str(source), str(voice), str(output), target_ratio=ratio,
                 audio_strategy="dynamic_duck", original_volume=.15, cfg=cfg)
    streams = probe(output)
    video = next(item for item in streams if item["codec_type"] == "video")
    audio = [item for item in streams if item["codec_type"] == "audio"]
    assert len(audio) == 1
    assert (audio[0]["sample_rate"], audio[0]["channels"]) == ("48000", 2)
    assert (video["width"], video["height"]) == ((1080, 1920) if ratio else (320, 180))
    assert int(video["nb_read_frames"]) == 36
    assert float(video["duration"]) == pytest.approx(3, abs=1 / 12)
    assert float(audio[0]["duration"]) == pytest.approx(3, abs=.06)
    samples = array.array("f")
    samples.frombytes(run_media(["ffmpeg", "-v", "error", "-i", str(output), "-map", "0:a:0",
                                "-ac", "1", "-ar", "48000", "-f", "f32le", "-"]))
    if sys.byteorder != "little":
        samples.byteswap()
    before, during, after = [tone_level(samples, start) for start in (.3, 1.1, 2.5)]
    assert before > .001
    assert during < .65 * before
    assert after > .75 * before
