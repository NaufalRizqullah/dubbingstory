"""Layered 9:16 contracts; media checks run only when explicitly enabled on Kaggle.

    DUBBINGSTORY_FFMPEG_TESTS=1 python -m pytest -q tests/test_layered_vertical.py

No providers, TTS, model downloads, or external video downloads are used.
"""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from dubbingstory.render import video_render
from dubbingstory.render.vertical_layout import layer_dimensions, layout_settings


def config(**overrides):
    values = dict(render_vertical_layout="layered", render_foreground_zoom=25,
                  render_background_zoom=100, render_background_blur=50,
                  render_background_grayscale=1, render_ratios=["9:16"],
                  subtitle_style="classic", subtitle_alignment="auto",
                  tts_narration_fit="legacy", tts_use_timeline_scheduler=True,
                  render_original_volume=.15, render_burn_subtitles=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def metadata(**overrides):
    values = dict(width=1920, height=1080, display_width=1920, display_height=1080,
                  video_duration=5, video_duration_source="stream", duration=5,
                  video_start_time=0, audio_start_time=0, codec="h264",
                  video_stream_index=0, has_audio=True, audio_codec="aac")
    values.update(overrides)
    return values


def command(cfg=None, **kwargs):
    return video_render.build_render_command(
        "source.mp4", kwargs.pop("narration_audio", "voice.wav"), "out.mp4",
        kwargs.pop("metadata", metadata()), target_ratio=kwargs.pop("target_ratio", "9:16"),
        cfg=cfg or config(), **kwargs,
    )


def graph(cmd):
    return cmd[cmd.index("-filter_complex") + 1] if "-filter_complex" in cmd else ""


def maps(cmd):
    return [cmd[index + 1] for index, part in enumerate(cmd) if part == "-map"]


def test_default_layout_is_backward_compatible_padding():
    settings = layout_settings(SimpleNamespace())
    assert settings == {"layout": "pad", "foreground_zoom": 25,
                        "background_zoom": 100, "background_blur": 50,
                        "background_grayscale": 1}
    cmd = command(config(render_vertical_layout="pad"))
    assert "overlay=" not in graph(cmd)
    assert "gblur=" not in graph(cmd)
    assert "pad=1080:1920:" in graph(cmd)


@pytest.mark.parametrize("name,value", [
    ("vertical_layout", "unknown"), ("foreground_zoom", -.1),
    ("foreground_zoom", 100.1), ("foreground_zoom", float("nan")),
    ("background_zoom", -1), ("background_zoom", 101),
    ("background_zoom", float("inf")), ("background_blur", 0),
    ("background_blur", 101), ("background_blur", float("nan")),
    ("background_grayscale", -.1), ("background_grayscale", 1.1),
    ("background_grayscale", float("nan")),
])
def test_invalid_layout_settings_are_rejected(name, value):
    with pytest.raises(ValueError):
        layout_settings(config(**{"render_" + name: value}))


@pytest.mark.parametrize("zoom,expected", [(0, (1080, 608)), (25, (1664, 936)), (100, (3414, 1920))])
def test_zoom_interpolates_fit_to_fill_and_uses_even_dimensions(zoom, expected):
    assert layer_dimensions(metadata(), 1080, 1920, zoom) == expected


def test_layer_geometry_uses_display_dimensions_for_anamorphic_input():
    anamorphic = metadata(width=720, height=576, display_width=1024, display_height=576, sar="64:45")
    assert layer_dimensions(anamorphic, 1080, 1920, 25) == (1664, 936)


def test_phone_rotation_is_taken_from_display_geometry():
    rotated = metadata(width=1920, height=1080, display_width=1080, display_height=1920, rotation=90)
    assert layer_dimensions(rotated, 1080, 1920, 25) == (1080, 1920)


@pytest.mark.parametrize("ratio", [None, "source", "16:9", "1:1", "4:3"])
def test_layered_preset_does_not_change_other_output_ratios(ratio):
    layered = command(target_ratio=ratio)
    padded = command(config(render_vertical_layout="pad"), target_ratio=ratio)
    assert layered == padded


def test_layers_share_one_video_timeline_and_one_audio_mix():
    cmd = command(audio_strategy="dynamic_duck")
    filters = graph(cmd)
    assert "split=2" in filters and "overlay=" in filters
    assert "gblur=" in filters
    assert filters.count("[0:a:0]") == 1
    assert filters.count("[1:a:0]") == 1
    assert maps(cmd).count("[out]") == 1
    assert cmd.count("-i") == 2  # source video and narration, not two videos
    assert cmd.count("-c:v") == 1
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[cmd.index("-vsync") + 1] == "0"
    assert not any(option in cmd for option in ("-r", "-shortest", "-t"))
    assert "fps=" not in filters and "color=c=" not in filters


def test_caption_overlay_is_applied_after_layer_composition(monkeypatch):
    monkeypatch.setattr(video_render, "build_subtitle_filter", lambda *args, **kwargs: "subtitles=fixture.srt")
    filters = graph(command(subtitle_path="fixture.srt", burn_subs=True))
    assert filters.index("overlay=") < filters.index("subtitles=")
    assert filters.count("subtitles=") == 1


def test_export_reframe_copies_one_existing_aac_track_without_mixing():
    cmd = command(narration_audio=None)
    assert cmd.count("-i") == 1
    assert len(maps(cmd)) == 2
    assert "amix=" not in graph(cmd)
    assert "asplit=" not in graph(cmd)
    assert cmd[cmd.index("-c:a") + 1] == "copy"


def test_export_reframe_accepts_silent_video_without_inventing_audio():
    cmd = command(narration_audio=None, metadata=metadata(has_audio=False, audio_codec=""))
    assert cmd.count("-i") == 1
    assert len(maps(cmd)) == 1
    assert "amix=" not in graph(cmd)
    assert "anullsrc" not in graph(cmd)


@pytest.mark.parametrize("subcommand,arguments", [
    ("run", ["--input", "video.mp4", "--mode", "summary"]),
    ("render", ["--project", "existing"]),
    ("reframe", ["--input", "final_summary_id_16x9.mp4", "--output", "vertical.mp4"]),
])
def test_cli_propagates_layout_controls_without_running_pipeline(monkeypatch, subcommand, arguments):
    from dubbingstory import cli
    cfg = config(render_vertical_layout="pad")
    observed = {}
    monkeypatch.setattr(cli, "build_config", lambda **kwargs: cfg)
    monkeypatch.setattr(cli, "cmd_" + subcommand, lambda args, effective: observed.update(vars(effective)))
    monkeypatch.setattr("sys.argv", ["dubbingstory", subcommand, *arguments,
                                   "--ratio", "9:16", "--vertical-layout", "layered",
                                   "--foreground-zoom", "35", "--background-zoom", "90",
                                   "--background-blur", "20", "--background-grayscale", "0.4"])
    cli.main()
    assert observed["render_vertical_layout"] == "layered"
    assert observed["render_foreground_zoom"] == 35
    assert observed["render_background_zoom"] == 90
    assert observed["render_background_blur"] == 20
    assert observed["render_background_grayscale"] == .4
    assert observed["tts_narration_fit"] == "legacy"


def test_cli_reframe_does_not_require_narration_or_alignment(monkeypatch):
    from dubbingstory import cli
    cfg = config(subtitle_style="karaoke", tts_narration_fit="rewrite")
    observed = {}
    monkeypatch.setattr(cli, "build_config", lambda **kwargs: cfg)
    monkeypatch.setattr(video_render, "reframe_video", lambda source, output, **kwargs: observed.update(source=source, output=output) or output)
    monkeypatch.setattr(cli, "_configure_enhancements", lambda *args: pytest.fail("Reframe must not validate a TTS pipeline"))
    monkeypatch.setattr("sys.argv", ["dubbingstory", "reframe", "--input", "existing.mp4",
                                   "--output", "vertical.mp4", "--vertical-layout", "layered"])
    cli.main()
    assert observed == {"source": "existing.mp4", "output": "vertical.mp4"}


def test_reframe_publishes_completed_file_and_recipe_only_after_success(tmp_path, monkeypatch):
    source, output = tmp_path / "source.mp4", tmp_path / "out.mp4"
    source.write_bytes(b"original export")
    monkeypatch.setattr(video_render, "get_video_metadata", lambda path: metadata())
    monkeypatch.setattr(video_render, "_validate_tts_manifest", lambda *args: pytest.fail("No TTS validation for an existing export"))
    monkeypatch.setattr(video_render, "_prepare_subtitle", lambda *args: pytest.fail("No alignment for an existing export"))

    def fake_render(cmd, label):
        assert not output.exists()
        assert not Path(str(output) + ".render.json").exists()
        assert Path(cmd[-1]).parent != output.parent
        Path(cmd[-1]).write_bytes(b"complete encoded video")

    monkeypatch.setattr(video_render, "_run_ffmpeg", fake_render)
    assert video_render.reframe_video(str(source), str(output), cfg=config()) == str(output)
    recipe = json.loads(Path(str(output) + ".render.json").read_text(encoding="utf-8"))
    assert output.read_bytes() == b"complete encoded video"
    assert source.read_bytes() == b"original export"
    assert recipe["status"] == "complete" and recipe["operation"] == "reframe"
    assert recipe["layout"] == layout_settings(config())
    assert recipe["audio"] == "existing_mix"
    assert recipe["ratio"] == "9:16"
    assert not list(tmp_path.glob(".reframe-*"))


def test_reframe_failure_leaves_source_and_no_partial_export(tmp_path, monkeypatch):
    source, output = tmp_path / "source.mp4", tmp_path / "out.mp4"
    source.write_bytes(b"original export")
    monkeypatch.setattr(video_render, "get_video_metadata", lambda path: metadata())

    def fail_after_partial_write(cmd, label):
        Path(cmd[-1]).write_bytes(b"incomplete video")
        raise RuntimeError("fixture encoding failure")

    monkeypatch.setattr(video_render, "_run_ffmpeg", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="fixture encoding failure"):
        video_render.reframe_video(str(source), str(output), cfg=config())
    assert source.read_bytes() == b"original export"
    assert not output.exists()
    assert not Path(str(output) + ".render.json").exists()
    assert not list(tmp_path.glob(".reframe-*"))


@pytest.mark.parametrize("conflict", ["source", "output", "manifest"])
def test_reframe_never_overwrites_existing_source_export_or_recipe(tmp_path, monkeypatch, conflict):
    source, output = tmp_path / "source.mp4", tmp_path / "out.mp4"
    source.write_bytes(b"original export")
    if conflict == "source":
        output = source
    elif conflict == "output":
        output.write_bytes(b"previous output")
    else:
        Path(str(output) + ".render.json").write_text("previous recipe", encoding="utf-8")
    monkeypatch.setattr(video_render, "get_video_metadata", lambda *args: pytest.fail("Conflicts must be rejected before probing/rendering"))
    expected_error = ValueError if conflict == "source" else FileExistsError
    with pytest.raises(expected_error):
        video_render.reframe_video(str(source), str(output), cfg=config())
    assert source.read_bytes() == b"original export"
    if conflict == "output":
        assert output.read_bytes() == b"previous output"
    if conflict == "manifest":
        assert Path(str(output) + ".render.json").read_text(encoding="utf-8") == "previous recipe"


def test_reframe_rechecks_output_name_after_encoding(tmp_path, monkeypatch):
    source, output = tmp_path / "source.mp4", tmp_path / "out.mp4"
    source.write_bytes(b"original export")
    monkeypatch.setattr(video_render, "get_video_metadata", lambda path: metadata())

    def another_job_claims_output(cmd, label):
        Path(cmd[-1]).write_bytes(b"this job")
        output.write_bytes(b"another job")

    monkeypatch.setattr(video_render, "_run_ffmpeg", another_job_claims_output)
    with pytest.raises(FileExistsError):
        video_render.reframe_video(str(source), str(output), cfg=config())
    assert output.read_bytes() == b"another job"
    assert not Path(str(output) + ".render.json").exists()
    assert not list(tmp_path.glob(".reframe-*"))


MEDIA_TESTS = pytest.mark.skipif(
    os.environ.get("DUBBINGSTORY_FFMPEG_TESTS") != "1"
    or not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="Opt in on Kaggle with DUBBINGSTORY_FFMPEG_TESTS=1 and FFmpeg/FFprobe installed",
)


def run_media(arguments):
    return subprocess.run(arguments, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def probe_streams(path):
    return json.loads(run_media(["ffprobe", "-v", "error", "-count_frames", "-show_streams",
                                 "-of", "json", str(path)]).stdout)["streams"]


@MEDIA_TESTS
def test_layered_render_pixels_duration_frame_count_and_single_audio(tmp_path):
    from PIL import Image

    source, output = tmp_path / "source.mp4", tmp_path / "vertical.mp4"
    run_media(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=12:duration=1",
               "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-c:a", "aac", str(source)])
    video_render.reframe_video(str(source), str(output), cfg=config())
    before, after = probe_streams(source), probe_streams(output)
    source_video = next(item for item in before if item["codec_type"] == "video")
    final_video = next(item for item in after if item["codec_type"] == "video")
    assert (final_video["width"], final_video["height"]) == (1080, 1920)
    assert int(final_video["nb_read_frames"]) == int(source_video["nb_read_frames"])
    assert float(final_video["duration"]) == pytest.approx(float(source_video["duration"]), abs=1 / 12)
    assert len([item for item in after if item["codec_type"] == "audio"]) == 1

    # Stream copy must retain the existing mixed voice/music exactly once.
    audio = [run_media(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
                        "-f", "s16le", "-acodec", "pcm_s16le", "-"]).stdout for path in (source, output)]
    assert audio[0] == audio[1]
    frame = run_media(["ffmpeg", "-v", "error", "-i", str(output), "-frames:v", "1",
                       "-f", "image2pipe", "-vcodec", "png", "-"]).stdout
    image = Image.open(io.BytesIO(frame)).convert("RGB")
    background = list(image.crop((150, 80, 930, 300)).resize((60, 20)).getdata())
    foreground = list(image.crop((150, 650, 930, 1250)).resize((60, 40)).getdata())
    assert sum(max(pixel) - min(pixel) for pixel in background) / len(background) < 8
    assert sum(max(pixel) - min(pixel) for pixel in foreground) / len(foreground) > 20
    assert Path(str(output) + ".render.json").is_file()


@MEDIA_TESTS
def test_layered_vfr_preserves_frame_presentation_times(tmp_path):
    source, output = tmp_path / "vfr.mp4", tmp_path / "vertical.mp4"
    run_media(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=12:duration=1",
               "-vf", "select='if(lt(n,6),not(mod(n,2)),1)'", "-vsync", "vfr",
               "-an", "-c:v", "libx264", str(source)])
    video_render.reframe_video(str(source), str(output), cfg=config(render_background_blur=5))
    timestamps = []
    for path in (source, output):
        result = run_media(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
                            "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path)])
        timestamps.append([float(frame["best_effort_timestamp_time"]) for frame in json.loads(result.stdout)["frames"]])
    assert len(set(round(b - a, 3) for a, b in zip(timestamps[0], timestamps[0][1:]))) > 1
    assert timestamps[1] == pytest.approx(timestamps[0], abs=.001)
    assert not any(item["codec_type"] == "audio" for item in probe_streams(output))
