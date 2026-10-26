"""Saved summary recovery without providers, synthesis, or media execution."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dubbingstory.render import video_render


def config(**overrides):
    values = dict(render_ratios=["16:9", "9:16"], render_vertical_layout="layered",
                  render_audio_strategy="dynamic_duck", render_original_volume=.15,
                  render_burn_subtitles=False, subtitle_style="classic",
                  subtitle_alignment="auto", tts_narration_fit="legacy",
                  tts_use_timeline_scheduler=True)
    values.update(overrides)
    return SimpleNamespace(**values)


def artifacts(tmp_path):
    (tmp_path / "audio").mkdir()
    (tmp_path / "scripts").mkdir()
    for name in ("source.mp4", "summary_source.mp4", "audio/audio_id.wav",
                 "audio/summary_audio_id.wav", "scripts/script_id.srt",
                 "scripts/summary_script_id.srt"):
        (tmp_path / name).write_bytes(b"saved fixture")


def test_summary_rerender_reuses_saved_artifacts_and_both_ratios(tmp_path, monkeypatch):
    artifacts(tmp_path)
    full_manifest = tmp_path / "render_manifest.json"
    full_manifest.write_text("full render record", encoding="utf-8")
    calls = []
    monkeypatch.setattr(video_render, "render_final", lambda *args, **kwargs: calls.append((args, kwargs)))
    outputs = video_render.render_project(str(tmp_path), cfg=config(), mode="summary")
    assert [Path(path).name for path in outputs] == ["final_summary_id_16x9.mp4", "final_summary_id_canvas_9x16.mp4"]
    assert [entry[1]["target_ratio"] for entry in calls] == [None, "9:16"]
    for args, kwargs in calls:
        assert Path(args[0]).name == "summary_source.mp4"
        assert Path(args[1]).name == "summary_audio_id.wav"
        assert Path(args[3]).name == "summary_script_id.srt"
        assert kwargs["audio_strategy"] == "dynamic_duck"
    manifest = json.loads((tmp_path / "summary_render_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "summary"
    assert manifest["status"] == "complete"
    assert manifest["rendered_videos"] == outputs
    assert full_manifest.read_text(encoding="utf-8") == "full render record"


def test_default_mode_retains_full_artifact_selection(tmp_path, monkeypatch):
    artifacts(tmp_path)
    calls = []
    monkeypatch.setattr(video_render, "render_final", lambda *args, **kwargs: calls.append(args))
    outputs = video_render.render_project(str(tmp_path), ["16:9"], config(render_vertical_layout="pad"))
    assert Path(calls[0][0]).name == "source.mp4"
    assert Path(calls[0][1]).name == "audio_id.wav"
    assert Path(calls[0][3]).name == "script_id.srt"
    assert Path(outputs[0]).name == "final_id_16x9.mp4"
    assert not (tmp_path / "summary_render_manifest.json").exists()


@pytest.mark.parametrize("mode,prefix", [("full", "final_id"), ("summary", "final_summary_id")])
def test_canvas_render_keeps_original_output_separate(tmp_path, monkeypatch, mode, prefix):
    artifacts(tmp_path)
    original = tmp_path / f"{prefix}_9x16.mp4"
    original.write_bytes(b"existing original export")

    def render(source, audio, output, *args, **kwargs):
        Path(output).write_bytes(b"canvas export")

    monkeypatch.setattr(video_render, "render_final", render)
    outputs = video_render.render_project(str(tmp_path), ["9:16"], config(), mode=mode)
    assert [Path(path).name for path in outputs] == [f"{prefix}_canvas_9x16.mp4"]
    assert original.read_bytes() == b"existing original export"
    assert Path(outputs[0]).read_bytes() == b"canvas export"
    manifest_name = "summary_render_manifest.json" if mode == "summary" else "render_manifest.json"
    manifest = json.loads((tmp_path / manifest_name).read_text(encoding="utf-8"))
    assert manifest["rendered_videos"] == outputs


@pytest.mark.parametrize("missing", ["summary_source.mp4", "audio/summary_audio_id.wav"])
def test_summary_never_falls_back_to_full_assets(tmp_path, monkeypatch, missing):
    artifacts(tmp_path)
    (tmp_path / missing).unlink()
    monkeypatch.setattr(video_render, "render_final", lambda *args, **kwargs: pytest.fail("Missing summary must not render full artifacts"))
    with pytest.raises((FileNotFoundError, ValueError), match="summary"):
        video_render.render_project(str(tmp_path), cfg=config(), mode="summary")


def test_summary_attempts_every_ratio_and_reports_partial_failure(tmp_path, monkeypatch):
    artifacts(tmp_path)
    attempted = []

    def render(*args, **kwargs):
        attempted.append(kwargs["target_ratio"])
        if kwargs["target_ratio"] is None:
            raise RuntimeError("fixture audio filter failure")

    monkeypatch.setattr(video_render, "render_final", render)
    # Even classic/padded recovery must return failure to the notebook caller.
    with pytest.raises(RuntimeError, match="summary_render_manifest.json"):
        video_render.render_project(str(tmp_path), cfg=config(render_vertical_layout="pad"), mode="summary")
    assert attempted == [None, "9:16"]
    manifest = json.loads((tmp_path / "summary_render_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["errors"][0]["ratio"] == "16:9"
    assert [Path(path).name for path in manifest["rendered_videos"]] == ["final_summary_id_9x16.mp4"]


@pytest.mark.parametrize("scheduled,expected_source", [(True, "summary_source.mp4"),
                                                       (False, "summary_source_id.mp4"),
                                                       (None, "summary_source.mp4")])
def test_legacy_language_recut_requires_explicit_manifest(tmp_path, monkeypatch, scheduled, expected_source):
    artifacts(tmp_path)
    (tmp_path / "summary_source_id.mp4").write_bytes(b"legacy recut")
    if scheduled is not None:
        (tmp_path / "audio/summary_audio_id.manifest.json").write_text(
            json.dumps({"timeline_scheduled": scheduled}), encoding="utf-8")
    calls = []
    monkeypatch.setattr(video_render, "render_final", lambda *args, **kwargs: calls.append(args))
    video_render.render_project(str(tmp_path), ["16:9"], config(render_vertical_layout="pad"), mode="summary")
    assert Path(calls[0][0]).name == expected_source


def test_cli_summary_render_dispatch_does_not_run_pipeline(tmp_path, monkeypatch):
    from dubbingstory import cli
    from dubbingstory.tts import voice_manager

    cfg = config(project_output_dir=str(tmp_path))
    observed = {}
    monkeypatch.setattr(cli, "build_config", lambda **kwargs: cfg)
    monkeypatch.setattr(cli, "cmd_summary", lambda *args: pytest.fail("Recovery must not rerun the summary pipeline"))
    monkeypatch.setattr(voice_manager, "generate_all_audio", lambda **kwargs: pytest.fail("Recovery must not resynthesize audio"))
    monkeypatch.setattr(video_render, "render_project", lambda **kwargs: observed.update(kwargs) or ["done.mp4"])
    monkeypatch.setattr("sys.argv", ["dubbingstory", "render", "--project", "existing",
                                   "--mode", "summary", "--ratio", "16:9", "9:16",
                                   "--vertical-layout", "layered"])
    cli.main()
    assert observed["mode"] == "summary"
    assert observed["ratios"] == ["16:9", "9:16"]
    assert Path(observed["project_dir"]).name == "existing"
    assert observed["cfg"].render_vertical_layout == "layered"
