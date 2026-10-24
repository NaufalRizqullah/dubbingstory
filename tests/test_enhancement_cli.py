"""CLI integration checks; mocks avoid provider/network/media execution."""

import json
from types import SimpleNamespace

import pytest

from dubbingstory import cli


def config(**overrides):
    values = dict(render_ratios=["16:9"], subtitle_style="classic",
                  subtitle_alignment="auto", tts_narration_fit="legacy",
                  render_burn_subtitles=False, render_original_volume=0.15,
                  tts_use_timeline_scheduler=True)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_native_cli_promotes_strict_fit_and_preserves_burn_override(monkeypatch):
    cfg = config()
    observed = {}
    monkeypatch.setattr(cli, "build_config", lambda **kwargs: cfg)
    monkeypatch.setattr(cli, "cmd_run", lambda args, effective: observed.update(vars(effective)))
    monkeypatch.setattr("sys.argv", ["dubbingstory", "run", "--input", "video.mp4",
                                   "--ratio", "source", "--lang", "id",
                                   "--subtitle-style", "karaoke", "--burn-subtitles"])
    cli.main()
    assert observed["render_ratios"] == ["source"]
    assert observed["tts_narration_fit"] == "rewrite"
    assert observed["narration_languages"] == ["id"]
    assert observed["render_burn_subtitles"] is True


def test_legacy_defaults_are_not_promoted():
    cfg = config()
    cli._configure_enhancements(SimpleNamespace(), cfg)
    assert cfg.tts_narration_fit == "legacy"
    assert cfg.subtitle_style == "classic"


def test_config_values_survive_absent_cli_options():
    cfg = config(subtitle_style="word", tts_narration_fit="error", render_burn_subtitles=True)
    cli._configure_enhancements(SimpleNamespace(subtitle_style=None, burn_subtitles=None), cfg)
    assert cfg.subtitle_style == "word"
    assert cfg.tts_narration_fit == "error"
    assert cfg.render_burn_subtitles is True
    cli._configure_enhancements(SimpleNamespace(burn_subtitles=False), cfg)
    assert cfg.render_burn_subtitles is False


def test_native_cannot_explicitly_disable_overflow_validation():
    with pytest.raises(ValueError, match="requires --narration-fit"):
        cli._configure_enhancements(SimpleNamespace(narration_fit="legacy"), config(render_ratios=["source"]))


def test_animated_requires_timeline_scheduler():
    with pytest.raises(ValueError, match="timeline_scheduler"):
        cli._configure_enhancements(SimpleNamespace(), config(subtitle_style="karaoke", tts_use_timeline_scheduler=False))


def test_volume_zero_is_valid():
    cfg = config(render_original_volume=0)
    cli._configure_enhancements(SimpleNamespace(), cfg)
    assert cfg.render_original_volume == 0


def test_url_project_identity_distinguishes_youtube_video_ids():
    first = cli._resolve_project_name(SimpleNamespace(url="https://www.youtube.com/watch?v=abcdefghijk"))
    second = cli._resolve_project_name(SimpleNamespace(url="https://www.youtube.com/watch?v=lmnopqrstuv"))
    assert first != second
    assert cli._resolve_project_name(SimpleNamespace(project="existing", url="unused")) == "existing"


def test_rights_gate_precedes_project_creation(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("A project must not be created before the rights gate")
    monkeypatch.setattr(cli, "_setup_project_dir", unexpected)
    with pytest.raises(ValueError, match="i-have-rights"):
        cli.cmd_ingest(SimpleNamespace(input=None, url="https://www.youtube.com/watch?v=abcdefghijk", i_have_rights=False), config())


def test_full_dub_propagates_source_duration_and_refreshes_srt(tmp_path, monkeypatch):
    from dubbingstory.tts import voice_manager

    project = tmp_path / "sample"
    audio = project / "audio" / "audio_id.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"mock audio")
    captured = {}
    result = {"id": {"concat_path": str(audio), "placements": [
        {"scene_id": "scene_001", "actual_start": 2.5, "actual_end": 3.5}],
        "final_segments": [{"scene_id": "scene_001", "text": "Benda itu berubah.",
                            "start_time": 0, "end_time": 8}]}}

    def generate(**kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(voice_manager, "generate_all_audio", generate)
    monkeypatch.setattr(cli, "_video_timeline_duration", lambda path, **kwargs: 12.5)
    cfg = config(project_output_dir=str(tmp_path), render_ratios=["source"], tts_narration_fit="error")
    cli.cmd_dub(SimpleNamespace(project="sample", engine="edge"), cfg)
    assert captured["timeline_duration"] == 12.5
    assert captured["timeline_kind"] == "source"
    srt = (project / "scripts" / "script_id.srt").read_text(encoding="utf-8")
    assert "00:00:02,500 --> 00:00:03,500" in srt
    assert "Benda itu berubah." in srt


def test_local_ingest_reads_original_sidecar_path(tmp_path, monkeypatch):
    from dubbingstory.ingest import local_video, subtitle_reader

    original = tmp_path / "movie.mp4"
    original.write_bytes(b"mock media")
    captured = []
    monkeypatch.setattr(local_video, "validate_and_copy", lambda path, output: str(tmp_path / "source.mp4"))
    monkeypatch.setattr(subtitle_reader, "find_and_read_subtitles", lambda path, output: captured.append(path))
    cfg = config(project_output_dir=str(tmp_path / "out"), ingest_use_asr_fallback=False)
    cli.cmd_ingest(SimpleNamespace(input=str(original), url=None, project="movie", i_have_rights=False), cfg)
    assert captured == [str(original)]
    manifest = json.loads((tmp_path / "out" / "movie" / "ingest_manifest.json").read_text(encoding="utf-8"))
    assert manifest["video_path"].endswith("source.mp4")
