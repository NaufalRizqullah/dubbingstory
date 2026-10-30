"""CLI integration checks; mocks avoid provider/network/media execution."""

import hashlib
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


def test_dense_source_analysis_and_action_sync_are_additive_cli_options(monkeypatch):
    cfg = config()
    observed = {}
    monkeypatch.setattr(cli, "build_config", lambda **kwargs: cfg)
    monkeypatch.setattr(cli, "cmd_run", lambda args, effective: observed.update(vars(effective)))
    monkeypatch.setattr("sys.argv", ["dubbingstory", "run", "--input", "video.mp4",
                                   "--frame-interval", "0.5", "--action-sync", "--engine", "elevenlabs",
                                   "--word-alignment", "provider"])
    cli.main()
    assert observed["vision_temporal_enabled"] is True
    assert observed["vision_temporal_interval_seconds"] == .5
    assert observed["narration_action_sync"] is True
    assert observed["tts_narration_fit"] == "rewrite"
    assert observed["tts_engine"] == "elevenlabs"
    assert observed["subtitle_alignment"] == "provider"


def test_summary_keeps_existing_behavior_when_reusing_dense_config():
    cfg = config(vision_temporal_enabled=True, narration_action_sync=True)
    cli._configure_enhancements(SimpleNamespace(mode="summary"), cfg)
    assert cfg.vision_temporal_enabled is False
    assert cfg.narration_action_sync is False
    assert cfg.tts_narration_fit == "legacy"


@pytest.mark.parametrize("interval", [0, -.5, float("nan"), float("inf")])
def test_invalid_dense_interval_is_rejected(interval):
    with pytest.raises(ValueError, match="interval_seconds"):
        cli._configure_enhancements(SimpleNamespace(frame_interval=interval), config())


def test_explicit_dense_request_cannot_silently_affect_summary():
    with pytest.raises(ValueError, match="mode full"):
        cli._configure_enhancements(SimpleNamespace(mode="summary", action_sync=True), config())


def test_url_project_identity_distinguishes_youtube_video_ids():
    first = cli._resolve_project_name(SimpleNamespace(url="https://www.youtube.com/watch?v=abcdefghijk"))
    second = cli._resolve_project_name(SimpleNamespace(url="https://www.youtube.com/watch?v=lmnopqrstuv"))
    assert first != second
    assert cli._resolve_project_name(SimpleNamespace(project="existing", url="unused")) == "existing"


def test_narration_source_defaults_to_visual_and_accepts_audio():
    cfg = config(narration_source="visual")
    cli._configure_enhancements(SimpleNamespace(narration_source="audio"), cfg)
    assert cfg.narration_source == "audio"


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


@pytest.fixture
def temporal_narration_project(tmp_path):
    project = tmp_path / "sample"
    project.mkdir()
    source = project / "source.mp4"
    source.write_bytes(b"source fixture; no media decoder is used")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    timeline = project / "visual_timeline.json"
    timeline.write_text(json.dumps({"status": "complete", "provenance": {"source_sha256": source_hash}}),
                        encoding="utf-8")
    timeline_hash = hashlib.sha256(timeline.read_bytes()).hexdigest()
    storyboard = {"scenes": [], "visual_timeline": {"source_sha256": source_hash,
                                                   "artifact_sha256": timeline_hash}}
    (project / "storyboard.json").write_text(json.dumps(storyboard), encoding="utf-8")
    return project, source_hash, timeline_hash


@pytest.mark.parametrize("changed", ["source", "timeline", "incomplete", "storyboard_source", "missing_timeline"])
def test_narrate_rejects_stale_visual_evidence_before_llm(temporal_narration_project, monkeypatch, changed):
    from dubbingstory.story import script_writer, story_planner

    project, _, _ = temporal_narration_project
    timeline = project / "visual_timeline.json"
    if changed == "source":
        (project / "source.mp4").write_bytes(b"replacement source")
    elif changed in {"timeline", "incomplete"}:
        payload = json.loads(timeline.read_text(encoding="utf-8"))
        payload["status" if changed == "incomplete" else "events"] = "failed" if changed == "incomplete" else ["changed"]
        timeline.write_text(json.dumps(payload), encoding="utf-8")
    elif changed == "storyboard_source":
        path = project / "storyboard.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["visual_timeline"]["source_sha256"] = "not-the-current-source"
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        timeline.unlink()
    monkeypatch.setattr(story_planner, "load_story_artifacts",
                        lambda *a: pytest.fail("Stale evidence must fail before loading story context"))
    monkeypatch.setattr(script_writer, "generate_narration",
                        lambda **kw: pytest.fail("Stale evidence must fail before the narration LLM"))
    cfg = config(project_output_dir=str(project.parent), narration_action_sync=True)
    with pytest.raises(ValueError, match="rerun analyze"):
        cli.cmd_narrate(SimpleNamespace(project=project.name, lang=["id"]), cfg)
    assert not (project / "narration_plan.json").exists()
    assert not (project / "scripts" / "script_id.json").exists()


def test_narrate_persists_matching_visual_proof(temporal_narration_project, monkeypatch):
    from dubbingstory.story import narration_planner, narration_qa, script_writer, story_planner, subtitle_gen

    project, source_hash, timeline_hash = temporal_narration_project
    segments = [{"scene_id": "scene_001", "text": "Besi dipotong perlahan.",
                 "timeline_start": 0.0, "timeline_end": 10.0, "action_sync_enabled": True}]
    monkeypatch.setattr(story_planner, "load_story_artifacts", lambda *a: ({}, {}))
    monkeypatch.setattr(cli, "_video_timeline_duration", lambda *a, **kw: 10.0)
    monkeypatch.setattr(narration_planner, "build_narration_plan",
                        lambda *a, **kw: {"total_duration": 10.0, "action_sync_enabled": True})
    monkeypatch.setattr(script_writer, "generate_narration", lambda **kw: {"id": segments})
    monkeypatch.setattr(subtitle_gen, "generate_srt", lambda *a: None)
    monkeypatch.setattr(narration_qa, "evaluate_narration", lambda *a, **kw: {})
    cfg = config(project_output_dir=str(project.parent), narration_action_sync=True)
    result = cli.cmd_narrate(SimpleNamespace(project=project.name, lang=["id"]), cfg)
    assert result == {"id": segments}
    script = json.loads((project / "scripts" / "script_id.json").read_text(encoding="utf-8"))
    assert script["visual_source_sha256"] == source_hash
    assert script["visual_timeline_sha256"] == timeline_hash
    assert script["action_sync_enabled"] is True


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
