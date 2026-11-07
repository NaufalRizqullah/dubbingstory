"""Saved-evidence resume fixtures; run in Kaggle without model/media services."""

import json
from types import SimpleNamespace

import pytest

from dubbingstory import cli
from dubbingstory.story import checkpoint, story_planner


def evidence(tmp_path, *, temporal=False):
    (tmp_path / "source.mp4").write_bytes(b"source fixture, never decoded")
    scenes = [{"scene_id": "scene_001", "start_time": 0.0, "end_time": 5.0}]
    (tmp_path / "segment_manifest.json").write_text(json.dumps({"status": "segmented", "scenes": scenes}))
    board = {"scenes": scenes}
    if temporal:
        source_hash = checkpoint.file_sha256(str(tmp_path / "source.mp4"))
        timeline = tmp_path / "visual_timeline.json"
        timeline.write_text(json.dumps({"status": "complete", "provenance": {"source_sha256": source_hash}}))
        board["visual_timeline"] = {"source_sha256": source_hash,
                                    "artifact_sha256": checkpoint.file_sha256(str(timeline))}
    (tmp_path / "storyboard.json").write_text(json.dumps(board))
    return board


def test_verified_checkpoint_and_legacy_dense_resume(tmp_path):
    board = evidence(tmp_path, temporal=True)
    assert checkpoint.load_analysis_checkpoint(tmp_path, require_temporal=True) == board
    checkpoint.write_analysis_checkpoint(tmp_path)
    assert checkpoint.load_analysis_checkpoint(tmp_path, require_temporal=True) == board


@pytest.mark.parametrize("name", ["source.mp4", "segment_manifest.json", "storyboard.json", "visual_timeline.json"])
def test_modified_evidence_cannot_be_resumed(tmp_path, name):
    evidence(tmp_path, temporal=True)
    checkpoint.write_analysis_checkpoint(tmp_path)
    with (tmp_path / name).open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="resume analyze"):
        checkpoint.load_analysis_checkpoint(tmp_path, require_temporal=True)


def test_old_non_temporal_evidence_requires_source_proof(tmp_path):
    board = evidence(tmp_path)
    with pytest.raises(ValueError, match="no source checkpoint"):
        checkpoint.load_analysis_checkpoint(tmp_path)
    checkpoint.write_analysis_checkpoint(tmp_path)
    assert checkpoint.load_analysis_checkpoint(tmp_path) == board


def test_legacy_dense_scene_mismatch_is_rejected(tmp_path):
    evidence(tmp_path, temporal=True)
    (tmp_path / "segment_manifest.json").write_text(json.dumps({"scenes": []}))
    with pytest.raises(ValueError, match="coverage/timing"):
        checkpoint.load_analysis_checkpoint(tmp_path)


def test_plan_command_reuses_evidence_without_vision(tmp_path, monkeypatch):
    board = evidence(tmp_path, temporal=True)
    monkeypatch.setattr(cli, "_setup_project_dir", lambda *args: str(tmp_path))
    seen = []
    monkeypatch.setattr(story_planner, "generate_story_artifacts", lambda value, **kwargs: seen.append(value) or "planned")
    # No vision module is needed by this command.
    assert cli.cmd_plan(SimpleNamespace(project="sample"), SimpleNamespace(narration_action_sync=True)) == "planned"
    assert seen == [board]


def test_plan_cli_dispatch_is_additive(monkeypatch):
    monkeypatch.setattr(cli, "build_config", lambda **kwargs: SimpleNamespace())
    seen = []
    monkeypatch.setattr(cli, "cmd_plan", lambda args, cfg: seen.append(args.project))
    monkeypatch.setattr("sys.argv", ["dubbingstory", "plan", "--project", "saved"])
    cli.main()
    assert seen == ["saved"]
