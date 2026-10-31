"""Segmentation guards with fake media; execute in Kaggle, not on the dev PC."""

import json
from types import SimpleNamespace

import pytest

from dubbingstory import cli


def segment_fixture(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"mock image; never decoded")
    return {
        "status": "segmented", "total_scenes": 1,
        "scenes": [{"scene_id": "scene_001", "start_time": 0.0, "end_time": 30.04}],
        "keyframes": {"scene_001": [str(frame)]},
    }


@pytest.mark.parametrize("failure", ["empty", "failed", "count", "duplicate", "nan", "reversed",
                                   "overlap", "missing_frames", "missing_file", "empty_file"])
def test_invalid_segment_manifest_is_rejected_with_recovery(tmp_path, failure):
    manifest = segment_fixture(tmp_path)
    if failure == "empty":
        manifest.update(total_scenes=0, scenes=[], keyframes={})
    elif failure == "failed":
        manifest["status"] = "failed"
    elif failure == "count":
        manifest["total_scenes"] = 5
    elif failure == "duplicate":
        manifest["scenes"] *= 2
        manifest["total_scenes"] = 2
    elif failure == "nan":
        manifest["scenes"][0]["end_time"] = float("nan")
    elif failure == "reversed":
        manifest["scenes"][0].update(start_time=20, end_time=10)
    elif failure == "overlap":
        manifest["scenes"].append({"scene_id": "scene_002", "start_time": 10, "end_time": 40})
        manifest["total_scenes"] = 2
    elif failure == "missing_frames":
        manifest["keyframes"] = {}
    elif failure == "missing_file":
        (tmp_path / "frame.jpg").unlink()
    elif failure == "empty_file":
        (tmp_path / "frame.jpg").write_bytes(b"")
    with pytest.raises(ValueError, match="Rerun segment"):
        cli.validate_segment_manifest(manifest)


def test_keyframes_are_optional_for_video_upload_only(tmp_path):
    manifest = segment_fixture(tmp_path)
    cli.validate_segment_manifest(manifest)
    manifest["keyframes"] = {}
    cli.validate_segment_manifest(manifest, require_keyframes=False)
    with pytest.raises(ValueError, match="no keyframes"):
        cli.validate_segment_manifest(manifest)


@pytest.mark.parametrize("failure", ["no_scenes", "decode", "no_keyframes", "missing_keyframe"])
def test_failed_segment_attempt_invalidates_prior_success(tmp_path, monkeypatch, failure, capsys):
    from dubbingstory.segment import keyframes, scene_detect

    old = segment_fixture(tmp_path)
    manifest_path = tmp_path / "segment_manifest.json"
    manifest_path.write_text(json.dumps(old), encoding="utf-8")
    (tmp_path / "ingest_manifest.json").write_text(json.dumps({"video_path": "mock.mp4"}), encoding="utf-8")
    monkeypatch.setattr(cli, "_setup_project_dir", lambda *args: str(tmp_path))
    calls = []

    def detect(**kwargs):
        if failure == "decode":
            raise ValueError("No decoded frames")
        return [] if failure == "no_scenes" else old["scenes"]

    def extract(**kwargs):
        calls.append("extract")
        return {"scene_001": [str(tmp_path / "absent.jpg")]} if failure == "missing_keyframe" else {}

    monkeypatch.setattr(scene_detect, "detect_scenes", detect)
    monkeypatch.setattr(keyframes, "extract_all_scenes", extract)
    with pytest.raises(ValueError):
        cli.cmd_segment(SimpleNamespace(project="sample"), SimpleNamespace())
    result = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["scenes"] == []
    assert "Segmentation selesai" not in capsys.readouterr().out
    if failure in {"decode", "no_scenes"}:
        assert calls == []


def test_valid_segmentation_publishes_existing_scene_and_frame_contract(tmp_path, monkeypatch):
    from dubbingstory.segment import keyframes, scene_detect

    expected = segment_fixture(tmp_path)
    (tmp_path / "ingest_manifest.json").write_text(json.dumps({"video_path": "mock.mp4"}), encoding="utf-8")
    monkeypatch.setattr(cli, "_setup_project_dir", lambda *args: str(tmp_path))
    monkeypatch.setattr(scene_detect, "detect_scenes", lambda **kwargs: expected["scenes"])
    monkeypatch.setattr(keyframes, "extract_all_scenes", lambda **kwargs: expected["keyframes"])
    result = cli.cmd_segment(SimpleNamespace(project="sample"), SimpleNamespace())
    assert result == {**expected, "video_path": "mock.mp4"}
    assert json.loads((tmp_path / "segment_manifest.json").read_text(encoding="utf-8")) == result


def test_analyze_rejects_empty_manifest_before_provider_call(tmp_path, monkeypatch):
    from dubbingstory.vision import scene_understanding

    (tmp_path / "segment_manifest.json").write_text(
        json.dumps({"status": "segmented", "scenes": [], "keyframes": {}}), encoding="utf-8")
    monkeypatch.setattr(cli, "_setup_project_dir", lambda *args: str(tmp_path))

    def unexpected(**kwargs):
        pytest.fail("Empty segmentation must stop before vision/model initialization")

    monkeypatch.setattr(scene_understanding, "run_analysis", unexpected)
    with pytest.raises(ValueError, match="no scenes"):
        cli.cmd_analyze(SimpleNamespace(project="sample"), SimpleNamespace())
    assert not (tmp_path / "storyboard.json").exists()
