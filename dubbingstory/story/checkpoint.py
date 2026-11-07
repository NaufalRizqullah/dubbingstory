"""Validate saved visual evidence before resuming text-only story planning."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile

from dubbingstory.tts.word_alignment import file_sha256


def _read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError(f"Missing or invalid {path.name}; resume analyze first") from None
    if not isinstance(value, dict):
        raise ValueError(f"Invalid {path.name}; resume analyze first")
    return value


def _evidence_paths(project_dir) -> dict[str, Path]:
    root = Path(project_dir)
    paths = {key: root / name for key, name in (
        ("source_sha256", "source.mp4"), ("storyboard_sha256", "storyboard.json"),
        ("segment_sha256", "segment_manifest.json"),
    )}
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("Story planning requires source, segment manifest and storyboard; resume analyze first")
    return paths


def write_analysis_checkpoint(project_dir) -> None:
    """Publish a content-bound checkpoint only after storyboard publication."""
    paths = _evidence_paths(project_dir)
    root = Path(project_dir)
    checkpoint = {"schema_version": 1, "status": "complete",
                  **{key: file_sha256(str(path)) for key, path in paths.items()}}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root,
                                         prefix="analysis_checkpoint.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(checkpoint, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, root / "analysis_checkpoint.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_analysis_checkpoint(project_dir, *, require_temporal: bool = False) -> dict:
    """Reuse complete evidence, never invoke vision or accept another source's data.

    Older dense-analysis runs already contain a source-bound timeline and can
    resume even when they predate analysis_checkpoint.json.
    """
    paths = _evidence_paths(project_dir)
    root = Path(project_dir)
    board = _read_object(paths["storyboard_sha256"])
    segment = _read_object(paths["segment_sha256"])
    if segment.get("status", "segmented") != "segmented":
        raise ValueError("Segment manifest is incomplete; resume analyze first")
    board_scenes, segment_scenes = board.get("scenes"), segment.get("scenes")
    if not isinstance(board_scenes, list) or not board_scenes or not isinstance(segment_scenes, list):
        raise ValueError("Saved visual evidence has no scenes; resume analyze first")
    try:
        board_ids = [scene["scene_id"] for scene in board_scenes]
        segment_ids = [scene["scene_id"] for scene in segment_scenes]
        valid = (all(isinstance(sid, str) and sid for sid in board_ids)
                 and len(set(board_ids)) == len(board_ids) and board_ids == segment_ids)
        for actual, expected in zip(board_scenes, segment_scenes):
            for field in ("start_time", "end_time"):
                left, right = float(actual[field]), float(expected[field])
                valid = valid and math.isfinite(left) and math.isfinite(right) and abs(left - right) < 1e-6
            valid = valid and float(actual["end_time"]) > float(actual["start_time"])
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Storyboard and segment scene coverage/timing disagree; resume analyze first")
    source_hash = file_sha256(str(paths["source_sha256"]))
    proof_path = root / "analysis_checkpoint.json"
    if proof_path.exists():
        proof = _read_object(proof_path)
        if (proof.get("schema_version") != 1 or proof.get("status") != "complete"
                or any(proof.get(key) != (source_hash if key == "source_sha256" else file_sha256(str(path)))
                       for key, path in paths.items())):
            raise ValueError("Analysis checkpoint does not match saved evidence/source; resume analyze first")
    temporal = board.get("visual_timeline") or {}
    if require_temporal or temporal:
        timeline_path = root / "visual_timeline.json"
        timeline = _read_object(timeline_path)
        provenance = timeline.get("provenance")
        if (not isinstance(temporal, dict) or timeline.get("status") != "complete"
                or not isinstance(provenance, dict) or provenance.get("source_sha256") != source_hash
                or temporal.get("source_sha256") != source_hash
                or temporal.get("artifact_sha256") != file_sha256(str(timeline_path))):
            raise ValueError("Temporal evidence is incomplete or stale; resume analyze first")
    elif not proof_path.exists():
        raise ValueError("Older non-temporal evidence has no source checkpoint; resume analyze once")
    return board
