"""Dense evidence regressions for Kaggle: mocked providers/decoders, no live media."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from dubbingstory.segment import keyframes
from dubbingstory.vision import temporal_analysis as temporal
from dubbingstory.vision.schemas import build_timed_frames_schema, normalize_timed_frames


def scene(name="scene_001", start=0.0, end=2.5):
    return {"scene_id": name, "start_time": start, "end_time": end,
            "duration": end - start, "analysis": {"action": "Memotong batang"}}


def config(**overrides):
    values = {"vision_temporal_enabled": True, "vision_temporal_interval_seconds": 0.5,
              "vision_temporal_batch_size": 2, "vision_concurrency": 1,
              "vision_provider": "openai", "vision_openai_model": "mock-vl"}
    values.update(overrides)
    return SimpleNamespace(**values)


def observed(frame, action="Memotong batang", change="", confidence=0.9):
    return {**frame, "action": action, "state_change": change, "confidence": confidence}


def response(frames):
    return [{"frame_id": frame["frame_id"], "action": "Memotong batang",
             "state_change": "", "confidence": 0.9} for frame in frames]


def test_dense_sampling_keeps_half_second_frames_beyond_legacy_cap():
    frames = keyframes.plan_temporal_frames([scene(end=12.0)], 0.5)
    assert len(frames) == 24
    assert [frame["timestamp"] for frame in frames] == [i / 2 for i in range(24)]
    assert len({frame["frame_id"] for frame in frames}) == 24


def test_cut_boundaries_and_subsecond_shots_are_kept():
    frames = keyframes.plan_temporal_frames([scene("a", 0, 1), scene("b", 1, 1.1)], 0.5)
    assert [(frame["scene_id"], frame["timestamp"]) for frame in frames] == [("a", 0), ("a", 0.5), ("b", 1)]
    assert all(frame["scene_start"] <= frame["sample_start"] < frame["sample_end"] <= frame["scene_end"] for frame in frames)


@pytest.mark.parametrize("interval", [0, 0.01, 11, float("nan"), float("inf")])
def test_invalid_interval_is_rejected(interval):
    with pytest.raises(ValueError, match="interval"):
        keyframes.plan_temporal_frames([scene()], interval)


@pytest.mark.parametrize("limit", [-1, 1.5, True, float("nan"), float("inf")])
def test_invalid_sample_cap_is_rejected(limit):
    with pytest.raises(ValueError, match="max_frames"):
        keyframes.plan_temporal_frames([scene()], max_frames=limit)


def test_sample_cap_fails_instead_of_silently_skipping():
    with pytest.raises(ValueError, match="No samples were silently discarded"):
        keyframes.plan_temporal_frames([scene()], 0.5, 4)


@pytest.mark.parametrize("scenes", [[], [scene("same"), scene("same", 3, 4)],
                                   [scene("a", 0, 2), scene("b", 1, 3)],
                                   [scene(end=float("nan"))]])
def test_bad_source_windows_fail(scenes):
    with pytest.raises(ValueError):
        keyframes.plan_temporal_frames(scenes)


class Decoder:
    def __init__(self, *, drift=0, readable=True):
        self.drift, self.readable = drift, readable
        self.position, self.released = 0.0, False

    def isOpened(self):
        return True

    def set(self, key, value):
        self.position = value

    def get(self, key):
        return 30.0 if key == keyframes.cv2.CAP_PROP_FPS else self.position + self.drift * 1000

    def read(self):
        return self.readable, SimpleNamespace(shape=(480, 640, 3)) if self.readable else None

    def release(self):
        self.released = True


def test_decoder_timestamp_is_used_and_sample_window_follows_it(tmp_path, monkeypatch):
    decoder = Decoder(drift=0.02)
    monkeypatch.setattr(keyframes.cv2, "VideoCapture", lambda _: decoder)
    monkeypatch.setattr(keyframes.cv2, "imwrite", lambda *args: True)
    frames = keyframes.extract_temporal_frames_from_source("mock.mp4", keyframes.plan_temporal_frames([scene()]), str(tmp_path))
    assert decoder.released
    assert frames[1]["timestamp"] == pytest.approx(0.52)
    assert frames[1]["requested_timestamp"] == 0.5
    assert frames[1]["sample_start"] == pytest.approx(0.27)


def test_rounded_scene_start_advances_to_real_in_scene_frame(tmp_path, monkeypatch):
    class BoundaryDecoder(Decoder):
        reads = 0

        def read(self):
            self.reads += 1
            return super().read()

        def get(self, key):
            if key == keyframes.cv2.CAP_PROP_FPS:
                return 30
            return (29 / 30 if self.reads == 1 else 1.0) * 1000

    decoder = BoundaryDecoder()
    monkeypatch.setattr(keyframes.cv2, "VideoCapture", lambda _: decoder)
    monkeypatch.setattr(keyframes.cv2, "imwrite", lambda *args: True)
    samples = keyframes.plan_temporal_frames([scene(start=.967, end=1.1)])
    frames = keyframes.extract_temporal_frames_from_source("mock.mp4", samples, str(tmp_path))
    assert decoder.reads == 2
    assert frames[0]["timestamp"] == 1.0
    assert frames[0]["requested_timestamp"] == .967


@pytest.mark.parametrize("drift,readable", [(1.0, True), (float("nan"), True), (-0.01, True), (0, False)])
def test_unusable_decoder_clock_or_frame_fails_and_releases(tmp_path, monkeypatch, drift, readable):
    decoder = Decoder(drift=drift, readable=readable)
    monkeypatch.setattr(keyframes.cv2, "VideoCapture", lambda _: decoder)
    monkeypatch.setattr(keyframes.cv2, "imwrite", lambda *args: True)
    with pytest.raises(ValueError):
        keyframes.extract_temporal_frames_from_source("mock.mp4", keyframes.plan_temporal_frames([scene()]), str(tmp_path))
    assert decoder.released


def test_every_frame_id_is_required_and_response_order_is_canonical():
    schema = build_timed_frames_schema(["first", "second"])
    assert schema["properties"]["frames"]["required"] == ["first", "second"]
    value = {"action": "Memotong", "state_change": "", "confidence": 0.8}
    result = normalize_timed_frames({"frames": {"second": value, "first": value}}, ["first", "second"])
    assert [item["frame_id"] for item in result] == ["first", "second"]
    with pytest.raises(ValueError, match="every requested frame"):
        normalize_timed_frames({"frames": {"first": value}}, ["first", "second"])


@pytest.mark.parametrize("confidence", [True, -0.1, 1.1, float("nan"), "0.9"])
def test_frame_confidence_is_not_coerced(confidence):
    with pytest.raises(ValueError, match="confidence"):
        normalize_timed_frames({"frames": {"a": {"action": "", "state_change": "", "confidence": confidence}}}, ["a"])


def test_context_pressure_splits_all_frames_without_omission():
    from dubbingstory.vision.openai_vision import VisionContextBudgetError
    frames = keyframes.plan_temporal_frames([scene(end=2)])
    calls = []

    def analyze(batch, prompt, **kwargs):
        calls.append([frame["frame_id"] for frame in batch])
        if len(batch) > 1:
            raise VisionContextBudgetError("mock capacity")
        assert batch[0]["frame_id"] in prompt
        return response(batch)

    result = temporal.analyze_frame_batch(SimpleNamespace(analyze_timed_frames=analyze), frames, 512)
    assert [item["frame_id"] for item in result] == [item["frame_id"] for item in frames]
    assert len(calls) == 2 * len(frames) - 1


def test_service_failure_does_not_multiply_calls_by_splitting():
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("mock service failure")

    with pytest.raises(RuntimeError, match="service"):
        temporal.analyze_frame_batch(SimpleNamespace(analyze_timed_frames=fail), keyframes.plan_temporal_frames([scene()]), 512)
    assert len(calls) == 1


def test_missing_observation_retries_smaller_batches_then_fails_if_still_missing():
    calls = []

    def partial(batch, *args, **kwargs):
        calls.append(len(batch))
        return response(batch[:-1])

    with pytest.raises(ValueError, match="missing frame"):
        temporal.analyze_frame_batch(SimpleNamespace(analyze_timed_frames=partial), keyframes.plan_temporal_frames([scene(end=1)]), 512)
    assert calls == [2, 1]


def test_events_do_not_bridge_uncertain_frames_or_unobserved_gaps():
    frames = keyframes.plan_temporal_frames([scene(end=3)])
    observations = [observed(frame) for frame in frames]
    observations[2]["confidence"] = 0.1
    events = temporal.build_visual_events(observations)
    assert [len(event["frame_ids"]) for event in events] == [2, 3]
    gap_events = temporal.build_visual_events([observations[0], observations[-1]])
    assert len(gap_events) == 2
    assert all(event["timing_basis"] == "sample_window" for event in events)


def test_per_second_summaries_leave_unsampled_seconds_explicit():
    frames = keyframes.plan_temporal_frames([scene(end=2.5)], 2)
    rows = temporal.build_second_summaries([observed(frame) for frame in frames], 2.5)
    assert [row["status"] for row in rows] == ["observed", "unsampled", "observed"]
    assert rows[1]["observations"] == rows[1]["actions"] == []
    assert rows[2]["end_time"] == 2.5


@pytest.fixture
def stage(tmp_path, monkeypatch):
    from dubbingstory.vision import scene_understanding
    (tmp_path / "source.mp4").write_bytes(b"mock source")
    calls, extracts = [], []
    analyzer = SimpleNamespace(model="mock-vl")

    def analyze(frames, prompt, **kwargs):
        calls.append([frame["frame_id"] for frame in frames])
        return response(frames)

    def extract(source, planned, root, **kwargs):
        extracts.append(root)
        result = []
        for frame in planned:
            path = Path(root) / (frame["frame_id"] + ".jpg")
            path.write_bytes(b"mock image " + frame["frame_id"].encode())
            result.append({**frame, "path": str(path)})
        return result

    analyzer.analyze_timed_frames = analyze
    monkeypatch.setattr(scene_understanding, "_create_analyzer", lambda cfg: analyzer)
    monkeypatch.setattr(temporal, "extract_temporal_frames_from_source", extract)
    storyboard = {"scenes": [scene()]}
    segment = {"scenes": [scene()]}
    return SimpleNamespace(root=tmp_path, analyzer=analyzer, calls=calls, extracts=extracts,
                           storyboard=storyboard, segment=segment)


def run(stage, **settings):
    return temporal.run_temporal_analysis(stage.segment, stage.storyboard, str(stage.root), config(**settings))


def test_complete_stage_covers_all_frames_and_reuses_batches(stage):
    enriched = run(stage)
    artifact = json.loads((stage.root / "visual_timeline.json").read_text())
    assert artifact["status"] == "complete"
    assert artifact["completed_frames"] == artifact["expected_frames"] == 5
    assert len(artifact["per_second"]) == 3
    assert enriched["visual_timeline"]["artifact_sha256"] == temporal._file_hash(str(stage.root / "visual_timeline.json"))
    assert len(stage.calls) == 3
    assert "events" not in stage.storyboard["scenes"][0]["analysis"]
    assert enriched["scenes"][0]["analysis"]["events"]
    stage.calls.clear()
    run(stage)
    assert stage.calls == []
    assert len(stage.extracts) == 1


def test_model_and_source_changes_invalidate_analysis_cache(stage):
    run(stage)
    stage.calls.clear()
    stage.analyzer.model = "another-model"
    run(stage)
    assert len(stage.calls) == 3
    stage.calls.clear()
    (stage.root / "source.mp4").write_bytes(b"changed source")
    run(stage)
    assert len(stage.calls) == 3
    assert len(stage.extracts) == 2


def test_changed_extraction_version_invalidates_frames_and_observations(stage, monkeypatch):
    run(stage)
    stage.calls.clear()
    monkeypatch.setattr(temporal, "TEMPORAL_EXTRACTION_VERSION", temporal.TEMPORAL_EXTRACTION_VERSION + 1)
    run(stage)
    assert len(stage.extracts) == 2
    assert len(stage.calls) == 3
    artifact = json.loads((stage.root / "visual_timeline.json").read_text())
    assert artifact["provenance"]["extraction_version"] == temporal.TEMPORAL_EXTRACTION_VERSION


def test_extraction_failure_reports_frame_and_stage_without_raw_error(stage, monkeypatch):
    def fail(*args, **kwargs):
        raise keyframes.TemporalFrameExtractionError(
            "frame_unavailable", "unsafe upstream URL https://private.invalid/?token=fake-secret",
            details={"frame_id": "frame_000032", "scene_id": "scene_003",
                     "requested_timestamp": 30.028, "decoder_timestamp": float("nan"),
                     "scene_start": 20.028, "scene_end": 30.042,
                     "tolerance_seconds": 0.1, "raw_error": "fake-secret"})

    monkeypatch.setattr(temporal, "extract_temporal_frames_from_source", fail)
    with pytest.raises(RuntimeError, match="frame_extraction; code: frame_unavailable") as error:
        run(stage)
    artifact_text = (stage.root / "visual_timeline.json").read_text()
    artifact = json.loads(artifact_text)
    assert artifact["status"] == "failed"
    assert artifact["failure_stage"] == "frame_extraction"
    assert artifact["failure_details"]["frame_id"] == "frame_000032"
    assert artifact["failure_details"]["requested_timestamp"] == 30.028
    assert "decoder_timestamp" not in artifact["failure_details"]
    assert "fake-secret" not in artifact_text + str(error.value)
    assert stage.calls == []


def test_provider_failure_diagnostics_do_not_publish_arbitrary_exception_text(stage):
    def fail(*args, **kwargs):
        raise RuntimeError("fake-secret-provider-response")

    stage.analyzer.analyze_timed_frames = fail
    with pytest.raises(RuntimeError, match="frame_analysis_failed") as error:
        run(stage)
    artifact_text = (stage.root / "visual_timeline.json").read_text()
    assert json.loads(artifact_text)["failure_stage"] == "frame_analysis"
    assert "fake-secret-provider-response" not in artifact_text + str(error.value)


def test_modified_frame_or_timestamp_is_reextracted_from_source(stage):
    run(stage)
    manifest_path = next((stage.root / "temporal_frames").glob("*/frames.json"))
    manifest = json.loads(manifest_path.read_text())
    manifest["frames"][0]["timestamp"] = 99
    manifest_path.write_text(json.dumps(manifest))
    run(stage)
    assert len(stage.extracts) == 2
    manifest = json.loads(manifest_path.read_text())
    Path(manifest["frames"][0]["path"]).write_bytes(b"altered image")
    run(stage)
    assert len(stage.extracts) == 3


def test_corrupt_batch_cache_is_recomputed(stage):
    run(stage)
    cache = next((stage.root / "temporal_cache").glob("*.json"))
    payload = json.loads(cache.read_text())
    payload["observations"] = []
    cache.write_text(json.dumps(payload))
    stage.calls.clear()
    run(stage)
    assert len(stage.calls) == 1


def test_failed_stage_invalidates_old_completion_and_preserves_input(stage):
    run(stage)
    unchanged = deepcopy(stage.storyboard)
    stage.analyzer.model = "uncached-failing-model"

    def fail(*args, **kwargs):
        raise RuntimeError("mock outage")

    stage.analyzer.analyze_timed_frames = fail
    with pytest.raises(RuntimeError, match="did not complete"):
        run(stage)
    artifact = json.loads((stage.root / "visual_timeline.json").read_text())
    assert artifact["status"] == "failed"
    assert stage.storyboard == unchanged


def test_even_preflight_failure_invalidates_old_completion(stage):
    run(stage)
    with pytest.raises(ValueError, match="max_frames"):
        run(stage, vision_temporal_max_frames=1)
    assert json.loads((stage.root / "visual_timeline.json").read_text())["status"] == "failed"


def test_storyboard_missing_scene_is_not_published_as_complete(stage):
    stage.storyboard["scenes"] = []
    with pytest.raises(ValueError, match="every segmented scene"):
        run(stage)
    assert stage.calls == stage.extracts == []


@pytest.mark.parametrize("settings", [{"vision_temporal_enabled": False}, {"pipeline_mode": "summary"}])
def test_disabled_or_summary_mode_does_not_create_artifacts(stage, settings):
    result = run(stage, **settings)
    assert result is stage.storyboard
    assert stage.calls == stage.extracts == []
    assert not (stage.root / "visual_timeline.json").exists()


def test_openai_timed_method_keeps_all_images_and_uses_exact_schema(tmp_path, monkeypatch):
    from dubbingstory.vision.openai_vision import OpenAIVisionAnalyzer
    analyzer = object.__new__(OpenAIVisionAnalyzer)
    frames = [{"frame_id": str(i), "path": str(tmp_path / f"{i}.jpg")} for i in range(3)]
    for frame in frames:
        Path(frame["path"]).write_bytes(b"mock")
    captured = {}
    monkeypatch.setattr(analyzer, "_messages_for_scene", lambda paths, prompt: captured.update(paths=paths) or [prompt])

    def call(messages, **kwargs):
        captured.update(kwargs)
        return {"frames": {row["frame_id"]: {key: row[key] for key in ("action", "state_change", "confidence")}
                           for row in response(frames)}}

    monkeypatch.setattr(analyzer, "_call_with_retry", call)
    result = analyzer.analyze_timed_frames(frames, "inspect every frame", 1024)
    assert captured["paths"] == [frame["path"] for frame in frames]
    assert captured["response_schema"]["properties"]["frames"]["required"] == ["0", "1", "2"]
    assert len(result) == 3
