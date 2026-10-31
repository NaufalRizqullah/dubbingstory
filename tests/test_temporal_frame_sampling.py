"""Dense seek regressions for Kaggle; fake frames/decoder only, no media IO."""

import math
from types import SimpleNamespace

import pytest

from dubbingstory.segment import keyframes


class RoundedDecoder:
    """Model a backend rounding seek time to a frame index, including EOF."""

    def __init__(self, timestamps=None, fps=24):
        self.timestamps = timestamps if timestamps is not None else [i / fps for i in range(721)]
        self.fps = fps
        self.index = 0
        self.timestamp = 0.0
        self.released = False
        self.seeks, self.reads = [], 0

    def isOpened(self):
        return True

    def set(self, prop, milliseconds):
        assert prop == keyframes.cv2.CAP_PROP_POS_MSEC
        seconds = milliseconds / 1000
        self.seeks.append(seconds)
        self.index = round(seconds * self.fps)
        return True

    def read(self):
        self.reads += 1
        if self.index >= len(self.timestamps):
            return False, None
        self.timestamp = self.timestamps[self.index]
        frame = SimpleNamespace(shape=(480, 640, 3), number=self.index)
        self.index += 1
        return True, frame

    def get(self, prop):
        if prop == keyframes.cv2.CAP_PROP_FPS:
            return self.fps
        assert prop == keyframes.cv2.CAP_PROP_POS_MSEC
        return self.timestamp * 1000

    def release(self):
        self.released = True


def install_decoder(monkeypatch, decoder):
    saved = []
    monkeypatch.setattr(keyframes.cv2, "VideoCapture", lambda path: decoder)
    monkeypatch.setattr(keyframes.cv2, "imwrite", lambda path, frame, options: saved.append(frame.number) or True)
    return saved


def sample(requested, start, end, frame_id="frame_000001", scene_id="scene_001"):
    return {"frame_id": frame_id, "scene_id": scene_id, "timestamp": requested,
            "requested_timestamp": requested, "scene_start": start, "scene_end": end,
            "sample_start": max(start, requested - .5), "sample_end": min(end, requested + .5)}


def test_reported_fractional_video_endpoint_retains_every_planned_sample(tmp_path, monkeypatch):
    decoder = RoundedDecoder()
    saved = install_decoder(monkeypatch, decoder)
    scenes = [{"scene_id": f"scene_{i + 1:03d}", "start_time": start, "end_time": end}
              for i, (start, end) in enumerate([(0, 10.014), (10.014, 20.028), (20.028, 30.042)])]
    plan = keyframes.plan_temporal_frames(scenes, interval_seconds=1)
    records = keyframes.extract_temporal_frames_from_source("fake.mp4", plan, str(tmp_path))
    assert len(records) == len(plan) == len(saved) == 33
    assert [row["frame_id"] for row in records] == [row["frame_id"] for row in plan]
    assert records[-1]["requested_timestamp"] == 30.028
    assert records[-1]["timestamp"] == 30.0
    assert saved[-1] == 720
    assert decoder.released
    for row in records:
        assert row["scene_start"] <= row["timestamp"] < row["scene_end"]
        assert abs(row["timestamp"] - row["requested_timestamp"]) <= .1
        assert row["sample_start"] <= row["timestamp"] <= row["sample_end"]


def test_next_scene_seek_is_recovered_using_last_actual_frame_before_cut(tmp_path, monkeypatch):
    decoder = RoundedDecoder()
    saved = install_decoder(monkeypatch, decoder)
    records = keyframes.extract_temporal_frames_from_source(
        "fake.mp4", [sample(.99, 0, 1)], str(tmp_path))
    assert records[0]["requested_timestamp"] == .99
    assert records[0]["timestamp"] == pytest.approx(round(23 / 24, 6))
    assert saved == [23]
    assert len(decoder.seeks) == 2


def test_rounding_at_scene_start_reads_forward_without_relabelling_timestamp(tmp_path, monkeypatch):
    decoder = RoundedDecoder(fps=30)
    install_decoder(monkeypatch, decoder)
    records = keyframes.extract_temporal_frames_from_source(
        "fake.mp4", [sample(.967, .967, 1.1)], str(tmp_path))
    assert records[0]["requested_timestamp"] == .967
    assert records[0]["timestamp"] == 1.0
    assert decoder.reads == 2


def test_recovery_uses_measured_vfr_clock_not_index_divided_by_fps(tmp_path, monkeypatch):
    decoder = RoundedDecoder(timestamps=[0, .037, .091, .119], fps=30)
    saved = install_decoder(monkeypatch, decoder)
    # Initial seek rounds to index 4 (EOF); the previous actual frame is at .119,
    # which differs from index/fps=.1. Preserve that measured source clock.
    records = keyframes.extract_temporal_frames_from_source(
        "fake.mp4", [sample(.14, 0, .16)], str(tmp_path))
    assert saved == [3]
    assert records[0]["timestamp"] == .119
    assert records[0]["requested_timestamp"] == .14


def test_no_actual_frame_in_scene_is_not_replaced_with_adjacent_scene(tmp_path, monkeypatch):
    decoder = RoundedDecoder()
    saved = install_decoder(monkeypatch, decoder)
    with pytest.raises(keyframes.TemporalFrameExtractionError) as raised:
        keyframes.extract_temporal_frames_from_source(
            "fake.mp4", [sample(.01, .01, .02)], str(tmp_path))
    assert raised.value.code == "sample_unavailable"
    assert raised.value.details["requested_timestamp"] == .01
    assert saved == [] and decoder.released


def test_unreadable_media_is_not_replaced_by_last_frame_metadata(tmp_path, monkeypatch):
    decoder = RoundedDecoder(timestamps=[])
    install_decoder(monkeypatch, decoder)
    with pytest.raises(keyframes.TemporalFrameExtractionError, match="Cannot decode") as raised:
        keyframes.extract_temporal_frames_from_source(
            "fake.mp4", [sample(30.028, 20.028, 30.042)], str(tmp_path))
    assert raised.value.code == "sample_unavailable"
    assert decoder.reads == 2 and decoder.released


@pytest.mark.parametrize("clock,code", [(float("nan"), "decoder_clock_invalid"),
                                        (-.1, "decoder_clock_invalid"), (8, "decoder_clock_drift")])
def test_invalid_clock_is_fatal_even_when_frame_exists(tmp_path, monkeypatch, clock, code):
    decoder = RoundedDecoder(timestamps=[clock] * 721)
    saved = install_decoder(monkeypatch, decoder)
    with pytest.raises(keyframes.TemporalFrameExtractionError) as raised:
        keyframes.extract_temporal_frames_from_source("fake.mp4", [sample(1, 0, 2)], str(tmp_path))
    assert raised.value.code == code
    assert len(decoder.seeks) == 1
    assert saved == [] and decoder.released
    assert all(not isinstance(value, float) or math.isfinite(value) for value in raised.value.details.values())


def test_error_details_do_not_include_paths_or_arbitrary_text():
    error = keyframes.TemporalFrameExtractionError("sample_unavailable", "Cannot decode", details={
        "frame_id": "frame_000001", "scene_id": "scene_001", "requested_timestamp": 3.5,
        "path": "/private/path", "api_key": "fake-secret", "decoder_timestamp": float("nan"),
    })
    assert error.details == {"frame_id": "frame_000001", "scene_id": "scene_001", "requested_timestamp": 3.5}
