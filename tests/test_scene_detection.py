"""Scene detection regressions for Kaggle, using fake decoders (no media IO)."""
from types import SimpleNamespace

import pytest

from dubbingstory.segment import scene_detect


class Timecode:
    def __init__(self, frame, fps=24, seconds=None):
        self.frame, self.fps, self.seconds = frame, fps, seconds

    def get_frames(self):
        return self.frame

    def get_seconds(self):
        return self.frame / self.fps if self.seconds is None else self.seconds

    def __add__(self, frames):
        seconds = None if self.seconds is None else self.seconds + frames / self.fps
        return Timecode(self.frame + frames, self.fps, seconds)


def install_decoder(monkeypatch, *, processed=721, declared=721, intervals=None, error=None,
                    last_position=None):
    """Mirror no-cut behavior: the manager returns a range only when asked."""
    video = SimpleNamespace(duration=None if declared is None else Timecode(declared),
                            position=Timecode(processed - 1) if last_position is None else last_position)
    calls = []

    class Manager:
        def add_detector(self, detector):
            calls.append(("detector", detector))

        def detect_scenes(self, *, video):
            calls.append(("decode", video))
            if error is not None:
                raise error
            return processed

        def get_scene_list(self, *, start_in_scene=False):
            calls.append(("ranges", start_in_scene))
            if intervals is not None:
                return intervals
            return [(Timecode(0), Timecode(processed))] if start_in_scene else []

    monkeypatch.setattr(scene_detect, "open_video", lambda path: video)
    monkeypatch.setattr(scene_detect, "SceneManager", Manager)
    monkeypatch.setattr(scene_detect, "_get_detector", lambda *args: "fake-detector")
    return calls


def test_no_cut_video_retains_whole_decoded_range(monkeypatch, capsys):
    calls = install_decoder(monkeypatch)
    scenes = scene_detect.detect_scenes("fake.mp4")
    assert ("ranges", True) in calls
    assert len(scenes) == 1
    assert scenes[0]["start_frame"] == 0
    assert scenes[0]["end_frame"] == 721
    assert scenes[0]["start_time"] == 0
    assert scenes[0]["end_time"] == pytest.approx(30.042)
    assert scenes[0]["parent_scene_id"] == "scene_001"
    output = capsys.readouterr().out
    assert "Decoded frames: 721; metadata estimate: 721" in output
    assert "No scene cuts detected" in output


def test_no_cut_video_uses_existing_long_shot_analysis_windows(monkeypatch):
    install_decoder(monkeypatch)
    scenes = scene_detect.detect_scenes("fake.mp4", max_duration=15)
    assert len(scenes) == 3
    assert scenes[0]["start_frame"] == 0 and scenes[-1]["end_frame"] == 721
    assert scenes[0]["start_time"] == 0 and scenes[-1]["end_time"] == pytest.approx(30.042)
    assert [scene["window_index"] for scene in scenes] == [0, 1, 2]
    assert all(scene["parent_scene_id"] == "scene_001" and scene["window_count"] == 3 for scene in scenes)
    for before, after in zip(scenes, scenes[1:]):
        assert before["end_time"] == after["start_time"]
        assert before["end_frame"] == after["start_frame"]


def test_detected_cuts_keep_merge_and_identity_contract(monkeypatch):
    ranges = [(Timecode(0), Timecode(24)), (Timecode(24), Timecode(120)),
              (Timecode(120), Timecode(240))]
    install_decoder(monkeypatch, processed=240, declared=240, intervals=ranges)
    scenes = scene_detect.detect_scenes("fake.mp4", min_duration=2)
    assert [(s["start_frame"], s["end_frame"]) for s in scenes] == [(0, 120), (120, 240)]
    assert [s["scene_id"] for s in scenes] == ["scene_001", "scene_002"]
    assert [s["parent_scene_id"] for s in scenes] == ["scene_001", "scene_002"]


@pytest.mark.parametrize("processed", [0, -1])
def test_no_frames_fail_without_metadata_only_fallback(monkeypatch, processed):
    calls = install_decoder(monkeypatch, processed=processed)
    with pytest.raises(ValueError, match="decoded no frames"):
        scene_detect.detect_scenes("unreadable.mp4")
    assert not any(call[0] == "ranges" for call in calls)


def test_significantly_short_decode_is_not_published_as_full_source(monkeypatch):
    install_decoder(monkeypatch, processed=240, declared=721)
    with pytest.raises(ValueError, match="stopped early.*240.*721"):
        scene_detect.detect_scenes("truncated.mp4")


@pytest.mark.parametrize("declared", [None, 0, 722])
def test_unknown_or_slightly_approximate_count_uses_decoded_endpoint(monkeypatch, declared):
    install_decoder(monkeypatch, processed=721, declared=declared)
    scenes = scene_detect.detect_scenes("fake.mp4")
    assert scenes[-1]["end_frame"] == 721


def test_vfr_timecode_endpoint_need_not_equal_decoded_image_count(monkeypatch):
    # Modern PySceneDetect can index timecodes using PTS while processed_frames
    # counts decoded images. They must not be compared as the same measurement.
    install_decoder(monkeypatch, processed=721, declared=721,
                    intervals=[(Timecode(0), Timecode(750))], last_position=Timecode(749))
    scenes = scene_detect.detect_scenes("vfr.mp4")
    assert scenes[-1]["end_time"] == 31.25
    assert scenes[-1]["end_frame"] == 750


@pytest.mark.parametrize("intervals", [
    [],
    [(Timecode(0), Timecode(0))],
    [(Timecode(1), Timecode(721))],
    [(Timecode(0), Timecode(100)), (Timecode(101), Timecode(721))],
    [(Timecode(0), Timecode(721, seconds=float("nan")))],
    [(Timecode(0), Timecode(700))],
])
def test_empty_or_invalid_detected_ranges_fail(monkeypatch, intervals):
    install_decoder(monkeypatch, intervals=intervals)
    with pytest.raises(ValueError):
        scene_detect.detect_scenes("fake.mp4")


def test_decode_exception_prevents_scene_splitting(monkeypatch, tmp_path):
    install_decoder(monkeypatch, error=RuntimeError("decoder failed"))
    monkeypatch.setattr(scene_detect, "split_video", lambda *args: pytest.fail("must not split failed decode"))
    with pytest.raises(RuntimeError, match="decoder failed"):
        scene_detect.detect_and_split("fake.mp4", str(tmp_path))
