"""Action placement contracts. Run in Kaggle; no provider/media process needed."""
import hashlib
import json
import wave
from types import SimpleNamespace

import pytest

from dubbingstory.tts.action_alignment import place_anchored_speech
from dubbingstory.tts import voice_manager


def segment(start=5.0, end=6.0):
    event = {"event_id": "e1", "scene_id": "scene_001", "start_time": start,
             "end_time": end, "action": "Besi dipotong", "state_change": "Besi terpisah",
             "confidence": .9, "frame_ids": ["f10", "f11"]}
    return {"scene_id": "scene_001", "text": "Besi dipotong perlahan.",
            "timeline_start": 0.0, "timeline_end": 10.0,
            "action_sync_enabled": True, "visual_events": [event], "anchor_event_ids": ["e1"],
            "action_anchors": [{"event_id": "e1", "phrase": "dipotong", "start_time": 999}]}


def words():
    return [{"word": "Besi", "start": 0.0, "end": .8},
            {"word": "dipotong", "start": 1.0, "end": 1.8},
            {"word": "perlahan.", "start": 2.0, "end": 2.8}]


def test_start_places_measured_word_inside_canonical_event_not_llm_time():
    start, report = place_anchored_speech(segment(), words(), earliest_start=0, latest_start=7)
    assert start == pytest.approx(4.5)
    assert report["status"] == "complete"
    assert report["anchors"][0]["spoken_time"] == pytest.approx(5.5)
    assert report["anchors"][0]["start_time"] == 5


def test_previous_speech_boundary_is_never_crossed():
    start, report = place_anchored_speech(segment(), words(), earliest_start=5.8, latest_start=7, tolerance=.2)
    assert start >= 5.8
    assert report["status"] == "partial"
    assert report["anchors"][0]["error_seconds"] == pytest.approx(.8)


def test_conflicting_anchors_report_miss_instead_of_stretching_speech():
    item = segment(5, 5.5)
    item["text"] = "Besi dipotong lalu dirapikan."
    item["visual_events"].append({**item["visual_events"][0], "event_id": "e2", "start_time": 9,
                                  "end_time": 9.5, "action": "Besi dirapikan"})
    item["anchor_event_ids"].append("e2")
    item["action_anchors"].append({"event_id": "e2", "phrase": "dirapikan"})
    timing = words()[:2] + [{"word": "lalu", "start": 2, "end": 2.5},
                            {"word": "dirapikan", "start": 3, "end": 3.5}]
    start, report = place_anchored_speech(item, timing, earliest_start=0, latest_start=6, tolerance=0)
    assert 0 <= start <= 6
    assert report["status"] == "partial"
    assert all(not item["matched"] for item in report["anchors"])
    assert timing[-1]["start"] == 3


def test_no_eligible_event_is_not_reported_as_validated_alignment():
    item = segment()
    item.update(visual_events=[], anchor_event_ids=[], action_anchors=[])
    start, report = place_anchored_speech(item, words(), earliest_start=1, latest_start=7)
    assert start == 1
    assert report["status"] == "no_anchors"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_bounds_fail(value):
    with pytest.raises(ValueError):
        place_anchored_speech(segment(), words(), earliest_start=value, latest_start=7)


def write_wav(path, duration):
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0\0" * round(duration * 24000))


class MetadataVoice:
    def get_engine_name(self):
        return "fake"

    def synthesize(self, **kwargs):
        pytest.fail("Action sync must request timing even with captions disabled")

    def synthesize_with_metadata(self, output_path, **kwargs):
        write_wav(output_path, 3)
        return {"word_timings": words(), "timing_source": "fake"}


@pytest.mark.parametrize("style,burn", [("classic", False), ("karaoke", False), ("karaoke", True)])
def test_dub_exports_same_action_and_word_timing_without_burn(tmp_path, monkeypatch, style, burn):
    scripts, audio = tmp_path / "scripts", tmp_path / "audio"
    scripts.mkdir()
    item = segment()
    (scripts / "script_id.txt").write_text("[scene_001] " + item["text"], encoding="utf-8")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source fixture; no media decoder is used")
    timeline = tmp_path / "visual_timeline.json"
    timeline.write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    proof = {"visual_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
             "visual_timeline_sha256": hashlib.sha256(timeline.read_bytes()).hexdigest()}
    (scripts / "script_id.json").write_text(
        json.dumps({"segments": [item], "total_duration": 10, **proof}), encoding="utf-8")
    monkeypatch.setattr(voice_manager, "get_engine", lambda *a: MetadataVoice())
    monkeypatch.setattr(voice_manager, "_render_timeline_canvas",
                        lambda **kw: write_wav(kw["output_path"], kw["total_duration"]))
    cfg = SimpleNamespace(narration_action_sync=True, narration_action_sync_policy="error",
                          tts_narration_fit="error", subtitle_style=style, subtitle_alignment="auto",
                          render_burn_subtitles=burn, tts_normalize_loudness=False)
    result = voice_manager.generate_all_audio(
        str(scripts), str(audio), cfg=cfg, timeline_duration=10, timeline_source_path=str(source))["id"]
    assert result["placements"][0]["actual_start"] == pytest.approx(4.5)
    assert result["action_sync_validated"] is True
    assert result["word_timings"][1]["start"] == pytest.approx(5.5)
    assert (scripts / "script_id.words.json").is_file()
    report = json.loads((audio / "audio_id.actions.json").read_text())
    assert report["validated"] is True
    assert report["audio_sha256"] == result["audio_sha256"]
    assert report["source_sha256"] == proof["visual_source_sha256"]
    saved_script = json.loads((scripts / "script_id.json").read_text(encoding="utf-8"))
    assert all(saved_script[key] == value for key, value in proof.items())


@pytest.mark.parametrize("changed", ["source", "timeline", "missing_timeline", "missing_proof"])
def test_dub_rejects_stale_visual_proof_before_synthesis(tmp_path, monkeypatch, changed):
    scripts, audio = tmp_path / "scripts", tmp_path / "audio"
    scripts.mkdir()
    item = segment()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original source")
    timeline = tmp_path / "visual_timeline.json"
    timeline.write_text(json.dumps({"status": "complete", "events": ["original"]}), encoding="utf-8")
    structured = {"segments": [item], "total_duration": 10,
                  "visual_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "visual_timeline_sha256": hashlib.sha256(timeline.read_bytes()).hexdigest()}
    if changed == "source":
        source.write_bytes(b"replacement source")
    elif changed == "timeline":
        timeline.write_text(json.dumps({"status": "complete", "events": ["revised"]}), encoding="utf-8")
    elif changed == "missing_timeline":
        timeline.unlink()
    else:
        structured.pop("visual_timeline_sha256")
    (scripts / "script_id.txt").write_text("[scene_001] " + item["text"], encoding="utf-8")
    (scripts / "script_id.json").write_text(json.dumps(structured), encoding="utf-8")
    monkeypatch.setattr(voice_manager, "get_engine", lambda *a: MetadataVoice())
    monkeypatch.setattr(voice_manager, "_generate_timeline_audio",
                        lambda **kw: pytest.fail("Stale visual proof must fail before TTS scheduling"))
    cfg = SimpleNamespace(narration_action_sync=True, tts_narration_fit="error", subtitle_style="classic")
    with pytest.raises(ValueError, match="does not match the current source/visual timeline"):
        voice_manager.generate_all_audio(
            str(scripts), str(audio), cfg=cfg, timeline_duration=10, timeline_source_path=str(source))
    manifest = json.loads((audio / "audio_id.manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["strict_timing_validated"] is False
    assert not (audio / "audio_id.wav").exists()


def test_error_policy_never_publishes_missed_action_audio(tmp_path, monkeypatch):
    item = segment(0, .2)
    monkeypatch.setattr(voice_manager, "_render_timeline_canvas", lambda **kw: pytest.fail("Must stop before render"))
    cfg = SimpleNamespace(narration_action_sync=True, narration_action_sync_policy="error",
                          narration_action_sync_tolerance=.1, tts_narration_fit="error", subtitle_alignment="auto")
    with pytest.raises(RuntimeError, match="spoken action cannot fit"):
        voice_manager._generate_timeline_audio(engine=MetadataVoice(), language="id", segments=[item],
            total_duration=10, output_path=str(tmp_path / "out.wav"), audio_dir=str(tmp_path / "raw"),
            speaking_rate=1, cfg=cfg, fail_log_path=str(tmp_path / "failed.log"))
    assert not (tmp_path / "out.wav").exists()
