"""Measured fitting and artifact regressions with fake TTS; no external services."""
import json
import wave
from types import SimpleNamespace

import pytest

from dubbingstory.tts import voice_manager as voices
from dubbingstory.tts.word_alignment import file_sha256, text_sha256, wav_duration
from dubbingstory.render.video_render import _validate_tts_manifest


def write_wav(path, duration):
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0\0" * round(duration * 24000))


class FakeTTS:
    def __init__(self, durations):
        self.durations = durations
        self.texts = []

    def get_engine_name(self):
        return "fake"

    def synthesize(self, text, output_path, **kwargs):
        self.texts.append(text)
        duration = self.durations[text]
        if isinstance(duration, Exception):
            raise duration
        write_wav(output_path, duration)
        return output_path


def prepare(tmp_path, monkeypatch, durations, policy="legacy", languages=("id",)):
    scripts, audio = tmp_path / "scripts", tmp_path / "audio"
    scripts.mkdir()
    audio.mkdir()
    engine = FakeTTS(durations)
    monkeypatch.setattr(voices, "get_engine", lambda *args: engine)
    monkeypatch.setattr(voices, "_get_audio_duration", wav_duration)
    monkeypatch.setattr(voices, "_render_timeline_canvas", lambda **kwargs: write_wav(kwargs["output_path"], kwargs["total_duration"]))
    segments = [{"scene_id": "first", "text": "Awal", "timeline_start": 0, "timeline_end": 2.5},
                {"scene_id": "second", "text": "Akhir", "timeline_start": 2.5, "timeline_end": 5}]
    for language in languages:
        (scripts / f"script_{language}.json").write_text(json.dumps({"segments": segments, "total_duration": 5}), encoding="utf-8")
        (scripts / f"script_{language}.txt").write_text("[first] Awal\n[second] Akhir\n", encoding="utf-8")
    cfg = SimpleNamespace(tts_narration_fit=policy, tts_fit_retries=2, tts_min_gap=.2,
                          tts_normalize_loudness=False, subtitle_style="classic")
    return scripts, audio, engine, cfg


def test_partial_legacy_tts_preserves_scripts_for_retry(tmp_path, monkeypatch):
    scripts, audio, engine, cfg = prepare(tmp_path, monkeypatch, {"Awal": RuntimeError("provider unavailable"), "Akhir": 1})
    original = (scripts / "script_id.json").read_bytes()
    result = voices.generate_all_audio(str(scripts), str(audio), cfg=cfg, timeline_source_path=str(tmp_path / "missing.mp4"))
    assert (scripts / "script_id.json").read_bytes() == original
    assert "Awal" in (scripts / "script_id.txt").read_text(encoding="utf-8")
    assert result["id"]["status"] == "partial"
    assert result["id"]["failed_segment_ids"] == ["first"]
    assert result["id"]["strict_timing_validated"] is False


def test_measured_rewrite_resynthesizes_only_overlong_beat(tmp_path, monkeypatch):
    from dubbingstory.story import script_writer
    scripts, audio, engine, cfg = prepare(tmp_path, monkeypatch, {"Awal": 3, "Ringkas": 1, "Akhir": 1}, policy="rewrite")
    monkeypatch.setattr(script_writer, "shorten_for_measured_duration", lambda segment, **kwargs: {**segment, "text": "Ringkas"})
    result = voices.generate_all_audio(str(scripts), str(audio), cfg=cfg, timeline_duration=5)["id"]
    assert engine.texts == ["Awal", "Ringkas", "Akhir"]
    assert result["strict_timing_validated"] is True
    assert result["complete_coverage"] is True
    assert result["placements"][1]["actual_start"] == 2.5
    assert json.loads((scripts / "script_id.json").read_text(encoding="utf-8"))["segments"][0]["text"] == "Ringkas"


def test_overflow_stops_before_canvas_and_invalidates_old_completion(tmp_path, monkeypatch):
    scripts, audio, engine, cfg = prepare(tmp_path, monkeypatch, {"Awal": 3, "Akhir": 1}, policy="error")
    (audio / "audio_id.wav").write_bytes(b"old output")
    monkeypatch.setattr(voices, "_render_timeline_canvas", lambda **kwargs: pytest.fail("Overflow must not reach canvas rendering"))
    with pytest.raises(RuntimeError, match="exceeds window"):
        voices.generate_all_audio(str(scripts), str(audio), cfg=cfg, timeline_duration=5)
    assert not (audio / "audio_id.wav").exists()
    assert json.loads((audio / "audio_id.manifest.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("rendered_duration", [4.9, 5.1, 0.0, float("nan"), float("inf")])
def test_invalid_canvas_duration_cannot_publish_audio_or_rewritten_script(tmp_path, monkeypatch, rendered_duration):
    scripts, audio, engine, cfg = prepare(tmp_path, monkeypatch, {"Awal": 1, "Akhir": 1}, policy="error")
    original_script = (scripts / "script_id.json").read_bytes()
    original_text = (scripts / "script_id.txt").read_bytes()
    (audio / "audio_id.wav").write_bytes(b"previous completed audio")

    def measured_duration(path):
        return rendered_duration if str(path).endswith(".synthesizing.wav") else wav_duration(path)

    monkeypatch.setattr(voices, "_get_audio_duration", measured_duration)
    with pytest.raises(ValueError, match="Rendered narration duration") as error:
        voices.generate_all_audio(str(scripts), str(audio), cfg=cfg, timeline_duration=5)
    assert "expected=" in str(error.value)
    assert "actual=" in str(error.value)
    assert "delta=" in str(error.value)
    assert engine.texts == ["Awal", "Akhir"]  # Every individual beat fit first.
    assert not (audio / "audio_id.wav").exists()
    assert not (audio / "audio_id.synthesizing.wav").exists()
    assert (scripts / "script_id.json").read_bytes() == original_script
    assert (scripts / "script_id.txt").read_bytes() == original_text
    manifest = json.loads((audio / "audio_id.manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["strict_timing_validated"] is False


def test_explicit_language_avoids_other_cached_scripts(tmp_path, monkeypatch):
    scripts, audio, engine, cfg = prepare(tmp_path, monkeypatch, {"Awal": 1, "Akhir": 1}, languages=("id", "en"))
    cfg.requested_languages = ["id"]
    result = voices.generate_all_audio(str(scripts), str(audio), cfg=cfg)
    assert set(result) == {"id"}
    assert engine.texts == ["Awal", "Akhir"]


@pytest.mark.parametrize("first_duration, status", [(1, "complete"), (RuntimeError("provider unavailable"), "partial")])
def test_plain_legacy_manifest_reports_actual_coverage(tmp_path, monkeypatch, first_duration, status):
    scripts, audio, engine, cfg = prepare(tmp_path, monkeypatch, {"Awal": first_duration, "Akhir": 1})
    (scripts / "script_id.json").unlink()
    original = (scripts / "script_id.txt").read_bytes()
    monkeypatch.setattr(voices, "_concatenate_audio", lambda paths, output: write_wav(output, len(paths)))
    result = voices.generate_all_audio(str(scripts), str(audio), cfg=cfg)["id"]
    manifest = json.loads((audio / "audio_id.manifest.json").read_text(encoding="utf-8"))
    assert result["status"] == manifest["status"] == status
    assert manifest["strict_timing_validated"] is False
    assert manifest["complete_coverage"] is (status == "complete")
    assert manifest["failed_segment_ids"] == ([] if status == "complete" else ["first"])
    assert (scripts / "script_id.txt").read_bytes() == original


def test_native_manifest_rejects_legacy_or_per_beat_overrun(tmp_path):
    audio = tmp_path / "audio_id.wav"
    write_wav(audio, 5)
    manifest_path = tmp_path / "audio_id.manifest.json"
    manifest = {"schema_version": 1, "status": "complete", "audio_sha256": file_sha256(str(audio)),
                "timeline_duration": 5, "overflow_seconds": 0, "complete_coverage": True,
                "fit_policy": "legacy", "strict_timing_validated": False,
                "expected_segment_ids": ["first"], "failed_segment_ids": [],
                "placements": [{"scene_id": "first", "planned_start": 0, "planned_end": 2,
                                "actual_start": 0, "actual_end": 3}]}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="strict per-beat"):
        _validate_tts_manifest(str(audio), 5, None)
    manifest.update(fit_policy="error", strict_timing_validated=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="visual beat window"):
        _validate_tts_manifest(str(audio), 5, None)
