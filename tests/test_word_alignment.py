"""Offline tests for the Kaggle execution environment; no service calls."""

import asyncio
import os
import sys
import tempfile
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import patch

from dubbingstory.tts import word_alignment as alignment
from dubbingstory.tts.base_engine import BaseTTSEngine
from dubbingstory.tts.edge_tts_engine import EdgeTTSEngine


def _write_wav(path, sample=b"\x00\x00", seconds=2):
    with wave.open(path, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(sample * int(24000 * seconds))


class WordAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.audio = os.path.join(self.directory.name, "narasi.wav")
        _write_wav(self.audio)
        self.words = [
            {"text": "Lihat", "start": .10, "end": .32},
            {"text": "hasilnya", "start": .65, "end": 1.42},
        ]

    def test_preserves_measured_pause_and_restores_display_punctuation(self):
        words = alignment.validate_word_timings(self.words, "Lihat hasilnya!", 2)
        self.assertEqual(words[1]["text"], "hasilnya!")
        self.assertEqual(words[1]["spoken_text"], "hasilnya")
        self.assertEqual(words[1]["start"], .65)
        self.assertEqual(words[0]["end"], .32)

    def test_rejects_missing_words_and_numeric_expansion_without_mapping(self):
        for text in ("Lihat hasilnya sekarang", "Lihat 2 hasilnya"):
            with self.subTest(text=text), self.assertRaises(alignment.WordAlignmentError):
                alignment.validate_word_timings(self.words, text, 2)

    def test_rejects_sentence_boundaries_invalid_times_and_overlap(self):
        invalid = [
            [{"text": "dua kata", "start": 0, "end": 1}],
            [{"text": "kata", "start": float("nan"), "end": 1}],
            [{"text": "kata", "start": 0, "end": float("inf")}],
            [{"text": "kata", "start": 1, "end": 1}],
            [{"text": "kata", "start": 0, "end": 2.1}],
            [{"text": "satu", "start": 0, "end": 1}, {"text": "dua", "start": .5, "end": 1.5}],
        ]
        for words in invalid:
            with self.subTest(words=words), self.assertRaises(alignment.WordAlignmentError):
                alignment.validate_word_timings(words, duration=2)

    def test_audio_transform_applies_speed_before_timeline_offset(self):
        result = alignment.transform_word_timings(self.words, offset=5, speed_factor=1.1, scene_id="beat_2")
        self.assertAlmostEqual(result[1]["start"], 5 + .65 / 1.1)
        self.assertAlmostEqual(result[1]["end"], 5 + 1.42 / 1.1)
        self.assertEqual(result[1]["scene_id"], "beat_2")
        self.assertEqual(self.words[1]["start"], .65)

    def test_payload_can_relocate_but_cannot_reuse_changed_audio_or_text(self):
        payload = alignment.build_word_payload(self.words, self.audio, "Lihat hasilnya", "id")
        relocated = os.path.join(self.directory.name, "kaggle.wav")
        _write_wav(relocated)
        os.remove(self.audio)
        self.assertEqual(len(alignment.validate_word_payload(payload, relocated, "Lihat hasilnya")), 2)
        with self.assertRaisesRegex(alignment.WordAlignmentError, "Narration changed"):
            alignment.validate_word_payload(payload, relocated, "Lihat hasil lain")
        _write_wav(relocated, sample=b"\x01\x00")
        with self.assertRaisesRegex(alignment.WordAlignmentError, "Audio changed"):
            alignment.validate_word_payload(payload, relocated, "Lihat hasilnya")

    def test_edge_boundaries_do_not_import_or_use_forced_alignment(self):
        with patch.object(alignment, "align_words", side_effect=AssertionError("Unexpected aligner")):
            words, source = alignment.resolve_word_timings(
                "Lihat hasilnya", self.audio, "id", SimpleNamespace(subtitle_alignment="auto"),
                provider_words=self.words, provider_source="edge",
            )
        self.assertEqual(source, "edge")
        self.assertEqual(words[1]["start"], .65)

    def test_explicit_edge_fails_instead_of_estimating_or_switching_backend(self):
        with patch.object(alignment, "align_words", side_effect=AssertionError("Unexpected aligner")):
            with self.assertRaises(alignment.WordAlignmentError):
                alignment.resolve_word_timings("Lihat hasilnya", self.audio, "id", SimpleNamespace(subtitle_alignment="edge"))

    def test_forced_alignment_rejects_old_whisperx_before_loading_models(self):
        fake = SimpleNamespace(load_align_model=lambda **kwargs: self.fail("Old aligner loaded"))
        with patch.dict(sys.modules, {"whisperx": fake}), patch.object(alignment.metadata, "version", return_value="3.8.5"):
            with self.assertRaisesRegex(alignment.WordAlignmentError, "3.8.6"):
                alignment.align_words("Lihat hasilnya", self.audio)

    def test_whisperx_uses_known_text_and_never_interpolates_missing_word(self):
        calls = []

        def fake_align(segments, model, metadata, audio, device, **kwargs):
            calls.append((segments, kwargs))
            return {"word_segments": [
                {"word": "Lihat", "start": .1, "end": .32, "score": .9},
                {"word": "hasilnya", "start": .65, "end": 1.42},
            ]}

        fake = SimpleNamespace(load_align_model=lambda **kwargs: (object(), {}), align=fake_align)
        with patch.dict(sys.modules, {"whisperx": fake}), patch.object(alignment.metadata, "version", return_value="3.8.6"), patch.dict(alignment._ALIGN_MODELS, {}, clear=True):
            with self.assertRaisesRegex(alignment.WordAlignmentError, "unaligned word"):
                alignment.align_words("Lihat hasilnya", self.audio)
        self.assertEqual(calls[0][0][0]["text"], "Lihat hasilnya")
        self.assertEqual(calls[0][1]["interpolate_method"], "ignore")

    def test_supported_whisperx_aligns_with_model_metadata_after_version_check(self):
        model = object()
        model_metadata = {"language": "id", "type": "huggingface"}
        calls = []

        def fake_align(segments, supplied_model, supplied_metadata, audio, device, **kwargs):
            calls.append((supplied_model, supplied_metadata, audio, device, kwargs))
            return {"word_segments": [
                {"word": "Lihat", "start": .1, "end": .32, "score": .9},
                {"word": "hasilnya", "start": .65, "end": 1.42, "score": .8},
            ]}

        fake = SimpleNamespace(load_align_model=lambda **kwargs: (model, model_metadata), align=fake_align)
        with patch.dict(sys.modules, {"whisperx": fake}), patch.object(alignment.metadata, "version", return_value="3.8.6"), patch.dict(alignment._ALIGN_MODELS, {}, clear=True):
            words = alignment.align_words("Lihat hasilnya", self.audio, language="id")
        self.assertEqual([word["text"] for word in words], ["Lihat", "hasilnya"])
        self.assertIs(calls[0][0], model)
        self.assertIs(calls[0][1], model_metadata)
        self.assertEqual(calls[0][4]["interpolate_method"], "ignore")

    def test_forced_alignment_quality_gate(self):
        words = [{**word, "score": .01} for word in self.words]
        with patch.object(alignment, "align_words", return_value=words):
            with self.assertRaisesRegex(alignment.WordAlignmentError, "low-confidence"):
                alignment.resolve_word_timings("Lihat hasilnya", self.audio, "id", SimpleNamespace(subtitle_alignment="whisperx"))


class EdgeMetadataTests(unittest.TestCase):
    def test_word_metadata_comes_from_audio_request_and_uses_seconds(self):
        calls = []

        class Communication:
            def __init__(self, text, voice, **kwargs):
                calls.append((text, voice, kwargs))

            async def stream(self):
                yield {"type": "audio", "data": b"audio-chunk"}
                yield {"type": "SentenceBoundary", "text": "Lihat hasilnya", "offset": 0, "duration": 20000000}
                yield {"type": "WordBoundary", "text": "Lihat", "offset": 1000000, "duration": 2200000}

        engine = object.__new__(EdgeTTSEngine)
        with tempfile.TemporaryDirectory() as directory, patch("dubbingstory.tts.edge_tts_engine.edge_tts", SimpleNamespace(Communicate=Communication), create=True):
            path = os.path.join(directory, "audio.mp3")
            words = asyncio.run(engine._async_synthesize_with_metadata("Lihat hasilnya", "id-ID-ArdiNeural", "+10%", path))
            with open(path, "rb") as audio:
                self.assertEqual(audio.read(), b"audio-chunk")
        self.assertEqual(calls[0][2]["boundary"], "WordBoundary")
        self.assertEqual(calls[0][2]["rate"], "+10%")
        self.assertEqual(len(words), 1)
        self.assertAlmostEqual(words[0]["start"], .1)
        self.assertAlmostEqual(words[0]["end"], .32)

    def test_base_extension_preserves_audio_only_engine_contract(self):
        class LegacyEngine(BaseTTSEngine):
            def synthesize(self, text, output_path, language="id", voice_id=None, speaking_rate=1):
                return output_path

            def list_voices(self, language="id"):
                return []

            def get_engine_name(self):
                return "legacy"

        self.assertEqual(LegacyEngine().synthesize_with_metadata("teks", "audio.wav"), {
            "audio_path": "audio.wav", "word_timings": [], "timing_source": None,
        })


if __name__ == "__main__":
    unittest.main()
