"""Provider contract regressions for Kaggle; all service/SDK calls are mocked.

These tests are authored for the execution environment and are not run on the
development computer. No test requires credentials, models, or live services.
"""

import base64
from datetime import timedelta
import io
import queue
from types import SimpleNamespace
from unittest.mock import Mock
import wave

import pytest
import requests

from dubbingstory.tts import azure_tts_engine as azure
from dubbingstory.tts import elevenlabs_tts_engine as eleven
from dubbingstory.tts import word_alignment as alignment
from dubbingstory.tts.base_engine import BaseTTSEngine
from dubbingstory.tts.google_chirp_tts_engine import GoogleChirpTTSEngine


def wav_bytes(seconds=2):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b"\x00\x00" * int(seconds * 24000))
    return buffer.getvalue()


def char_alignment(text, *, start=0.1, step=0.07):
    return {
        "characters": list(text),
        "character_start_times_seconds": [start + i * step for i in range(len(text))],
        "character_end_times_seconds": [start + (i + 1) * step for i in range(len(text))],
    }


def eleven_payload(text="Bor itu."):
    return {
        "audio_base64": base64.b64encode(b"\x00\x00" * 24000).decode("ascii"),
        "alignment": char_alignment(text),
        "normalized_alignment": char_alignment(text),
    }


def response(status, payload=None, headers=None):
    return SimpleNamespace(status_code=status, headers=headers or {},
                           json=lambda: payload, close=Mock())


def eleven_engine(**overrides):
    return eleven.ElevenLabsTTSEngine(
        api_key="test-placeholder", custom_voices={"id": "configured_voice"}, **overrides)


@pytest.mark.parametrize("provider", ["edge", "azure", "elevenlabs"])
@pytest.mark.parametrize("backend", ["auto", "provider"])
def test_native_provider_words_use_one_shared_resolver(tmp_path, monkeypatch, provider, backend):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(wav_bytes())
    words = [{"text": "Bor", "start": 0.1, "end": 0.3},
             {"text": "itu", "start": 0.8, "end": 1.1}]
    monkeypatch.setattr(alignment, "align_words", Mock(side_effect=AssertionError("No forced alignment")))
    actual, source = alignment.resolve_word_timings(
        "Bor itu.", str(audio), "id", SimpleNamespace(subtitle_alignment=backend), words, provider)
    assert source == provider
    assert actual[1]["start"] == 0.8
    assert actual[1]["text"] == "itu."


def test_explicit_provider_does_not_estimate_missing_boundaries(monkeypatch):
    monkeypatch.setattr(alignment, "align_words", Mock(side_effect=AssertionError("No fallback")))
    with pytest.raises(alignment.WordAlignmentError, match="Native word timestamps"):
        alignment.resolve_word_timings("Bor itu.", "unused.wav", "id",
                                       SimpleNamespace(subtitle_alignment="provider"))
    with pytest.raises(alignment.WordAlignmentError, match="another TTS provider"):
        alignment.resolve_word_timings("Bor itu.", "unused.wav", "id",
                                       SimpleNamespace(subtitle_alignment="edge"), [], "azure")


def test_audio_only_provider_falls_back_to_real_alignment(tmp_path, monkeypatch):
    words = [{"text": "Bor", "start": .1, "end": .3, "score": .9},
             {"text": "itu.", "start": .7, "end": 1, "score": .9}]
    forced = Mock(return_value=words)
    monkeypatch.setattr(alignment, "align_words", forced)
    result, source = alignment.resolve_word_timings(
        "Bor itu.", str(tmp_path / "unused.wav"), "id", SimpleNamespace(subtitle_alignment="auto"))
    assert result == words and source == "whisperx"
    forced.assert_called_once()


def test_character_alignment_keeps_observed_pauses_and_unicode():
    raw = char_alignment("Besi, déjà!")
    # Measured pause belongs between words; it must not be spread across letters.
    raw["character_start_times_seconds"][6:] = [value + .3 for value in raw["character_start_times_seconds"][6:]]
    raw["character_end_times_seconds"][6:] = [value + .3 for value in raw["character_end_times_seconds"][6:]]
    words = eleven.words_from_character_alignment(raw, "Besi, déjà!", 2)
    assert [word["text"] for word in words] == ["Besi,", "déjà!"]
    assert words[1]["start"] == raw["character_start_times_seconds"][6]
    assert words[0]["end"] == raw["character_end_times_seconds"][4]


@pytest.mark.parametrize("kind", ["missing", "length", "nan", "negative", "reverse", "beyond", "expanded"])
def test_character_alignment_rejects_incomplete_or_untrustworthy_timing(kind):
    raw = char_alignment("Baut")
    text = "Baut"
    if kind == "missing":
        raw = None
    elif kind == "length":
        raw["character_start_times_seconds"].pop()
    elif kind == "nan":
        raw["character_end_times_seconds"][1] = float("nan")
    elif kind == "negative":
        raw["character_start_times_seconds"][0] = -.1
    elif kind == "reverse":
        raw["character_end_times_seconds"][1] = .01
    elif kind == "beyond":
        raw["character_end_times_seconds"][-1] = 5
    else:
        raw = char_alignment("dua")
        text = "2"
    with pytest.raises(alignment.WordAlignmentError):
        eleven.words_from_character_alignment(raw, text, 2)


def test_elevenlabs_wav_and_metadata_share_the_same_request(tmp_path, monkeypatch):
    replies = response(200, eleven_payload())
    post = Mock(return_value=replies)
    monkeypatch.setattr(eleven.requests, "post", post)
    engine = eleven_engine()
    engine.set_context("Benda ini belum selesai.", "Lalu pasang bautnya.")
    target = tmp_path / "speech.wav"
    result = engine.synthesize_with_metadata("Bor itu.", str(target), speaking_rate=1.02)
    assert result["timing_source"] == "elevenlabs"
    assert len(result["word_timings"]) == 2
    assert alignment.wav_duration(str(target)) == 1.0
    post.assert_called_once()
    options = post.call_args.kwargs
    assert options["params"] == {"output_format": "pcm_24000"}
    assert options["allow_redirects"] is False
    assert options["json"]["text"] == "Bor itu."
    assert options["json"]["previous_text"] == "Benda ini belum selesai."
    assert options["json"]["next_text"] == "Lalu pasang bautnya."
    assert "language_code" not in options["json"]
    assert options["json"]["voice_settings"]["speed"] == 1.02
    replies.close.assert_called_once()


def test_normalized_speech_mismatch_does_not_claim_original_timing(tmp_path, monkeypatch):
    payload = eleven_payload("2")
    payload["normalized_alignment"] = char_alignment("dua")
    monkeypatch.setattr(eleven.requests, "post", Mock(return_value=response(200, payload)))
    result = eleven_engine().synthesize_with_metadata("2", str(tmp_path / "speech.wav"))
    assert result["word_timings"] == [] and result["timing_source"] is None
    assert "forced alignment" in result["timing_error"]


def test_missing_or_other_language_voice_fails_before_provider_call(tmp_path, monkeypatch):
    post = Mock(side_effect=AssertionError("No provider call"))
    monkeypatch.setattr(eleven.requests, "post", post)
    with pytest.raises(ValueError, match="voice ID"):
        eleven_engine().synthesize("Text", str(tmp_path / "x.wav"), language="en")
    post.assert_not_called()


def test_invalid_pcm_is_not_published(tmp_path, monkeypatch):
    payload = eleven_payload()
    payload["audio_base64"] = base64.b64encode(b"x").decode("ascii")
    monkeypatch.setattr(eleven.requests, "post", Mock(return_value=response(200, payload)))
    target = tmp_path / "x.wav"
    with pytest.raises(RuntimeError, match="incomplete"):
        eleven_engine().synthesize("Bor itu.", str(target))
    assert not target.exists()


def test_elevenlabs_retries_are_bounded_and_diagnostics_are_sanitized(monkeypatch):
    post = Mock(side_effect=requests.RequestException("sensitive provider request"))
    monkeypatch.setattr(eleven.requests, "post", post)
    sleep = Mock()
    monkeypatch.setattr(eleven.time, "sleep", sleep)
    with pytest.raises(RuntimeError) as caught:
        eleven_engine(max_attempts=3)._request("configured_voice", {"text": "Bor"})
    assert post.call_count == 3 and sleep.call_count == 2
    assert "sensitive" not in str(caught.value)


def test_elevenlabs_honors_bounded_retry_after_and_never_retries_auth_errors(monkeypatch):
    busy = response(429, headers={"Retry-After": "99999"})
    ok = response(200, eleven_payload())
    post = Mock(side_effect=[busy, ok])
    monkeypatch.setattr(eleven.requests, "post", post)
    sleep = Mock()
    monkeypatch.setattr(eleven.time, "sleep", sleep)
    eleven_engine()._request("configured_voice", {"text": "Bor"})
    sleep.assert_called_once_with(15.0)
    assert busy.close.call_count == ok.close.call_count == 1
    denied = response(401, {"detail": "not safe to expose"})
    post.reset_mock(side_effect=True)
    post.return_value = denied
    with pytest.raises(RuntimeError, match="HTTP 401") as caught:
        eleven_engine()._request("configured_voice", {"text": "Bor"})
    assert post.call_count == 1 and "not safe" not in str(caught.value)


def sdk_fake(events=None, results=None):
    instances = []
    success = SimpleNamespace(reason="done", audio_data=wav_bytes())
    pending_results = list(results or [success])

    class Synthesizer:
        def __init__(self, **kwargs):
            self.callback = None
            self.stopped = False
            self.synthesis_word_boundary = SimpleNamespace(connect=self.connect, disconnect=self.disconnect)
            instances.append(self)

        def connect(self, callback):
            self.callback = callback

        def disconnect(self, callback):
            assert self.callback is callback
            self.callback = None

        def speak_ssml_async(self, text):
            def get():
                for event in events or []:
                    self.callback(event)
                return pending_results.pop(0)
            return SimpleNamespace(get=get)

        def stop_speaking_async(self):
            self.stopped = True

    sdk = SimpleNamespace(
        SpeechConfig=lambda **kwargs: SimpleNamespace(set_speech_synthesis_output_format=lambda fmt: None),
        SpeechSynthesisOutputFormat=SimpleNamespace(Riff24Khz16BitMonoPcm="wav24"),
        SpeechSynthesizer=Synthesizer,
        SpeechSynthesisBoundaryType=SimpleNamespace(Word="word"),
        ResultReason=SimpleNamespace(SynthesizingAudioCompleted="done"),
    )
    return sdk, instances


def azure_engine(**options):
    return azure.AzureTTSEngine(subscription_key="test-placeholder", region="test-region",
                               rate_limit_per_minute=0, **options)


def test_azure_sdk_captures_native_words_and_ignores_punctuation(tmp_path, monkeypatch):
    events = [SimpleNamespace(boundary_type="word", text="Bor", audio_offset=1000000,
                              duration=timedelta(seconds=.2)),
              SimpleNamespace(boundary_type="punctuation", text=",", audio_offset=3000000,
                              duration=timedelta(0)),
              SimpleNamespace(boundary_type="word", text="itu", audio_offset=8000000,
                              duration=timedelta(seconds=.4))]
    sdk, instances = sdk_fake(events)
    monkeypatch.setattr(azure.AzureTTSEngine, "_load_speech_sdk", staticmethod(lambda: sdk))
    engine = azure_engine(metadata_backend="sdk")
    result = engine.synthesize_with_metadata("Bor itu.", str(tmp_path / "voice.wav"))
    assert result["timing_source"] == "azure"
    assert result["word_timings"][1]["start"] == .8
    assert result["word_timings"][1]["end"] == pytest.approx(1.2)
    assert len(instances) == 1 and instances[0].callback is None
    assert engine.get_capabilities()["native_word_timings"] is True


def test_azure_uses_event_text_for_unicode_not_guessed_character_offsets(tmp_path, monkeypatch):
    event = SimpleNamespace(boundary_type="word", text="déjà", audio_offset=1000000,
                            duration=timedelta(seconds=.5), text_offset=999, word_length=999)
    sdk, _ = sdk_fake([event])
    monkeypatch.setattr(azure.AzureTTSEngine, "_load_speech_sdk", staticmethod(lambda: sdk))
    result = azure_engine().synthesize_with_metadata("déjà", str(tmp_path / "voice.wav"))
    assert result["word_timings"][0]["text"] == "déjà"


def test_azure_auto_without_sdk_keeps_audio_only_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(azure.AzureTTSEngine, "_load_speech_sdk", staticmethod(lambda: None))
    engine = azure_engine()
    synth = Mock(return_value="voice.wav")
    monkeypatch.setattr(engine, "synthesize", synth)
    result = engine.synthesize_with_metadata("Bor itu.", str(tmp_path / "voice.wav"))
    assert result["word_timings"] == [] and result["timing_source"] is None
    assert engine.get_capabilities()["requires_alignment_extra"] is True
    synth.assert_called_once()
    with pytest.raises(RuntimeError, match="optional azure extra"):
        azure_engine(metadata_backend="sdk").synthesize_with_metadata("Bor itu.", "unused.wav")


def test_azure_timeout_stops_sdk_without_publishing_or_rest_fallback(tmp_path, monkeypatch):
    sdk, instances = sdk_fake()
    monkeypatch.setattr(azure.AzureTTSEngine, "_load_speech_sdk", staticmethod(lambda: sdk))
    monkeypatch.setattr(azure.threading, "Thread", lambda **kwargs: SimpleNamespace(start=lambda: None))
    monkeypatch.setattr(azure.queue, "Queue", lambda **kwargs: SimpleNamespace(get=Mock(side_effect=queue.Empty)))
    monkeypatch.setattr(azure.requests, "post", Mock(side_effect=AssertionError("No REST retry")))
    target = tmp_path / "timeout.wav"
    with pytest.raises(RuntimeError, match="timed out"):
        azure_engine().synthesize_with_metadata("Bor itu.", str(target))
    assert instances[0].stopped and instances[0].callback is None
    assert not target.exists()


def test_azure_retry_creates_fresh_boundary_collection(tmp_path, monkeypatch):
    canceled = SimpleNamespace(reason="canceled", cancellation_details=SimpleNamespace(
        error_code=SimpleNamespace(name="TooManyRequests")))
    done = SimpleNamespace(reason="done", audio_data=wav_bytes())
    event = SimpleNamespace(boundary_type="word", text="Bor", audio_offset=1000000,
                            duration=timedelta(seconds=.4))
    sdk, instances = sdk_fake([event], [canceled, done])
    monkeypatch.setattr(azure.AzureTTSEngine, "_load_speech_sdk", staticmethod(lambda: sdk))
    monkeypatch.setattr(azure.time, "sleep", Mock())
    result = azure_engine(max_retries=2).synthesize_with_metadata("Bor", str(tmp_path / "retry.wav"))
    assert len(instances) == 2 and len(result["word_timings"]) == 1


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_provider_timeouts_fail_before_requests(timeout):
    with pytest.raises(ValueError, match="timeout_seconds"):
        azure_engine(timeout_seconds=timeout)
    with pytest.raises(ValueError, match="timeout_seconds"):
        eleven_engine(timeout_seconds=timeout)


def test_base_capabilities_and_context_do_not_require_native_alignment():
    class AudioOnly(BaseTTSEngine):
        def synthesize(self, *args, **kwargs):
            return "audio.wav"
        def list_voices(self, language="id"):
            return []
        def get_engine_name(self):
            return "offline"
    engine = AudioOnly()
    engine.set_context("previous", "next")
    assert engine.get_capabilities()["native_word_timings"] is False
    assert engine.synthesize_with_metadata("text", "audio.wav")["word_timings"] == []


def chirp_without_credentials():
    engine = object.__new__(GoogleChirpTTSEngine)
    engine.voices = dict(GoogleChirpTTSEngine.DEFAULT_VOICES)
    engine.timeout_seconds = 60
    engine.texttospeech = SimpleNamespace(
        SynthesisInput=lambda **kwargs: kwargs,
        VoiceSelectionParams=lambda **kwargs: kwargs,
        AudioConfig=lambda **kwargs: kwargs,
        AudioEncoding=SimpleNamespace(LINEAR16="pcm"),
    )
    engine.client = SimpleNamespace(synthesize_speech=Mock(return_value=SimpleNamespace(audio_content=wav_bytes())))
    return engine


def test_chirp_keeps_audio_only_metadata_and_bounds_sdk_request(tmp_path):
    engine = chirp_without_credentials()
    result = engine.synthesize_with_metadata("Bor itu.", str(tmp_path / "chirp.wav"))
    assert result["word_timings"] == [] and result["timing_source"] is None
    options = engine.client.synthesize_speech.call_args.kwargs
    assert options["timeout"] == 60 and options["retry"] is None
    assert options["voice"]["language_code"] == "id-ID"
    assert engine.get_capabilities()["requires_alignment_extra"] is True


def test_chirp_service_failure_is_sanitized_and_does_not_publish(tmp_path):
    engine = chirp_without_credentials()
    engine.client.synthesize_speech.side_effect = RuntimeError("sensitive provider details")
    target = tmp_path / "chirp.wav"
    with pytest.raises(RuntimeError, match="Chirp synthesis failed") as caught:
        engine.synthesize("Bor itu.", str(target))
    assert "sensitive" not in str(caught.value) and not target.exists()


@pytest.mark.parametrize("text,rate", [("", 1), ("Bor", 0), ("Bor", -1),
                                      ("Bor", float("nan")), ("Bor", float("inf"))])
def test_piper_rejects_invalid_input_before_downloading_a_model(text, rate, monkeypatch):
    from dubbingstory.tts.piper_tts_engine import PiperTTSEngine

    engine = object.__new__(PiperTTSEngine)
    engine.voices = {"id": "id_ID-news_tts-medium"}
    download = Mock(side_effect=AssertionError("Invalid input must not download a model"))
    monkeypatch.setattr(engine, "_ensure_model_downloaded", download)
    with pytest.raises(ValueError):
        engine.synthesize(text, "unused.wav", speaking_rate=rate)
    download.assert_not_called()
