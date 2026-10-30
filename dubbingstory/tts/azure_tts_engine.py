"""Official Azure Speech TTS: REST audio and optional SDK word boundaries.

Designed for the Azure Speech Free (F0) tier as well as Standard (S0).  The
engine uses the documented real-time REST endpoint and writes RIFF PCM WAV
output so it drops into the existing DubbingStory timeline scheduler.
"""

from __future__ import annotations

import html
import importlib
import io
import math
import os
import queue
import tempfile
import threading
import time
import wave
from typing import Any

import requests

from dubbingstory.tts.base_engine import BaseTTSEngine


class AzureTTSEngine(BaseTTSEngine):
    DEFAULT_VOICES = {
        "id": "id-ID-ArdiNeural",
        "en": "en-US-GuyNeural",
    }
    LANGUAGE_CODES = {"id": "id-ID", "en": "en-US"}

    def __init__(
        self,
        *,
        subscription_key: str | None = None,
        region: str | None = None,
        endpoint: str | None = None,
        custom_voices: dict[str, str] | None = None,
        output_format: str = "riff-24khz-16bit-mono-pcm",
        timeout_seconds: float = 60.0,
        rate_limit_per_minute: int = 20,
        max_retries: int = 4,
        metadata_backend: str = "auto",
    ) -> None:
        self.subscription_key = (subscription_key or os.getenv("AZURE_SPEECH_KEY", "")).strip()
        self.region = (region or os.getenv("AZURE_SPEECH_REGION", "")).strip()
        self.endpoint = (endpoint or os.getenv("AZURE_SPEECH_ENDPOINT", "")).strip()
        self.output_format = output_format or "riff-24khz-16bit-mono-pcm"
        self.timeout_seconds = float(timeout_seconds)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Azure timeout_seconds must be positive and finite")
        self.rate_limit_per_minute = max(0, int(rate_limit_per_minute or 0))
        self.max_retries = max(1, min(5, int(max_retries or 1)))
        if metadata_backend not in {"auto", "sdk", "alignment"}:
            raise ValueError("Azure metadata_backend must be auto, sdk or alignment")
        self.metadata_backend = metadata_backend
        self.voices = dict(self.DEFAULT_VOICES)
        if custom_voices:
            self.voices.update({str(k): str(v) for k, v in custom_voices.items() if v})
        self._last_request_at = 0.0

    def get_engine_name(self) -> str:
        return "azure"

    @staticmethod
    def _load_speech_sdk():
        try:
            return importlib.import_module("azure.cognitiveservices.speech")
        except (ImportError, OSError):
            return None

    def get_capabilities(self) -> dict:
        native = (self.metadata_backend != "alignment" and not self.endpoint
                  and self._load_speech_sdk() is not None)
        return {**super().get_capabilities(), "native_word_timings": native,
                "timing_backend": "provider" if native else "whisperx",
                "requires_alignment_extra": not native}

    def list_voices(self, language: str = "id") -> list[dict[str, Any]]:
        voice = self.voices.get(language) or self.voices.get("id")
        return [
            {
                "voice_id": voice,
                "language": language,
                "description": "Azure Speech standard neural voice",
            }
        ]

    def _synthesis_url(self) -> str:
        if self.endpoint:
            return self.endpoint.rstrip("/")
        if not self.region:
            raise RuntimeError(
                "AZURE_SPEECH_REGION is required for Azure TTS. "
                "Create an Azure Speech resource (F0 is supported) and set the region."
            )
        return f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1"

    def _check_credentials(self) -> None:
        if not self.subscription_key:
            raise RuntimeError(
                "AZURE_SPEECH_KEY is required for Azure TTS. "
                "Use the key from your Azure Speech resource."
            )

    def _throttle(self) -> None:
        # Azure F0 real-time TTS currently allows 20 transactions per 60 seconds.
        # A gentle client-side interval avoids avoidable HTTP 429s during multi-beat
        # narration synthesis. Set rate_limit_per_minute=0 to disable for S0.
        if self.rate_limit_per_minute <= 0 or self._last_request_at <= 0:
            return
        interval = 60.0 / float(self.rate_limit_per_minute)
        remaining = interval - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)

    @staticmethod
    def _rate_percent(speaking_rate: float) -> int:
        rate = float(speaking_rate)
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("TTS speaking_rate must be positive and finite")
        rate = max(0.75, min(1.25, rate))
        return int(round((rate - 1.0) * 100.0))

    def _ssml(self, text: str, language: str, voice: str, speaking_rate: float) -> str:
        lang_code = self.LANGUAGE_CODES.get(language, language or "id-ID")
        rate = self._rate_percent(speaking_rate)
        safe_text = html.escape(str(text or "").strip(), quote=False)
        safe_voice = html.escape(voice, quote=True)
        safe_lang = html.escape(lang_code, quote=True)
        return (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{safe_lang}">'
            f'<voice name="{safe_voice}"><prosody rate="{rate:+d}%">{safe_text}</prosody></voice>'
            '</speak>'
        )

    def synthesize(
        self,
        text: str,
        output_path: str,
        language: str = "id",
        voice_id: str | None = None,
        speaking_rate: float = 1.0,
    ) -> str:
        self._check_credentials()
        text = str(text or "").strip()
        if not text:
            raise ValueError("Azure TTS text is empty")

        voice = voice_id or self.voices.get(language) or self.voices.get("id")
        if not voice:
            raise ValueError(f"No Azure TTS voice configured for language={language!r}")

        url = self._synthesis_url()
        headers = {
            "Ocp-Apim-Subscription-Key": self.subscription_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": self.output_format,
            "User-Agent": "DubbingStory-AzureTTS",
        }
        body = self._ssml(text, language, voice, speaking_rate).encode("utf-8")

        response = None
        for attempt in range(self.max_retries):
            self._throttle()
            self._last_request_at = time.monotonic()
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    data=body,
                    timeout=self.timeout_seconds,
                    allow_redirects=False,
                )
            except requests.RequestException:
                if attempt + 1 >= self.max_retries:
                    raise RuntimeError("Azure TTS request failed after bounded retries. Check network access and Speech resource configuration.") from None
                time.sleep(min(8.0, 1.5 * (2 ** attempt)))
                continue

            try:
                if response.status_code == 200 and response.content:
                    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                    with open(output_path, "wb") as handle:
                        handle.write(response.content)
                    return output_path

                retryable = response.status_code == 429 or 500 <= response.status_code < 600
                if retryable and attempt + 1 < self.max_retries:
                    retry_after = response.headers.get("Retry-After", "")
                    try:
                        wait = float(retry_after) if retry_after else 1.5 * (2 ** attempt)
                    except (TypeError, ValueError):
                        wait = 1.5 * (2 ** attempt)
                    if not math.isfinite(wait):
                        wait = 1.5 * (2 ** attempt)
                    time.sleep(max(1.0, min(30.0, wait)))
                    continue

                raise RuntimeError(
                    f"Azure TTS failed HTTP {response.status_code}. Check credentials, voice, quota and region; provider response details were omitted."
                )
            finally:
                response.close()

        raise RuntimeError("Azure TTS failed after bounded retries.")

    def synthesize_with_metadata(
        self, text: str, output_path: str, language: str = "id",
        voice_id: str | None = None, speaking_rate: float = 1.0,
    ) -> dict:
        """Use boundaries from the same SDK request as the PCM waveform.

        ``auto`` preserves REST installations without the SDK. A configured
        custom REST endpoint also stays on REST: it is not a WebSocket SDK URL.
        The common alignment layer can then align that audio with WhisperX.
        A failed SDK request is never retried through another provider.
        """
        sdk = self._load_speech_sdk() if self.metadata_backend != "alignment" else None
        if self.metadata_backend == "sdk" and (sdk is None or self.endpoint):
            raise RuntimeError(
                "Azure SDK word timing requires the optional azure extra and a region-based "
                "Speech resource without a custom REST endpoint. Otherwise select metadata_backend=alignment."
            )
        if self.metadata_backend == "alignment" or sdk is None or self.endpoint:
            return super().synthesize_with_metadata(text, output_path, language, voice_id, speaking_rate)

        self._check_credentials()
        if not self.region:
            raise RuntimeError("AZURE_SPEECH_REGION is required for Azure SDK word timing.")
        text = str(text or "").strip()
        if not text:
            raise ValueError("Azure TTS text is empty")
        if not math.isfinite(float(speaking_rate)) or float(speaking_rate) <= 0:
            raise ValueError("TTS speaking_rate must be positive and finite")
        voice = voice_id or self.voices.get(language)
        if not voice:
            raise ValueError(f"No Azure TTS voice configured for language={language!r}")

        result, words = self._synthesize_sdk(sdk, self._ssml(text, language, voice, speaking_rate))
        payload = bytes(result.audio_data)
        try:
            with wave.open(io.BytesIO(payload), "rb") as audio:
                duration = audio.getnframes() / audio.getframerate()
                if audio.getcomptype() != "NONE" or not math.isfinite(duration) or duration <= 0:
                    raise ValueError("empty or non-PCM data")
        except (wave.Error, EOFError, ValueError, ZeroDivisionError):
            raise RuntimeError("Azure SDK returned unreadable PCM audio.") from None

        # Some voices omit boundaries. Retain the valid audio and let the shared
        # resolver either force-align it or fail under explicit provider mode.
        from dubbingstory.tts.word_alignment import WordAlignmentError, validate_word_timings
        timing_error = None
        try:
            words = validate_word_timings(words, text, duration)
        except WordAlignmentError:
            words = []
            timing_error = "Azure word boundaries are incomplete or do not match the final narration."

        directory = os.path.dirname(output_path) or "."
        os.makedirs(directory, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".wav", delete=False) as handle:
                temporary = handle.name
                handle.write(payload)
            os.replace(temporary, output_path)
        finally:
            if temporary and os.path.isfile(temporary):
                os.remove(temporary)
        return {"audio_path": output_path, "word_timings": words,
                "timing_source": "azure" if words else None, "timing_error": timing_error}

    def _synthesize_sdk(self, sdk, ssml: str):
        """Bound blocking SDK futures; only publish audio after completion."""
        try:
            speech_config = sdk.SpeechConfig(subscription=self.subscription_key, region=self.region)
            speech_config.set_speech_synthesis_output_format(sdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm)
        except Exception:
            raise RuntimeError("Could not configure Azure Speech SDK; check the resource region and SDK runtime.") from None
        for attempt in range(self.max_retries):
            words = []
            try:
                synthesizer = sdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
            except Exception:
                raise RuntimeError("Could not initialize Azure Speech synthesis; check the SDK runtime and resource configuration.") from None

            def on_word(event):
                if event.boundary_type != sdk.SpeechSynthesisBoundaryType.Word:
                    return
                try:
                    start = float(event.audio_offset) / 10_000_000
                    duration = event.duration.total_seconds()
                    words.append({"text": str(event.text), "start": start, "end": start + duration})
                except (AttributeError, TypeError, ValueError):
                    # Missing data remains missing; do not infer word durations.
                    return

            synthesizer.synthesis_word_boundary.connect(on_word)
            self._throttle()
            self._last_request_at = time.monotonic()
            completed = queue.Queue(maxsize=1)

            def wait_result():
                try:
                    completed.put((synthesizer.speak_ssml_async(ssml).get(), None))
                except Exception:
                    completed.put((None, "SDK synthesis failed; check resource configuration and network access."))

            # ResultFuture.get() has no timeout. A daemon waiter avoids hanging
            # the pipeline or writing late output after the deadline expires.
            waiter = threading.Thread(target=wait_result, daemon=True)
            waiter.start()
            try:
                result, failure = completed.get(timeout=self.timeout_seconds)
            except queue.Empty:
                try:
                    synthesizer.stop_speaking_async()
                except Exception:
                    pass
                raise RuntimeError("Azure SDK synthesis timed out. No completed audio was published; retry explicitly.") from None
            finally:
                try:
                    synthesizer.synthesis_word_boundary.disconnect(on_word)
                except Exception:
                    # An SDK shutdown error must not conceal a timeout or a
                    # completed result. No callback ever writes output files.
                    pass
            if failure:
                raise RuntimeError(failure) from None
            if result.reason == sdk.ResultReason.SynthesizingAudioCompleted:
                return result, words
            details = getattr(result, "cancellation_details", None)
            error_code = getattr(getattr(details, "error_code", None), "name", "")
            retryable = error_code in {"TooManyRequests", "ConnectionFailure", "ServiceTimeout", "ServiceError", "ServiceUnavailable"}
            if retryable and attempt + 1 < self.max_retries:
                time.sleep(min(8.0, 1.5 * (2 ** attempt)))
                continue
            raise RuntimeError("Azure SDK synthesis was canceled. Check the Speech resource, voice, quota and network; provider details were omitted.")
        raise RuntimeError("Azure SDK synthesis failed after bounded retries.")
