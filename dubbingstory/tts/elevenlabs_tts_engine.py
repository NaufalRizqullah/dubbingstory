"""Optional ElevenLabs TTS adapter with measured character-to-word alignment.

Uses the existing requests dependency, never a guessed voice ID. PCM output is
wrapped in WAV without trimming/resampling, keeping the provider timeline intact.
"""

from __future__ import annotations

import base64
import binascii
import math
import os
import re
import tempfile
import time
import wave
from urllib.parse import quote

import requests

from dubbingstory.tts.base_engine import BaseTTSEngine
from dubbingstory.tts.word_alignment import WordAlignmentError, validate_word_timings


def words_from_character_alignment(alignment: dict, text: str, duration: float) -> list[dict]:
    """Aggregate supplied character intervals; never distribute time uniformly."""
    if not isinstance(alignment, dict):
        raise WordAlignmentError("ElevenLabs character alignment is missing.")
    characters = alignment.get("characters")
    starts = alignment.get("character_start_times_seconds")
    ends = alignment.get("character_end_times_seconds")
    if (not isinstance(characters, list) or not isinstance(starts, list)
            or not isinstance(ends, list) or not characters
            or len(characters) != len(starts) or len(characters) != len(ends)):
        raise WordAlignmentError("ElevenLabs alignment arrays have inconsistent lengths.")
    if any(not isinstance(char, str) or len(char) != 1 for char in characters):
        raise WordAlignmentError("ElevenLabs alignment must contain individual characters.")
    try:
        starts = [float(value) for value in starts]
        ends = [float(value) for value in ends]
    except (TypeError, ValueError):
        raise WordAlignmentError("ElevenLabs character timestamps are invalid.") from None
    previous_start = 0.0
    previous_end = 0.0
    for start, end in zip(starts, ends):
        if (not math.isfinite(start) or not math.isfinite(end) or start < 0
                or end < start or start < previous_start - 0.001
                or end < previous_end - 0.001 or end > duration + 0.02):
            raise WordAlignmentError("ElevenLabs character intervals are invalid or out of order.")
        previous_start, previous_end = start, end
    transcript = "".join(characters)
    words = []
    for match in re.finditer(r"\S+", transcript):
        # Punctuation is retained for display. The lexical validator ignores
        # punctuation-only events and validates all spoken words against text.
        words.append({"text": match.group(), "start": starts[match.start()],
                      "end": ends[match.end() - 1]})
    return validate_word_timings(words, text, duration)


class ElevenLabsTTSEngine(BaseTTSEngine):
    API_URL = "https://api.elevenlabs.io/v1/text-to-speech"
    SAMPLE_RATE = 24000

    def __init__(
        self, *, api_key: str | None = None, custom_voices: dict[str, str] | None = None,
        model: str = "eleven_multilingual_v2", timeout_seconds: float = 60.0,
        max_attempts: int = 3,
    ) -> None:
        self.api_key = str(api_key or os.getenv("ELEVENLABS_API_KEY", "")).strip()
        self.voices = {str(key): str(value).strip() for key, value in (custom_voices or {}).items() if value}
        self.model = str(model or "eleven_multilingual_v2").strip()
        self.timeout_seconds = float(timeout_seconds)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("ElevenLabs timeout_seconds must be positive and finite")
        self.max_attempts = max(1, min(5, int(max_attempts)))
        self._previous_text = ""
        self._next_text = ""

    def get_engine_name(self) -> str:
        return "elevenlabs"

    def get_capabilities(self) -> dict:
        return {**super().get_capabilities(), "native_word_timings": True,
                "timing_backend": "provider", "requires_alignment_extra": False,
                "contextual_synthesis": True}

    def set_context(self, previous_text: str = "", next_text: str = "") -> None:
        # Bound optional context independently from spoken content. It is sent
        # as context fields, never prepended/appended to the synthesized text.
        self._previous_text = str(previous_text or "")[-2000:]
        self._next_text = str(next_text or "")[:2000]

    def list_voices(self, language: str = "id") -> list[dict]:
        voice = self.voices.get(language)
        return ([{"voice_id": voice, "language": language,
                  "description": "Configured ElevenLabs voice"}] if voice else [])

    def synthesize(
        self, text: str, output_path: str, language: str = "id",
        voice_id: str | None = None, speaking_rate: float = 1.0,
    ) -> str:
        return self.synthesize_with_metadata(text, output_path, language, voice_id, speaking_rate)["audio_path"]

    def synthesize_with_metadata(
        self, text: str, output_path: str, language: str = "id",
        voice_id: str | None = None, speaking_rate: float = 1.0,
    ) -> dict:
        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is required for --engine elevenlabs. Configure it in the execution environment's secrets.")
        text = str(text or "").strip()
        if not text:
            raise ValueError("ElevenLabs TTS text is empty")
        voice = str(voice_id or self.voices.get(language) or "").strip()
        if not voice:
            raise ValueError(f"Configure an ElevenLabs voice ID for language={language!r}; voice IDs from Edge/Azure/Piper/Chirp cannot be reused.")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", voice):
            raise ValueError("ElevenLabs voice ID contains unsupported characters")
        rate = float(speaking_rate)
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("TTS speaking_rate must be positive and finite")
        # Conservative narration speed; timing already reflects this setting.
        rate = min(1.2, max(0.7, rate))
        body = {"text": text, "model_id": self.model,
                "voice_settings": {"speed": rate}}
        if self._previous_text:
            body["previous_text"] = self._previous_text
        if self._next_text:
            body["next_text"] = self._next_text
        # Multilingual v2 does not support language_code. Let the chosen voice
        # and narration text determine its language; do not send ignored flags.
        if self.model != "eleven_multilingual_v2":
            body["language_code"] = language
        payload = self._request(voice, body)
        try:
            pcm = base64.b64decode(payload["audio_base64"], validate=True)
        except (KeyError, TypeError, ValueError, binascii.Error):
            raise RuntimeError("ElevenLabs returned invalid audio data.") from None
        if not pcm or len(pcm) % 2:
            raise RuntimeError("ElevenLabs returned empty or incomplete 16-bit PCM audio.")
        duration = len(pcm) / (2 * self.SAMPLE_RATE)
        words = []
        timing_error = None
        # Prefer the actual normalized spoken text. If it expands numbers or
        # abbreviations, exact lexical validation must fail, not conceal it by
        # switching back to an original-text alignment with different words.
        alignment = payload.get("normalized_alignment")
        if alignment is None:
            alignment = payload.get("alignment")
        try:
            words = words_from_character_alignment(alignment, text, duration)
        except WordAlignmentError:
            timing_error = "ElevenLabs timestamps are incomplete or normalized speech differs from narration; forced alignment is required."

        directory = os.path.dirname(output_path) or "."
        os.makedirs(directory, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".wav", delete=False) as handle:
                temporary = handle.name
            with wave.open(temporary, "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(self.SAMPLE_RATE)
                audio.writeframes(pcm)
            os.replace(temporary, output_path)
        finally:
            if temporary and os.path.isfile(temporary):
                os.remove(temporary)
        return {"audio_path": output_path, "word_timings": words,
                "timing_source": "elevenlabs" if words else None, "timing_error": timing_error}

    def _request(self, voice: str, body: dict) -> dict:
        url = f"{self.API_URL}/{quote(voice, safe='')}/with-timestamps"
        for attempt in range(self.max_attempts):
            try:
                response = requests.post(
                    url, params={"output_format": "pcm_24000"},
                    headers={"xi-api-key": self.api_key, "Content-Type": "application/json"},
                    json=body, timeout=self.timeout_seconds, allow_redirects=False,
                )
            except requests.RequestException:
                if attempt + 1 < self.max_attempts:
                    time.sleep(min(8.0, 1.5 * 2 ** attempt))
                    continue
                raise RuntimeError("ElevenLabs request failed after bounded retries. Check network access and service availability.") from None
            try:
                if response.status_code == 200:
                    try:
                        payload = response.json()
                    except ValueError:
                        raise RuntimeError("ElevenLabs returned invalid JSON.") from None
                    if not isinstance(payload, dict):
                        raise RuntimeError("ElevenLabs returned an invalid response object.")
                    return payload
                retryable = response.status_code == 429 or 500 <= response.status_code < 600
                if retryable and attempt + 1 < self.max_attempts:
                    try:
                        delay = float(response.headers.get("Retry-After", 1.5 * 2 ** attempt))
                    except (TypeError, ValueError):
                        delay = 1.5 * 2 ** attempt
                    if not math.isfinite(delay):
                        delay = 1.5 * 2 ** attempt
                    time.sleep(max(0.0, min(15.0, delay)))
                    continue
                raise RuntimeError(
                    f"ElevenLabs failed HTTP {response.status_code}. Check API access, configured voice/model and quota; provider details were omitted."
                )
            finally:
                response.close()
        raise RuntimeError("ElevenLabs synthesis failed after bounded retries.")
