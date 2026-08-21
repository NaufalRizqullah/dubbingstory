"""Google Cloud Text-to-Speech Chirp 3 HD engine.

Authentication uses Google Cloud Application Default Credentials (ADC), not the
Gemini ``GOOGLE_API_KEY``.  In notebooks set ``GOOGLE_APPLICATION_CREDENTIALS``
to a service-account JSON file or otherwise configure ADC before using
``--engine chirp``.
"""

from __future__ import annotations

import os

from dubbingstory.tts.base_engine import BaseTTSEngine


class GoogleChirpTTSEngine(BaseTTSEngine):
    DEFAULT_VOICES = {
        "id": "id-ID-Chirp3-HD-Charon",
        "en": "en-US-Chirp3-HD-Charon",
    }
    LANGUAGE_CODES = {"id": "id-ID", "en": "en-US"}

    VOICE_CATALOG = {
        "id": [
            {"voice_id": "id-ID-Chirp3-HD-Charon", "description": "Indonesian male, balanced narration"},
            {"voice_id": "id-ID-Chirp3-HD-Fenrir", "description": "Indonesian male, expressive"},
            {"voice_id": "id-ID-Chirp3-HD-Puck", "description": "Indonesian male, conversational"},
            {"voice_id": "id-ID-Chirp3-HD-Aoede", "description": "Indonesian female, clear"},
            {"voice_id": "id-ID-Chirp3-HD-Leda", "description": "Indonesian female, storytelling"},
        ],
        "en": [
            {"voice_id": "en-US-Chirp3-HD-Charon", "description": "US English male"},
            {"voice_id": "en-US-Chirp3-HD-Aoede", "description": "US English female"},
        ],
    }

    def __init__(self, custom_voices: dict[str, str] | None = None, api_endpoint: str | None = None):
        try:
            from google.cloud import texttospeech
        except ImportError as exc:
            raise RuntimeError(
                "google-cloud-texttospeech is required for --engine chirp. "
                "Install: pip install google-cloud-texttospeech"
            ) from exc

        self.texttospeech = texttospeech
        client_options = {"api_endpoint": api_endpoint} if api_endpoint else None
        try:
            self.client = texttospeech.TextToSpeechClient(client_options=client_options)
        except Exception as exc:
            raise RuntimeError(
                "Could not initialize Google Cloud TTS. Configure Application "
                "Default Credentials (GOOGLE_APPLICATION_CREDENTIALS / ADC)."
            ) from exc

        self.voices = dict(self.DEFAULT_VOICES)
        if custom_voices:
            self.voices.update(custom_voices)

    def get_engine_name(self) -> str:
        return "chirp"

    def list_voices(self, language: str = "id") -> list[dict]:
        return self.VOICE_CATALOG.get(language, [])

    def synthesize(
        self,
        text: str,
        output_path: str,
        language: str = "id",
        voice_id: str | None = None,
        speaking_rate: float = 1.0,
    ) -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("Cannot synthesize empty text")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        tts = self.texttospeech
        voice_name = voice_id or self.voices.get(language, self.DEFAULT_VOICES["en"])
        language_code = self.LANGUAGE_CODES.get(language, "en-US")

        # Pace control is supported by current Chirp 3 HD synthesis.  Clamp to
        # a conservative narration range; duration fitting remains a scheduler
        # concern rather than aggressively speeding up the voice.
        rate = max(0.85, min(1.15, float(speaking_rate or 1.0)))
        response = self.client.synthesize_speech(
            input=tts.SynthesisInput(text=text),
            voice=tts.VoiceSelectionParams(language_code=language_code, name=voice_name),
            audio_config=tts.AudioConfig(
                audio_encoding=tts.AudioEncoding.LINEAR16,
                speaking_rate=rate,
            ),
        )

        with open(output_path, "wb") as handle:
            handle.write(response.audio_content)
        return output_path
