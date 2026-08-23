"""Official Azure Speech Text-to-Speech REST engine.

Designed for the Azure Speech Free (F0) tier as well as Standard (S0).  The
engine uses the documented real-time REST endpoint and writes RIFF PCM WAV
output so it drops into the existing DubbingStory timeline scheduler.
"""

from __future__ import annotations

import html
import os
import time
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
    ) -> None:
        self.subscription_key = (subscription_key or os.getenv("AZURE_SPEECH_KEY", "")).strip()
        self.region = (region or os.getenv("AZURE_SPEECH_REGION", "")).strip()
        self.endpoint = (endpoint or os.getenv("AZURE_SPEECH_ENDPOINT", "")).strip()
        self.output_format = output_format or "riff-24khz-16bit-mono-pcm"
        self.timeout_seconds = max(5.0, float(timeout_seconds or 60.0))
        self.rate_limit_per_minute = max(0, int(rate_limit_per_minute or 0))
        self.max_retries = max(1, int(max_retries or 1))
        self.voices = dict(self.DEFAULT_VOICES)
        if custom_voices:
            self.voices.update({str(k): str(v) for k, v in custom_voices.items() if v})
        self._last_request_at = 0.0

    def get_engine_name(self) -> str:
        return "azure"

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
        rate = max(0.75, min(1.25, float(speaking_rate or 1.0)))
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
                )
            except requests.RequestException as exc:
                if attempt + 1 >= self.max_retries:
                    raise RuntimeError(f"Azure TTS request failed: {exc}") from exc
                time.sleep(min(8.0, 1.5 * (2 ** attempt)))
                continue

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
                except ValueError:
                    wait = 1.5 * (2 ** attempt)
                time.sleep(max(1.0, min(30.0, wait)))
                continue

            detail = response.text[:500] if response.text else "empty response"
            raise RuntimeError(
                f"Azure TTS failed HTTP {response.status_code}: {detail}"
            )

        detail = response.text[:500] if response is not None and response.text else "no response"
        raise RuntimeError(f"Azure TTS failed after retries: {detail}")
