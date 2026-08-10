"""
dubbingstory.tts.edge_tts_engine — Microsoft Edge TTS integration

Edge-TTS is a free, high-quality neural TTS system using Microsoft Edge's
online text-to-speech service.
- Free (no API key required)
- High quality neural voices
- Supports 300+ voices in 100+ languages
- Native Indonesian support (id-ID-ArdiNeural, id-ID-GadisNeural)
- Supports SSML-like rate/pitch/volume adjustment

https://github.com/rany2/edge-tts
"""

import asyncio
import os
import subprocess

from dubbingstory.tts.base_engine import BaseTTSEngine

# Lazy import for edge_tts
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False


class EdgeTTSEngine(BaseTTSEngine):
    """
    Edge-TTS engine — free, high-quality, online neural TTS.

    Uses Microsoft Edge's read-aloud service. No API key needed.
    Requires internet connection.
    """

    DEFAULT_VOICES = {
        "id": "id-ID-ArdiNeural",
        "en": "en-US-GuyNeural",
    }

    VOICE_CATALOG = {
        "id": [
            {
                "voice_id": "id-ID-ArdiNeural",
                "language": "id",
                "gender": "Male",
                "description": "Indonesian male voice — warm, natural narrator tone",
            },
            {
                "voice_id": "id-ID-GadisNeural",
                "language": "id",
                "gender": "Female",
                "description": "Indonesian female voice — clear, friendly tone",
            },
        ],
        "en": [
            {
                "voice_id": "en-US-GuyNeural",
                "language": "en",
                "gender": "Male",
                "description": "US English male voice — casual, natural",
            },
            {
                "voice_id": "en-US-JennyNeural",
                "language": "en",
                "gender": "Female",
                "description": "US English female voice — warm, professional",
            },
            {
                "voice_id": "en-US-AriaNeural",
                "language": "en",
                "gender": "Female",
                "description": "US English female voice — expressive, lively",
            },
            {
                "voice_id": "en-GB-RyanNeural",
                "language": "en",
                "gender": "Male",
                "description": "British English male voice — articulate narrator",
            },
        ],
    }

    def __init__(self, custom_voices: dict[str, str] | None = None):
        """
        Initialize Edge-TTS engine.

        Parameters
        ----------
        custom_voices : dict[str, str] | None
            Override default voice mapping, e.g. {"id": "id-ID-GadisNeural"}.
        """
        if not EDGE_TTS_AVAILABLE:
            raise RuntimeError(
                "❌ edge-tts is not installed.\n"
                "   Install: pip install edge-tts"
            )
        self.voices = {**self.DEFAULT_VOICES}
        if custom_voices:
            self.voices.update(custom_voices)

    def get_engine_name(self) -> str:
        return "edge"

    def synthesize(
        self,
        text: str,
        output_path: str,
        language: str = "id",
        voice_id: str | None = None,
        speaking_rate: float = 1.0,
    ) -> str:
        """
        Synthesize text to speech using Edge-TTS.

        Edge-TTS produces MP3 natively. We convert to WAV via ffmpeg
        for consistency with the rest of the pipeline.

        Parameters
        ----------
        text : str
            Text to synthesize.
        output_path : str
            Path to save the output audio file (.wav).
        language : str
            Language code ("id", "en").
        voice_id : str | None
            Specific Edge-TTS voice name. If None, use default for language.
        speaking_rate : float
            Speed multiplier (1.0 = normal). Mapped to Edge-TTS rate format.

        Returns
        -------
        str
            Path to the generated audio file.
        """
        voice = voice_id or self.voices.get(language, self.DEFAULT_VOICES.get("en"))

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Edge-TTS rate format: "+0%", "+50%", "-20%", etc.
        rate_percent = int((speaking_rate - 1.0) * 100)
        rate_str = f"{rate_percent:+d}%"

        # Edge-TTS outputs MP3 — we'll convert to WAV after
        mp3_path = output_path + ".tmp.mp3"

        try:
            # Run the async synthesis in a sync context
            asyncio.run(self._async_synthesize(text, voice, rate_str, mp3_path))
        except Exception as e:
            # Clean up partial file
            if os.path.exists(mp3_path):
                os.remove(mp3_path)
            raise RuntimeError(f"❌ Edge-TTS synthesis failed: {e}") from e

        if not os.path.exists(mp3_path):
            raise RuntimeError(
                f"❌ Edge-TTS did not create output file: {mp3_path}"
            )

        # Convert MP3 → WAV for pipeline consistency
        self._mp3_to_wav(mp3_path, output_path)

        # Clean up temp MP3
        if os.path.exists(mp3_path):
            os.remove(mp3_path)

        return output_path

    async def _async_synthesize(
        self,
        text: str,
        voice: str,
        rate: str,
        output_path: str,
    ) -> None:
        """Async Edge-TTS synthesis."""
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save(output_path)

    @staticmethod
    def _mp3_to_wav(mp3_path: str, wav_path: str) -> None:
        """Convert MP3 to WAV using ffmpeg."""
        cmd = [
            "ffmpeg", "-y",
            "-i", mp3_path,
            "-acodec", "pcm_s16le",
            "-ar", "24000",
            "-ac", "1",
            wav_path,
        ]
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "❌ ffmpeg not found. Install ffmpeg to convert Edge-TTS output.\n"
                "   Download: https://ffmpeg.org/download.html"
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="replace")[:300] if e.stderr else ""
            raise RuntimeError(f"❌ MP3→WAV conversion failed: {stderr}") from e

    def list_voices(self, language: str = "id") -> list[dict]:
        """List available Edge-TTS voices for a language."""
        return self.VOICE_CATALOG.get(language, [])
