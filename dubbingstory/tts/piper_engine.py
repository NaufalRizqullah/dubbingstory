"""
dubbingstory.tts.piper_engine — Piper TTS integration (MVP)

Piper is a fast, local neural TTS system.
- MIT license
- CPU-only (no GPU needed)
- Indonesian voice: id_ID-news_tts-medium
- English voices: en_US-lessac-medium, en_GB-semaine-medium

https://github.com/rhasspy/piper
"""

import os
import subprocess
import wave

from dubbingstory.tts.base_engine import BaseTTSEngine


class PiperTTSEngine(BaseTTSEngine):
    """
    Piper TTS engine — fast, local, CPU-only.

    Uses the piper-tts CLI tool. Models are auto-downloaded
    on first use.
    """

    DEFAULT_VOICES = {
        "id": "id_ID-news_tts-medium",
        "en": "en_US-lessac-medium",
    }

    def __init__(self, model_dir: str | None = None):
        """
        Initialize Piper TTS engine.

        Parameters
        ----------
        model_dir : str | None
            Directory for model storage. If None, uses default
            piper cache directory.
        """
        self.model_dir = model_dir

    def get_engine_name(self) -> str:
        return "piper"

    def synthesize(
        self,
        text: str,
        output_path: str,
        language: str = "id",
        voice_id: str | None = None,
        speaking_rate: float = 1.0,
    ) -> str:
        """
        Synthesize text to speech using Piper.

        Calls piper CLI: echo "text" | piper --model voice --output_file out.wav
        """
        voice = voice_id or self.DEFAULT_VOICES.get(language, self.DEFAULT_VOICES["en"])

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Build piper command
        cmd = [
            "piper",
            "--model", voice,
            "--output_file", output_path,
        ]

        if speaking_rate != 1.0:
            # Piper supports --length_scale (inverse of speed: 0.5 = 2x speed)
            length_scale = 1.0 / speaking_rate
            cmd.extend(["--length_scale", f"{length_scale:.2f}"])

        if self.model_dir:
            cmd.extend(["--data-dir", self.model_dir])

        try:
            result = subprocess.run(
                cmd,
                input=text,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "❌ Piper TTS not found.\n"
                "   Install: pip install piper-tts\n"
                "   Or download from: https://github.com/rhasspy/piper/releases"
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr[:300] if e.stderr else ""
            raise RuntimeError(f"❌ Piper TTS failed: {stderr}")

        if not os.path.exists(output_path):
            raise RuntimeError(f"❌ Piper did not create output file: {output_path}")

        return output_path

    def list_voices(self, language: str = "id") -> list[dict]:
        """List available Piper voices."""
        voices = {
            "id": [
                {
                    "voice_id": "id_ID-news_tts-medium",
                    "language": "id",
                    "description": "Indonesian news-style voice (medium quality)",
                    "sample_rate": 22050,
                },
            ],
            "en": [
                {
                    "voice_id": "en_US-lessac-medium",
                    "language": "en",
                    "description": "US English voice (medium quality)",
                    "sample_rate": 22050,
                },
                {
                    "voice_id": "en_GB-semaine-medium",
                    "language": "en",
                    "description": "British English voice (medium quality)",
                    "sample_rate": 22050,
                },
                {
                    "voice_id": "en_US-amy-medium",
                    "language": "en",
                    "description": "US English female voice (medium quality)",
                    "sample_rate": 22050,
                },
            ],
        }
        return voices.get(language, [])

    def get_audio_duration(self, audio_path: str) -> float:
        """Get duration of a WAV file in seconds."""
        try:
            with wave.open(audio_path, "r") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / rate
        except Exception:
            return 0.0
