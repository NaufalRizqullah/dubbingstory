"""
dubbingstory.tts.base_engine — Abstract TTS engine interface

All TTS engines must implement this interface for pluggability.
"""

from abc import ABC, abstractmethod


class BaseTTSEngine(ABC):
    """Abstract base class for TTS engines."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: str,
        language: str = "id",
        voice_id: str | None = None,
        speaking_rate: float = 1.0,
    ) -> str:
        """
        Synthesize text to speech audio.

        Parameters
        ----------
        text : str
            Text to synthesize.
        output_path : str
            Path to save the audio file.
        language : str
            Language code ("id", "en").
        voice_id : str | None
            Specific voice to use. If None, use default for language.
        speaking_rate : float
            Speed multiplier (1.0 = normal).

        Returns
        -------
        str
            Path to the generated audio file.
        """
        ...

    @abstractmethod
    def list_voices(self, language: str = "id") -> list[dict]:
        """
        List available voices for a language.

        Returns
        -------
        list[dict]
            List of voice info dicts with at least "voice_id" and "description".
        """
        ...

    @abstractmethod
    def get_engine_name(self) -> str:
        """Return the engine name."""
        ...

    def estimate_duration(self, text: str, language: str = "id") -> float:
        """
        Estimate audio duration for a given text.

        Rough estimate based on average speaking rates:
        - Indonesian: ~3 words/second
        - English: ~2.5 words/second

        Parameters
        ----------
        text : str
            Text to estimate duration for.
        language : str
            Language code.

        Returns
        -------
        float
            Estimated duration in seconds.
        """
        words = len(text.split())
        rates = {
            "id": 3.0,  # Indonesian tends to be faster
            "en": 2.5,
        }
        rate = rates.get(language, 2.5)
        return words / rate
