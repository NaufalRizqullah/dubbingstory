"""
dubbingstory.utils.timing — Timestamp and duration utilities
"""


def seconds_to_timecode(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm timecode."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def timecode_to_seconds(timecode: str) -> float:
    """Convert HH:MM:SS.mmm or HH:MM:SS,mmm timecode to seconds."""
    timecode = timecode.replace(",", ".")
    parts = timecode.split(":")

    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    else:
        return float(timecode)


def estimate_narration_duration(
    text: str,
    language: str = "id",
    words_per_second: float | None = None,
) -> float:
    """
    Estimate narration duration based on word count.

    Average speaking rates:
    - Indonesian: ~3.0 words/second
    - English: ~2.5 words/second
    """
    word_count = len(text.split())
    if words_per_second is None:
        rates = {"id": 3.0, "en": 2.5}
        words_per_second = rates.get(language, 2.5)
    return word_count / words_per_second if words_per_second > 0 else 0.0
