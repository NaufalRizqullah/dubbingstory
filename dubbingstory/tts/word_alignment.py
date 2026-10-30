"""Measured word timings for generated narration, never uniform interpolation.

All public timing values are seconds relative to the associated waveform unless
explicitly transformed into the final video timeline. Heavy alignment libraries
are imported only when requested; Edge-only captions do not require WhisperX.
"""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
import math
import os
import re
import unicodedata
import wave
from pathlib import Path


class WordAlignmentError(ValueError):
    """Audio/text cannot be represented by trustworthy word timestamps."""


_ALIGN_MODELS: dict[tuple, tuple] = {}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold())


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def wav_duration(path: str) -> float:
    try:
        with wave.open(str(path), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
    except (OSError, EOFError, wave.Error, ZeroDivisionError) as exc:
        raise WordAlignmentError("Word alignment requires a readable PCM WAV from the TTS pipeline.") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise WordAlignmentError("Word alignment requires nonempty audio.")
    return duration


def validate_word_timings(
    words: list[dict], text: str | None = None, duration: float | None = None
) -> list[dict]:
    """Validate order, nonzero intervals, and exact lexical text coverage.

    Punctuation/case differences are harmless; absent words, numeric expansions,
    sentence-level events, and missing timestamps require explicit correction or
    realignment. Existing timing values are not redistributed or interpolated.
    """
    if not isinstance(words, list) or not words:
        raise WordAlignmentError("No word timestamps available. Regenerate with Edge word boundaries or use WhisperX alignment.")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise WordAlignmentError("The associated audio duration must be positive and finite.")
    normalized = []
    previous_end = 0.0
    for index, word in enumerate(words):
        if not isinstance(word, dict):
            raise WordAlignmentError(f"Word {index + 1} is not a timestamp record.")
        label = str(word.get("text", word.get("word", ""))).strip()
        if not _tokens(label):
            # A punctuation-only event does not describe a spoken word.
            continue
        if len(label.split()) > 1:
            raise WordAlignmentError(f"Boundary {index + 1} contains multiple words; sentence timing is insufficient.")
        try:
            start, end = float(word["start"]), float(word["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WordAlignmentError(f"Word {index + 1} has no valid start/end. Spell out unsupported numbers or abbreviations and regenerate narration.") from exc
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise WordAlignmentError(f"Invalid interval for word {index + 1}.")
        if start < previous_end - 0.001:
            raise WordAlignmentError(f"Overlapping or out-of-order interval for word {index + 1}.")
        if duration is not None and end > duration + 0.02:
            raise WordAlignmentError(f"Word {index + 1} exceeds the associated audio duration.")
        item = {**word, "text": label, "start": start, "end": end}
        item.pop("word", None)
        normalized.append(item)
        previous_end = end
    if not normalized:
        raise WordAlignmentError("No spoken words have timestamps.")
    if text is not None:
        observed = [token for word in normalized for token in _tokens(word["text"])]
        if observed != _tokens(text):
            raise WordAlignmentError("Word timestamps do not cover the final narration text exactly. Regenerate or realign the final audio; no estimated word timing was substituted.")
        # Keep display punctuation when the provider uses the same word breaks.
        display_words = text.split()
        if len(display_words) == len(normalized) and all(
            _tokens(display) == _tokens(word["text"])
            for display, word in zip(display_words, normalized)
        ):
            for display, word in zip(display_words, normalized):
                word["spoken_text"] = word["text"]
                word["text"] = display
    return normalized


def transform_word_timings(
    words: list[dict], offset: float = 0.0, speed_factor: float = 1.0,
    scene_id: str | None = None,
) -> list[dict]:
    """Apply only post-synthesis speed changes, then final placement offset."""
    if not math.isfinite(speed_factor) or speed_factor <= 0 or not math.isfinite(offset) or offset < 0:
        raise WordAlignmentError("Invalid audio timing transform.")
    result = []
    for word in validate_word_timings(words):
        item = {**word, "start": offset + word["start"] / speed_factor,
                "end": offset + word["end"] / speed_factor}
        if scene_id is not None:
            item["scene_id"] = scene_id
        result.append(item)
    return result


def align_words(
    text: str, audio_path: str, language: str = "id", backend: str = "whisperx",
    device: str = "cpu", model_name: str | None = None,
) -> list[dict]:
    """Force-align known final text against a clean, fitted TTS WAV.

    Model downloads occur only when this optional runtime path is selected.
    WhisperX's interpolation is explicitly disabled: unalignable words fail
    validation instead of inheriting guessed times from neighbouring words.
    """
    if backend != "whisperx":
        raise WordAlignmentError(f"Unsupported forced alignment backend: {backend}")
    try:
        import whisperx
    except ImportError as exc:
        raise WordAlignmentError("WhisperX alignment is unavailable. Install the optional alignment extra in the execution environment, or use Edge word boundaries (--word-alignment edge).") from exc
    # 3.8.6 fixes the 'ignore' interpolation path and includes the Indonesian
    # model. Refuse older code rather than accepting fabricated timestamps.
    try:
        installed = metadata.version("whisperx")
    except metadata.PackageNotFoundError as exc:
        raise WordAlignmentError("Cannot verify WhisperX version. Install the project's alignment extra in the execution environment.") from exc
    release = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:\+[^\s]+)?", installed)
    if not release or not (3, 8, 6) <= tuple(map(int, release.groups())) < (4, 0, 0):
        raise WordAlignmentError("Word alignment requires WhisperX >=3.8.6,<4 to disable timing interpolation reliably. Update the optional alignment extra in the execution environment.")
    duration = wav_duration(audio_path)
    key = (language, device, model_name)
    try:
        if key not in _ALIGN_MODELS:
            options = {"language_code": language, "device": device}
            if model_name:
                options["model_name"] = model_name
            _ALIGN_MODELS[key] = whisperx.load_align_model(**options)
        model, model_metadata = _ALIGN_MODELS[key]
        result = whisperx.align(
            [{"text": text, "start": 0.0, "end": duration}],
            model, model_metadata, audio_path, device,
            interpolate_method="ignore", return_char_alignments=False,
        )
    except Exception as exc:
        raise WordAlignmentError(f"WhisperX could not align the narration ({type(exc).__name__}). Check the language model and alignment environment.") from exc
    words = result.get("word_segments", [])
    # Missing scores can indicate words with no alignable characters. Never
    # accept timestamps filled by a provider version's interpolation behaviour.
    for word in words:
        if _tokens(str(word.get("word", ""))) and "score" not in word:
            raise WordAlignmentError("WhisperX returned an unaligned word. Spell out numbers/abbreviations in the spoken narration and regenerate.")
    return validate_word_timings(words, text=text, duration=duration)


def resolve_word_timings(
    text: str, audio_path: str, language: str, cfg,
    provider_words: list[dict] | None = None, provider_source: str | None = None,
) -> tuple[list[dict], str]:
    """Choose provider boundaries or optional forced alignment with validation."""
    backend = getattr(cfg, "subtitle_alignment", "auto")
    if backend not in {"auto", "provider", "edge", "whisperx"}:
        raise WordAlignmentError(f"Unsupported word alignment selection: {backend}")
    provider_error = None
    if backend == "edge" and provider_source not in {None, "edge"}:
        raise WordAlignmentError("Edge alignment was selected for another TTS provider. Use --word-alignment auto or provider when switching engines.")
    if backend in {"auto", "provider", "edge"} and provider_words:
        try:
            source = provider_source or ("edge" if backend == "edge" else "provider")
            return validate_word_timings(provider_words, text, wav_duration(audio_path)), source
        except WordAlignmentError as exc:
            provider_error = str(exc)
    if backend == "provider":
        raise WordAlignmentError(provider_error or "Native word timestamps are missing. Enable this engine's metadata support, or use --word-alignment auto with the optional alignment extra.")
    if backend == "edge":
        raise WordAlignmentError(provider_error or "Edge word boundaries are missing. Use a compatible edge-tts version, regenerate TTS, or select --word-alignment whisperx.")
    try:
        words = align_words(
            text, audio_path, language,
            device=getattr(cfg, "subtitle_alignment_device", "cpu"),
            model_name=getattr(cfg, "subtitle_alignment_model", None),
        )
    except WordAlignmentError as exc:
        if provider_error:
            raise WordAlignmentError(f"Provider boundaries failed validation: {provider_error} Alignment fallback failed: {exc}") from exc
        raise
    minimum_score = float(getattr(cfg, "subtitle_alignment_min_score", 0.05))
    if not math.isfinite(minimum_score) or not 0 <= minimum_score <= 1:
        raise WordAlignmentError("subtitle.alignment_min_score must be between 0 and 1.")
    if any(not math.isfinite(float(word.get("score", 0))) or float(word.get("score", 0)) < minimum_score for word in words):
        raise WordAlignmentError("Forced alignment contains low-confidence words; review the narration/audio before generating animated captions.")
    return words, "whisperx"


def build_word_payload(
    words: list[dict], audio_path: str, text: str, language: str,
    placements: list[dict] | None = None,
) -> dict:
    """Build a versioned artifact bound to the final audio and narration."""
    return {
        "schema_version": 1, "language": language,
        "audio_path": str(Path(audio_path).resolve()),
        "audio_sha256": file_sha256(audio_path), "text_sha256": text_sha256(text),
        "text": text, "placements": placements or [],
        "words": validate_word_timings(words, text=text, duration=wav_duration(audio_path)),
    }


def validate_word_payload(
    payload: dict | str, audio_path: str | None = None, text: str | None = None,
) -> list[dict]:
    """Reject stale timing before a caption-only re-render.

    Callers may supply the relocated project's audio path (e.g. on Kaggle),
    avoiding any requirement that the original machine's path still exists.
    """
    if isinstance(payload, (str, os.PathLike)):
        with open(payload, encoding="utf-8") as source:
            payload = json.load(source)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise WordAlignmentError("Unsupported timing manifest version. Regenerate TTS/alignment.")
    audio_path = audio_path or payload.get("audio_path")
    if not audio_path or not os.path.isfile(audio_path):
        raise WordAlignmentError("The timing manifest audio is missing; provide the current project audio or regenerate TTS.")
    if file_sha256(audio_path) != payload.get("audio_sha256"):
        raise WordAlignmentError("Audio changed since word alignment. Regenerate timings before rendering.")
    final_text = payload.get("text", "") if text is None else text
    if text_sha256(final_text) != payload.get("text_sha256"):
        raise WordAlignmentError("Narration changed since word alignment. Regenerate TTS and timings.")
    return validate_word_timings(payload.get("words", []), final_text, wav_duration(audio_path))
