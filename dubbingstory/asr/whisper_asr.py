"""
dubbingstory.asr.whisper_asr — Faster-Whisper based ASR module

Provides automatic speech recognition to generate transcripts from raw video
audio when pre-existing subtitle files are unavailable. The generated transcript
is formatted to be compatible with the existing subtitle_reader output, with
added word-level timestamps for more granular analysis if needed.

Dependencies:
- faster-whisper
- ffmpeg / ffprobe (already used elsewhere)

The module is structured to lazily load the heavy ASR model on first use, so
importing the module is cheap even when ASR is not actually invoked.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from dubbingstory.utils.ffmpeg_helpers import extract_audio, get_video_duration

# Lazy import for faster_whisper to avoid immediate heavy dependency load
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WhisperConfig:
    """Configuration for Faster-Whisper ASR processing."""

    model_name: str = "base"           # tiny | base | small | medium | large | large-v3
    language: str | None = None        # None = auto-detect
    device: str = "cpu"                # "cpu" | "cuda"
    compute_type: str = "int8"         # "float32", "float16", "int8_float16", "int8"
    task: str = "transcribe"           # "transcribe" | "translate"


# ──────────────────────────────────────────────────────────────────────────────
# FasterWhisperASR class
# ──────────────────────────────────────────────────────────────────────────────

class FasterWhisperASR:
    """
    Wrapper around Faster-Whisper that produces subtitle-like entries with
    millisecond timestamps and word-level segmentation.

    Output format will include segments with 'text' and 'words' keys:

        [
            {
                "index": 1,
                "start_ms": 0,
                "end_ms": 5000,
                "start_seconds": 0.0,
                "end_seconds": 5.0,
                "text": "Hello world",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5},
                    {"word": "world", "start": 0.6, "end": 1.0},
                ]
            },
            ...
        ]
    """

    def __init__(self, config: WhisperConfig | None = None) -> None:
        if not FASTER_WHISPER_AVAILABLE:
            raise RuntimeError(
                "❌ Faster-Whisper is not installed. Please install it:"
                "   pip install faster-whisper"
            )
        self.config = config or WhisperConfig()
        self._model: Any = None  # Lazily loaded

    # ── Model loading ──────────────────────────────────────────────────────

    def _load_model(self) -> Any:
        """Lazily load the Faster-Whisper model."""
        if self._model is not None:
            return self._model

        model_name = self.config.model_name
        device = self.config.device
        compute_type = self.config.compute_type

        print(f"   🎙️ Loading Faster-Whisper model: {model_name} "
              f"(device={device}, compute_type={compute_type})...")

        start_time = time.time()
        try:
            self._model = WhisperModel(model_name, device=device, compute_type=compute_type)
            print(f"   ✅ Faster-Whisper model loaded in {time.time() - start_time:.2f}s.")
            return self._model
        except Exception as e:
            raise RuntimeError(
                f"❌ Failed to load Faster-Whisper model '{model_name}': {e}\n"
                f"   Check model_name, device ('{device}'), and compute_type ('{compute_type}')."
            ) from e

    # ── Audio preparation ──────────────────────────────────────────────────

    @staticmethod
    def prepare_audio_for_transcription(video_path: str, work_dir: str) -> str:
        """
        Extract the audio track from a video file as a WAV.
        Faster-Whisper can typically handle various audio formats directly,
        but extracting to WAV can sometimes prevent issues and ensures consistency.

        Parameters
        ----------
        video_path : str
            Path to the source video file.
        work_dir : str
            Directory to store the extracted audio file. Will be created if
            it does not exist.

        Returns
        -------
        str
            Path to the extracted audio file (WAV).
        """
        os.makedirs(work_dir, exist_ok=True)
        audio_path = os.path.join(work_dir, "asr_audio.wav")

        # Skip extraction if already done (cache)
        if os.path.exists(audio_path):
            print(f"      ⏩ Audio already extracted: {audio_path}")
            return audio_path

        # Extract audio using the existing ffmpeg helper
        print(f"      🔊 Extracting audio from video...")
        extract_audio(video_path, audio_path)
        print(f"      ✅ Audio extracted: {audio_path}")
        return audio_path

    # ── Transcription ──────────────────────────────────────────────────────

    def transcribe(self, video_path: str, work_dir: str) -> list[dict]:
        """
        Transcribe the audio of a video file into subtitle-like entries.

        Parameters
        ----------
        video_path : str
            Path to the source video file.
        work_dir : str
            Directory to store intermediate audio files.

        Returns
        -------
        list[dict]
            List of subtitle entries with millisecond timestamps and word-level data.
        """
        duration = get_video_duration(video_path)
        if duration <= 0:
            raise RuntimeError(
                "❌ Cannot determine video duration; ASR aborted."
            )

        # Extract audio to WAV for consistent, reliable transcription
        audio_source_path = self.prepare_audio_for_transcription(video_path, work_dir)

        model = self._load_model()

        print("      ⏳ Decoding audio & extracting features (no output yet)...")

        # Use tqdm for progress bar, mirroring the reference implementation
        from tqdm import tqdm

        segments_generator, info = model.transcribe(
            audio_source_path,
            language=self.config.language,
            task=self.config.task,
            beam_size=5,
            word_timestamps=True,
            # Additional options from reference: compute_type handled in model load
        )

        total_dur = round(info.duration, 2)
        progress = tqdm(
            total=total_dur,
            unit="s",
            desc="      Transkripsi",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}]",
        )

        all_segments = []
        for segment in segments_generator:
            progress.update(min(segment.end, total_dur) - progress.n)
            all_segments.append(segment)

        progress.update(total_dur - progress.n) # snap ke 100% saat selesai
        progress.close()

        entries = self._segments_to_entries(all_segments)
        print(f"   ✅ ASR produced {len(entries)} transcript entries "
              f"(duration: {duration:.1f}s)")
        return entries

    # ── Output normalization ──────────────────────────────────────────────

    @staticmethod
    def _segments_to_entries(segments: list[Any]) -> list[dict]:
        """
        Normalize Faster-Whisper segments to subtitle-reader format,
        including word-level data.
        """
        entries: list[dict] = []
        for i, seg in enumerate(segments):
            text = seg.text.strip()
            if not text:
                continue

            start = float(seg.start)
            end = float(seg.end)

            # Faster-Whisper segments might have issues; ensure end > start
            if end <= start:
                end = start + 0.1

            words_data = []
            if seg.words:
                for w in seg.words:
                    if w.word.strip(): # Only add non-empty words
                        words_data.append({
                            "word": w.word.strip(),
                            "start": float(w.start),
                            "end": float(w.end),
                        })

            entries.append({
                "index": i + 1,
                "start_ms": int(start * 1000),
                "end_ms": int(end * 1000),
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "text": text,
                "words": words_data,
                "style": "Default",
            })
        return entries


# ──────────────────────────────────────────────────────────────────────────────
# Convenience entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def transcribe_video(
    video_path: str,
    project_dir: str,
    *,
    model_name: str = "base",
    language: str | None = None,
    device: str = "cpu",
    compute_type: str = "int8",
) -> dict | None:
    """
    High-level helper: transcribe a video and return a subtitle-reader-shaped
    dict ready to plug into the existing pipeline.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    project_dir : str
        Project output directory (used as work dir for audio + transcript).
    model_name : str
        Whisper model size.
    language : str | None
        Force a specific language, or None for auto-detection.
    device : str
        "cpu" or "cuda".
    compute_type : str
        Compute type for Faster-Whisper (e.g., "float16", "int8").

    Returns
    -------
    dict or None
        Same shape as ``dubbingstory.ingest.subtitle_reader.find_and_read_subtitles``,
        but with a "words" key in each entry, and "source": "asr".

            {
                "source_files": [...],   # empty list (generated)
                "primary_file": "...",   # path to saved transcript
                "entries": [...],
                "context_string": "...",
                "total_lines": N,
                "source": "asr",         # NEW field for source tracking
            }

        Returns None if ASR is unavailable or fails.
    """
    from dubbingstory.ingest.subtitle_reader import subtitles_to_context_string

    work_dir = os.path.join(project_dir, "asr")
    os.makedirs(work_dir, exist_ok=True)

    config = WhisperConfig(
        model_name=model_name,
        language=language,
        device=device,
        compute_type=compute_type,
    )

    try:
        asr = FasterWhisperASR(config)
        entries = asr.transcribe(video_path, work_dir)
    except Exception as e:
        print(f"   ⚠️ ASR failed: {e}")
        return None

    if not entries:
        print("   ⚠️ ASR produced no usable transcript.")
        return None

    context_string = subtitles_to_context_string(entries)

    # Persist the transcript for debugging / re-use
    transcript_path = os.path.join(work_dir, "asr_transcript.json")
    with open(transcript_path, "w", encoding="utf-8") as f:
        # Save full entries with word-level data
        json.dump(entries, f, ensure_ascii=False, indent=2)

    return {
        "source_files": [],
        "primary_file": transcript_path,
        "entries": entries,
        "context_string": context_string,
        "total_lines": len(entries),
        "source": "asr",
    }
