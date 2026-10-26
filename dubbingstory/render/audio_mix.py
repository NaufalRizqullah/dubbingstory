"""Audio mixing on a finite video-owned timeline, shared by all render paths."""

from __future__ import annotations

import math
import os
import subprocess


def build_audio_filter(
    *, duration: float, has_original_audio: bool, strategy: str = "dynamic_duck",
    original_volume: float = 0.15, narration_volume: float = 1.0,
    duck_threshold: float = 0.025, duck_ratio: float = 8.0,
    duck_attack_ms: float = 20.0, duck_release_ms: float = 350.0,
    original_start_offset: float = 0.0,
) -> str:
    """Build [out] audio spanning the video, even with missing/short source audio."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("A positive measured video duration is required for mixing")
    if any(not math.isfinite(value) for value in (original_volume, narration_volume, original_start_offset,
                                                  duck_threshold, duck_ratio, duck_attack_ms, duck_release_ms)):
        raise ValueError("Audio levels, offsets and ducking controls must be finite numbers")
    strategy = (strategy or "dynamic_duck").lower()
    if strategy not in {"replace", "mute_original", "duck", "dynamic_duck"}:
        raise ValueError(f"Unsupported audio strategy: {strategy}")
    timeline = f"apad,atrim=duration={duration:.6f}"
    mix_original = has_original_audio and original_volume > 0 and strategy not in {"replace", "mute_original"}
    # FFmpeg 4.x cannot always resolve channel layouts through the split ->
    # sidechaincompress -> amix branches (stereo AAC + mono TTS). Constrain both
    # mixed inputs explicitly; narration-only output keeps its existing layout.
    mix_format = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
    narration_format = f",{mix_format}" if mix_original else ""
    filters = [f"[1:a:0]asetpts=PTS-STARTPTS,aresample=48000,volume={max(0.0, narration_volume):.6f}{narration_format},{timeline}[narr]"]
    if not mix_original:
        filters.append("[narr]anull[out]")
        return ";".join(filters)
    # Preserve original audio's offset from the first video frame. Resetting
    # both streams independently would move sound effects on offset sources.
    offset_filter = ""
    if original_start_offset < 0:
        offset_filter = f"atrim=start={-original_start_offset:.6f},asetpts=PTS-STARTPTS,"
    elif original_start_offset > 0:
        offset_filter = f"adelay={original_start_offset * 1000:.3f}:all=1,"
    filters.append(f"[0:a:0]asetpts=PTS-STARTPTS,aresample=48000,{offset_filter}volume={max(0.0, original_volume):.6f},{mix_format},{timeline}[orig]")
    if strategy == "dynamic_duck":
        # The compressor uses packed double samples. Pin its main AND detector
        # layouts/rates before negotiation, then convert back for the mix branch.
        compressor_format = "aformat=sample_fmts=dbl:sample_rates=48000:channel_layouts=stereo"
        filters.extend([
            "[narr]asplit=2[narrmix][side_in]",
            f"[orig]{compressor_format}[compress_main]",
            f"[side_in]{compressor_format}[compress_side]",
            "[compress_main][compress_side]sidechaincompress="
            f"threshold={max(0.00098, min(1.0, duck_threshold)):.5f}:"
            f"ratio={max(1.0, min(20.0, duck_ratio)):.2f}:"
            f"attack={max(0.01, min(2000.0, duck_attack_ms)):.2f}:"
            f"release={max(0.01, min(9000.0, duck_release_ms)):.2f}[ducked]",
            f"[ducked]{mix_format}[duckmix]",
            "[duckmix][narrmix]amix=inputs=2:duration=first:normalize=0[out]",
        ])
    else:
        filters.append("[orig][narr]amix=inputs=2:duration=first:normalize=0[out]")
    return ";".join(filters)


def mix_narration_with_original(
    original_video: str, narration_audio: str, output_path: str,
    strategy: str = "dynamic_duck", original_volume: float = 0.15,
    narration_volume: float = 1.0, dynamic_original_volume: float | None = None,
    duck_threshold: float = 0.025, duck_ratio: float = 8.0,
    duck_attack_ms: float = 20.0, duck_release_ms: float = 350.0,
) -> str:
    """Mix without shortening video; retain an explicit legacy volume override."""
    from dubbingstory.ingest.local_video import get_video_metadata

    metadata = get_video_metadata(original_video)
    duration = float(metadata.get("video_duration") or metadata.get("duration") or 0)
    graph = build_audio_filter(
        duration=duration, has_original_audio=metadata.get("has_audio", False), strategy=strategy,
        original_volume=original_volume if dynamic_original_volume is None else dynamic_original_volume,
        narration_volume=narration_volume, duck_threshold=duck_threshold, duck_ratio=duck_ratio,
        duck_attack_ms=duck_attack_ms, duck_release_ms=duck_release_ms,
        original_start_offset=float(metadata.get("audio_start_time", 0)) - float(metadata.get("video_start_time", 0)),
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cmd = ["ffmpeg", "-y", "-copyts", "-itsoffset", f"{-float(metadata.get('video_start_time', 0)):.9f}",
           "-i", original_video, "-i", narration_audio,
           "-filter_complex", graph, "-map", "0:v:0", "-map", "[out]",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-avoid_negative_ts", "disabled", "-movflags", "+faststart", output_path]
    _run_ffmpeg(cmd, strategy)
    return output_path


def _run_ffmpeg(cmd: list[str], label: str = "") -> None:
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg is required for rendering") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace")[-1200:] if exc.stderr else ""
        raise RuntimeError(f"FFmpeg failed ({label}): {stderr}") from exc
