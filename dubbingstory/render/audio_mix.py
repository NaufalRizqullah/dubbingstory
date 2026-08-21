"""
dubbingstory.render.audio_mix — Mix narration with the video's original sound.

Strategies
----------
``dynamic_duck``
    V2 default.  Original ambience stays audible and is compressed only while
    narration is active.  This makes intentional narration gaps feel alive.
``duck``
    Legacy fixed-volume original track + narration.
``replace`` / ``mute_original``
    Narration only.
"""

from __future__ import annotations

import os
import subprocess


def mix_narration_with_original(
    original_video: str,
    narration_audio: str,
    output_path: str,
    strategy: str = "dynamic_duck",
    original_volume: float = 0.15,
    narration_volume: float = 1.0,
    dynamic_original_volume: float = 0.85,
    duck_threshold: float = 0.025,
    duck_ratio: float = 8.0,
    duck_attack_ms: float = 20.0,
    duck_release_ms: float = 350.0,
) -> str:
    """Mix narration audio with original video audio."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    strategy = (strategy or "dynamic_duck").strip().lower()

    if strategy == "replace":
        return _replace_audio(original_video, narration_audio, output_path, narration_volume)
    if strategy == "mute_original":
        return _mute_and_add(original_video, narration_audio, output_path, narration_volume)
    if strategy == "dynamic_duck":
        return _dynamic_duck_audio(
            original_video,
            narration_audio,
            output_path,
            original_base_vol=dynamic_original_volume,
            narration_vol=narration_volume,
            threshold=duck_threshold,
            ratio=duck_ratio,
            attack_ms=duck_attack_ms,
            release_ms=duck_release_ms,
        )
    return _duck_audio(
        original_video,
        narration_audio,
        output_path,
        original_volume,
        narration_volume,
    )


def _dynamic_duck_audio(
    video_path: str,
    narration_path: str,
    output_path: str,
    *,
    original_base_vol: float,
    narration_vol: float,
    threshold: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
) -> str:
    """Side-chain compress original sound only while narration is present.

    The narration signal is split: one copy controls the compressor and one is
    mixed back into the output.  During timeline gaps the side-chain is silent,
    so the original workshop/road/ambient sound naturally returns.
    """
    filter_complex = (
        f"[0:a]volume={max(0.0, original_base_vol):.3f}[orig];"
        f"[1:a]volume={max(0.0, narration_vol):.3f},asplit=2[narrmix][side];"
        f"[orig][side]sidechaincompress="
        f"threshold={max(0.00098, min(1.0, threshold)):.5f}:"
        f"ratio={max(1.0, ratio):.2f}:"
        f"attack={max(0.01, attack_ms):.2f}:"
        f"release={max(0.01, release_ms):.2f}[ducked];"
        "[ducked][narrmix]amix=inputs=2:duration=first:normalize=0[out]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", narration_path,
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[out]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path,
    ]
    _run_ffmpeg(cmd, "dynamic_duck")
    return output_path


def _duck_audio(
    video_path: str,
    narration_path: str,
    output_path: str,
    original_vol: float,
    narration_vol: float,
) -> str:
    """Legacy fixed-volume ducking."""
    filter_complex = (
        f"[0:a]volume={original_vol}[orig];"
        f"[1:a]volume={narration_vol}[narr];"
        "[orig][narr]amix=inputs=2:duration=first:dropout_transition=3:normalize=0[out]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", narration_path,
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[out]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path,
    ]
    _run_ffmpeg(cmd, "duck_mix")
    return output_path


def _replace_audio(
    video_path: str,
    narration_path: str,
    output_path: str,
    narration_vol: float,
) -> str:
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", narration_path,
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-af", f"volume={narration_vol}",
        "-shortest",
        output_path,
    ]
    _run_ffmpeg(cmd, "replace_audio")
    return output_path


def _mute_and_add(
    video_path: str,
    narration_path: str,
    output_path: str,
    narration_vol: float,
) -> str:
    return _replace_audio(video_path, narration_path, output_path, narration_vol)


def _run_ffmpeg(cmd: list[str], label: str = "") -> None:
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg is required for audio mixing") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace")[-1200:] if exc.stderr else ""
        raise RuntimeError(f"FFmpeg failed ({label}): {stderr}") from exc
