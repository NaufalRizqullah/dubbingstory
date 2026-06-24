"""
dubbingstory.render.audio_mix — Mix narration audio with original video

Supports multiple mixing strategies:
- duck: Lower original volume when narration plays
- replace: Complete replacement of original audio
- mute_original: Mute original, only narration
"""

import os
import subprocess


def mix_narration_with_original(
    original_video: str,
    narration_audio: str,
    output_path: str,
    strategy: str = "duck",
    original_volume: float = 0.15,
    narration_volume: float = 1.0,
) -> str:
    """
    Mix narration audio with original video audio.

    Parameters
    ----------
    original_video : str
        Path to the original video file.
    narration_audio : str
        Path to the narration audio file.
    output_path : str
        Path to save the mixed output.
    strategy : str
        Mixing strategy: "duck", "replace", or "mute_original".
    original_volume : float
        Volume level for original audio (0.0-1.0), used in "duck" mode.
    narration_volume : float
        Volume level for narration audio (0.0-1.0).

    Returns
    -------
    str
        Path to the mixed audio file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if strategy == "replace":
        return _replace_audio(original_video, narration_audio, output_path, narration_volume)
    elif strategy == "mute_original":
        return _mute_and_add(original_video, narration_audio, output_path, narration_volume)
    else:
        return _duck_audio(
            original_video, narration_audio, output_path,
            original_volume, narration_volume,
        )


def _duck_audio(
    video_path: str,
    narration_path: str,
    output_path: str,
    original_vol: float,
    narration_vol: float,
) -> str:
    """
    Mix by ducking original audio and overlaying narration.

    Uses FFmpeg amix filter to combine both audio tracks.
    """
    filter_complex = (
        f"[0:a]volume={original_vol}[orig];"
        f"[1:a]volume={narration_vol}[narr];"
        f"[orig][narr]amix=inputs=2:duration=longest:dropout_transition=3[out]"
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
    """Replace original audio entirely with narration."""
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
    """Mute original audio, add only narration."""
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

    _run_ffmpeg(cmd, "mute_and_add")
    return output_path


def _run_ffmpeg(cmd: list[str], label: str = "") -> None:
    """Run FFmpeg command with error handling."""
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace")[:500] if e.stderr else ""
        raise RuntimeError(
            f"FFmpeg failed ({label}): {stderr}"
        ) from e
