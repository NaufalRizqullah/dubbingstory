"""
dubbingstory.utils.ffmpeg_helpers — FFmpeg utility functions

Adapted from opensource-clipping/clipping/studio/ffmpeg_utils.py
"""

import subprocess


def run_ffmpeg(cmd: list[str], label: str = "") -> None:
    """Run an FFmpeg command, raise on failure."""
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
            f"FFmpeg failed{' (' + label + ')' if label else ''}: {stderr}"
        ) from e


def format_seconds(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using FFprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        video_path,
    ]

    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True, text=True,
        )
        import json
        data = json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 0))
    except Exception:
        return 0.0


def extract_audio(video_path: str, output_path: str) -> str:
    """Extract audio from video as WAV."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        output_path,
    ]
    run_ffmpeg(cmd, "extract_audio")
    return output_path
