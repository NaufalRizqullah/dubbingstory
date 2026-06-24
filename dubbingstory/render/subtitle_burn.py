"""
dubbingstory.render.subtitle_burn — Burn subtitles into video

Optional subtitle burn-in using FFmpeg subtitles filter.
"""

import os
import subprocess


def burn_subtitles(
    video_path: str,
    subtitle_path: str,
    output_path: str,
    font_size: int = 24,
    font_color: str = "white",
    outline_color: str = "black",
    outline_width: int = 2,
    position: str = "bottom",
) -> str:
    """
    Burn subtitles into a video using FFmpeg.

    Parameters
    ----------
    video_path : str
        Input video path.
    subtitle_path : str
        Path to the .srt subtitle file.
    output_path : str
        Path for the output video.
    font_size : int
        Subtitle font size.
    font_color : str
        Subtitle text color.
    outline_color : str
        Outline/border color.
    outline_width : int
        Outline width.
    position : str
        "bottom" or "center".

    Returns
    -------
    str
        Path to the output video.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Escape path for FFmpeg subtitles filter (Windows paths need special handling)
    safe_sub_path = subtitle_path.replace("\\", "/").replace(":", "\\:")

    # Build subtitles filter
    margin_v = 30 if position == "bottom" else 0
    alignment = 2 if position == "bottom" else 5  # SSA alignment values

    style = (
        f"FontSize={font_size},"
        f"PrimaryColour=&H00FFFFFF,"  # White
        f"OutlineColour=&H00000000,"  # Black outline
        f"BorderStyle=1,"
        f"Outline={outline_width},"
        f"MarginV={margin_v},"
        f"Alignment={alignment}"
    )

    vf = f"subtitles='{safe_sub_path}':force_style='{style}'"

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "copy",
        output_path,
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace")[:500] if e.stderr else ""
        raise RuntimeError(f"FFmpeg subtitle burn failed: {stderr}") from e

    return output_path
