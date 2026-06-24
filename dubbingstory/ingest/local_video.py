"""
dubbingstory.ingest.local_video — Local video file validation & metadata

Handles:
  - File existence and format validation
  - Copy/symlink to project directory
  - Extract basic metadata (duration, resolution, codec)
"""

import os
import shutil
import subprocess
import json


SUPPORTED_FORMATS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".m4v"}


def validate_video(path: str) -> bool:
    """Check if file exists and has a supported video format."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported video format: {ext}\n"
            f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    return True


def get_video_metadata(path: str) -> dict:
    """
    Extract video metadata using FFprobe.

    Returns
    -------
    dict
        Keys: duration, width, height, fps, codec, has_audio, filesize_mb
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        probe = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"⚠️ FFprobe gagal: {e}")
        return {
            "duration": 0,
            "width": 0,
            "height": 0,
            "fps": 0,
            "codec": "unknown",
            "has_audio": False,
            "filesize_mb": os.path.getsize(path) / (1024 * 1024),
        }

    # Find video stream
    video_stream = None
    has_audio = False
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video" and not video_stream:
            video_stream = stream
        if stream.get("codec_type") == "audio":
            has_audio = True

    metadata = {
        "duration": float(probe.get("format", {}).get("duration", 0)),
        "width": int(video_stream.get("width", 0)) if video_stream else 0,
        "height": int(video_stream.get("height", 0)) if video_stream else 0,
        "fps": 0,
        "codec": video_stream.get("codec_name", "unknown") if video_stream else "unknown",
        "has_audio": has_audio,
        "filesize_mb": round(
            float(probe.get("format", {}).get("size", 0)) / (1024 * 1024), 2
        ),
    }

    # Parse FPS from r_frame_rate (e.g., "30/1" or "30000/1001")
    if video_stream:
        r_fps = video_stream.get("r_frame_rate", "0/1")
        try:
            num, den = r_fps.split("/")
            metadata["fps"] = round(int(num) / int(den), 2)
        except (ValueError, ZeroDivisionError):
            metadata["fps"] = 0

    return metadata


def validate_and_copy(path: str, project_dir: str) -> str:
    """
    Validate a local video and copy it to the project directory.

    Parameters
    ----------
    path : str
        Path to the source video file.
    project_dir : str
        Project output directory.

    Returns
    -------
    str
        Path to the video in the project directory.
    """
    validate_video(path)

    # Get metadata
    metadata = get_video_metadata(path)
    print(f"   📋 Metadata:")
    print(f"      Duration : {metadata['duration']:.1f}s")
    print(f"      Resolution: {metadata['width']}x{metadata['height']}")
    print(f"      FPS      : {metadata['fps']}")
    print(f"      Codec    : {metadata['codec']}")
    print(f"      Audio    : {'Yes' if metadata['has_audio'] else 'No'}")
    print(f"      Size     : {metadata['filesize_mb']:.1f} MB")

    # Copy to project directory
    dest = os.path.join(project_dir, "source.mp4")
    abs_path = os.path.abspath(path)
    abs_dest = os.path.abspath(dest)

    if abs_path != abs_dest:
        print(f"   📁 Copying to project directory...")
        shutil.copy2(path, dest)
    else:
        print(f"   📁 File already in project directory.")

    # Save metadata
    meta_path = os.path.join(project_dir, "video_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return dest
