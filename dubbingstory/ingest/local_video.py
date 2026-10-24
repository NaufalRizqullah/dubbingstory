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
import math
import hashlib
import tempfile

from .sources import SourceIngestError


SUPPORTED_FORMATS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".m4v"}


def _number(value, default=0.0):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _ratio(value, default=0.0):
    try:
        numerator, denominator = str(value).replace(":", "/").split("/")
        return _number(numerator) / _number(denominator)
    except (ValueError, ZeroDivisionError):
        return default


def _stream_duration(stream: dict) -> float:
    duration = _number(stream.get("duration"))
    if duration > 0:
        return duration
    return max(0, _number(stream.get("duration_ts")) * _ratio(stream.get("time_base")))


def metadata_from_probe(probe: dict, filesize: int = 0) -> dict:
    """Parse FFprobe output without subprocesses (also useful to offline tests).

    ``duration`` retains its historical container meaning. ``video_duration``
    and ``video_start_time`` describe the video timeline; display dimensions
    include sample aspect ratio and right-angle phone rotation.
    """
    streams = probe.get("streams") or []
    video = next((item for item in streams if item.get("codec_type") == "video"
                  and not (item.get("disposition") or {}).get("attached_pic")), {})
    audio = next((item for item in streams if item.get("codec_type") == "audio"), {})
    container = probe.get("format") or {}
    container_duration = _number(container.get("duration"))
    video_duration = _stream_duration(video)
    duration_source = "stream" if video_duration > 0 else "container"
    if video_duration <= 0:
        video_duration = container_duration if video else 0
    rotation = _number((video.get("tags") or {}).get("rotate"))
    for side_data in video.get("side_data_list") or []:
        if "rotation" in side_data:
            rotation = _number(side_data["rotation"])
            break
    rotation %= 360
    width = int(_number(video.get("width")))
    height = int(_number(video.get("height")))
    sar_text = video.get("sample_aspect_ratio") or "1:1"
    sar = _ratio(sar_text, 1.0) or 1.0
    display_width, display_height = int(round(width * sar)), height
    if math.isclose(rotation % 180, 90, abs_tol=0.1):
        display_width, display_height = display_height, display_width
    nominal_fps = _ratio(video.get("r_frame_rate"))
    average_fps = _ratio(video.get("avg_frame_rate"))
    return {
        "duration": container_duration or video_duration,
        "container_duration": container_duration,
        "video_duration": video_duration, "video_duration_source": duration_source,
        "width": width, "height": height,
        "display_width": display_width, "display_height": display_height,
        "rotation": rotation, "sar": sar_text,
        "sample_aspect_ratio": sar_text,
        "display_aspect_ratio": video.get("display_aspect_ratio") or "",
        "fps": round(nominal_fps, 2), "avg_fps": average_fps,
        "r_frame_rate": video.get("r_frame_rate") or "0/1",
        "avg_frame_rate": video.get("avg_frame_rate") or "0/1",
        # Different rates are a warning, not definitive detection of VFR.
        "vfr_possible": bool(nominal_fps and average_fps and abs(nominal_fps - average_fps) > 0.01),
        "video_time_base": video.get("time_base") or "",
        "video_start_time": _number(video.get("start_time"), _number(container.get("start_time"))),
        "container_start_time": _number(container.get("start_time")),
        "video_duration_ts": video.get("duration_ts"),
        "video_frames": video.get("nb_frames"),
        "codec": video.get("codec_name") or "unknown",
        "video_stream_index": video.get("index"),
        "pixel_format": video.get("pix_fmt") or "",
        "color_transfer": video.get("color_transfer") or "",
        "color_primaries": video.get("color_primaries") or "",
        "has_audio": bool(audio), "audio_codec": audio.get("codec_name") or "",
        "audio_duration": _stream_duration(audio),
        "audio_start_time": _number(audio.get("start_time"), _number(container.get("start_time"))),
        "filesize_mb": round((_number(container.get("size")) or filesize) / (1024 * 1024), 2),
    }


def validate_video(path: str) -> bool:
    """Check if file exists and has a supported video format."""
    if not os.path.isfile(path):
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
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        print("⚠️ FFprobe could not read video metadata. Check the media and FFprobe installation.")
        return metadata_from_probe({}, os.path.getsize(path))

    return metadata_from_probe(probe, os.path.getsize(path))


def require_video_timeline(path: str, metadata: dict) -> dict:
    """Resolve an exact video-owned duration for strict native/animated output.

    Container duration may include a longer audio tail. If the video stream has
    no duration, inspect its packet presentation timestamps instead of using
    container duration or an average-frame-rate estimate. This does not decode
    or modify media. Unknown final packet timing is an actionable failure.
    """
    result = dict(metadata)
    if result.get("video_duration_source") == "container":
        index = result.get("video_stream_index")
        selector = str(index) if isinstance(index, int) else "v:0"
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", selector,
                 "-show_packets", "-show_entries", "packet=pts_time,duration_time",
                 "-of", "json", path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
                text=True, timeout=120,
            )
            packets = json.loads(probe.stdout).get("packets") or []
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            raise SourceIngestError("invalid_timeline", "Could not determine the video timeline. Use a source with valid stream timestamps; no container-duration guess was used.") from exc
        first_pts = None
        last_pts = None
        last_duration = 0.0
        for packet in packets:
            pts = _number(packet.get("pts_time"), float("nan"))
            if not math.isfinite(pts):
                raise SourceIngestError("invalid_timeline", "Video packets lack presentation timestamps; provide a source with a valid video timeline.")
            duration = _number(packet.get("duration_time"))
            first_pts = pts if first_pts is None else min(first_pts, pts)
            if last_pts is None or pts > last_pts:
                last_pts, last_duration = pts, duration
            elif pts == last_pts:
                last_duration = max(last_duration, duration)
        if first_pts is None or last_pts is None or last_duration <= 0:
            raise SourceIngestError("invalid_timeline", "The final video packet has no measured duration. Native output cannot safely infer the end from audio or average FPS.")
        result.update(video_duration=last_pts + last_duration - first_pts,
                      video_start_time=first_pts, video_duration_source="packets")
    duration = _number(result.get("video_duration") or result.get("duration"))
    if duration <= 0:
        raise SourceIngestError("invalid_timeline", "A measured positive video duration is required for native output.")
    return result


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

    def content_hash(filename: str) -> str:
        digest = hashlib.sha256()
        with open(filename, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    source_hash = content_hash(path)
    dest = os.path.join(project_dir, "source.mp4")
    meta_path = os.path.join(project_dir, "video_metadata.json")
    if not os.path.isfile(dest) and os.path.exists(meta_path):
        raise SourceIngestError("cache_invalid", "This project has source metadata but its video is missing. Re-ingest into a new project to avoid mixing stale artifacts.")
    if os.path.isfile(dest) and content_hash(dest) != source_hash:
        raise SourceIngestError("cache_conflict", "This project contains a different source video. Use another --project name to preserve its existing artifacts.")
    if os.path.isfile(dest) and os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as handle:
                existing = json.load(handle)
        except (OSError, ValueError):
            existing = {}
        if isinstance(existing, dict) and existing.get("sha256"):
            if existing["sha256"] != source_hash:
                raise SourceIngestError("cache_invalid", "The project source no longer matches its metadata. Re-ingest into a new project.")
            print("   📁 Reusing verified source video.")
            return dest

    # Get metadata
    metadata = get_video_metadata(path)
    if metadata.get("width", 0) <= 0 or metadata.get("video_duration", 0) <= 0:
        raise SourceIngestError("invalid_media", "Local media did not pass video validation. Check the video and FFprobe installation.")
    metadata.update({
        "schema_version": 2, "provider": "local", "source_id": source_hash,
        "source_identity": f"local:{source_hash}", "request_identity": f"local:{source_hash}",
        "canonical_url": None, "sha256": source_hash,
        "source_filename": os.path.basename(path),
    })
    print(f"   📋 Metadata:")
    print(f"      Duration : {metadata['duration']:.1f}s")
    print(f"      Resolution: {metadata['width']}x{metadata['height']}")
    print(f"      FPS      : {metadata['fps']}")
    print(f"      Codec    : {metadata['codec']}")
    print(f"      Audio    : {'Yes' if metadata['has_audio'] else 'No'}")
    print(f"      Size     : {metadata['filesize_mb']:.1f} MB")

    # Copy to project directory
    abs_path = os.path.abspath(path)
    abs_dest = os.path.abspath(dest)

    os.makedirs(project_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ingest-", dir=project_dir) as staging:
        staged_metadata = os.path.join(staging, "video_metadata.json")
        with open(staged_metadata, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        if abs_path != abs_dest and not os.path.exists(dest):
            print("   📁 Copying to project directory...")
            staged_source = os.path.join(staging, "source.mp4")
            shutil.copy2(path, staged_source)
            if content_hash(staged_source) != source_hash:
                raise SourceIngestError("invalid_media", "Local source changed during ingest. Retry with a stable input file.")
            if os.path.exists(dest):
                raise SourceIngestError("cache_conflict", "Another source appeared in this project during ingest. Use another project.")
            os.replace(staged_source, dest)
        else:
            print("   📁 File already in project directory.")
        os.replace(staged_metadata, meta_path)

    return dest
