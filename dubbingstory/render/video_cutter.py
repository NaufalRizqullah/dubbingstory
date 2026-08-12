"""
dubbingstory.render.video_cutter — Cut and concatenate video scenes for summary mode

Uses FFmpeg to extract selected scenes and concatenate them into
a single highlight/recap video.
"""

import os
import subprocess
import tempfile


def _extract_clip(
    source_video: str,
    start_time: float,
    duration: float,
    output_path: str,
) -> bool:
    """
    Extract a single clip from the source video using FFmpeg.

    Uses input seeking (-ss before -i) for speed, with re-encoding
    to ensure clean cuts at arbitrary timestamps.
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_time:.3f}",
        "-i", source_video,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        output_path,
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace")[:300] if e.stderr else ""
        print(f"      ❌ FFmpeg clip extraction failed: {stderr}")
        return False


def cut_and_concat(
    source_video: str,
    selected_scenes: list[dict],
    output_path: str,
) -> str:
    """
    Cut selected scenes from source video and concatenate them into
    a single summary video using FFmpeg concat demuxer (hard-cut).

    Parameters
    ----------
    source_video : str
        Path to the original full video.
    selected_scenes : list[dict]
        Selected scenes (must have start_time, end_time, duration keys).
        Expected to be sorted chronologically.
    output_path : str
        Path for the concatenated output video.

    Returns
    -------
    str
        Path to the output summary video.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if not selected_scenes:
        raise ValueError("No scenes provided for cutting.")

    print(f"\n   ✂️  Cutting {len(selected_scenes)} scenes from source video...")

    # Create temp directory for individual clips
    clips_dir = os.path.join(os.path.dirname(output_path), "_summary_clips")
    os.makedirs(clips_dir, exist_ok=True)

    clip_paths = []

    for i, scene in enumerate(selected_scenes):
        scene_id = scene.get("scene_id", f"clip_{i+1:03d}")
        clip_path = os.path.join(clips_dir, f"{scene_id}.mp4")

        start = scene["start_time"]
        duration = scene["duration"]

        print(f"      ✂️  {scene_id}: {start:.1f}s → {start + duration:.1f}s ({duration:.1f}s)")

        success = _extract_clip(source_video, start, duration, clip_path)
        if success and os.path.exists(clip_path):
            clip_paths.append(clip_path)
        else:
            print(f"      ⚠️  Skipping {scene_id} (extraction failed)")

    if not clip_paths:
        raise RuntimeError("All clip extractions failed. Cannot create summary.")

    # Concatenate clips using FFmpeg concat demuxer
    print(f"\n   🔗 Concatenating {len(clip_paths)} clips...")

    concat_list_path = os.path.join(clips_dir, "concat_list.txt")
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for clip_path in clip_paths:
            # FFmpeg concat demuxer requires escaped paths
            safe_path = clip_path.replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")

    concat_cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        output_path,
    ]

    try:
        subprocess.run(
            concat_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace")[:500] if e.stderr else ""
        raise RuntimeError(f"FFmpeg concat failed: {stderr}") from e

    # Cleanup temp clips
    for clip_path in clip_paths:
        try:
            os.remove(clip_path)
        except OSError:
            pass
    try:
        os.remove(concat_list_path)
        os.rmdir(clips_dir)
    except OSError:
        pass

    if not os.path.exists(output_path):
        raise RuntimeError(f"Concat output not found: {output_path}")

    # Get final duration
    duration_s = _get_video_duration(output_path)
    print(f"   ✅ Summary video created: {output_path} ({duration_s:.1f}s)")

    return output_path


def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]

    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        return float(result.stdout.decode().strip())
    except Exception:
        return 0.0
