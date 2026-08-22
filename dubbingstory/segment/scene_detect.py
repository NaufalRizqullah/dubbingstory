"""
dubbingstory.segment.scene_detect — Scene detection using PySceneDetect

Detects scene boundaries in a video and splits it into individual
scene files using FFmpeg.
"""

import os

from scenedetect import detect, AdaptiveDetector, ContentDetector, ThresholdDetector
from scenedetect import split_video_ffmpeg


def _get_detector(detector_type: str, threshold: float):
    """Create a PySceneDetect detector instance."""
    if detector_type == "adaptive":
        return AdaptiveDetector(adaptive_threshold=threshold)
    elif detector_type == "content":
        return ContentDetector(threshold=threshold * 10)  # Content uses ~27 default
    elif detector_type == "threshold":
        return ThresholdDetector(threshold=threshold * 4)  # Threshold uses ~12 default
    else:
        print(f"⚠️ Unknown detector '{detector_type}', falling back to adaptive.")
        return AdaptiveDetector(adaptive_threshold=threshold)


def detect_scenes(
    video_path: str,
    detector: str = "adaptive",
    threshold: float = 3.0,
    min_duration: float = 2.0,
    max_duration: float | None = None,
    merge_short: bool = True,
) -> list[dict]:
    """
    Detect scene boundaries in a video.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    detector : str
        Detection method: "adaptive", "content", or "threshold".
    threshold : float
        Sensitivity threshold for scene detection.
    min_duration : float
        Minimum scene duration in seconds.
    merge_short : bool
        If True, merge scenes shorter than min_duration into neighbors.

    Returns
    -------
    list[dict]
        List of scene dicts:
        {
            "scene_id": "scene_001",
            "scene_index": 0,
            "start_time": 0.0,
            "end_time": 7.5,
            "duration": 7.5,
            "start_frame": 0,
            "end_frame": 225,
        }
    """
    print(f"   🔍 Detecting scenes (method: {detector}, threshold: {threshold})...")

    det = _get_detector(detector, threshold)
    scene_list = detect(video_path, det)

    print(f"   📊 Raw scenes detected: {len(scene_list)}")

    # Convert to our format
    scenes = []
    for i, (start, end) in enumerate(scene_list):
        start_sec = start.get_seconds()
        end_sec = end.get_seconds()
        duration = end_sec - start_sec

        scenes.append({
            "scene_id": f"scene_{i + 1:03d}",
            "scene_index": i,
            "start_time": round(start_sec, 3),
            "end_time": round(end_sec, 3),
            "duration": round(duration, 3),
            "start_frame": start.get_frames(),
            "end_frame": end.get_frames(),
        })

    # Merge short scenes first, then split abnormally long continuous shots into
    # bounded analysis/selectable windows. This preserves visual coverage for
    # long takes without sending a huge frame set into one vision request.
    if merge_short and min_duration > 0:
        scenes = _merge_short_scenes(scenes, min_duration)
        print(f"   📊 Scenes after merging (min {min_duration}s): {len(scenes)}")

    if max_duration and max_duration > 0:
        before = len(scenes)
        scenes = _split_long_scenes(scenes, float(max_duration))
        if len(scenes) != before:
            print(f"   📊 Scenes after long-shot windowing (max {max_duration:.1f}s): {len(scenes)}")

    return scenes


def _merge_short_scenes(scenes: list[dict], min_duration: float) -> list[dict]:
    """
    Merge scenes shorter than min_duration into their neighbors.

    Strategy: merge into the previous scene (extend its end_time).
    """
    if not scenes:
        return scenes

    merged = [scenes[0]]

    for scene in scenes[1:]:
        if scene["duration"] < min_duration:
            # Extend previous scene
            merged[-1]["end_time"] = scene["end_time"]
            merged[-1]["end_frame"] = scene["end_frame"]
            merged[-1]["duration"] = round(
                merged[-1]["end_time"] - merged[-1]["start_time"], 3
            )
        else:
            merged.append(scene)

    # Check if first scene is too short and merge forward
    if len(merged) > 1 and merged[0]["duration"] < min_duration:
        merged[1]["start_time"] = merged[0]["start_time"]
        merged[1]["start_frame"] = merged[0]["start_frame"]
        merged[1]["duration"] = round(
            merged[1]["end_time"] - merged[1]["start_time"], 3
        )
        merged = merged[1:]

    # Re-index
    for i, scene in enumerate(merged):
        scene["scene_id"] = f"scene_{i + 1:03d}"
        scene["scene_index"] = i

    return merged




def _split_long_scenes(scenes: list[dict], max_duration: float) -> list[dict]:
    """Split long detected shots into near-equal windows no longer than max_duration."""
    import math

    if not scenes or max_duration <= 0:
        return scenes

    result: list[dict] = []
    for source in scenes:
        duration = float(source.get("duration", 0.0) or 0.0)
        if duration <= max_duration:
            item = dict(source)
            item.setdefault("source_scene_id", source.get("scene_id"))
            result.append(item)
            continue

        parts = max(2, int(math.ceil(duration / max_duration)))
        start = float(source["start_time"])
        end = float(source["end_time"])
        start_frame = int(source.get("start_frame", 0) or 0)
        end_frame = int(source.get("end_frame", start_frame) or start_frame)
        frame_span = max(0, end_frame - start_frame)

        for part in range(parts):
            a_ratio = part / parts
            b_ratio = (part + 1) / parts
            a = start + duration * a_ratio
            b = end if part == parts - 1 else start + duration * b_ratio
            item = dict(source)
            item.update({
                "start_time": round(a, 3),
                "end_time": round(b, 3),
                "duration": round(b - a, 3),
                "start_frame": start_frame + int(round(frame_span * a_ratio)),
                "end_frame": end_frame if part == parts - 1 else start_frame + int(round(frame_span * b_ratio)),
                "source_scene_id": source.get("scene_id"),
                "source_scene_part": part + 1,
                "source_scene_parts": parts,
            })
            result.append(item)

    for i, scene in enumerate(result):
        scene["scene_id"] = f"scene_{i + 1:03d}"
        scene["scene_index"] = i
    return result


def split_video(
    video_path: str,
    scene_list: list[dict],
    output_dir: str,
) -> list[str]:
    """
    Split video into individual scene files using FFmpeg.

    Parameters
    ----------
    video_path : str
        Path to the source video.
    scene_list : list[dict]
        Scene list from detect_scenes().
    output_dir : str
        Directory to save scene files.

    Returns
    -------
    list[str]
        Paths to the split scene files.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"   ✂️ Splitting video into {len(scene_list)} scenes...")

    # Build scenedetect-compatible scene list for split_video_ffmpeg
    from scenedetect import FrameTimecode
    import subprocess

    scene_paths = []

    for scene in scene_list:
        scene_id = scene["scene_id"]
        output_path = os.path.join(output_dir, f"{scene_id}.mp4")

        start = scene["start_time"]
        duration = scene["duration"]

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", video_path,
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-avoid_negative_ts", "make_zero",
            output_path,
        ]

        try:
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            scene_paths.append(output_path)
            print(f"      ✅ {scene_id}: {start:.1f}s - {scene['end_time']:.1f}s ({duration:.1f}s)")
        except subprocess.CalledProcessError as e:
            print(f"      ❌ {scene_id}: FFmpeg error")
            stderr = e.stderr.decode("utf-8", errors="replace")[:200] if e.stderr else ""
            print(f"         {stderr}")

    return scene_paths


def detect_and_split(
    video_path: str,
    output_dir: str,
    detector: str = "adaptive",
    threshold: float = 3.0,
    min_duration: float = 2.0,
    max_duration: float | None = None,
    merge_short: bool = True,
) -> list[dict]:
    """
    Convenience function: detect scenes and split video in one call.

    Returns the scene list with added 'file_path' key.
    """
    scenes = detect_scenes(
        video_path,
        detector=detector,
        threshold=threshold,
        min_duration=min_duration,
        max_duration=max_duration,
        merge_short=merge_short,
    )

    scene_paths = split_video(video_path, scenes, output_dir)

    # Add file paths to scene dicts
    for scene, path in zip(scenes, scene_paths):
        scene["file_path"] = path

    return scenes
