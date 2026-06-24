"""
dubbingstory.segment.keyframes — Keyframe extraction from scene videos

Extracts representative frames from each scene for visual analysis.
Critical for understanding video content — changes between keyframes
reveal the process/action happening in the scene.
"""

import os
import cv2


def extract_keyframes(
    scene_path: str,
    output_dir: str,
    scene_id: str,
    strategy: str = "distributed",
    interval_seconds: float = 2.0,
    max_per_scene: int = 7,
) -> list[str]:
    """
    Extract keyframes from a single scene video.

    Parameters
    ----------
    scene_path : str
        Path to the scene video file.
    output_dir : str
        Directory to save keyframe images.
    scene_id : str
        Scene identifier (e.g., "scene_001").
    strategy : str
        "distributed" — extract at evenly spaced positions (start, 25%, 50%, 75%, end).
        "interval" — extract every N seconds.
    max_per_scene : int
        Maximum keyframes to extract per scene.

    Returns
    -------
    list[str]
        List of keyframe image paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(scene_path)
    if not cap.isOpened():
        print(f"      ⚠️ Could not open video: {scene_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / fps

    if total_frames <= 0:
        cap.release()
        return []

    # Determine frame positions to extract
    if strategy == "distributed":
        frame_positions = _get_distributed_positions(total_frames, max_per_scene)
    elif strategy == "interval":
        frame_positions = _get_interval_positions(
            total_frames, fps, interval_seconds, max_per_scene
        )
    else:
        frame_positions = _get_distributed_positions(total_frames, max_per_scene)

    # Extract frames
    keyframe_paths = []

    for idx, frame_pos in enumerate(frame_positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()

        if not ret:
            continue

        # Save as JPEG
        filename = f"{scene_id}_kf{idx:02d}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        keyframe_paths.append(filepath)

    cap.release()

    return keyframe_paths


def _get_distributed_positions(total_frames: int, count: int) -> list[int]:
    """
    Get evenly distributed frame positions.

    For count=5: start, 25%, 50%, 75%, end
    """
    if total_frames <= 1:
        return [0]

    count = min(count, total_frames)
    if count <= 1:
        return [0]

    positions = []
    for i in range(count):
        pos = int(i * (total_frames - 1) / (count - 1))
        positions.append(pos)

    return positions


def _get_interval_positions(
    total_frames: int,
    fps: float,
    interval_seconds: float,
    max_count: int,
) -> list[int]:
    """Get frame positions at fixed time intervals."""
    interval_frames = int(fps * interval_seconds)
    if interval_frames <= 0:
        interval_frames = 1

    positions = list(range(0, total_frames, interval_frames))

    # Always include the last frame
    if positions[-1] != total_frames - 1:
        positions.append(total_frames - 1)

    # Cap at max_count
    if len(positions) > max_count:
        # Evenly sample from the interval positions
        step = len(positions) / max_count
        sampled = [positions[int(i * step)] for i in range(max_count)]
        positions = sampled

    return positions


def extract_all_scenes(
    scenes: list[dict],
    scenes_dir: str,
    output_dir: str,
    strategy: str = "distributed",
    interval_seconds: float = 2.0,
    max_per_scene: int = 7,
) -> dict[str, list[str]]:
    """
    Extract keyframes for all scenes.

    Parameters
    ----------
    scenes : list[dict]
        Scene list from scene_detect.
    scenes_dir : str
        Directory containing scene video files.
    output_dir : str
        Directory to save all keyframes.
    strategy : str
        Keyframe extraction strategy.
    max_per_scene : int
        Maximum keyframes per scene.

    Returns
    -------
    dict[str, list[str]]
        Mapping of scene_id → list of keyframe paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"   🖼️ Extracting keyframes (strategy: {strategy})...")

    all_keyframes: dict[str, list[str]] = {}

    for scene in scenes:
        scene_id = scene["scene_id"]
        scene_path = scene.get("file_path")

        if not scene_path:
            scene_path = os.path.join(scenes_dir, f"{scene_id}.mp4")

        if not os.path.exists(scene_path):
            print(f"      ⚠️ {scene_id}: file not found, skip.")
            all_keyframes[scene_id] = []
            continue

        # Create per-scene keyframe directory
        scene_kf_dir = os.path.join(output_dir, scene_id)
        keyframes = extract_keyframes(
            scene_path=scene_path,
            output_dir=scene_kf_dir,
            scene_id=scene_id,
            strategy=strategy,
            interval_seconds=interval_seconds,
            max_per_scene=max_per_scene,
        )

        all_keyframes[scene_id] = keyframes
        print(f"      ✅ {scene_id}: {len(keyframes)} keyframes ({scene['duration']:.1f}s)")

    total = sum(len(kf) for kf in all_keyframes.values())
    print(f"   📊 Total keyframes: {total} across {len(scenes)} scenes")

    return all_keyframes
