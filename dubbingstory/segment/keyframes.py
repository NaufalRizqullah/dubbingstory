"""
dubbingstory.segment.keyframes — Keyframe extraction from scene videos

Extracts representative frames from each scene for visual analysis.
Critical for understanding video content — changes between keyframes
reveal the process/action happening in the scene.
"""

import os
import math
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


def extract_keyframes_from_source(
    video_path: str,
    scenes: list[dict],
    output_dir: str,
    strategy: str = "distributed",
    interval_seconds: float = 2.0,
    max_per_scene: int = 7,
    max_edge: int = 768,
    jpeg_quality: int = 88,
    shared_boundary_anchor: bool = True,
    adaptive: bool = True,
) -> dict[str, list[str]]:
    """Extract keyframes directly from the source video using timestamps."""
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"   🖼️ Extracting keyframes directly from source ({strategy})...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"      ⚠️ Could not open video: {video_path}")
        return {}
        
    all_keyframes = {}

    # V2.1: adjacent analysis windows from the SAME detected parent shot share
    # one exact timestamp at their internal boundary. Hard-cut neighbors never
    # share frames, avoiding evidence contamination across shots.
    start_anchors: dict[str, float] = {}
    end_anchors: dict[str, float] = {}
    if shared_boundary_anchor:
        ordered = sorted(scenes, key=lambda item: float(item.get("start_time", 0) or 0))
        for left, right in zip(ordered, ordered[1:]):
            left_parent = str(left.get("parent_scene_id", left.get("source_scene_id", "")) or "")
            right_parent = str(right.get("parent_scene_id", right.get("source_scene_id", "")) or "")
            boundary = float(left.get("end_time", 0) or 0)
            contiguous = abs(boundary - float(right.get("start_time", 0) or 0)) <= 0.05
            if left_parent and left_parent == right_parent and contiguous:
                end_anchors[str(left.get("scene_id"))] = boundary
                start_anchors[str(right.get("scene_id"))] = boundary

    for scene in scenes:
        sid = scene["scene_id"]
        start = scene["start_time"]
        end = scene["end_time"]
        
        # Avoid exact boundaries
        margin = min(0.25, max(0.0, (end - start) * 0.05))
        a = start + margin
        b = max(a, end - margin)
        
        duration = max(0.0, b - a)
        if adaptive:
            # Adaptive: ~1 frame per 2.5s, minimum 2, capped by max_per_scene
            count = max(2, min(int(max_per_scene), int(round(duration / 2.5))))
            if duration <= 2.0:
                count = 1
        else:
            count = max(1, int(max_per_scene))

        if strategy == "interval" and interval_seconds > 0:
            timestamps = np.arange(a, b + 1e-6, float(interval_seconds), dtype=float)
            if len(timestamps) == 0 or timestamps[-1] < b - 1e-3:
                timestamps = np.append(timestamps, b)
            if len(timestamps) > count:
                idx = np.linspace(0, len(timestamps) - 1, count).round().astype(int)
                timestamps = timestamps[idx]
        else:
            timestamps = np.linspace(a, b, count)

        if len(timestamps) >= 2 and shared_boundary_anchor:
            if sid in start_anchors:
                timestamps[0] = start_anchors[sid]
            if sid in end_anchors:
                timestamps[-1] = end_anchors[sid]

        paths = []
        for i, ts in enumerate(timestamps):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
                
            # Resize
            h, w = frame.shape[:2]
            scale = min(1.0, max_edge / max(h, w))
            if scale < 1.0:
                nw = max(32, int(round((w * scale) / 32) * 32))
                nh = max(32, int(round((h * scale) / 32) * 32))
                frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

            path = os.path.join(output_dir, sid, f"{sid}_kf{i:02d}.jpg")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
            paths.append(path)
            
        all_keyframes[sid] = paths
        print(f"      ✅ {sid}: {len(paths)} keyframes")
        
    cap.release()
    return all_keyframes


def extract_all_scenes(
    scenes: list[dict],
    scenes_dir: str,
    output_dir: str,
    strategy: str = "distributed",
    interval_seconds: float = 2.0,
    max_per_scene: int = 7,
    source_video: str = "",
    max_edge: int = 768,
    jpeg_quality: int = 88,
    shared_boundary_anchor: bool = True,
    adaptive: bool = True,
) -> dict[str, list[str]]:
    """
    Extract keyframes for all scenes.
    
    If source_video is provided, it extracts directly from it.

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
    if source_video and os.path.exists(source_video):
        return extract_keyframes_from_source(
            video_path=source_video,
            scenes=scenes,
            output_dir=output_dir,
            strategy=strategy,
            interval_seconds=interval_seconds,
            max_per_scene=max_per_scene,
            max_edge=max_edge,
            jpeg_quality=jpeg_quality,
            shared_boundary_anchor=shared_boundary_anchor,
            adaptive=adaptive,
        )

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


def plan_temporal_frames(scenes: list[dict], interval_seconds: float = 0.5,
                         max_frames: int = 0) -> list[dict]:
    """Plan dense samples on source time, without the legacy keyframe cap.

    Scene intervals are half-open, so a hard-cut boundary belongs to the next
    scene. Every scene gets its first sample and then interval-spaced samples;
    very short scenes therefore still have evidence. A resource limit rejects
    the plan instead of silently dropping requested samples.
    """
    interval = float(interval_seconds)
    if not math.isfinite(interval) or interval < 0.1 or interval > 10:
        raise ValueError("Temporal frame interval must be finite and between 0.1 and 10 seconds")
    if (isinstance(max_frames, bool) or not isinstance(max_frames, (int, float))
            or not math.isfinite(max_frames) or int(max_frames) != max_frames or max_frames < 0):
        raise ValueError("Temporal max_frames must be a nonnegative integer (0 means unlimited)")
    if not scenes:
        raise ValueError("Temporal analysis requires at least one scene")
    result = []
    seen = set()
    previous_end = 0.0
    for scene in scenes:
        sid = str(scene.get("scene_id", ""))
        if not sid or sid in seen:
            raise ValueError("Temporal analysis requires unique, nonempty scene IDs")
        seen.add(sid)
        start, end = float(scene["start_time"]), float(scene["end_time"])
        if not all(math.isfinite(t) for t in (start, end)) or start < 0 or end <= start:
            raise ValueError(f"Invalid source interval for {sid}")
        if start < previous_end - 0.001:
            raise ValueError("Temporal scenes must be chronological and nonoverlapping")
        previous_end = end
        count = max(1, int(math.ceil((end - start) / interval - 1e-9)))
        if max_frames and len(result) + count > max_frames:
            raise ValueError(
                f"Temporal sampling exceeds max_frames={max_frames}; increase the limit "
                "or interval_seconds. No samples were silently discarded."
            )
        for index in range(count):
            timestamp = start + index * interval
            result.append({
                "frame_id": f"frame_{len(result):06d}", "scene_id": sid,
                "requested_timestamp": round(timestamp, 6),
                "timestamp": round(timestamp, 6),
                "scene_start": start, "scene_end": end,
                "sample_start": max(start, timestamp - interval / 2),
                "sample_end": min(end, timestamp + interval / 2),
            })
    return result


def extract_temporal_frames_from_source(video_path: str, samples: list[dict],
                                        output_dir: str, max_edge: int = 768,
                                        jpeg_quality: int = 88) -> list[dict]:
    """Decode every planned frame and retain its measured source timestamp.

    Failed reads/writes and an unusable decoder clock are fatal. These samples
    drive spoken-action placement, so the legacy best-effort skipping policy is
    inappropriate here. The original video is never modified.
    """
    if not samples:
        raise ValueError("No temporal frames were planned")
    if int(max_edge) < 32 or not 1 <= int(jpeg_quality) <= 100:
        raise ValueError("Invalid temporal frame image settings")
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError("Cannot open source video for temporal sampling")
    records = []
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        tolerance = max(0.1, 2 / fps) if math.isfinite(fps) and fps > 0 else 0.1
        previous_timestamp = -1.0
        for sample in samples:
            requested = float(sample["requested_timestamp"])
            cap.set(cv2.CAP_PROP_POS_MSEC, requested * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f"Cannot decode temporal frame {sample['frame_id']} at {requested:.3f}s")
            timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            # Scene manifests round boundaries to milliseconds. A seek may
            # return the preceding frame (e.g. 0.966667 for a 0.967 boundary).
            # Read forward within a small bound instead of relabelling that
            # frame's clock or accepting evidence from the previous scene.
            for _ in range(2):
                if not math.isfinite(timestamp) or timestamp >= float(sample["scene_start"]) or timestamp < 0:
                    break
                if abs(timestamp - requested) > tolerance:
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise ValueError(f"Cannot decode a frame inside scene for {sample['frame_id']}")
                timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if (not math.isfinite(timestamp) or timestamp < 0
                    or abs(timestamp - requested) > tolerance
                    or timestamp < previous_timestamp - 0.001
                    or timestamp < float(sample["scene_start"]) - 1e-6
                    or timestamp >= float(sample["scene_end"])):
                raise ValueError(f"Unreliable decoder timestamp at {sample['frame_id']}; temporal analysis stopped")
            previous_timestamp = timestamp
            height, width = frame.shape[:2]
            scale = min(1.0, int(max_edge) / max(height, width))
            if scale < 1:
                frame = cv2.resize(frame, (max(32, round(width * scale / 32) * 32),
                                          max(32, round(height * scale / 32) * 32)),
                                   interpolation=cv2.INTER_AREA)
            path = os.path.join(output_dir, sample["frame_id"] + ".jpg")
            if not cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]):
                raise ValueError(f"Cannot save temporal frame {sample['frame_id']}")
            delta = timestamp - requested
            records.append({**sample, "timestamp": round(timestamp, 6), "path": path,
                            "sample_start": max(float(sample["scene_start"]), sample["sample_start"] + delta),
                            "sample_end": min(float(sample["scene_end"]), sample["sample_end"] + delta)})
    finally:
        cap.release()
    return records
