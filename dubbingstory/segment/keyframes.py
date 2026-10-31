"""
dubbingstory.segment.keyframes — Keyframe extraction from scene videos

Extracts representative frames from each scene for visual analysis.
Critical for understanding video content — changes between keyframes
reveal the process/action happening in the scene.
"""

import os
import math
import re
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


TEMPORAL_EXTRACTION_VERSION = 2


class TemporalFrameExtractionError(ValueError):
    """Source-time extraction failure with bounded, nonsecret diagnostic fields."""

    def __init__(self, code: str, message: str, *, details: dict | None = None):
        self.code = code
        self.details = {}
        for key, value in (details or {}).items():
            if key in {"frame_id", "scene_id"}:
                if isinstance(value, str) and re.fullmatch(r"(?:frame|scene)_\d+", value):
                    self.details[key] = value[:64]
            elif key in {"requested_timestamp", "decoder_timestamp", "scene_start", "scene_end", "tolerance_seconds"}:
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    self.details[key] = value
        super().__init__(message)


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


def _read_temporal_sample(cap, sample: dict, *, tolerance: float, fps: float,
                          previous_timestamp: float):
    """Recover seek rounding at scene/end boundaries using only decoded clocks.

    Seeking in milliseconds can land after the last frame or in the next scene.
    One bounded backward seek/forward scan may recover a nearby frame in this
    scene. The requested timestamp remains unchanged; no timestamp is invented.
    """
    requested = float(sample["requested_timestamp"])
    start, end = float(sample["scene_start"]), float(sample["scene_end"])
    details = {key: sample[key] for key in
               ("frame_id", "scene_id", "requested_timestamp", "scene_start", "scene_end")}
    details["tolerance_seconds"] = tolerance

    def failure(code, message, timestamp=None):
        return TemporalFrameExtractionError(code, message, details={
            **details, "decoder_timestamp": timestamp,
        })

    def check_clock(timestamp, prior=None):
        if (not math.isfinite(timestamp) or timestamp < 0
                or (prior is not None and timestamp < prior - 1e-6)):
            raise failure("decoder_clock_invalid", "Temporal decoder clock is invalid or moved backwards", timestamp)

    def acceptable(timestamp):
        return (start <= timestamp < end and abs(timestamp - requested) <= tolerance
                and timestamp >= previous_timestamp - 0.001)

    scan_limit = min(256, max(8, math.ceil(2 * tolerance * fps) + 4)) if math.isfinite(fps) and fps > 0 else 16
    cap.set(cv2.CAP_PROP_POS_MSEC, requested * 1000.0)
    ok, frame = cap.read()
    timestamp = None
    if ok and frame is not None:
        timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        check_clock(timestamp)
        if abs(timestamp - requested) > tolerance:
            raise failure("decoder_clock_drift", "Temporal decoder timestamp is too far from the requested sample", timestamp)
        if acceptable(timestamp):
            return frame, timestamp
        # Rounded scene starts may seek to the previous scene's last frame.
        for _ in range(scan_limit):
            if timestamp >= start:
                break
            ok, candidate = cap.read()
            if not ok or candidate is None:
                break
            current = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            check_clock(current, timestamp)
            if current <= timestamp:
                raise failure("decoder_clock_stalled", "Temporal decoder clock did not advance", current)
            timestamp, frame = current, candidate
            if acceptable(timestamp):
                return frame, timestamp
            if timestamp > requested + tolerance:
                break

    # Use an earlier *time*, not frame_count/fps: that arithmetic is not a
    # reliable source timestamp on VFR video. Never reuse an unrelated sample.
    seek_time = max(0.0, requested - tolerance)
    cap.set(cv2.CAP_PROP_POS_MSEC, seek_time * 1000.0)
    best, last = None, None
    for _ in range(scan_limit):
        ok, candidate = cap.read()
        if not ok or candidate is None:
            break
        current = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        check_clock(current, last)
        if last is not None and current <= last:
            raise failure("decoder_clock_stalled", "Temporal decoder clock did not advance", current)
        last = current
        timestamp = current
        if current >= end or current > requested + tolerance:
            break
        if acceptable(current) and (best is None or abs(current - requested) < abs(best[1] - requested)):
            best = (candidate, current)
        if best is not None and current >= requested:
            break
    if best is not None:
        return best
    raise failure("sample_unavailable", "Cannot decode a nearby frame inside the temporal sample's scene", timestamp)


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
        raise TemporalFrameExtractionError("source_open_failed", "Cannot open source video for temporal sampling")
    records = []
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        tolerance = max(0.1, 2 / fps) if math.isfinite(fps) and fps > 0 else 0.1
        previous_timestamp = -1.0
        for sample in samples:
            requested = float(sample["requested_timestamp"])
            frame, timestamp = _read_temporal_sample(
                cap, sample, tolerance=tolerance, fps=fps, previous_timestamp=previous_timestamp)
            previous_timestamp = timestamp
            height, width = frame.shape[:2]
            scale = min(1.0, int(max_edge) / max(height, width))
            if scale < 1:
                frame = cv2.resize(frame, (max(32, round(width * scale / 32) * 32),
                                          max(32, round(height * scale / 32) * 32)),
                                   interpolation=cv2.INTER_AREA)
            path = os.path.join(output_dir, sample["frame_id"] + ".jpg")
            if not cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]):
                raise TemporalFrameExtractionError("frame_write_failed", "Cannot save temporal frame", details={
                    **sample, "decoder_timestamp": timestamp, "tolerance_seconds": tolerance,
                })
            delta = timestamp - requested
            records.append({**sample, "timestamp": round(timestamp, 6), "path": path,
                            "sample_start": max(float(sample["scene_start"]), sample["sample_start"] + delta),
                            "sample_end": min(float(sample["scene_end"]), sample["sample_end"] + delta)})
    finally:
        cap.release()
    return records
