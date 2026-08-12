"""
dubbingstory.story.scene_selector — Select most important scenes for summary mode

Scores and ranks scenes from storyboard.json, then selects the top scenes
that fit within a target duration to create a highlight recap.
"""

import json
import os


# Narrative roles ranked by importance for summary
ROLE_WEIGHTS = {
    "introduction": 0.9,
    "setup": 0.6,
    "process": 0.4,
    "climax": 1.0,
    "result": 0.9,
    "conclusion": 0.8,
}


def _score_scene(scene: dict, total_scenes: int) -> float:
    """
    Calculate a composite importance score for a single scene.

    Scoring formula:
        score = (narration_importance * 0.4) +
                (confidence * 0.2) +
                (narrative_role_weight * 0.2) +
                (duration_fitness * 0.1) +
                (position_bonus * 0.1)

    Parameters
    ----------
    scene : dict
        A scene entry from storyboard.json.
    total_scenes : int
        Total number of scenes in the video.

    Returns
    -------
    float
        Score between 0.0 and 1.0.
    """
    analysis = scene.get("analysis", {})
    narrative = scene.get("narrative", {})

    # 1. Narration importance (from temporal flow analysis)
    importance = narrative.get("importance", 0.5)

    # 2. Vision confidence
    confidence = analysis.get("confidence", 0.5)

    # 3. Narrative role weight
    role = narrative.get("role", "process")
    role_weight = ROLE_WEIGHTS.get(role, 0.4)

    # 4. Duration fitness — penalize very short (<3s) or very long (>30s) scenes
    duration = scene.get("duration", 5.0)
    if duration < 3.0:
        duration_fitness = 0.3
    elif duration > 30.0:
        duration_fitness = 0.5
    else:
        duration_fitness = 1.0

    # 5. Position bonus — first and last scenes get a bonus
    scene_index = scene.get("scene_index", 0) if "scene_index" in scene else 0
    # Try to extract index from scene_id
    if scene_index == 0 and scene.get("scene_id", "").startswith("scene_"):
        try:
            scene_index = int(scene["scene_id"].split("_")[1]) - 1
        except (IndexError, ValueError):
            pass

    position_bonus = 0.5
    if total_scenes > 0:
        if scene_index == 0:  # First scene
            position_bonus = 0.9
        elif scene_index == total_scenes - 1:  # Last scene
            position_bonus = 0.8
        elif scene_index == 1:  # Second scene (often setup)
            position_bonus = 0.6

    # Composite score
    score = (
        importance * 0.4
        + confidence * 0.2
        + role_weight * 0.2
        + duration_fitness * 0.1
        + position_bonus * 0.1
    )

    return round(min(max(score, 0.0), 1.0), 4)


def select_scenes(
    storyboard: dict,
    target_duration: int | None = None,
    max_scenes: int | None = None,
    min_score: float = 0.3,
) -> list[dict]:
    """
    Select the most important scenes for a video summary.

    Strategy:
    1. Score all scenes
    2. Filter by minimum score
    3. Sort by score (descending)
    4. Pick top scenes that fit within target_duration
    5. Re-sort by timestamp (chronological order)

    Parameters
    ----------
    storyboard : dict
        Full storyboard data from vision analysis.
    target_duration : int | None
        Target summary duration in seconds. If None, auto-calculate
        (~10-15% of original, clamped to 60-120s range).
    max_scenes : int | None
        Maximum number of scenes. If None, auto-determine.
    min_score : float
        Minimum score to consider a scene (0.0-1.0).

    Returns
    -------
    list[dict]
        Selected scenes with added 'summary_score' key, in chronological order.
    """
    scenes = storyboard.get("scenes", [])
    total_scenes = len(scenes)
    total_duration = storyboard.get("total_duration", 0)

    if not scenes:
        return []

    # Auto-calculate target duration if not specified
    if target_duration is None:
        # ~10-15% of original, clamped to 60-120s
        auto_target = total_duration * 0.12
        target_duration = int(max(60, min(auto_target, 120)))

    # Auto-calculate max scenes if not specified
    if max_scenes is None:
        # Roughly 1 scene per 8-12 seconds of target
        max_scenes = max(3, min(target_duration // 10, 15))

    print(f"   📊 Scene selection:")
    print(f"      Original: {total_scenes} scenes, {total_duration:.0f}s total")
    print(f"      Target:   ~{target_duration}s, max {max_scenes} scenes")

    # Score all scenes
    scored = []
    for scene in scenes:
        score = _score_scene(scene, total_scenes)
        entry = {**scene, "summary_score": score}
        scored.append(entry)

    # Filter by minimum score
    candidates = [s for s in scored if s["summary_score"] >= min_score]
    print(f"      Candidates (score >= {min_score}): {len(candidates)}/{total_scenes}")

    if not candidates:
        # Fallback: take top 5 scenes regardless of min_score
        candidates = sorted(scored, key=lambda s: s["summary_score"], reverse=True)[:5]
        print(f"      ⚠️ No scenes above threshold, using top {len(candidates)} fallback")

    # Sort by score descending
    candidates.sort(key=lambda s: s["summary_score"], reverse=True)

    # Greedy selection: pick top scenes until target_duration is reached
    selected = []
    cumulative_duration = 0.0

    for scene in candidates:
        if len(selected) >= max_scenes:
            break
        if cumulative_duration + scene["duration"] > target_duration * 1.2:
            # Allow 20% overshoot, but don't go too far
            continue
        selected.append(scene)
        cumulative_duration += scene["duration"]

    # Re-sort by start_time (chronological)
    selected.sort(key=lambda s: s.get("start_time", 0))

    summary_duration = sum(s["duration"] for s in selected)
    print(f"      Selected: {len(selected)} scenes, {summary_duration:.0f}s total")

    # Print selection details
    for s in selected:
        role = s.get("narrative", {}).get("role", "?")
        print(
            f"         {s['scene_id']}: "
            f"{s['start_time']:.1f}s-{s['end_time']:.1f}s "
            f"({s['duration']:.1f}s) "
            f"score={s['summary_score']:.2f} role={role}"
        )

    return selected


def save_summary_manifest(
    selected_scenes: list[dict],
    storyboard: dict,
    project_dir: str,
    target_duration: int | None = None,
) -> str:
    """
    Save the summary scene selection to a manifest file.

    Returns
    -------
    str
        Path to the saved manifest.
    """
    summary_duration = sum(s["duration"] for s in selected_scenes)

    manifest = {
        "mode": "summary",
        "original_scenes": storyboard.get("total_scenes", 0),
        "original_duration": storyboard.get("total_duration", 0),
        "selected_scenes": len(selected_scenes),
        "summary_duration": round(summary_duration, 2),
        "target_duration": target_duration,
        "video_title": storyboard.get("video_title", ""),
        "video_summary": storyboard.get("video_summary", ""),
        "narrative_arc": storyboard.get("narrative_arc", ""),
        "scenes": [
            {
                "scene_id": s["scene_id"],
                "start_time": s["start_time"],
                "end_time": s["end_time"],
                "duration": s["duration"],
                "summary_score": s["summary_score"],
                "narrative_role": s.get("narrative", {}).get("role", ""),
                "action": s.get("analysis", {}).get("action", ""),
                "likely_context": s.get("analysis", {}).get("likely_context", ""),
            }
            for s in selected_scenes
        ],
    }

    manifest_path = os.path.join(project_dir, "summary_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest_path
