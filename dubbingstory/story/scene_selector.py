"""Story-aware scene selection for summary mode.

The selector no longer asks only "which scenes are visually strong?".  It also
uses global story-plan metadata to preserve causal beats and bridge scenes, while
respecting the requested duration much more strictly.
"""

from __future__ import annotations

import json
import os
from typing import Any


ROLE_WEIGHTS = {
    "introduction": 0.85,
    "setup": 0.78,
    "problem": 0.95,
    "process": 0.58,
    "bridge": 0.62,
    "context": 0.45,
    "complication": 0.90,
    "turning_point": 0.98,
    "climax": 1.00,
    "result": 0.92,
    "resolution": 0.92,
    "conclusion": 0.88,
}


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _story_role_map(story_plan: dict | None) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for item in (story_plan or {}).get("scene_roles", []) or []:
        if isinstance(item, dict) and item.get("scene_id"):
            result[str(item["scene_id"])] = item
    return result


def _score_scene(scene: dict, total_scenes: int, story_meta: dict | None = None) -> float:
    analysis = scene.get("analysis", {}) or {}
    narrative = scene.get("narrative", {}) or {}
    story_meta = story_meta or {}

    temporal_importance = _clamp01(narrative.get("importance", 0.5), 0.5)
    confidence = _clamp01(analysis.get("confidence", 0.5), 0.5)
    role = str(story_meta.get("role", narrative.get("role", "process")) or "process")
    role_weight = ROLE_WEIGHTS.get(role, 0.5)
    story_importance = _clamp01(story_meta.get("story_importance", temporal_importance), temporal_importance)
    causal_importance = _clamp01(
        story_meta.get("causal_importance", narrative.get("causal_importance", 0.0)), 0.0
    )
    bridge_importance = _clamp01(
        story_meta.get("bridge_importance", narrative.get("bridge_importance", 0.0)), 0.0
    )

    duration = float(scene.get("duration", 5.0) or 5.0)
    if duration < 2.0:
        duration_fitness = 0.35
    elif duration > 25.0:
        duration_fitness = 0.55
    else:
        duration_fitness = 1.0

    scene_index = 0
    scene_id = str(scene.get("scene_id", ""))
    if scene_id.startswith("scene_"):
        try:
            scene_index = int(scene_id.split("_")[1]) - 1
        except (IndexError, ValueError):
            scene_index = 0
    position_bonus = 0.45
    if total_scenes:
        if scene_index == 0:
            position_bonus = 0.85
        elif scene_index == total_scenes - 1:
            position_bonus = 0.80

    # Story and causal importance dominate. Vision confidence prevents broken
    # analyses from winning, but does not determine narrative value by itself.
    score = (
        temporal_importance * 0.14
        + confidence * 0.08
        + role_weight * 0.14
        + story_importance * 0.27
        + causal_importance * 0.18
        + bridge_importance * 0.10
        + duration_fitness * 0.05
        + position_bonus * 0.04
    )
    if bool(story_meta.get("must_keep", False)):
        score = min(1.0, score + 0.12)
    return round(_clamp01(score), 4)


def _is_valid_scene(scene: dict) -> bool:
    analysis = scene.get("analysis", {}) or {}
    confidence = _clamp01(analysis.get("confidence", 1.0), 1.0)
    action_text = str(analysis.get("action", "") or "").lower()
    if confidence < 0.1:
        return False
    return not any(marker in action_text for marker in ("failed to analyze", "unable to analyze", "skipped in summary"))


def _fits(current: float, scene: dict, limit: float) -> bool:
    return current + float(scene.get("duration", 0) or 0) <= limit + 1e-6


def select_scenes(
    storyboard: dict,
    target_duration: int | None = None,
    max_scenes: int | None = None,
    min_score: float = 0.3,
    *,
    story_plan: dict | None = None,
    duration_tolerance: float = 1.05,
) -> list[dict]:
    """Select a causally coherent set of scenes under a duration budget."""
    scenes = storyboard.get("scenes", []) or []
    total_scenes = len(scenes)
    total_duration = float(storyboard.get("total_duration", 0) or 0)
    if not scenes:
        return []

    if target_duration is None:
        target_duration = int(max(60, min(total_duration * 0.12, 120)))
    if max_scenes is None:
        max_scenes = max(3, min(int(target_duration // 10), 18))

    duration_tolerance = max(1.0, min(float(duration_tolerance or 1.0), 1.15))
    hard_limit = float(target_duration) * duration_tolerance
    role_map = _story_role_map(story_plan)

    print("   📊 Story-aware scene selection:")
    print(f"      Original: {total_scenes} scenes, {total_duration:.0f}s")
    print(f"      Target:   {target_duration}s (hard limit {hard_limit:.0f}s), max {max_scenes} scenes")

    scored: list[dict] = []
    for scene in scenes:
        sid = str(scene.get("scene_id", ""))
        story_meta = role_map.get(sid, {})
        entry = {
            **scene,
            "summary_score": _score_scene(scene, total_scenes, story_meta),
            "story_role": story_meta.get("role", scene.get("narrative", {}).get("role", "process")),
            "story_importance": _clamp01(story_meta.get("story_importance", 0.5), 0.5),
            "causal_importance": _clamp01(story_meta.get("causal_importance", 0.0), 0.0),
            "bridge_importance": _clamp01(story_meta.get("bridge_importance", 0.0), 0.0),
            "must_keep": bool(story_meta.get("must_keep", False)),
        }
        scored.append(entry)

    valid = [s for s in scored if _is_valid_scene(s)]
    candidates = [s for s in valid if s["summary_score"] >= min_score]
    if not candidates:
        candidates = sorted(valid or scored, key=lambda s: s["summary_score"], reverse=True)[: max_scenes]
        print(f"      ⚠️ No candidates above threshold; fallback to top {len(candidates)}")
    print(f"      Candidates: {len(candidates)}/{total_scenes}")

    by_time = sorted(candidates, key=lambda s: float(s.get("start_time", 0) or 0))
    selected: list[dict] = []
    selected_ids: set[str] = set()
    current = 0.0

    def add(scene: dict) -> bool:
        nonlocal current
        sid = str(scene.get("scene_id", ""))
        if sid in selected_ids or len(selected) >= max_scenes:
            return False
        if not _fits(current, scene, hard_limit):
            return False
        selected.append(scene)
        selected_ids.add(sid)
        current += float(scene.get("duration", 0) or 0)
        return True

    # 1) Mandatory global-story anchors first, strongest first.  We still obey
    # the hard duration budget; "must_keep" is a priority, not a license to
    # overshoot by 20% like the old selector.
    mandatory = sorted(
        [s for s in by_time if s.get("must_keep")],
        key=lambda s: (s["summary_score"], s.get("story_importance", 0)),
        reverse=True,
    )
    for scene in mandatory:
        add(scene)

    # 2) Ensure broad story-arc coverage.  One representative per meaningful
    # role prevents a recap from collapsing into several adjacent machining
    # shots while losing the problem or payoff.
    role_order = [
        "setup", "problem", "process", "complication", "turning_point",
        "climax", "resolution",
    ]
    for role in role_order:
        options = [
            s for s in by_time
            if s["story_role"] == role and str(s.get("scene_id")) not in selected_ids
        ]
        if options:
            add(max(options, key=lambda s: (s["summary_score"], s.get("causal_importance", 0))))

    # 3) Timeline windows provide coverage even when story-role labels are weak.
    remaining_slots = max(0, max_scenes - len(selected))
    if remaining_slots and by_time:
        timeline_end = max(total_duration, max(float(s.get("end_time", 0) or 0) for s in by_time))
        window_count = min(max_scenes, len(by_time))
        window_size = timeline_end / max(1, window_count)
        for index in range(window_count):
            if len(selected) >= max_scenes:
                break
            start = index * window_size
            end = timeline_end if index == window_count - 1 else (index + 1) * window_size
            options = [
                s for s in by_time
                if str(s.get("scene_id")) not in selected_ids
                and start <= float(s.get("start_time", 0) or 0) < end
            ]
            if options:
                add(max(options, key=lambda s: s["summary_score"]))

    # 4) Fill with high-value scenes.
    for scene in sorted(by_time, key=lambda s: s["summary_score"], reverse=True):
        if len(selected) >= max_scenes:
            break
        add(scene)

    # 5) Bridge repair: when two selected beats are far apart, prefer inserting
    # a high bridge/causal scene between them if it fits.  This is deliberately
    # conservative so it cannot blow the duration budget.
    selected.sort(key=lambda s: float(s.get("start_time", 0) or 0))
    changed = True
    while changed and len(selected) < max_scenes:
        changed = False
        for left, right in zip(selected, selected[1:]):
            gap_candidates = [
                s for s in by_time
                if str(s.get("scene_id")) not in selected_ids
                and float(left.get("end_time", 0) or 0) <= float(s.get("start_time", 0) or 0)
                and float(s.get("end_time", 0) or 0) <= float(right.get("start_time", 0) or 0)
                and (s.get("bridge_importance", 0) >= 0.55 or s.get("causal_importance", 0) >= 0.70)
            ]
            if not gap_candidates:
                continue
            bridge = max(
                gap_candidates,
                key=lambda s: (s.get("bridge_importance", 0) + s.get("causal_importance", 0), s["summary_score"]),
            )
            if add(bridge):
                selected.sort(key=lambda s: float(s.get("start_time", 0) or 0))
                changed = True
                break

    selected.sort(key=lambda s: float(s.get("start_time", 0) or 0))
    summary_duration = sum(float(s.get("duration", 0) or 0) for s in selected)
    print(f"      Selected: {len(selected)} scenes, {summary_duration:.1f}s")
    for scene in selected:
        print(
            f"         {scene['scene_id']}: {scene.get('start_time', 0):.1f}s-"
            f"{scene.get('end_time', 0):.1f}s ({scene.get('duration', 0):.1f}s) "
            f"score={scene['summary_score']:.2f} role={scene.get('story_role')} "
            f"causal={scene.get('causal_importance', 0):.2f} bridge={scene.get('bridge_importance', 0):.2f}"
        )
    return selected


def save_summary_manifest(
    selected_scenes: list[dict],
    storyboard: dict,
    project_dir: str,
    target_duration: int | None = None,
) -> str:
    summary_duration = sum(float(s.get("duration", 0) or 0) for s in selected_scenes)
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
                "summary_score": s.get("summary_score", 0),
                "story_role": s.get("story_role", s.get("narrative", {}).get("role", "")),
                "story_importance": s.get("story_importance", 0),
                "causal_importance": s.get("causal_importance", 0),
                "bridge_importance": s.get("bridge_importance", 0),
                "action": s.get("analysis", {}).get("action", ""),
                "likely_context": s.get("analysis", {}).get("likely_context", ""),
            }
            for s in selected_scenes
        ],
    }
    manifest_path = os.path.join(project_dir, "summary_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest_path
