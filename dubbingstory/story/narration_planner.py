"""Continuity-aware narration planning.

The planner separates *what should be explained* from the final prose.  It also
computes explicit word budgets from the available video duration so the TTS
stage does not need to manufacture long silent pads per scene.
"""

from __future__ import annotations

import json
from typing import Any



PLAN_REFINEMENT_PROMPT = """You are planning a coherent video narration, not writing the narration itself.

GLOBAL STORY PLAN:
{story_plan_json}

STORY MEMORY / CONTINUITY FACTS:
{story_memory_json}

VIDEO MODE: {mode}

SCENES THAT WILL ACTUALLY APPEAR, in final chronological order:
{scenes_json}

For every scene_id, decide the narrative intent and what factual information
must be explained so that the entire narration feels like one continuous story.
Use cause/effect and process progression. Do not fill space with generic hype.

Return ONLY a JSON array with one item for every supplied scene_id:
[
  {{
    "scene_id": "scene_001",
    "intent": "what this beat should accomplish",
    "must_explain": ["specific supported fact"],
    "continuity_from_previous": "what this continues/resolves",
    "transition_strategy": "none|cause|contrast|progression|time|payoff|bridge",
    "avoid_repeating": ["information already covered earlier"]
  }}
]
"""


def _scene_role_map(story_plan: dict | None) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for item in (story_plan or {}).get("scene_roles", []) or []:
        if isinstance(item, dict) and item.get("scene_id"):
            result[str(item["scene_id"])] = item
    return result


def _coverage_for_role(role: str, importance: float, base: float) -> float:
    role = (role or "process").lower()
    modifier = {
        "setup": 0.95,
        "problem": 0.95,
        "turning_point": 0.95,
        "climax": 0.95,
        "resolution": 0.90,
        "complication": 0.90,
        "process": 0.80,
        "bridge": 0.62,
        "context": 0.60,
    }.get(role, 0.78)
    importance_factor = 0.78 + 0.22 * max(0.0, min(1.0, importance))
    return max(0.50, min(0.96, base * modifier / 0.80 * importance_factor))



ROLE_PRIORITY = {
    "climax": 9,
    "turning_point": 8,
    "problem": 7,
    "complication": 6,
    "resolution": 5,
    "setup": 4,
    "process": 3,
    "bridge": 2,
    "context": 1,
}


def _role_for_scene(scene: dict, roles: dict[str, dict]) -> tuple[str, float]:
    scene_id = str(scene.get("scene_id", ""))
    meta = roles.get(scene_id, {}) or {}
    narrative = scene.get("narrative", {}) or {}
    role = str(meta.get("role", narrative.get("role", "process")) or "process")
    importance = float(meta.get("story_importance", narrative.get("importance", 0.5)) or 0.5)
    return role, max(0.0, min(1.0, importance))


def _group_scenes_into_beats(
    scenes: list[dict],
    roles: dict[str, dict],
    cfg,
) -> list[list[dict]]:
    """Group adjacent shots into narration beats.

    Editors cut visuals much more frequently than a human narrator starts a new
    thought.  Grouping prevents the voice from sounding like it resets at every
    shot while still keeping strong plot anchors distinct.
    """
    if not scenes:
        return []
    enabled = bool(getattr(cfg, "narration_group_scenes", True))
    if not enabled:
        return [[scene] for scene in scenes]

    target = max(4.0, float(getattr(cfg, "narration_target_beat_duration", 11.0) or 11.0))
    maximum = max(target, float(getattr(cfg, "narration_max_beat_duration", 18.0) or 18.0))
    anchors = {"problem", "complication", "turning_point", "climax", "resolution"}

    groups: list[list[dict]] = []
    current: list[dict] = []
    current_duration = 0.0

    def flush() -> None:
        nonlocal current, current_duration
        if current:
            groups.append(current)
        current = []
        current_duration = 0.0

    for scene in scenes:
        duration = max(0.1, float(scene.get("duration", 0) or 0.1))
        role, importance = _role_for_scene(scene, roles)

        # A strong story anchor starts a fresh thought once the current beat has
        # enough substance. Very short setup/process shots can still lead into it.
        if current and role in anchors and current_duration >= max(4.0, target * 0.55):
            flush()

        if current and current_duration + duration > maximum:
            flush()

        current.append(scene)
        current_duration += duration

        # End a complete thought around the target duration. Keep tiny bridge /
        # process shots together; isolate very important anchors after they land.
        if current_duration >= target:
            if role in anchors or importance >= 0.9 or current_duration >= maximum * 0.82:
                flush()

    flush()
    return groups


def _dominant_role(group: list[dict], roles: dict[str, dict]) -> tuple[str, float]:
    ranked = []
    for scene in group:
        role, importance = _role_for_scene(scene, roles)
        ranked.append((ROLE_PRIORITY.get(role, 0), importance, role))
    if not ranked:
        return "process", 0.5
    _, importance, role = max(ranked)
    return role, max(_role_for_scene(scene, roles)[1] for scene in group)

def build_narration_plan(
    storyboard: dict,
    scenes: list[dict],
    *,
    story_plan: dict | None,
    story_memory: dict | None,
    cfg,
    mode: str,
) -> dict:
    """Return narration beats with timeline, continuity and explicit word budgets."""
    target_wpm = float(getattr(cfg, "narration_target_wpm", 160) or 160)
    base_coverage = float(getattr(cfg, "narration_speech_coverage", 0.84) or 0.84)
    min_wpm = float(getattr(cfg, "narration_min_wpm", 135) or 135)
    max_wpm = float(getattr(cfg, "narration_max_wpm", 175) or 175)

    roles = _scene_role_map(story_plan)
    groups = _group_scenes_into_beats(scenes, roles, cfg)
    entries: list[dict[str, Any]] = []
    cumulative = 0.0

    for index, group in enumerate(groups):
        first = group[0]
        last = group[-1]
        covered_ids = [str(scene.get("scene_id", "")) for scene in group]
        beat_id = covered_ids[0] or f"beat_{index + 1:03d}"
        role, importance = _dominant_role(group, roles)

        if mode == "summary":
            duration = sum(max(0.1, float(scene.get("duration", 0) or 0.1)) for scene in group)
            timeline_start = cumulative
            timeline_end = timeline_start + duration
            cumulative = timeline_end
        else:
            timeline_start = float(first.get("start_time", cumulative) or cumulative)
            source_end = float(last.get("end_time", timeline_start) or timeline_start)
            duration = max(
                0.1,
                source_end - timeline_start,
                sum(max(0.1, float(scene.get("duration", 0) or 0.1)) for scene in group),
            )
            timeline_end = timeline_start + duration
            cumulative = timeline_end

        coverage = _coverage_for_role(role, importance, base_coverage)
        target_words = max(5, round(duration * target_wpm / 60.0 * coverage))
        min_words = max(4, round(duration * min_wpm / 60.0 * max(0.48, coverage - 0.12)))
        max_words = max(target_words + 3, round(duration * max_wpm / 60.0 * min(1.0, coverage + 0.10)))

        reasons: list[str] = []
        evidence_fallback: list[str] = []
        for scene in group:
            sid = str(scene.get("scene_id", ""))
            meta = roles.get(sid, {}) or {}
            narrative = scene.get("narrative", {}) or {}
            analysis = scene.get("analysis", {}) or {}
            for value in (meta.get("reason"), narrative.get("cue")):
                value = str(value or "").strip()
                if value and value not in reasons:
                    reasons.append(value)
            for value in (analysis.get("action"), analysis.get("likely_context")):
                value = str(value or "").strip()
                if value and value not in evidence_fallback:
                    evidence_fallback.append(value)

        must_explain = (reasons or evidence_fallback)[:4]
        entries.append(
            {
                "scene_id": beat_id,
                "covers_scene_ids": covered_ids,
                "scene_order": index,
                "story_role": role,
                "story_importance": round(max(0.0, min(1.0, importance)), 3),
                "timeline_start": round(timeline_start, 3),
                "timeline_end": round(timeline_end, 3),
                "available_duration": round(duration, 3),
                "speech_coverage_target": round(coverage, 3),
                "target_words": target_words,
                "min_words": min_words,
                "max_words": max_words,
                "intent": reasons[0] if reasons else f"Explain how this beat advances the story",
                "must_explain": must_explain,
                "continuity_from_previous": "",
                "transition_strategy": "progression" if index else "none",
                "avoid_repeating": [],
            }
        )

    _refine_plan_with_gemini(
        entries,
        scenes=scenes,
        story_plan=story_plan or {},
        story_memory=story_memory or {},
        cfg=cfg,
        mode=mode,
    )

    if mode == "summary":
        total_duration = sum(float(e["available_duration"]) for e in entries)
    else:
        total_duration = max((float(e.get("timeline_end", 0) or 0) for e in entries), default=0.0)
    total_target_words = sum(int(e["target_words"]) for e in entries)
    return {
        "mode": mode,
        "target_wpm": target_wpm,
        "speech_coverage_target": base_coverage,
        "total_duration": round(total_duration, 3),
        "total_target_words": total_target_words,
        "source_scene_count": len(scenes),
        "narration_beat_count": len(entries),
        "segments": entries,
    }


def _refine_plan_with_gemini(
    entries: list[dict],
    *,
    scenes: list[dict],
    story_plan: dict,
    story_memory: dict,
    cfg,
    mode: str,
) -> None:
    if not bool(getattr(cfg, "narration_plan_llm_refine", True)):
        return
    api_key = getattr(cfg, "api_key_gemini", "")
    if not api_key or not entries:
        return

    scene_map = {str(s.get("scene_id")): s for s in scenes}
    compact_scenes = []
    for entry in entries:
        covered = [scene_map.get(sid, {}) for sid in entry.get("covers_scene_ids", [entry["scene_id"]])]
        analyses = [(scene.get("analysis", {}) or {}) for scene in covered]
        compact_scenes.append(
            {
                "scene_id": entry["scene_id"],
                "covers_scene_ids": entry.get("covers_scene_ids", [entry["scene_id"]]),
                "story_role": entry["story_role"],
                "target_words": entry["target_words"],
                "actions": [a.get("action", "") for a in analyses if a.get("action")],
                "changes": [a.get("changes", "") for a in analyses if a.get("changes")],
                "likely_context": [a.get("likely_context", "") for a in analyses if a.get("likely_context")],
                "transcript": " ".join(str(a.get("transcript_text", "") or "") for a in analyses).strip(),
            }
        )

    prompt = PLAN_REFINEMENT_PROMPT.format(
        story_plan_json=json.dumps(story_plan, ensure_ascii=False, indent=2)[:24000],
        story_memory_json=json.dumps(story_memory, ensure_ascii=False, indent=2)[:12000],
        mode=mode,
        scenes_json=json.dumps(compact_scenes, ensure_ascii=False, indent=2),
    )

    from dubbingstory.vision.gemini_analyzer import GeminiVideoAnalyzer

    model = getattr(cfg, "story_model", None) or getattr(cfg, "vision_gemini_model", "gemini-2.5-flash")
    fallback_model = getattr(cfg, "vision_gemini_fallback_model", "gemini-2.0-flash")
    analyzer = GeminiVideoAnalyzer(api_key=api_key, model=model, fallback_model=fallback_model)

    try:
        from google.genai import types

        response = analyzer.client.models.generate_content(
            model=model,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        raw = json.loads(getattr(response, "text", "") or "[]")
        if not isinstance(raw, list):
            return
        by_id = {
            str(item.get("scene_id")): item
            for item in raw
            if isinstance(item, dict) and item.get("scene_id")
        }
        for entry in entries:
            extra = by_id.get(entry["scene_id"])
            if not extra:
                continue
            if extra.get("intent"):
                entry["intent"] = str(extra["intent"])
            for key in ("continuity_from_previous", "transition_strategy"):
                if extra.get(key):
                    entry[key] = str(extra[key])
            for key in ("must_explain", "avoid_repeating"):
                if isinstance(extra.get(key), list):
                    entry[key] = [str(v) for v in extra[key] if str(v).strip()][:6]
        print("   🧭 Narration planner: continuity intents refined")
    except Exception as exc:
        print(f"   ⚠️ Narration planner refinement skipped: {exc}")
