"""Prompt builders for continuity-aware narration."""

from __future__ import annotations

import json


V2_NARRATION_PROMPT = """You are a professional video storyteller.

The viewer must understand the STORY, not receive a mechanical description of
frames. Explain what is happening, why it matters, what changes, and how the
current beat connects to what came before. Use only supplied evidence.

MODE: {mode}
LANGUAGE: {language_label}
STYLE NAME: {style_name}
STYLE TONE: {tone}

GLOBAL STORY PLAN:
{story_plan_json}

STORY MEMORY / FACTS THAT MUST STAY CONSISTENT:
{story_memory_json}

STYLE RULES (secondary to factuality and continuity):
{rules}

SCENES + NARRATION PLAN, in final playback order:
{scenes_json}

HARD RULES:
0. Every `text` value MUST be written entirely in {language_label}. Do not
    answer in English when the requested language is Bahasa Indonesia. Translate
    supplied English evidence into the requested language; never copy its wording.
1. Return one narration item for every supplied NARRATION BEAT scene_id, in the same order.
   A beat may cover multiple visual scene IDs; narrate them as one thought.
2. Tell one continuous story. A later segment must sound like a continuation,
   not a fresh reaction to a new image.
3. Prioritize: PROCESS/CHANGE -> REASON -> CONSEQUENCE -> NEXT STEP.
4. Transcript/dialogue is stronger evidence for meaning and motivation than a
   visual guess. Visual facts are stronger evidence for visible actions/objects.
5. Never invent component names, character identities, relationships, causes,
   or outcomes. If unsupported, describe only what is known.
6. Avoid generic filler and repeated hype such as "Lihat ini!", "Luar biasa!",
   "Amazing!", "Gila!", "Kalian pasti gak nyangka", "Wow", or equivalents.
7. Exclamation marks should be rare (normally no more than 1 in 5 sentences).
8. Do not say "in this scene", "pada scene ini", "terlihat pada frame", or
   other production-language unless uncertainty genuinely requires hedging.
9. Natural transitions are welcome only when they add meaning: cause, contrast,
   progression, payoff, time, or bridge. Do not force a transition phrase into
   every segment.
10. Respect each segment's target_words and NEVER exceed max_words. Also avoid
    under-writing below min_words unless the evidence is genuinely insufficient.
11. Opening: establish the actual goal/problem, not a generic clickbait promise.
12. Closing: resolve/pay off what the narration has been tracking.
13. Keep Indonesian natural and conversational when language is Indonesian;
    avoid translating English storytelling idioms literally.
14. Do not repeat a fact listed in avoid_repeating unless needed for clarity.
15. If confidence is below {hedging_threshold}, hedge only the uncertain fact.

Return ONLY a valid JSON array:
[
  {{
    "scene_id": "scene_001",
    "text": "natural narration",
    "importance": 0.8
  }}
]
"""


REWRITE_PROMPT = """You are revising an existing video narration after QA.

LANGUAGE: {language_label}
QA ISSUES:
{issues_json}

GLOBAL STORY PLAN:
{story_plan_json}

NARRATION PLAN:
{narration_plan_json}

CURRENT NARRATION:
{current_json}

Rewrite the narration to fix the QA issues while preserving factual accuracy,
scene order, scene_id values, and the per-segment min/target/max word budgets.
Reduce generic reactions, repeated sentence openings, and disconnected scene-by-
scene commentary. Make cause/effect and process continuity explicit when the
provided evidence supports it. Do not invent missing facts.

Return ONLY the corrected JSON array with scene_id, text, importance.
"""


# Compatibility aliases retained for callers outside script_writer.
NARRATION_PROMPT = V2_NARRATION_PROMPT
SUMMARY_NARRATION_PROMPT = V2_NARRATION_PROMPT


def _language_label(language: str) -> str:
    return {"id": "Bahasa Indonesia", "en": "English"}.get(language, language)


def _rules(style_config: dict) -> str:
    return "\n".join(f"- {rule}" for rule in style_config.get("rules", []) or []) or "- Write naturally and accurately"


def _beat_payload(scene_map: dict[str, dict], plan_entry: dict) -> dict:
    covered_ids = [str(sid) for sid in plan_entry.get("covers_scene_ids", []) if sid]
    if not covered_ids:
        covered_ids = [str(plan_entry.get("scene_id", ""))]
    covered = [scene_map.get(scene_id, {}) for scene_id in covered_ids]

    evidence = []
    for scene in covered:
        analysis = scene.get("analysis", {}) or {}
        evidence.append(
            {
                "scene_id": scene.get("scene_id"),
                "transcript": analysis.get("transcript_text", ""),
                "action": analysis.get("action", ""),
                "changes": analysis.get("changes", ""),
                "people": analysis.get("people", ""),
                "visible_objects": (analysis.get("visible_objects", []) or [])[:10],
                "environment": analysis.get("environment", ""),
                "likely_context": analysis.get("likely_context", ""),
                "character_goal": analysis.get("character_goal", ""),
                "state_before": analysis.get("state_before", ""),
                "state_after": analysis.get("state_after", ""),
                "cause": analysis.get("cause", ""),
                "effect": analysis.get("effect", ""),
                "confidence": analysis.get("confidence", 0),
            }
        )

    return {
        "scene_id": plan_entry.get("scene_id"),
        "covers_scene_ids": covered_ids,
        "duration_seconds": round(float(plan_entry.get("available_duration", 0) or 0), 2),
        "timeline_start": plan_entry.get("timeline_start"),
        "timeline_end": plan_entry.get("timeline_end"),
        "story_role": plan_entry.get("story_role", "process"),
        "intent": plan_entry.get("intent", ""),
        "continuity_from_previous": plan_entry.get("continuity_from_previous", ""),
        "transition_strategy": plan_entry.get("transition_strategy", ""),
        "must_explain": plan_entry.get("must_explain", []),
        "avoid_repeating": plan_entry.get("avoid_repeating", []),
        "min_words": plan_entry.get("min_words", 3),
        "target_words": plan_entry.get("target_words", 8),
        "max_words": plan_entry.get("max_words", 12),
        "evidence_by_visual_scene": evidence,
    }


def build_v2_narration_prompt(
    *,
    scenes: list[dict],
    narration_plan: dict,
    story_plan: dict,
    story_memory: dict,
    language: str,
    style: str,
    style_config: dict,
    hedging_threshold: float,
    mode: str,
) -> str:
    scene_map = {str(scene.get("scene_id")): scene for scene in scenes}
    payload = [
        _beat_payload(scene_map, plan_entry)
        for plan_entry in narration_plan.get("segments", []) or []
    ]

    return V2_NARRATION_PROMPT.format(
        mode=mode,
        language_label=_language_label(language),
        style_name=style_config.get("name", style),
        tone=style_config.get("tone", "natural storytelling"),
        story_plan_json=json.dumps(story_plan, ensure_ascii=False, indent=2)[:26000],
        story_memory_json=json.dumps(story_memory, ensure_ascii=False, indent=2)[:14000],
        rules=_rules(style_config),
        scenes_json=json.dumps(payload, ensure_ascii=False, indent=2),
        hedging_threshold=hedging_threshold,
    )


def build_rewrite_prompt(
    *,
    current_segments: list[dict],
    qa: dict,
    narration_plan: dict,
    story_plan: dict,
    language: str,
) -> str:
    return REWRITE_PROMPT.format(
        language_label=_language_label(language),
        issues_json=json.dumps(qa.get("issues", []), ensure_ascii=False, indent=2),
        story_plan_json=json.dumps(story_plan, ensure_ascii=False, indent=2)[:22000],
        narration_plan_json=json.dumps(narration_plan, ensure_ascii=False, indent=2)[:26000],
        current_json=json.dumps(current_segments, ensure_ascii=False, indent=2),
    )


def _compat_plan(storyboard: dict, scenes: list[dict], mode: str) -> dict:
    """Small compatibility plan when this module is called directly."""
    cumulative = 0.0
    entries = []
    for index, scene in enumerate(scenes):
        duration = float(scene.get("duration", 0) or 0)
        start = cumulative if mode == "summary" else float(scene.get("start_time", 0) or 0)
        if mode == "summary":
            cumulative += duration
        analysis = scene.get("analysis", {}) or {}
        narrative = scene.get("narrative", {}) or {}
        target_words = max(4, round(duration * 160 / 60 * 0.8))
        entries.append(
            {
                "scene_id": scene.get("scene_id"),
                "available_duration": duration,
                "timeline_start": start,
                "timeline_end": start + duration,
                "story_role": narrative.get("role", "process"),
                "intent": narrative.get("cue", "") or analysis.get("likely_context", "") or analysis.get("action", ""),
                "must_explain": [analysis.get("action", "")] if analysis.get("action") else [],
                "avoid_repeating": [],
                "continuity_from_previous": narrative.get("connects_to_next", ""),
                "transition_strategy": "progression" if index else "none",
                "min_words": max(3, round(target_words * 0.72)),
                "target_words": target_words,
                "max_words": max(target_words + 2, round(target_words * 1.18)),
            }
        )
    return {"mode": mode, "segments": entries}


def build_narration_prompt(
    storyboard: dict,
    language: str,
    style: str,
    style_config: dict,
    hedging_threshold: float = 0.5,
) -> str:
    scenes = storyboard.get("scenes", []) or []
    return build_v2_narration_prompt(
        scenes=scenes,
        narration_plan=_compat_plan(storyboard, scenes, "full"),
        story_plan={"premise": storyboard.get("video_summary", ""), "scene_roles": []},
        story_memory={},
        language=language,
        style=style,
        style_config=style_config,
        hedging_threshold=hedging_threshold,
        mode="full",
    )


def build_summary_narration_prompt(
    storyboard: dict,
    selected_scenes: list[dict],
    language: str,
    style: str,
    style_config: dict,
    hedging_threshold: float = 0.5,
) -> str:
    return build_v2_narration_prompt(
        scenes=selected_scenes,
        narration_plan=_compat_plan(storyboard, selected_scenes, "summary"),
        story_plan={"premise": storyboard.get("video_summary", ""), "scene_roles": []},
        story_memory={},
        language=language,
        style=style,
        style_config=style_config,
        hedging_threshold=hedging_threshold,
        mode="summary",
    )
