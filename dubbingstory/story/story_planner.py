"""Global story understanding for DubbingStory v2.

This module turns factual scene cards + transcript context into two persistent
artifacts:

* ``story_plan.json``   — premise, goal/conflict, story arc and per-scene roles
* ``story_memory.json`` — stable characters/objects/threads for continuity

The planner uses Gemini when available and always has a deterministic fallback,
so local vision runs do not become unusable if the planning API is unavailable.
"""

from __future__ import annotations

import json
import os
from typing import Any

from dubbingstory.story.scene_cards import build_scene_cards, compact_scene_cards
from dubbingstory.story.scene_continuity import build_scene_deltas
from dubbingstory.story.scene_signature import build_scene_signatures
from dubbingstory.story.gemini_json import (
    generate_story_json, raise_if_strict_text_error, strict_text_routing,
    text_llm_available, validate_scene_items, validate_string_list,
)


STORY_PLANNER_PROMPT = """You are the global story planner for a video narration system.

Your job is NOT to write narration. First understand the complete video.
Use transcript/dialogue as the primary semantic source when present, and visual
facts as supporting evidence. Never invent names, relationships, motivations,
object purposes, or causes that are not supported by the supplied evidence.

VIDEO
Title: {video_title}
Description: {video_description}
Existing visual summary: {video_summary}
Existing arc label: {narrative_arc}
Transcript source: {transcript_source}

SCENE CARDS (chronological):
{scene_cards_json}

Return ONLY one JSON object with exactly these top-level keys:
{{
  "premise": "What the video is fundamentally about",
  "main_goal": "Main goal/process, or empty if unsupported",
  "central_conflict": "Main problem/challenge, or empty if unsupported",
  "story_arc": {{
    "setup": ["scene_001"],
    "problem": [],
    "process": [],
    "complication": [],
    "turning_point": [],
    "climax": [],
    "resolution": []
  }},
  "scene_roles": [
    {{
      "scene_id": "scene_001",
      "role": "setup|problem|process|bridge|complication|turning_point|climax|resolution|context",
      "story_importance": 0.0,
      "causal_importance": 0.0,
      "bridge_importance": 0.0,
      "must_keep": false,
      "plot_thread": "short stable thread label",
      "reason": "why this scene matters to the story"
    }}
  ],
  "characters": [
    {{
      "id": "character_1",
      "label": "evidence-based stable label",
      "goal": "",
      "state": "",
      "last_seen_scene": "scene_001"
    }}
  ],
  "important_objects": [
    {{
      "id": "object_1",
      "label": "stable object label",
      "purpose": "only if supported",
      "state": "",
      "history": ["scene_001: ..."]
    }}
  ],
  "unresolved_threads": ["..."],
  "story_memory": {{
    "known_facts": ["facts that future writing must preserve"],
    "continuity_rules": ["identity/object/goal facts that must stay consistent"],
    "open_questions": ["questions not yet resolved by evidence"]
  }}
}}

Rules:
- Include every supplied scene_id exactly once in scene_roles.
- A visually flashy scene is not automatically story-important.
- Mark bridge scenes when removing them would make cause/effect or process flow confusing.
- Prefer causal explanation over generic excitement.
- If evidence is uncertain, leave the field empty or state uncertainty in the reason.
"""


def _clip_json_payload(cards: list[dict], max_chars: int) -> str:
    """Fit prompt payload to a conservative character budget without losing order."""
    if max_chars <= 0:
        max_chars = 60000

    rendered: list[dict] = []
    current = 2
    for card in cards:
        text = json.dumps(card, ensure_ascii=False)
        if rendered and current + len(text) + 2 > max_chars:
            break
        rendered.append(card)
        current += len(text) + 2
    return json.dumps(rendered, ensure_ascii=False, indent=2)


def _scene_role_defaults(storyboard: dict) -> list[dict]:
    roles: list[dict] = []
    scenes = storyboard.get("scenes", []) or []
    for index, scene in enumerate(scenes):
        narrative = scene.get("narrative", {}) or {}
        role = str(narrative.get("role", "process") or "process")
        if role == "introduction":
            role = "setup"
        elif role == "result" or role == "conclusion":
            role = "resolution"
        importance = float(narrative.get("importance", 0.5) or 0.5)
        roles.append(
            {
                "scene_id": scene.get("scene_id"),
                "role": role,
                "story_importance": importance,
                "causal_importance": min(1.0, importance * 0.8),
                "bridge_importance": 0.5 if role in {"process", "bridge"} else 0.2,
                "must_keep": index == 0 or index == len(scenes) - 1,
                "plot_thread": "main",
                "reason": narrative.get("cue", "") or narrative.get("connects_to_next", ""),
            }
        )
    return roles


def fallback_story_artifacts(storyboard: dict) -> tuple[dict, dict]:
    """Build usable story artifacts without another LLM call."""
    roles = _scene_role_defaults(storyboard)
    plan = {
        "premise": storyboard.get("video_summary", "") or storyboard.get("video_title", ""),
        "main_goal": "",
        "central_conflict": "",
        "story_arc": {
            "setup": [r["scene_id"] for r in roles if r["role"] == "setup"],
            "problem": [r["scene_id"] for r in roles if r["role"] == "problem"],
            "process": [r["scene_id"] for r in roles if r["role"] == "process"],
            "complication": [r["scene_id"] for r in roles if r["role"] == "complication"],
            "turning_point": [r["scene_id"] for r in roles if r["role"] == "turning_point"],
            "climax": [r["scene_id"] for r in roles if r["role"] == "climax"],
            "resolution": [r["scene_id"] for r in roles if r["role"] == "resolution"],
        },
        "scene_roles": roles,
        "characters": [],
        "important_objects": [],
        "unresolved_threads": [],
    }
    memory = {
        "known_facts": [storyboard.get("video_summary", "")] if storyboard.get("video_summary") else [],
        "continuity_rules": [],
        "open_questions": [],
        "characters": [],
        "important_objects": [],
    }
    return plan, memory


def _normalize_story_artifacts(raw: dict, storyboard: dict) -> tuple[dict, dict]:
    fallback_plan, fallback_memory = fallback_story_artifacts(storyboard)
    if not isinstance(raw, dict):
        return fallback_plan, fallback_memory

    plan = {
        "premise": str(raw.get("premise", "") or fallback_plan["premise"]),
        "main_goal": str(raw.get("main_goal", "") or ""),
        "central_conflict": str(raw.get("central_conflict", "") or ""),
        "story_arc": raw.get("story_arc") if isinstance(raw.get("story_arc"), dict) else fallback_plan["story_arc"],
        "scene_roles": raw.get("scene_roles") if isinstance(raw.get("scene_roles"), list) else [],
        "characters": raw.get("characters") if isinstance(raw.get("characters"), list) else [],
        "important_objects": raw.get("important_objects") if isinstance(raw.get("important_objects"), list) else [],
        "unresolved_threads": raw.get("unresolved_threads") if isinstance(raw.get("unresolved_threads"), list) else [],
    }

    valid_scene_ids = [s.get("scene_id") for s in storyboard.get("scenes", []) or []]
    supplied = {
        str(item.get("scene_id")): item
        for item in plan["scene_roles"]
        if isinstance(item, dict) and item.get("scene_id") in valid_scene_ids
    }
    defaults = {item["scene_id"]: item for item in fallback_plan["scene_roles"]}

    normalized_roles: list[dict] = []
    for scene_id in valid_scene_ids:
        base = defaults.get(scene_id, {})
        item = supplied.get(scene_id, {})
        normalized_roles.append(
            {
                "scene_id": scene_id,
                "role": str(item.get("role", base.get("role", "process")) or "process"),
                "story_importance": _clamp01(item.get("story_importance", base.get("story_importance", 0.5))),
                "causal_importance": _clamp01(item.get("causal_importance", base.get("causal_importance", 0.4))),
                "bridge_importance": _clamp01(item.get("bridge_importance", base.get("bridge_importance", 0.2))),
                "must_keep": bool(item.get("must_keep", base.get("must_keep", False))),
                "plot_thread": str(item.get("plot_thread", base.get("plot_thread", "main")) or "main"),
                "reason": str(item.get("reason", base.get("reason", "")) or ""),
            }
        )
    plan["scene_roles"] = normalized_roles

    memory_raw = raw.get("story_memory") if isinstance(raw.get("story_memory"), dict) else {}
    memory = {
        "known_facts": _string_list(memory_raw.get("known_facts")),
        "continuity_rules": _string_list(memory_raw.get("continuity_rules")),
        "open_questions": _string_list(memory_raw.get("open_questions")),
        "characters": plan["characters"],
        "important_objects": plan["important_objects"],
        "unresolved_threads": plan["unresolved_threads"],
    }
    if not any(memory.values()):
        memory = fallback_memory
    return plan, memory


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _story_response_validator(storyboard: dict):
    expected_ids = {str(scene.get("scene_id")) for scene in storyboard.get("scenes", []) or []}

    def validator(payload: Any) -> str | None:
        if not isinstance(payload, dict) or not isinstance(payload.get("premise"), str) or not payload["premise"].strip():
            return "story plan must be an object with a nonempty premise"
        error = validate_scene_items(payload.get("scene_roles"), expected_ids)
        if error:
            return error
        for item in payload["scene_roles"]:
            if not isinstance(item.get("role"), str) or not item["role"].strip():
                return "every scene role requires a nonempty role"
            for key in ("story_importance", "causal_importance", "bridge_importance"):
                value = item.get(key)
                if type(value) not in (int, float) or not 0 <= value <= 1:
                    return "scene importance fields must be numbers from zero to one"
        arc, memory = payload.get("story_arc"), payload.get("story_memory")
        if not isinstance(arc, dict) or not isinstance(memory, dict):
            return "story_arc and story_memory must be objects"
        for ids in arc.values():
            if not validate_string_list(ids) or not set(ids) <= expected_ids:
                return "story_arc may only reference supplied scene IDs"
        for key in ("known_facts", "continuity_rules", "open_questions"):
            if not validate_string_list(memory.get(key)):
                return "story_memory fields must be arrays of strings"
        for key in ("characters", "important_objects"):
            value = payload.get(key)
            if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
                return "characters and important_objects must be arrays of objects"
        if not validate_string_list(payload.get("unresolved_threads")):
            return "unresolved_threads must be an array of strings"
        return None

    return validator


def generate_story_artifacts(
    storyboard: dict,
    *,
    subtitle_context: dict | None,
    cfg,
    project_dir: str | None = None,
) -> tuple[list[dict], dict, dict]:
    """Build scene cards + global story plan/memory and optionally persist them."""
    cards = build_scene_cards(storyboard, subtitle_context=subtitle_context)
    deltas = build_scene_deltas(cards)
    signatures = build_scene_signatures(cards, deltas)
    plan, memory = fallback_story_artifacts(storyboard)

    enabled = bool(getattr(cfg, "story_enabled", True))
    if enabled and text_llm_available(cfg):
        compact = compact_scene_cards(cards)
        delta_map = {str(item.get("scene_id")): item for item in deltas if item.get("scene_id")}
        for item in compact:
            sid = str(item.get("scene_id", ""))
            delta = delta_map.get(sid, {}) or {}
            item["continuity"] = {
                "relationship": delta.get("relationship", "uncertain"),
                "new_information_only": delta.get("new_information_only", []),
                "state_change": delta.get("state_change", {}),
            }
        max_chars = int(getattr(cfg, "story_prompt_max_chars", 60000) or 60000)
        prompt = STORY_PLANNER_PROMPT.format(
            video_title=storyboard.get("video_title", ""),
            video_description=storyboard.get("video_description", ""),
            video_summary=storyboard.get("video_summary", ""),
            narrative_arc=storyboard.get("narrative_arc", ""),
            transcript_source=storyboard.get("video_transcript_source", "none"),
            scene_cards_json=(json.dumps(compact, ensure_ascii=False, indent=2)
                              if strict_text_routing(cfg) else _clip_json_payload(compact, max_chars=max_chars)),
        )

        try:
            raw = generate_story_json(
                cfg,
                prompt,
                task_name="story-planner",
                temperature=0.15,
                model_attr="story_model",
                response_validator=_story_response_validator(storyboard) if strict_text_routing(cfg) else None,
            )
            plan, memory = _normalize_story_artifacts(raw, storyboard)
            print("   🧠 Story planner: global story plan generated")
        except Exception as exc:
            raise_if_strict_text_error(cfg, exc, "story-planner")
            print(f"   ⚠️ Story planner failed; using temporal fallback: {exc}")
    elif enabled:
        print("   ⚠️ Story planner: text provider unavailable; using temporal fallback")

    if project_dir:
        _save_json(os.path.join(project_dir, "scene_cards.json"), cards)
        _save_json(os.path.join(project_dir, "scene_delta.json"), deltas)
        _save_json(os.path.join(project_dir, "scene_signatures.json"), signatures)
        _save_json(os.path.join(project_dir, "story_plan.json"), plan)
        _save_json(os.path.join(project_dir, "story_memory.json"), memory)

    return cards, plan, memory


def rebuild_evidence_artifacts(
    storyboard: dict,
    *,
    subtitle_context: dict | None,
    project_dir: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Rebuild cards/deltas/signatures after on-demand vision promotion."""
    cards = build_scene_cards(storyboard, subtitle_context=subtitle_context)
    deltas = build_scene_deltas(cards)
    signatures = build_scene_signatures(cards, deltas)
    _save_json(os.path.join(project_dir, "scene_cards.json"), cards)
    _save_json(os.path.join(project_dir, "scene_delta.json"), deltas)
    _save_json(os.path.join(project_dir, "scene_signatures.json"), signatures)
    return cards, deltas, signatures


def load_evidence_artifacts(project_dir: str, storyboard: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Load V2.1 evidence artifacts, rebuilding deterministically if missing."""
    cards_path = os.path.join(project_dir, "scene_cards.json")
    delta_path = os.path.join(project_dir, "scene_delta.json")
    signature_path = os.path.join(project_dir, "scene_signatures.json")
    cards = _load_json(cards_path, [])
    deltas = _load_json(delta_path, [])
    signatures = _load_json(signature_path, [])
    if not cards:
        cards = build_scene_cards(storyboard)
    if not deltas:
        deltas = build_scene_deltas(cards)
    if not signatures:
        signatures = build_scene_signatures(cards, deltas)
    return cards, deltas, signatures


def load_story_artifacts(project_dir: str, storyboard: dict) -> tuple[dict, dict]:
    """Load persisted artifacts, falling back to storyboard-derived versions."""
    plan_path = os.path.join(project_dir, "story_plan.json")
    memory_path = os.path.join(project_dir, "story_memory.json")
    fallback_plan, fallback_memory = fallback_story_artifacts(storyboard)

    plan = _load_json(plan_path, fallback_plan)
    memory = _load_json(memory_path, fallback_memory)
    return plan, memory


def _save_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _load_json(path: str, fallback: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return fallback
