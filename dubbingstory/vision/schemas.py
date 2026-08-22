"""Strict JSON schemas and normalization helpers for vision outputs.

The schemas are intentionally conservative so local OpenAI-compatible servers
(vLLM in particular) can constrain generation at decode time instead of relying
only on prompt instructions.

Version 4 also budgets the *total* scene response so a normal deep response is
realistically able to finish inside the 640-token scene budget.  A separate,
smaller rescue schema is used only after a detected runaway/second truncation.
"""

from __future__ import annotations

from typing import Any

VISION_SCHEMA_VERSION = "vision-jsonschema-v4-budgeted"


def _score_schema() -> dict[str, Any]:
    return {"type": "number", "minimum": 0.0, "maximum": 1.0}


def _string_schema(max_length: int = 120) -> dict[str, Any]:
    return {"type": "string", "maxLength": max_length}


def _deep_scene_schema(*, rescue: bool = False) -> dict[str, Any]:
    """Build the stable-key deep schema with different output budgets.

    Keeping the same keys matters because downstream story code already reads
    these fields.  Only the allowed verbosity changes between normal and rescue
    mode.
    """
    if rescue:
        limits = {
            "visible_items": 4,
            "visible_item_chars": 32,
            "people": 48,
            "action": 80,
            "changes": 80,
            "environment": 48,
            "likely_context": 80,
            "character_goal": 48,
            "state_before": 48,
            "state_after": 48,
            "cause": 48,
            "effect": 48,
            "unresolved_question": 48,
            "text_items": 2,
            "text_item_chars": 32,
        }
    else:
        limits = {
            "visible_items": 6,
            "visible_item_chars": 48,
            "people": 80,
            "action": 120,
            "changes": 120,
            "environment": 80,
            "likely_context": 120,
            "character_goal": 80,
            "state_before": 80,
            "state_after": 80,
            "cause": 80,
            "effect": 80,
            "unresolved_question": 80,
            "text_items": 3,
            "text_item_chars": 48,
        }

    properties: dict[str, Any] = {
        "visible_objects": {
            "type": "array",
            "items": _string_schema(limits["visible_item_chars"]),
            "maxItems": limits["visible_items"],
        },
        "people": _string_schema(limits["people"]),
        "action": _string_schema(limits["action"]),
        "changes": _string_schema(limits["changes"]),
        "environment": _string_schema(limits["environment"]),
        "likely_context": _string_schema(limits["likely_context"]),
        "character_goal": _string_schema(limits["character_goal"]),
        "state_before": _string_schema(limits["state_before"]),
        "state_after": _string_schema(limits["state_after"]),
        "cause": _string_schema(limits["cause"]),
        "effect": _string_schema(limits["effect"]),
        "unresolved_question": _string_schema(limits["unresolved_question"]),
        "text_visible": {
            "type": "array",
            "items": _string_schema(limits["text_item_chars"]),
            "maxItems": limits["text_items"],
        },
        "confidence": _score_schema(),
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }


DEEP_SCENE_SCHEMA: dict[str, Any] = _deep_scene_schema(rescue=False)
DEEP_SCENE_RESCUE_SCHEMA: dict[str, Any] = _deep_scene_schema(rescue=True)


CHEAP_SCENE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": _string_schema(160),
        "visual_change": _score_schema(),
        "story_relevance": _score_schema(),
        "salience": _score_schema(),
        "confidence": _score_schema(),
    },
    "required": [
        "action",
        "visual_change",
        "story_relevance",
        "salience",
        "confidence",
    ],
    "additionalProperties": False,
}


def build_temporal_flow_schema(scene_ids: list[str]) -> dict[str, Any]:
    """Build a bounded temporal schema for exactly one analysis chunk."""
    count = len(scene_ids)
    scene_id_schema: dict[str, Any] = {"type": "string"}
    if scene_ids:
        scene_id_schema["enum"] = list(scene_ids)

    enriched_item = {
        "type": "object",
        "properties": {
            "scene_id": scene_id_schema,
            "narrative_role": {
                "type": "string",
                "enum": [
                    "introduction",
                    "setup",
                    "problem",
                    "process",
                    "bridge",
                    "climax",
                    "result",
                    "conclusion",
                ],
            },
            "connects_to_next": _string_schema(180),
            "narration_importance": _score_schema(),
            "causal_importance": _score_schema(),
            "bridge_importance": _score_schema(),
            "narration_cue": _string_schema(180),
        },
        "required": [
            "scene_id",
            "narrative_role",
            "connects_to_next",
            "narration_importance",
            "causal_importance",
            "bridge_importance",
            "narration_cue",
        ],
        "additionalProperties": False,
    }

    scenes_schema: dict[str, Any] = {
        "type": "array",
        "items": enriched_item,
        "maxItems": max(count, 1),
    }
    if count:
        scenes_schema["minItems"] = count
        scenes_schema["maxItems"] = count

    return {
        "type": "object",
        "properties": {
            "video_summary": _string_schema(800),
            "narrative_arc": {
                "type": "string",
                "enum": ["repair_process", "cooking", "tutorial", "event", "story", "other", "unknown"],
            },
            "domain": _string_schema(80),
            "scenes_enriched": scenes_schema,
        },
        "required": ["video_summary", "narrative_arc", "domain", "scenes_enriched"],
        "additionalProperties": False,
    }


def _dedupe_strings(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        clean = " ".join(item.split()).strip()
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(clean)
        if len(result) >= limit:
            break
    return result


def _clamp_score(value: Any, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, score))


def normalize_scene_analysis(data: Any, kind: str = "deep") -> dict[str, Any]:
    """Defensive post-parse normalization, including dedupe/caps."""
    if not isinstance(data, dict):
        raise ValueError(f"Vision response must be a JSON object, got {type(data).__name__}")

    if kind == "cheap":
        return {
            "action": str(data.get("action", "") or "").strip(),
            "visual_change": _clamp_score(data.get("visual_change")),
            "story_relevance": _clamp_score(data.get("story_relevance")),
            "salience": _clamp_score(data.get("salience")),
            "confidence": _clamp_score(data.get("confidence")),
        }

    string_fields = (
        "people",
        "action",
        "changes",
        "environment",
        "likely_context",
        "character_goal",
        "state_before",
        "state_after",
        "cause",
        "effect",
        "unresolved_question",
    )
    result: dict[str, Any] = {
        "visible_objects": _dedupe_strings(data.get("visible_objects"), 6),
        "text_visible": _dedupe_strings(data.get("text_visible"), 3),
        "confidence": _clamp_score(data.get("confidence")),
    }
    for field in string_fields:
        result[field] = str(data.get(field, "") or "").strip()
    return result
