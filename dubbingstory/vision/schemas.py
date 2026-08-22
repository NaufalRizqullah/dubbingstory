"""Strict JSON schemas and normalization helpers for vision outputs.

The schemas are intentionally conservative so local OpenAI-compatible servers
(vLLM in particular) can constrain generation at decode time instead of relying
only on prompt instructions.
"""

from __future__ import annotations

from typing import Any

VISION_SCHEMA_VERSION = "vision-jsonschema-v3"


def _score_schema() -> dict[str, Any]:
    return {"type": "number", "minimum": 0.0, "maximum": 1.0}


def _string_schema(max_length: int = 240) -> dict[str, Any]:
    return {"type": "string", "maxLength": max_length}


DEEP_SCENE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "visible_objects": {
            "type": "array",
            "items": _string_schema(96),
            "maxItems": 8,
        },
        "people": _string_schema(240),
        "action": _string_schema(240),
        "changes": _string_schema(240),
        "environment": _string_schema(240),
        "likely_context": _string_schema(240),
        "character_goal": _string_schema(240),
        "state_before": _string_schema(240),
        "state_after": _string_schema(240),
        "cause": _string_schema(240),
        "effect": _string_schema(240),
        "unresolved_question": _string_schema(240),
        "text_visible": {
            "type": "array",
            "items": _string_schema(80),
            "maxItems": 4,
        },
        "confidence": _score_schema(),
    },
    "required": [
        "visible_objects",
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
        "text_visible",
        "confidence",
    ],
    "additionalProperties": False,
}


CHEAP_SCENE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": _string_schema(240),
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
            "connects_to_next": _string_schema(320),
            "narration_importance": _score_schema(),
            "causal_importance": _score_schema(),
            "bridge_importance": _score_schema(),
            "narration_cue": _string_schema(320),
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
            "video_summary": _string_schema(1200),
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
    """Defensive post-parse normalization, including dedupe/caps.

    Decode-time JSON schema is the primary guardrail. This normalization is a
    second line of defense and also protects optional schema-fallback servers.
    """
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
        "visible_objects": _dedupe_strings(data.get("visible_objects"), 8),
        "text_visible": _dedupe_strings(data.get("text_visible"), 4),
        "confidence": _clamp_score(data.get("confidence")),
    }
    for field in string_fields:
        result[field] = str(data.get(field, "") or "").strip()
    return result
