"""Build compact, story-oriented scene cards from the visual storyboard.

Scene cards are deliberately factual.  They are the boundary between vision and
story generation: the vision layer says what is visible / heard; later LLM
stages decide how those facts form a story.
"""

from __future__ import annotations

from typing import Any


def _overlapping_transcript(
    subtitle_context: dict | None,
    start_time: float,
    end_time: float,
    max_chars: int = 700,
) -> str:
    if not subtitle_context:
        return ""

    lines: list[str] = []
    for entry in subtitle_context.get("entries", []) or []:
        sub_start = float(entry.get("start_seconds", 0) or 0)
        sub_end = float(entry.get("end_seconds", 0) or 0)
        if sub_start < end_time and sub_end > start_time:
            text = str(entry.get("text", "")).strip()
            if text:
                lines.append(text)

    text = " ".join(lines).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def build_scene_cards(
    storyboard: dict,
    subtitle_context: dict | None = None,
) -> list[dict[str, Any]]:
    """Convert storyboard scenes into compact story-planning records.

    No new facts are invented here.  Optional state/cause/effect fields are
    copied only when the vision model supplied them.
    """
    cards: list[dict[str, Any]] = []

    for index, scene in enumerate(storyboard.get("scenes", []) or []):
        analysis = scene.get("analysis", {}) or {}
        narrative = scene.get("narrative", {}) or {}
        start = float(scene.get("start_time", 0) or 0)
        end = float(scene.get("end_time", start) or start)

        transcript_text = str(analysis.get("transcript_text", "") or "").strip()
        if not transcript_text:
            transcript_text = _overlapping_transcript(subtitle_context, start, end)

        card = {
            "scene_id": scene.get("scene_id", f"scene_{index + 1:03d}"),
            "scene_index": index,
            "start_time": start,
            "end_time": end,
            "duration": float(scene.get("duration", max(0.0, end - start)) or 0),
            "selected_for_deep_analysis": bool(analysis.get("selected", True)),
            "transcript": transcript_text,
            "visual": {
                "people": analysis.get("people", ""),
                "visible_objects": analysis.get("visible_objects", []) or [],
                "action": analysis.get("action", ""),
                "changes": analysis.get("changes", ""),
                "environment": analysis.get("environment", ""),
                "text_visible": analysis.get("text_visible", []) or [],
            },
            "context": {
                "likely_context": analysis.get("likely_context", ""),
                "character_goal": analysis.get("character_goal", ""),
                "state_before": analysis.get("state_before", ""),
                "state_after": analysis.get("state_after", ""),
                "cause": analysis.get("cause", ""),
                "effect": analysis.get("effect", ""),
                "unresolved_question": analysis.get("unresolved_question", ""),
            },
            "existing_narrative": {
                "role": narrative.get("role", "process"),
                "importance": float(narrative.get("importance", 0.5) or 0.5),
                "cue": narrative.get("cue", ""),
                "connects_to_next": narrative.get("connects_to_next", ""),
            },
            "confidence": float(analysis.get("confidence", 0) or 0),
        }
        cards.append(card)

    return cards


def compact_scene_cards(
    cards: list[dict[str, Any]],
    *,
    max_transcript_chars: int = 350,
) -> list[dict[str, Any]]:
    """Create a lower-token representation for Gemini planning prompts."""
    compact: list[dict[str, Any]] = []
    for card in cards:
        transcript = str(card.get("transcript", "") or "")
        if len(transcript) > max_transcript_chars:
            transcript = transcript[: max_transcript_chars - 1].rstrip() + "…"

        visual = card.get("visual", {}) or {}
        context = card.get("context", {}) or {}
        existing = card.get("existing_narrative", {}) or {}
        compact.append(
            {
                "scene_id": card.get("scene_id"),
                "start": round(float(card.get("start_time", 0) or 0), 2),
                "duration": round(float(card.get("duration", 0) or 0), 2),
                "transcript": transcript,
                "action": visual.get("action", ""),
                "changes": visual.get("changes", ""),
                "people": visual.get("people", ""),
                "objects": (visual.get("visible_objects", []) or [])[:10],
                "likely_context": context.get("likely_context", ""),
                "goal": context.get("character_goal", ""),
                "state_before": context.get("state_before", ""),
                "state_after": context.get("state_after", ""),
                "cause": context.get("cause", ""),
                "effect": context.get("effect", ""),
                "existing_role": existing.get("role", "process"),
                "existing_importance": existing.get("importance", 0.5),
                "confidence": card.get("confidence", 0),
            }
        )
    return compact
