"""Normalized retrieval signatures for DubbingStory V2.1 scene windows."""

from __future__ import annotations

from typing import Any


SIGNATURE_VERSION = "v2.1.0"


def _clean(value: Any) -> str:
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value if str(v).strip())
    return " ".join(str(value or "").strip().split())


def build_scene_signatures(
    cards: list[dict[str, Any]],
    deltas: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build a compact evidence-oriented text representation for retrieval.

    Signatures deliberately mix visual action/state, continuity delta and audio
    context.  Transcript alone is never used as the representation.
    """
    delta_map = {str(d.get("scene_id")): d for d in (deltas or []) if d.get("scene_id")}
    signatures: list[dict[str, Any]] = []

    for card in cards:
        sid = str(card.get("scene_id", ""))
        visual = card.get("visual", {}) or {}
        context = card.get("context", {}) or {}
        delta = delta_map.get(sid, {}) or {}

        parts: list[str] = []
        fields = [
            ("ACTION", visual.get("action")),
            ("CHANGE", visual.get("changes")),
            ("STATE_BEFORE", context.get("state_before")),
            ("STATE_AFTER", context.get("state_after")),
            ("GOAL", context.get("character_goal")),
            ("CONTEXT", context.get("likely_context")),
            ("PEOPLE", visual.get("people")),
            ("OBJECTS", visual.get("visible_objects")),
            ("CONTINUITY", delta.get("relationship")),
            ("NEW_INFO", delta.get("new_information_only")),
            ("AUDIO", card.get("transcript")),
        ]
        for label, value in fields:
            text = _clean(value)
            if text:
                parts.append(f"{label}: {text}")

        confidence = float(card.get("confidence", 0) or 0)
        deep = bool(card.get("selected_for_deep_analysis", True))
        action = _clean(visual.get("action"))
        skipped = not deep or "skipped in summary" in action.lower()
        evidence_quality = 0.0 if skipped else max(0.0, min(1.0, confidence))

        signatures.append(
            {
                "scene_id": sid,
                "parent_scene_id": card.get("parent_scene_id"),
                "window_index": card.get("window_index", 0),
                "window_count": card.get("window_count", 1),
                "start_time": float(card.get("start_time", 0) or 0),
                "end_time": float(card.get("end_time", 0) or 0),
                "duration": float(card.get("duration", 0) or 0),
                "signature": "\n".join(parts),
                "signature_version": SIGNATURE_VERSION,
                "confidence": round(confidence, 3),
                "evidence_quality": round(evidence_quality, 3),
                "deep_evidence": deep and not skipped,
                "relationship": delta.get("relationship", "uncertain"),
                "state_change": delta.get("state_change", {"before": "", "after": ""}),
                "new_information_only": delta.get("new_information_only", []),
            }
        )

    return signatures
