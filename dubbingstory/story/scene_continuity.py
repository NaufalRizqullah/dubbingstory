"""Local continuity / scene-delta extraction for DubbingStory V2.1.

The module intentionally starts with deterministic evidence.  It compares
adjacent scene windows, canonicalizes a few common entity labels, and records
what persisted vs. what changed.  This gives story planning and retrieval a
stable continuity layer without asking Qwen to re-describe the same footage.
"""

from __future__ import annotations

import re
from typing import Any


_RELATIONSHIPS = {
    "setup",
    "continuation",
    "progression",
    "transition",
    "result",
    "reaction",
    "parallel",
    "new_topic",
    "uncertain",
}

_ALIAS_GROUPS = {
    "mechanic": {"mechanic", "technician", "worker", "operator", "repairman", "man", "person"},
    "woman": {"woman", "female", "lady"},
    "component": {"component", "part", "piece", "workpiece", "item"},
    "machine": {"machine", "machinery", "equipment"},
    "vehicle": {"vehicle", "car", "truck", "motorcycle", "bike"},
}

_TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)


def _text_tokens(value: Any) -> set[str]:
    text = str(value or "").lower()
    return {token for token in _TOKEN_RE.findall(text) if len(token) > 2}


def _canonical_label(value: str) -> str:
    tokens = _text_tokens(value)
    for canonical, aliases in _ALIAS_GROUPS.items():
        if tokens & aliases:
            return canonical
    return " ".join(sorted(tokens)) if tokens else ""


def _entity_labels(card: dict) -> set[str]:
    visual = card.get("visual", {}) or {}
    labels: set[str] = set()

    people = str(visual.get("people", "") or "").strip()
    if people:
        canonical = _canonical_label(people)
        if canonical:
            labels.add(canonical)

    for obj in visual.get("visible_objects", []) or []:
        canonical = _canonical_label(str(obj or ""))
        if canonical:
            labels.add(canonical)
    return labels


def _jaccard(a: Any, b: Any) -> float:
    left = _text_tokens(a)
    right = _text_tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _same_parent(previous: dict | None, current: dict) -> bool:
    if not previous:
        return False
    p = str(previous.get("parent_scene_id", "") or "")
    c = str(current.get("parent_scene_id", "") or "")
    return bool(p and c and p == c)


def _relationship(previous: dict | None, current: dict, next_card: dict | None) -> tuple[str, float]:
    if previous is None:
        return "setup", 0.95

    p_visual = previous.get("visual", {}) or {}
    c_visual = current.get("visual", {}) or {}
    p_context = previous.get("context", {}) or {}
    c_context = current.get("context", {}) or {}

    same_parent = _same_parent(previous, current)
    entity_overlap = _entity_labels(previous) & _entity_labels(current)
    action_sim = _jaccard(p_visual.get("action", ""), c_visual.get("action", ""))
    state_link = _jaccard(p_context.get("state_after", ""), c_context.get("state_before", ""))
    change_text = " ".join(
        str(v or "")
        for v in (
            c_visual.get("changes", ""),
            c_context.get("state_before", ""),
            c_context.get("state_after", ""),
            c_context.get("effect", ""),
        )
    ).strip()

    if same_parent and (entity_overlap or action_sim >= 0.25):
        if change_text and (state_link >= 0.15 or _jaccard(c_context.get("state_before", ""), c_context.get("state_after", "")) < 0.65):
            return "progression", 0.9
        return "continuation", 0.86

    gap = max(0.0, float(current.get("start_time", 0) or 0) - float(previous.get("end_time", 0) or 0))
    if entity_overlap and action_sim >= 0.25:
        return ("progression" if change_text else "continuation"), 0.78

    c_role = str((current.get("existing_narrative", {}) or {}).get("role", "") or "").lower()
    if c_role in {"result", "resolution", "conclusion"}:
        return "result", 0.75
    if c_role in {"reaction"}:
        return "reaction", 0.72

    if gap <= 1.0 and (entity_overlap or action_sim >= 0.12):
        return "transition", 0.65

    if next_card and _entity_labels(previous) & _entity_labels(next_card):
        return "parallel", 0.58

    if not entity_overlap and action_sim < 0.08:
        return "new_topic", 0.7
    return "uncertain", 0.45


def build_scene_deltas(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one local continuity record per scene card in chronological order."""
    ordered = sorted(cards, key=lambda c: (float(c.get("start_time", 0) or 0), int(c.get("scene_index", 0) or 0)))
    deltas: list[dict[str, Any]] = []

    for index, current in enumerate(ordered):
        previous = ordered[index - 1] if index else None
        next_card = ordered[index + 1] if index + 1 < len(ordered) else None
        c_visual = current.get("visual", {}) or {}
        c_context = current.get("context", {}) or {}
        p_visual = (previous or {}).get("visual", {}) or {}
        p_context = (previous or {}).get("context", {}) or {}

        current_entities = _entity_labels(current)
        previous_entities = _entity_labels(previous) if previous else set()
        persistent = sorted(current_entities & previous_entities)
        new_entities = sorted(current_entities - previous_entities)

        relation, relation_confidence = _relationship(previous, current, next_card)
        if relation not in _RELATIONSHIPS:
            relation = "uncertain"

        current_action = str(c_visual.get("action", "") or "").strip()
        previous_action = str(p_visual.get("action", "") or "").strip()
        new_action = current_action if _jaccard(previous_action, current_action) < 0.72 else ""

        before = str(c_context.get("state_before", "") or "").strip()
        after = str(c_context.get("state_after", "") or "").strip()
        if not before and previous:
            before = str(p_context.get("state_after", "") or "").strip()

        new_info: list[str] = []
        for value in (
            new_action,
            str(c_visual.get("changes", "") or "").strip(),
            str(c_context.get("effect", "") or "").strip(),
        ):
            if value and value.lower() not in {v.lower() for v in new_info}:
                new_info.append(value)
        if before and after and _jaccard(before, after) < 0.8:
            state_text = f"{before} -> {after}"
            if state_text.lower() not in {v.lower() for v in new_info}:
                new_info.append(state_text)

        visual_confidence = float(current.get("confidence", 0) or 0)
        evidence_quality = visual_confidence if current.get("selected_for_deep_analysis", True) else 0.0
        confidence = max(0.0, min(1.0, relation_confidence * 0.65 + evidence_quality * 0.35))

        deltas.append(
            {
                "scene_id": current.get("scene_id"),
                "parent_scene_id": current.get("parent_scene_id"),
                "window_index": current.get("window_index", 0),
                "continuation_from": previous.get("scene_id") if previous else None,
                "relationship": relation,
                "persistent_entities": persistent,
                "new_entities": new_entities,
                "new_action": new_action,
                "state_change": {"before": before, "after": after},
                "new_information_only": new_info[:6],
                "same_parent_window": _same_parent(previous, current),
                "action_similarity_from_previous": round(_jaccard(previous_action, current_action), 3),
                "confidence": round(confidence, 3),
                "evidence_quality": round(evidence_quality, 3),
            }
        )

    return deltas
