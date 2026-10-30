"""Ground spoken action phrases in timestamped visual evidence.

Only the vision timeline supplies times. Text models may name an existing event
and quote a phrase from their narration; they cannot create or move an event.
"""

from __future__ import annotations

import math
import re
import unicodedata


def normalized_tokens(text: str) -> list[str]:
    """Unicode words, insensitive to case and surrounding punctuation."""
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)*", unicodedata.normalize("NFKC", str(text)).casefold())


def phrase_word_span(words: list[dict], phrase: str) -> tuple[int, int]:
    """Find a unique contiguous phrase; return word indexes [first, stop).

    A provider may put punctuation or several tokens in one word record. The
    returned span still indexes the original records, not normalized tokens.
    Repeated phrases are rejected rather than selecting an arbitrary occurrence.
    """
    wanted = normalized_tokens(phrase)
    if not wanted:
        raise ValueError("An action anchor requires a nonempty spoken phrase")
    tokens: list[str] = []
    owners: list[int] = []
    for index, word in enumerate(words):
        parts = normalized_tokens(word.get("word", word.get("text", "")))
        tokens.extend(parts)
        owners.extend([index] * len(parts))
    matches = [i for i in range(len(tokens) - len(wanted) + 1)
               if tokens[i:i + len(wanted)] == wanted]
    if len(matches) != 1:
        reason = "absent" if not matches else "ambiguous (appears more than once)"
        raise ValueError(f"Action anchor phrase is {reason} in spoken text: {phrase!r}")
    first = matches[0]
    return owners[first], owners[first + len(wanted) - 1] + 1


def scene_visual_events(scene: dict) -> list[dict]:
    """Validate and copy optional source-time events without trusting LLM times."""
    raw = (scene.get("analysis", {}) or {}).get("events", [])
    if not isinstance(raw, list):
        raise ValueError("Visual events must be a list")
    if not raw:
        return []
    scene_id = str(scene.get("scene_id", ""))
    scene_start = float(scene.get("start_time", 0))
    scene_end = float(scene.get("end_time", scene_start))
    result = []
    seen = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"{scene_id}: visual event must be an object")
        event_id = str(item.get("event_id", "")).strip()
        start, end = float(item.get("start_time", -1)), float(item.get("end_time", -1))
        confidence = float(item.get("confidence", 0))
        action = str(item.get("action", "")).strip()
        frame_ids = item.get("frame_ids", [])
        if (not event_id or event_id in seen or not action
                or str(item.get("scene_id", scene_id)) != scene_id
                or not all(math.isfinite(v) for v in (scene_start, scene_end, start, end, confidence))
                or not scene_start <= start < end <= scene_end + 0.001
                or not 0 <= confidence <= 1
                or not isinstance(frame_ids, list) or not frame_ids
                or any(not isinstance(value, str) or not value.strip() for value in frame_ids)):
            raise ValueError(f"{scene_id}: invalid or unsupported visual event {event_id!r}")
        seen.add(event_id)
        result.append({
            "event_id": event_id, "scene_id": scene_id,
            "start_time": start, "end_time": min(end, scene_end),
            "action": action, "state_change": str(item.get("state_change", "") or ""),
            "confidence": confidence, "frame_ids": list(dict.fromkeys(frame_ids)),
            "timing_basis": str(item.get("timing_basis", "sample_window")),
        })
    return sorted(result, key=lambda item: (item["start_time"], item["end_time"], item["event_id"]))


def compact_visual_events(events: list[dict], *, required_ids=(), max_events: int = 8) -> dict:
    """Bound prompt evidence without changing the full timeline or anchor data.

    Keep required anchors and the beginning/end, then spread remaining evidence
    over the observed time range. Count omitted entries explicitly; this is a
    prompt representation, never a replacement for the saved observations.
    """
    ordered = sorted(events, key=lambda event: (event["start_time"], event["event_id"]))
    required = set(required_ids)
    known = {event["event_id"] for event in ordered}
    if required - known:
        raise ValueError("Required action events are absent from prompt evidence")
    limit = max(2, int(max_events), len(required))
    selected = {index for index, event in enumerate(ordered) if event["event_id"] in required}
    if ordered:
        for index in (0, len(ordered) - 1):
            if len(selected) < limit:
                selected.add(index)
    while len(selected) < min(limit, len(ordered)):
        remaining = [index for index in range(len(ordered)) if index not in selected]
        # Prefer uncovered time rather than many near-duplicate frames at the
        # start. Visible changes/confidence break ties, not source chronology.
        chosen = max(remaining, key=lambda index: (
            min(abs(ordered[index]["start_time"] - ordered[other]["start_time"])
                for other in selected),
            bool(ordered[index].get("state_change")), ordered[index]["confidence"], -index,
        ))
        selected.add(chosen)
    keys = ("event_id", "start_time", "end_time", "action", "state_change", "confidence")
    return {
        "visual_events": [{key: ordered[index][key] for key in keys} for index in sorted(selected)],
        "visual_event_count": len(ordered),
        "visual_events_omitted": len(ordered) - len(selected),
    }


def select_anchor_events(events: list[dict], *, minimum_confidence: float, limit: int = 1) -> list[str]:
    """Prefer well-supported changes, avoiding duplicate actions in one beat."""
    eligible = [event for event in events if event["confidence"] >= minimum_confidence]
    ranked = sorted(eligible, key=lambda event: (
        -bool(event.get("state_change")), -event["confidence"],
        -(event["end_time"] - event["start_time"]), event["start_time"],
    ))
    selected = []
    actions = set()
    for event in ranked:
        action = tuple(normalized_tokens(event["action"]))
        if action in actions:
            continue
        selected.append(event)
        actions.add(action)
        if len(selected) >= max(1, min(2, limit)):
            break
    return [event["event_id"] for event in sorted(selected, key=lambda event: event["start_time"])]


def canonical_action_anchors(raw: object, text: str, planned: dict) -> list[dict]:
    """Validate phrases and hydrate event times exclusively from the fixed plan."""
    supplied_events = planned.get("visual_events", [])
    required = planned.get("anchor_event_ids", [])
    if not isinstance(supplied_events, list) or not isinstance(required, list):
        raise ValueError("Visual events and required anchor IDs must be arrays")
    if any(not isinstance(event_id, str) for event_id in required):
        raise ValueError("Required anchor IDs must be strings")
    start, end = float(planned["timeline_start"]), float(planned["timeline_end"])
    if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
        raise ValueError("Action anchor beat requires a finite positive source-time window")
    covered = set(planned.get("covers_scene_ids", []))
    events = {}
    for event in supplied_events:
        if not isinstance(event, dict) or not isinstance(event.get("scene_id"), str):
            raise ValueError("Canonical visual event requires a source scene ID")
        if covered and event["scene_id"] not in covered:
            raise ValueError("Canonical visual event belongs to a different scene")
        # Reuse the source-event boundary validator for saved script artifacts.
        checked = scene_visual_events({
            "scene_id": event["scene_id"], "start_time": start, "end_time": end,
            "analysis": {"events": [event]},
        })[0]
        if checked["event_id"] in events:
            raise ValueError("Canonical visual event IDs must be unique")
        events[checked["event_id"]] = checked
    if raw is None:
        raw = []
    if not isinstance(raw, list) or len(raw) > 2:
        raise ValueError("action_anchors must be an array containing at most two items")
    if len(required) != len(set(required)) or any(event_id not in events for event_id in required):
        raise ValueError("Narration plan has invalid required visual event IDs")
    text_words = [{"word": token} for token in normalized_tokens(text)]
    previous_phrase_end = -1
    previous_event_start = -1.0
    result = []
    seen = set()
    for anchor in raw:
        if not isinstance(anchor, dict):
            raise ValueError("Each action anchor must be an object")
        event_id = str(anchor.get("event_id", ""))
        phrase = str(anchor.get("phrase", "")).strip()
        event = events.get(event_id)
        if event is None or event_id not in required or event_id in seen:
            raise ValueError(f"Unknown, unselected or duplicate action anchor event: {event_id!r}")
        if not 1 <= len(normalized_tokens(phrase)) <= 6:
            raise ValueError("An action anchor phrase must contain one to six spoken words")
        first, stop = phrase_word_span(text_words, phrase)
        if (first < previous_phrase_end or event["start_time"] < previous_event_start
                or not start <= event["start_time"] < event["end_time"] <= end + 0.001):
            raise ValueError("Action anchors must follow speech/source chronology inside the beat")
        result.append({
            "event_id": event_id, "phrase": phrase,
            "start_time": event["start_time"], "end_time": event["end_time"],
            "confidence": event["confidence"], "scene_id": event["scene_id"],
            "frame_ids": list(event["frame_ids"]),
            "timing_basis": event.get("timing_basis", "sample_window"),
        })
        previous_phrase_end, previous_event_start = stop, event["start_time"]
        seen.add(event_id)
    if seen != set(required):
        raise ValueError("Narration must quote one spoken action phrase for every selected visual event")
    return result


def preserve_action_phrases(text: str, anchors: list[dict]) -> None:
    """Duration/QA rewrites retain selected phrases and their spoken order."""
    words = [{"word": token} for token in normalized_tokens(text)]
    previous_stop = -1
    for anchor in anchors:
        first, stop = phrase_word_span(words, anchor["phrase"])
        if first < previous_stop:
            raise ValueError("A narration rewrite changed the order of required action phrases")
        previous_stop = stop
