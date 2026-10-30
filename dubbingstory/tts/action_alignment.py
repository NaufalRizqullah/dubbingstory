"""Place measured speech on source time using grounded spoken-action phrases.

Only the speech start may move. Video, word durations and the selected visual
events remain unchanged. Misses are reported rather than hidden by fake timing.
"""

from __future__ import annotations

import math
from statistics import median

from dubbingstory.story.action_anchors import canonical_action_anchors, phrase_word_span


def _distance(value: float, start: float, end: float) -> float:
    return max(start - value, value - end, 0.0)


def place_anchored_speech(
    segment: dict,
    words: list[dict],
    *,
    earliest_start: float,
    latest_start: float,
    tolerance: float = 0.5,
) -> tuple[float, dict]:
    """Return a feasible start and measured anchor report for one speech clip.

    A phrase's first measured word is its synchronization point. All anchors
    constrain the same clip start, so multiple anchors cannot stretch speech.
    With conflicting constraints choose the smallest worst miss inside the
    permitted speech window and leave enforcement to the caller's policy.
    """
    if (not all(math.isfinite(value) for value in (earliest_start, latest_start, tolerance))
            or earliest_start < 0 or latest_start < earliest_start or not 0 <= tolerance <= 2):
        raise ValueError("Invalid speech placement bounds or action synchronization tolerance")
    if not segment.get("action_sync_enabled", False):
        raise ValueError("Action synchronization requires a freshly planned script; rerun analyze and narrate with action_sync enabled")
    anchors = canonical_action_anchors(segment.get("action_anchors"), segment["text"], segment)
    if not anchors:
        return earliest_start, {"status": "no_anchors", "anchors": [], "tolerance_seconds": tolerance}

    measured = []
    for anchor in anchors:
        first, stop = phrase_word_span(words, anchor["phrase"])
        offset = float(words[first]["start"])
        phrase_end = float(words[stop - 1]["end"])
        start, end = float(anchor["start_time"]), float(anchor["end_time"])
        if (not all(math.isfinite(value) for value in (offset, phrase_end, start, end))
                or offset < 0 or phrase_end < offset or end <= start):
            raise ValueError("Invalid measured word timing or visual action window")
        measured.append((anchor, offset, phrase_end, start - offset, end - offset))

    lower = max([earliest_start] + [item[3] - tolerance for item in measured])
    upper = min([latest_start] + [item[4] + tolerance for item in measured])
    preferred = median([(item[3] + item[4]) / 2 for item in measured])
    if lower <= upper:
        chosen = min(upper, max(lower, preferred))
    else:
        # The minimax solution for interval distance lies at a bound or halfway
        # between opposing interval bounds. Evaluate those finite candidates.
        limits = [item[3] for item in measured] + [item[4] for item in measured]
        candidates = {earliest_start, latest_start}
        candidates.update(min(latest_start, max(earliest_start, value)) for value in limits)
        candidates.update(min(latest_start, max(earliest_start, (left + right) / 2))
                          for left in limits for right in limits)

        def cost(value):
            misses = [_distance(value, item[3], item[4]) for item in measured]
            return max(misses), sum(misses), abs(value - preferred), value

        chosen = min(candidates, key=cost)

    reports = []
    for anchor, offset, phrase_end, _, _ in measured:
        spoken = chosen + offset
        error = _distance(spoken, anchor["start_time"], anchor["end_time"])
        reports.append({
            **anchor,
            "spoken_time": round(spoken, 6),
            "spoken_end": round(chosen + phrase_end, 6),
            "error_seconds": round(error, 6),
            "matched": error <= tolerance + 1e-6,
        })
    return chosen, {
        "status": "complete" if all(item["matched"] for item in reports) else "partial",
        "tolerance_seconds": tolerance,
        "placement_shift_seconds": round(chosen - earliest_start, 6),
        "anchors": reports,
    }
