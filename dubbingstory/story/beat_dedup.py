"""Cheap deterministic duplicate removal for story-first beats."""

from __future__ import annotations

import re
from typing import Any


_WORD_RE = re.compile(r"[\w]+", flags=re.UNICODE)


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(str(text or "").lower()))


def _bigrams(text: str) -> set[str]:
    compact = _normalize(text).replace(" ", "_")
    if len(compact) < 2:
        return {compact} if compact else set()
    return {compact[i : i + 2] for i in range(len(compact) - 1)}


def bigram_jaccard(left: str, right: str) -> float:
    a = _bigrams(left)
    b = _bigrams(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _beat_text(beat: dict[str, Any]) -> str:
    parts = [
        str(beat.get("intent", "") or ""),
        str(beat.get("text", "") or ""),
        " ".join(str(v) for v in beat.get("required_facts", []) or []),
    ]
    return " ".join(part for part in parts if part).strip()


def deduplicate_beats(beats: list[dict[str, Any]], threshold: float = 0.90) -> list[dict[str, Any]]:
    """Drop obvious near-duplicate beats while preserving chronological order."""
    kept: list[dict[str, Any]] = []
    for beat in beats:
        text = _beat_text(beat)
        duplicate_index = None
        duplicate_score = 0.0
        for index, existing in enumerate(kept):
            score = bigram_jaccard(text, _beat_text(existing))
            if score >= threshold and score > duplicate_score:
                duplicate_index = index
                duplicate_score = score

        if duplicate_index is None:
            kept.append(dict(beat))
            continue

        # Preserve the more important representative but do not reorder beats.
        existing = kept[duplicate_index]
        old_importance = float(existing.get("importance", 0) or 0)
        new_importance = float(beat.get("importance", 0) or 0)
        if new_importance > old_importance:
            replacement = dict(beat)
            replacement["dedup_replaced_beat_id"] = existing.get("beat_id")
            replacement["dedup_similarity"] = round(duplicate_score, 3)
            kept[duplicate_index] = replacement
        else:
            existing.setdefault("dedup_dropped_beat_ids", []).append(beat.get("beat_id"))
            existing["dedup_similarity"] = max(
                float(existing.get("dedup_similarity", 0) or 0), round(duplicate_score, 3)
            )

    for index, beat in enumerate(kept):
        beat["beat_id"] = f"beat_{index + 1:03d}"
        beat["beat_index"] = index
    return kept
