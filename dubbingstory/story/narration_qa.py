"""Deterministic QA checks for generated narration."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


REACTIVE_PHRASES = {
    "id": [
        "lihat ini", "luar biasa", "amazing", "gila", "keren banget",
        "bikin takjub", "pasti gak nyangka", "makin seru", "melongo",
    ],
    "en": [
        "look at this", "amazing", "incredible", "you won't believe",
        "insane", "wow", "check this out",
    ],
}


def evaluate_narration(
    segments: list[dict],
    narration_plan: dict | None,
    *,
    language: str,
) -> dict[str, Any]:
    plan_segments = (narration_plan or {}).get("segments", []) or []
    plan_by_id = {str(p.get("scene_id")): p for p in plan_segments}

    texts = [str(s.get("text", "") or "").strip() for s in segments if str(s.get("text", "") or "").strip()]
    full_text = " ".join(texts)
    words = re.findall(r"\b\w+\b", full_text, flags=re.UNICODE)
    sentence_like = [s for s in re.split(r"(?<=[.!?])\s+", full_text) if s.strip()]
    exclamations = full_text.count("!")
    exclamation_ratio = exclamations / max(1, len(sentence_like))

    reactive_hits: list[str] = []
    lower = full_text.lower()
    for phrase in REACTIVE_PHRASES.get(language, REACTIVE_PHRASES["en"]):
        count = lower.count(phrase)
        reactive_hits.extend([phrase] * count)

    openers = []
    for sentence in sentence_like:
        tokens = re.findall(r"\b\w+\b", sentence.lower(), flags=re.UNICODE)
        if tokens:
            openers.append(" ".join(tokens[:2]))
    opener_counts = Counter(openers)
    repeated_openers = {k: v for k, v in opener_counts.items() if v >= 3}

    supplied_ids = [str(s.get("scene_id", "")) for s in segments]
    missing_ids = [sid for sid in plan_by_id if sid not in supplied_ids]

    target_words = sum(int(p.get("target_words", 0) or 0) for p in plan_segments)
    word_ratio = len(words) / target_words if target_words else 1.0
    duration = float((narration_plan or {}).get("total_duration", 0) or 0)
    timeline_wpm = len(words) / duration * 60.0 if duration > 0 else 0.0

    issues: list[str] = []
    if exclamation_ratio > 0.25:
        issues.append(f"too_many_exclamations:{exclamation_ratio:.2f}")
    if reactive_hits and len(reactive_hits) > max(2, len(segments) // 5):
        issues.append(f"generic_reaction_phrases:{len(reactive_hits)}")
    if repeated_openers:
        issues.append("repetitive_sentence_openers")
    if target_words and word_ratio < 0.72:
        issues.append(f"narration_too_sparse:{word_ratio:.2f}")
    if target_words and word_ratio > 1.18:
        issues.append(f"narration_too_dense:{word_ratio:.2f}")
    if missing_ids:
        issues.append(f"missing_planned_scenes:{len(missing_ids)}")

    continuity_score = 1.0
    continuity_score -= min(0.35, len(missing_ids) * 0.05)
    continuity_score -= min(0.25, len(repeated_openers) * 0.05)
    continuity_score -= min(0.20, max(0.0, exclamation_ratio - 0.15))
    continuity_score = round(max(0.0, min(1.0, continuity_score)), 3)

    return {
        "segments": len(segments),
        "word_count": len(words),
        "target_word_count": target_words,
        "target_word_ratio": round(word_ratio, 3),
        "timeline_wpm": round(timeline_wpm, 1),
        "exclamation_ratio": round(exclamation_ratio, 3),
        "reactive_phrase_hits": reactive_hits,
        "repeated_sentence_openers": repeated_openers,
        "missing_scene_ids": missing_ids,
        "continuity_score": continuity_score,
        "issues": issues,
        "needs_rewrite": bool(issues),
    }
