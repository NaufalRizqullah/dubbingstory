"""Lightweight language-contract checks for narration generation.

This module intentionally avoids heavyweight language-detection dependencies.
It is not a general language classifier; it only catches obvious cases where a
requested Indonesian narration is returned in English (or vice versa).
"""

from __future__ import annotations

import re
from typing import Any


LANGUAGE_LABELS = {
    "id": "Bahasa Indonesia",
    "en": "English",
}

# Function/common words are more reliable than technical nouns for detecting a
# whole-response language mismatch. Keep these sets deliberately conservative.
_ID_MARKERS = {
    "yang", "dan", "untuk", "dengan", "dari", "pada", "ini", "itu",
    "sedang", "kemudian", "karena", "setelah", "sebelum", "agar", "sebagai",
    "akan", "sudah", "masih", "saat", "ketika", "hingga", "lalu", "namun",
    "bagian", "komponen", "proses", "dilakukan", "diperiksa", "dikerjakan",
    "dipasang", "diperbaiki", "dibuat", "menjadi", "tersebut", "berikutnya",
}

_EN_MARKERS = {
    "the", "and", "to", "of", "is", "are", "was", "were", "being", "with",
    "for", "from", "this", "that", "then", "because", "after", "before",
    "while", "into", "through", "during", "will", "has", "have", "had",
    "component", "process", "workpiece", "repair", "being", "next", "finally",
    "needs", "needed", "using", "used", "now", "once", "until", "where",
}

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", flags=re.UNICODE)


def language_label(language: str) -> str:
    return LANGUAGE_LABELS.get(str(language or "").lower(), str(language or ""))


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


def language_mismatch_reason(text: str, expected_language: str) -> str | None:
    """Return a reason when *text* is clearly dominated by the wrong language.

    The thresholds are intentionally conservative to allow product names,
    technical terms, and proper nouns from another language.
    """
    expected = str(expected_language or "").lower()
    if expected not in {"id", "en"}:
        return None

    tokens = _tokens(text)
    if len(tokens) < 6:
        return None

    id_hits = sum(token in _ID_MARKERS for token in tokens)
    en_hits = sum(token in _EN_MARKERS for token in tokens)
    token_count = len(tokens)

    if expected == "id":
        foreign_hits, expected_hits = en_hits, id_hits
        foreign_name = "English"
        expected_name = "Bahasa Indonesia"
    else:
        foreign_hits, expected_hits = id_hits, en_hits
        foreign_name = "Bahasa Indonesia"
        expected_name = "English"

    min_foreign_hits = 4 if token_count < 40 else max(5, int(token_count * 0.08))
    clearly_dominant = (
        foreign_hits >= min_foreign_hits
        and foreign_hits >= expected_hits + 2
        and foreign_hits >= max(4, int(expected_hits * 1.5))
    )
    if not clearly_dominant:
        return None

    return (
        f"language mismatch: expected {expected_name} ({expected}), but output appears "
        f"dominated by {foreign_name} markers "
        f"(expected_hits={expected_hits}, foreign_hits={foreign_hits}, tokens={token_count})"
    )


def _extract_text_values(payload: Any) -> list[str]:
    """Extract narration ``text`` values from accepted response shapes."""
    items: list[Any] = []
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        for key in ("segments", "narration", "items"):
            if isinstance(payload.get(key), list):
                items = payload[key]
                break
        if not items and isinstance(payload.get("text"), str):
            items = [payload]

    texts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        value = str(item.get("text", "") or "").strip()
        if value:
            texts.append(value)
    return texts


def payload_language_mismatch_reason(payload: Any, expected_language: str) -> str | None:
    """Validate narration payload language, per segment and across the response."""
    texts = _extract_text_values(payload)
    if not texts:
        return None

    # Catch a clearly wrong individual segment first so a mixed response is also
    # retried instead of being accepted because the aggregate happens to balance.
    for index, text in enumerate(texts):
        reason = language_mismatch_reason(text, expected_language)
        if reason:
            return f"segment[{index}] {reason}"

    return language_mismatch_reason(" ".join(texts), expected_language)


def narration_language_issue(segments: list[dict], expected_language: str) -> str | None:
    """QA-friendly wrapper for already-normalized narration segments."""
    return payload_language_mismatch_reason(segments, expected_language)
