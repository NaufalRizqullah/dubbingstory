"""Story-first beat planning for DubbingStory V2.1.

A beat describes *what the recap should communicate* before footage is fixed.
Semantic matching later answers which scene window can prove the beat.
"""

from __future__ import annotations

import json
import math
from typing import Any

from dubbingstory.story.beat_dedup import deduplicate_beats
from dubbingstory.story.gemini_json import generate_story_json


BEAT_PLANNER_VERSION = "v2.1.0"

BEAT_PLANNER_PROMPT = """You are planning story beats for a video recap.
Do NOT write final narration and do NOT assign footage.

GLOBAL STORY PLAN:
{story_plan_json}

STORY MEMORY / FACTS:
{story_memory_json}

CHRONOLOGICAL EVIDENCE SUMMARY:
{evidence_json}

TARGET RECAP DURATION: {target_duration:.1f} seconds
TARGET BEAT COUNT: {target_count}

Return ONLY a JSON array. Each item must have:
{{
  "beat_id": "beat_001",
  "story_role": "setup|problem|development|complication|turning_point|climax|resolution|bridge",
  "intent": "what this beat should accomplish",
  "required_facts": ["specific evidence-supported fact"],
  "forbidden_claims": ["claim that must not be invented"],
  "approx_ratio": 0.0,
  "importance": 0.0,
  "target_duration_sec": 10.0
}}

Rules:
- Return about {target_count} beats (minimum 3 when the story has enough material).
- Keep story order; approx_ratio must be non-decreasing from 0 to 1.
- required_facts must come from the supplied plan/evidence; do not invent facts.
- A beat is a meaningful story thought, not one beat per camera cut.
- Merge repetitive process steps and preserve problem -> progression -> payoff.
- Do not put scene IDs in intent/required_facts. Footage assignment happens later.
- If evidence is uncertain, make the beat safer/generic rather than inventing a cause or outcome.
"""


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _target_count(target_duration: float, cfg) -> int:
    beat_duration = max(6.0, float(getattr(cfg, "semantic_match_target_beat_duration", 11.0) or 11.0))
    minimum = max(3, int(getattr(cfg, "semantic_match_min_beats", 5) or 5))
    maximum = max(minimum, int(getattr(cfg, "semantic_match_max_beats", 20) or 20))
    return max(minimum, min(maximum, int(round(max(1.0, target_duration) / beat_duration))))


def _compact_evidence(signatures: list[dict], max_chars: int = 42000) -> list[dict]:
    items: list[dict] = []
    chars = 2
    for sig in sorted(signatures, key=lambda s: float(s.get("start_time", 0) or 0)):
        # Deep evidence is preferred. Keep screening-only entries out of the LLM
        # beat prompt because they are intentionally weak and may be promoted later.
        if not sig.get("deep_evidence", False):
            continue
        item = {
            "scene_id": sig.get("scene_id"),
            "ratio_hint": sig.get("start_time", 0),
            "evidence": str(sig.get("signature", "") or "")[:1200],
            "confidence": sig.get("confidence", 0),
        }
        rendered = json.dumps(item, ensure_ascii=False)
        if items and chars + len(rendered) > max_chars:
            break
        items.append(item)
        chars += len(rendered)
    return items


def _signature_fact(sig: dict) -> str:
    text = str(sig.get("signature", "") or "")
    for line in text.splitlines():
        if line.startswith(("ACTION:", "CHANGE:", "NEW_INFO:", "STATE_AFTER:")):
            value = line.split(":", 1)[1].strip()
            if value:
                return value
    return text.splitlines()[0].split(":", 1)[-1].strip() if text else ""


def _fallback_beats(
    story_plan: dict,
    signatures: list[dict],
    target_duration: float,
    target_count: int,
) -> list[dict]:
    deep = [s for s in signatures if s.get("deep_evidence", False) and str(s.get("signature", "") or "").strip()]
    deep.sort(key=lambda s: float(s.get("start_time", 0) or 0))
    if not deep:
        return []

    count = min(target_count, len(deep))
    if count <= 1:
        picks = [deep[0]]
    else:
        picks = []
        for i in range(count):
            pos = i * (len(deep) - 1) / max(1, count - 1)
            picks.append(deep[int(round(pos))])

    role_map = {
        str(item.get("scene_id")): item
        for item in story_plan.get("scene_roles", []) or []
        if isinstance(item, dict) and item.get("scene_id")
    }
    role_alias = {
        "process": "development",
        "context": "setup",
        "result": "resolution",
        "conclusion": "resolution",
    }

    beats: list[dict] = []
    duration_each = target_duration / max(1, len(picks))
    max_end = max(float(s.get("end_time", 0) or 0) for s in deep) or 1.0
    for index, sig in enumerate(picks):
        sid = str(sig.get("scene_id", ""))
        meta = role_map.get(sid, {}) or {}
        role = str(meta.get("role", "development") or "development")
        role = role_alias.get(role, role)
        fact = _signature_fact(sig)
        reason = str(meta.get("reason", "") or "").strip()
        intent = reason or (f"Explain the progression: {fact}" if fact else "Advance the main story with supported evidence")
        beats.append(
            {
                "beat_id": f"beat_{index + 1:03d}",
                "beat_index": index,
                "story_role": role,
                "intent": intent,
                "required_facts": [fact] if fact else [],
                "forbidden_claims": [],
                "approx_ratio": round(_clamp01(float(sig.get("start_time", 0) or 0) / max_end), 3),
                "importance": round(_clamp01(meta.get("story_importance", sig.get("confidence", 0.5)), 0.5), 3),
                "target_duration_sec": round(duration_each, 2),
                "planner_source": "deterministic_fallback",
            }
        )
    return beats


def _normalize(raw: Any, target_duration: float, target_count: int) -> list[dict]:
    if isinstance(raw, dict):
        raw = raw.get("beats", raw.get("items", []))
    if not isinstance(raw, list):
        return []

    beats: list[dict] = []
    last_ratio = 0.0
    default_duration = target_duration / max(1, min(target_count, len(raw) or target_count))
    for index, item in enumerate(raw[: max(3, target_count + 4)]):
        if not isinstance(item, dict):
            continue
        intent = str(item.get("intent", "") or "").strip()
        facts = [str(v).strip() for v in item.get("required_facts", []) or [] if str(v).strip()][:6]
        if not intent and not facts:
            continue
        ratio = max(last_ratio, _clamp01(item.get("approx_ratio", index / max(1, target_count - 1))))
        last_ratio = ratio
        beats.append(
            {
                "beat_id": f"beat_{len(beats) + 1:03d}",
                "beat_index": len(beats),
                "story_role": str(item.get("story_role", "development") or "development"),
                "intent": intent or "Advance the supported story progression",
                "required_facts": facts,
                "forbidden_claims": [str(v).strip() for v in item.get("forbidden_claims", []) or [] if str(v).strip()][:6],
                "approx_ratio": round(ratio, 3),
                "importance": round(_clamp01(item.get("importance", 0.6), 0.6), 3),
                "target_duration_sec": round(max(4.0, float(item.get("target_duration_sec", default_duration) or default_duration)), 2),
                "planner_source": "gemini",
            }
        )
    return beats


def plan_story_beats(
    *,
    story_plan: dict,
    story_memory: dict,
    signatures: list[dict],
    target_duration: float,
    cfg,
) -> list[dict]:
    """Plan duration-aware story beats and deduplicate them before matching."""
    target_duration = max(20.0, float(target_duration or 0) or 60.0)
    target_count = _target_count(target_duration, cfg)
    beats: list[dict] = []

    if bool(getattr(cfg, "semantic_match_beat_planner_llm", True)) and getattr(cfg, "api_key_gemini", ""):
        evidence = _compact_evidence(signatures)
        prompt = BEAT_PLANNER_PROMPT.format(
            story_plan_json=json.dumps(story_plan, ensure_ascii=False, indent=2)[:26000],
            story_memory_json=json.dumps(story_memory, ensure_ascii=False, indent=2)[:14000],
            evidence_json=json.dumps(evidence, ensure_ascii=False, indent=2),
            target_duration=target_duration,
            target_count=target_count,
        )
        try:
            raw = generate_story_json(
                cfg,
                prompt,
                task_name="v21-beat-planner",
                temperature=0.2,
                model_attr="story_model",
            )
            beats = _normalize(raw, target_duration, target_count)
        except Exception as exc:
            print(f"   ⚠️ V2.1 beat planner LLM failed; deterministic fallback: {exc}")

    if not beats:
        beats = _fallback_beats(story_plan, signatures, target_duration, target_count)

    threshold = float(getattr(cfg, "semantic_match_beat_dedup_threshold", 0.90) or 0.90)
    beats = deduplicate_beats(beats, threshold=threshold)
    for beat in beats:
        beat["planner_version"] = BEAT_PLANNER_VERSION
    return beats
