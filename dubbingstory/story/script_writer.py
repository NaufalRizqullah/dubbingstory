"""Continuity-aware narration script generator for DubbingStory v2."""

from __future__ import annotations

import json
from typing import Any

from dubbingstory.story.narration_planner import build_narration_plan
from dubbingstory.story.narration_qa import evaluate_narration
from dubbingstory.story.prompt_templates import build_rewrite_prompt, build_v2_narration_prompt
from dubbingstory.story.story_planner import fallback_story_artifacts


def _style_config(cfg, style: str) -> dict:
    styles = getattr(cfg, "narration_styles", {}) or {}
    value = styles.get(style, {}) or {}
    if value:
        return value
    return {
        "name": style,
        "tone": "natural, informative storytelling",
        "rules": [
            "Explain cause, process, change and consequence",
            "Avoid repetitive reaction phrases",
        ],
        "examples": {"id": [], "en": []},
    }


def _client(cfg):
    from dubbingstory.vision.gemini_analyzer import GeminiVideoAnalyzer

    api_key = getattr(cfg, "api_key_gemini", "")
    if not api_key:
        raise ValueError("❌ GOOGLE_API_KEY not found for narration generation.")
    model = getattr(cfg, "narration_model", None) or getattr(cfg, "story_model", None) or getattr(cfg, "vision_gemini_model", "gemini-2.5-flash")
    fallback = getattr(cfg, "vision_gemini_fallback_model", "gemini-2.0-flash")
    return GeminiVideoAnalyzer(api_key=api_key, model=model, fallback_model=fallback), model


def _call_json(analyzer, model: str, prompt: str, *, temperature: float = 0.45) -> Any:
    from google.genai import types

    response = analyzer.client.models.generate_content(
        model=model,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=temperature,
        ),
    )
    text = getattr(response, "text", None)
    if not text:
        raise ValueError("Empty response from Gemini")
    return json.loads(text)


def _extract_segment_list(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("segments", "narration", "items"):
            if isinstance(payload.get(key), list):
                return [item for item in payload[key] if isinstance(item, dict)]
    raise ValueError("Narration response must be a JSON array")


def _plan_map(narration_plan: dict) -> dict[str, dict]:
    return {
        str(item.get("scene_id")): item
        for item in narration_plan.get("segments", []) or []
        if item.get("scene_id")
    }


def _validate_narration(
    segments: list[dict],
    scenes: list[dict],
    narration_plan: dict,
    language: str = "id",
) -> list[dict]:
    scene_map = {str(scene.get("scene_id")): scene for scene in scenes}
    plan_map = _plan_map(narration_plan)
    supplied = {
        str(seg.get("scene_id")): seg
        for seg in segments
        if seg.get("scene_id")
    }

    validated: list[dict] = []
    for planned in narration_plan.get("segments", []) or []:
        scene_id = str(planned.get("scene_id", ""))
        scene = scene_map.get(scene_id, {})
        raw = supplied.get(scene_id, {})
        text = str(raw.get("text", "") or "").strip()
        covered_ids = [str(sid) for sid in planned.get("covers_scene_ids", [scene_id]) if sid]
        if not text:
            covered_scenes = [scene_map.get(sid, {}) for sid in covered_ids]
            text = _fallback_beat_text(covered_scenes, language)

        word_count = len(text.split())
        target_wpm = float(narration_plan.get("target_wpm", 160) or 160)
        estimated = word_count / target_wpm * 60.0 if target_wpm > 0 else 0.0
        validated.append(
            {
                "scene_id": scene_id,
                "covers_scene_ids": covered_ids,
                "text": text,
                "word_count": word_count,
                "estimated_duration": round(estimated, 3),
                "start_time": float(planned.get("timeline_start", scene.get("start_time", 0)) or 0),
                "end_time": float(planned.get("timeline_end", scene.get("end_time", 0)) or 0),
                "timeline_start": float(planned.get("timeline_start", 0) or 0),
                "timeline_end": float(planned.get("timeline_end", 0) or 0),
                "target_words": int(planned.get("target_words", word_count) or word_count),
                "min_words": int(planned.get("min_words", 0) or 0),
                "max_words": int(planned.get("max_words", max(word_count, 1)) or max(word_count, 1)),
                "story_role": planned.get("story_role", "process"),
                "importance": float(raw.get("importance", planned.get("story_importance", 0.5)) or 0.5),
            }
        )
    return validated


def _fallback_text(scene: dict, language: str = "id") -> str:
    analysis = scene.get("analysis", {}) or {}
    action = str(analysis.get("action", "") or "").strip()
    context = str(analysis.get("likely_context", "") or "").strip()
    changes = str(analysis.get("changes", "") or "").strip()
    if language == "id":
        pieces = [p for p in [context, action, changes] if p and "unable to analyze" not in p.lower()]
        if pieces:
            return ". ".join(p.rstrip(".") for p in pieces[:2]) + "."
        return "Proses berlanjut ke tahap berikutnya."
    pieces = [p for p in [context, action, changes] if p and "unable to analyze" not in p.lower()]
    if pieces:
        return ". ".join(p.rstrip(".") for p in pieces[:2]) + "."
    return "The process continues to the next step."


def _fallback_beat_text(scenes: list[dict], language: str = "id") -> str:
    pieces: list[str] = []
    for scene in scenes:
        analysis = scene.get("analysis", {}) or {}
        for value in (analysis.get("likely_context"), analysis.get("action"), analysis.get("changes")):
            text = str(value or "").strip()
            if text and "unable to analyze" not in text.lower() and text not in pieces:
                pieces.append(text)
    if pieces:
        return ". ".join(item.rstrip(".") for item in pieces[:3]) + "."
    return "Proses berlanjut ke tahap berikutnya." if language == "id" else "The process continues to the next step."


def _fallback_narration(scenes: list[dict], narration_plan: dict, language: str) -> list[dict]:
    scene_map = {str(scene.get("scene_id")): scene for scene in scenes}
    raw = []
    for planned in narration_plan.get("segments", []) or []:
        covered_ids = [str(sid) for sid in planned.get("covers_scene_ids", [planned.get("scene_id")]) if sid]
        covered = [scene_map.get(sid, {}) for sid in covered_ids]
        raw.append(
            {
                "scene_id": planned.get("scene_id"),
                "text": _fallback_beat_text(covered, language),
                "importance": planned.get("story_importance", 0.5),
            }
        )
    return _validate_narration(raw, scenes, narration_plan, language)


def _maybe_rewrite(
    narration: list[dict],
    *,
    language: str,
    narration_plan: dict,
    story_plan: dict,
    analyzer,
    model: str,
    cfg,
    scenes: list[dict],
) -> tuple[list[dict], dict]:
    qa = evaluate_narration(narration, narration_plan, language=language)
    if not qa.get("needs_rewrite") or not bool(getattr(cfg, "narration_auto_rewrite", True)):
        return narration, qa

    try:
        prompt = build_rewrite_prompt(
            current_segments=[
                {"scene_id": item["scene_id"], "text": item["text"], "importance": item.get("importance", 0.5)}
                for item in narration
            ],
            qa=qa,
            narration_plan=narration_plan,
            story_plan=story_plan,
            language=language,
        )
        revised_raw = _extract_segment_list(_call_json(analyzer, model, prompt, temperature=0.25))
        revised = _validate_narration(revised_raw, scenes, narration_plan, language)
        revised_qa = evaluate_narration(revised, narration_plan, language=language)

        # Keep rewrite only when it improves issue count or word-budget accuracy.
        before = (len(qa.get("issues", [])), abs(1.0 - float(qa.get("target_word_ratio", 1.0))))
        after = (len(revised_qa.get("issues", [])), abs(1.0 - float(revised_qa.get("target_word_ratio", 1.0))))
        if after <= before:
            print(f"   🧪 {language.upper()} QA rewrite applied: {qa.get('issues', [])} -> {revised_qa.get('issues', [])}")
            return revised, revised_qa
    except Exception as exc:
        print(f"   ⚠️ {language.upper()} QA rewrite failed: {exc}")
    return narration, qa


def _generate(
    *,
    storyboard: dict,
    scenes: list[dict],
    languages: list[str],
    style: str,
    cfg,
    mode: str,
    story_plan: dict | None,
    story_memory: dict | None,
    narration_plan: dict | None,
) -> dict[str, list[dict]]:
    fallback_plan, fallback_memory = fallback_story_artifacts(storyboard)
    story_plan = story_plan or fallback_plan
    story_memory = story_memory or fallback_memory
    narration_plan = narration_plan or build_narration_plan(
        storyboard,
        scenes,
        story_plan=story_plan,
        story_memory=story_memory,
        cfg=cfg,
        mode=mode,
    )

    style_config = _style_config(cfg, style)
    hedging_threshold = float(getattr(cfg, "narration_hedging_threshold", 0.5) or 0.5)
    analyzer, model = _client(cfg)
    result: dict[str, list[dict]] = {}

    for language in languages:
        print(f"\n   ✍️ Generating {language.upper()} {mode.upper()} narration (story-aware)...")
        prompt = build_v2_narration_prompt(
            scenes=scenes,
            narration_plan=narration_plan,
            story_plan=story_plan,
            story_memory=story_memory,
            language=language,
            style=style,
            style_config=style_config,
            hedging_threshold=hedging_threshold,
            mode=mode,
        )
        try:
            raw = _extract_segment_list(_call_json(analyzer, model, prompt, temperature=0.45))
            narration = _validate_narration(raw, scenes, narration_plan, language)
            narration, qa = _maybe_rewrite(
                narration,
                language=language,
                narration_plan=narration_plan,
                story_plan=story_plan,
                analyzer=analyzer,
                model=model,
                cfg=cfg,
                scenes=scenes,
            )
            result[language] = narration
            print(
                f"   ✅ {language.upper()}: {len(narration)} beats, "
                f"{sum(s['word_count'] for s in narration)} words, "
                f"QA issues={len(qa.get('issues', []))}"
            )
        except Exception as exc:
            print(f"   ❌ {language.upper()} narration failed: {exc}")
            result[language] = _fallback_narration(scenes, narration_plan, language)

    return result


def generate_narration(
    storyboard: dict,
    languages: list[str] | None = None,
    style: str = "viral_fb",
    cfg=None,
    *,
    story_plan: dict | None = None,
    story_memory: dict | None = None,
    narration_plan: dict | None = None,
) -> dict[str, list[dict]]:
    if languages is None:
        languages = ["id", "en"]
    return _generate(
        storyboard=storyboard,
        scenes=storyboard.get("scenes", []) or [],
        languages=languages,
        style=style,
        cfg=cfg,
        mode="full",
        story_plan=story_plan,
        story_memory=story_memory,
        narration_plan=narration_plan,
    )


def generate_summary_narration(
    storyboard: dict,
    selected_scenes: list[dict],
    languages: list[str] | None = None,
    style: str = "viral_fb",
    cfg=None,
    *,
    story_plan: dict | None = None,
    story_memory: dict | None = None,
    narration_plan: dict | None = None,
) -> dict[str, list[dict]]:
    if languages is None:
        languages = ["id", "en"]
    return _generate(
        storyboard=storyboard,
        scenes=selected_scenes,
        languages=languages,
        style=style,
        cfg=cfg,
        mode="summary",
        story_plan=story_plan,
        story_memory=story_memory,
        narration_plan=narration_plan,
    )
