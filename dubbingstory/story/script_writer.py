"""
dubbingstory.story.script_writer — Narration script generator

Generates narration scripts from storyboard using Gemini AI.
Supports multiple languages and narration styles.
"""

import json

from dubbingstory.vision.gemini_analyzer import GeminiVideoAnalyzer
from dubbingstory.story.prompt_templates import build_narration_prompt


def generate_narration(
    storyboard: dict,
    languages: list[str] = None,
    style: str = "viral_fb",
    cfg=None,
) -> dict[str, list[dict]]:
    """
    Generate narration scripts from a storyboard.

    Parameters
    ----------
    storyboard : dict
        Storyboard data from vision analysis.
    languages : list[str]
        Languages to generate ("id", "en").
    style : str
        Narration style key.
    cfg : SimpleNamespace
        Config object.

    Returns
    -------
    dict[str, list[dict]]
        Mapping of language → list of narration segments:
        {
            "id": [
                {"scene_id": "scene_001", "text": "...", "estimated_duration": 5.0},
            ],
            "en": [...]
        }
    """
    if languages is None:
        languages = ["id", "en"]

    api_key = getattr(cfg, "api_key_gemini", "")
    if not api_key:
        raise ValueError("❌ GOOGLE_API_KEY not found for narration generation.")

    model = getattr(cfg, "vision_gemini_model", "gemini-2.5-flash")
    fallback = getattr(cfg, "vision_gemini_fallback_model", "gemini-2.0-flash")
    hedging_threshold = getattr(cfg, "narration_hedging_threshold", 0.5)

    # Load narration styles
    styles = getattr(cfg, "narration_styles", {})
    style_config = styles.get(style, {})

    if not style_config:
        print(f"   ⚠️ Style '{style}' not found in config, using minimal defaults.")
        style_config = {
            "name": style,
            "tone": "neutral",
            "rules": ["Write naturally"],
            "examples": {"id": [], "en": []},
        }

    analyzer = GeminiVideoAnalyzer(
        api_key=api_key,
        model=model,
        fallback_model=fallback,
    )

    all_narrations: dict[str, list[dict]] = {}

    for lang in languages:
        print(f"\n   ✍️ Generating {lang.upper()} narration (style: {style})...")

        prompt = build_narration_prompt(
            storyboard=storyboard,
            language=lang,
            style=style,
            style_config=style_config,
            hedging_threshold=hedging_threshold,
        )

        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.6,  # Slightly creative for narration
            )

            response = analyzer.client.models.generate_content(
                model=model,
                contents=[prompt],
                config=config,
            )

            text = getattr(response, "text", None)
            if not text:
                raise ValueError("Empty response from Gemini.")

            segments = json.loads(text)

            # Validate and enrich with timing
            narration = _validate_narration(segments, storyboard)
            all_narrations[lang] = narration

            total_words = sum(s.get("word_count", 0) for s in narration)
            print(f"   ✅ {lang.upper()}: {len(narration)} segments, ~{total_words} words")

        except Exception as e:
            print(f"   ❌ {lang.upper()} narration failed: {e}")
            all_narrations[lang] = _generate_fallback_narration(storyboard, lang)

    return all_narrations


def _validate_narration(
    segments: list[dict],
    storyboard: dict,
) -> list[dict]:
    """Validate and enrich narration segments with timing data."""
    scene_map = {
        s["scene_id"]: s
        for s in storyboard.get("scenes", [])
    }

    validated = []
    for seg in segments:
        scene_id = seg.get("scene_id", "")
        scene = scene_map.get(scene_id, {})

        entry = {
            "scene_id": scene_id,
            "text": seg.get("text", ""),
            "word_count": seg.get("word_count", len(seg.get("text", "").split())),
            "estimated_duration": seg.get("estimated_duration", scene.get("duration", 5.0)),
            "start_time": scene.get("start_time", 0),
            "end_time": scene.get("end_time", 0),
            "importance": seg.get("importance", 0.5),
        }

        # Skip empty narrations
        if entry["text"].strip():
            validated.append(entry)

    return validated


def _generate_fallback_narration(
    storyboard: dict,
    language: str,
) -> list[dict]:
    """Generate minimal fallback narration when AI fails."""
    segments = []

    for scene in storyboard.get("scenes", []):
        scene_id = scene["scene_id"]
        action = scene.get("analysis", {}).get("action", "")
        context = scene.get("analysis", {}).get("likely_context", "")

        if language == "id":
            text = f"Pada bagian ini, {action.lower() if action else 'aktivitas berlangsung'}."
        else:
            text = f"In this segment, {action.lower() if action else 'activity takes place'}."

        segments.append({
            "scene_id": scene_id,
            "text": text,
            "word_count": len(text.split()),
            "estimated_duration": scene.get("duration", 5.0),
            "start_time": scene.get("start_time", 0),
            "end_time": scene.get("end_time", 0),
            "importance": 0.5,
        })

    return segments
