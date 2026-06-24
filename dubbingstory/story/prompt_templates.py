"""
dubbingstory.story.prompt_templates — LLM prompts for narration generation

Generates narration scripts in multiple languages and styles
from the storyboard.json data.
"""


NARRATION_PROMPT = """You are a professional video narrator creating a {style_name} narration.

VIDEO CONTEXT:
Title: {video_title}
Summary: {video_summary}
Narrative arc: {narrative_arc}
Domain: {domain}
Total duration: {total_duration}s

NARRATION STYLE:
Tone: {tone}
Language: {language_label}
Rules:
{rules}

EXAMPLES of this style:
{examples}

SCENES TO NARRATE:
{scenes_json}

CRITICAL RULES:
1. Write narration text for EACH scene based on its visual analysis and narrative cue
2. Narration must match the scene duration — roughly 2-3 words per second
3. Scene duration is provided; keep narration proportional
4. Use hedging language for low-confidence scenes (confidence < {hedging_threshold}):
   - Indonesian: "terlihat...", "kemungkinan...", "tampaknya..."
   - English: "appears to...", "what seems to be...", "likely..."
5. Narration should FLOW naturally from one scene to the next
6. Do NOT describe UI elements, watermarks, or video metadata
7. Focus on the STORY, not frame-by-frame description
8. For scenes with very low importance (< 0.3), keep narration brief or skip
9. Every narration segment must be standalone meaningful

Return ONLY a valid JSON array:
[
  {{
    "scene_id": "scene_001",
    "text": "The narration text for this scene...",
    "word_count": 15,
    "estimated_duration": 5.0,
    "importance": 0.8,
    "notes": "Any special notes for TTS"
  }}
]"""


def build_narration_prompt(
    storyboard: dict,
    language: str,
    style: str,
    style_config: dict,
    hedging_threshold: float = 0.5,
) -> str:
    """
    Build the narration generation prompt.

    Parameters
    ----------
    storyboard : dict
        Storyboard data.
    language : str
        Target language code ("id" or "en").
    style : str
        Narration style key.
    style_config : dict
        Style definition from narration_styles.yaml.
    hedging_threshold : float
        Confidence threshold below which to use hedging language.
    """
    import json

    language_labels = {
        "id": "Bahasa Indonesia",
        "en": "English",
    }

    # Build rules string
    rules = style_config.get("rules", [])
    rules_str = "\n".join(f"- {r}" for r in rules)

    # Build examples
    examples_list = style_config.get("examples", {}).get(language, [])
    examples_str = "\n".join(f'  "{ex}"' for ex in examples_list)

    # Build scenes JSON (simplified for prompt)
    scenes_for_prompt = []
    for scene in storyboard.get("scenes", []):
        scenes_for_prompt.append({
            "scene_id": scene["scene_id"],
            "start_time": scene["start_time"],
            "end_time": scene["end_time"],
            "duration": scene["duration"],
            "action": scene.get("analysis", {}).get("action", ""),
            "changes": scene.get("analysis", {}).get("changes", ""),
            "likely_context": scene.get("analysis", {}).get("likely_context", ""),
            "confidence": scene.get("analysis", {}).get("confidence", 0),
            "narrative_role": scene.get("narrative", {}).get("role", ""),
            "narration_cue": scene.get("narrative", {}).get("cue", ""),
            "importance": scene.get("narrative", {}).get("importance", 0.5),
        })

    return NARRATION_PROMPT.format(
        style_name=style_config.get("name", style),
        video_title=storyboard.get("video_title", ""),
        video_summary=storyboard.get("video_summary", ""),
        narrative_arc=storyboard.get("narrative_arc", ""),
        domain=storyboard.get("domain", ""),
        total_duration=storyboard.get("total_duration", 0),
        tone=style_config.get("tone", ""),
        language_label=language_labels.get(language, language),
        rules=rules_str,
        examples=examples_str,
        scenes_json=json.dumps(scenes_for_prompt, indent=2, ensure_ascii=False),
        hedging_threshold=hedging_threshold,
    )
