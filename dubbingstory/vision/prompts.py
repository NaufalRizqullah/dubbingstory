"""
dubbingstory.vision.prompts — Prompt templates for visual analysis

Anti-hallucination focused prompts for scene understanding.
"""


SCENE_ANALYSIS_PROMPT = """You are analyzing a video scene from a {domain} video.
You are given {n_frames} keyframes extracted from a single scene spanning {time_range}.

CRITICAL RULES:
- Do NOT invent facts that are not visible in the frames
- Use hedging language when uncertain: "appears to be...", "likely...", "seems to..."
- If you cannot determine something, explicitly say so
- Focus on what IS visible, not what you think SHOULD be there

{subtitle_context}

Analyze and describe (KEEP ALL DESCRIPTIONS EXTREMELY BRIEF, 1 short sentence max):
1. visible_objects: List ALL visible objects, tools, materials, text/signs
2. people: Number of people, their clothing, posture, activity
3. action: What is happening in this scene
4. changes: What changed between the first and last frame
5. environment: Indoor/outdoor, lighting, workspace type
6. likely_context: What step in a larger process this might represent
7. text_visible: Any text, signs, labels, or watermarks visible
8. confidence: Your overall confidence in this analysis (0.0 to 1.0)

{context_hint}

Return ONLY a valid JSON object (no markdown, no explanation):
{{
  "scene_id": "{scene_id}",
  "time_range": "{time_range}",
  "visible_objects": ["object1", "object2"],
  "people": "description of people present",
  "action": "what is happening",
  "changes": "what changed between frames",
  "environment": "description of setting",
  "likely_context": "what step this represents",
  "text_visible": ["any text seen"],
  "confidence": 0.0
}}"""

CHEAP_SCENE_ANALYSIS_PROMPT = """You are screening a video scene from a {domain} video.
You are given {n_frames} keyframes spanning {time_range}.

Briefly describe the scene and score its visual salience (how important or distinct this scene is) from 0.0 to 1.0.

Return ONLY a valid JSON object:
{{
  "scene_id": "{scene_id}",
  "action": "brief description of action",
  "visual_change": 0.8,
  "salience": 0.9,
  "confidence": 0.9
}}"""


TEMPORAL_FLOW_PROMPT = """You are building a narrative understanding of a complete video.
Below are per-scene analyses from a {domain} video with {n_scenes} scenes.

Each scene has been independently analyzed with keyframes. Your job is to:
1. Understand the TEMPORAL FLOW — what happens from start to finish
2. Identify the NARRATIVE ARC — beginning, process, climax, result
3. Add INTER-SCENE RELATIONSHIPS — how scenes connect to each other
4. Rate NARRATIVE IMPORTANCE — which scenes are most important to narrate

{subtitle_context}

Scene analyses:
{scene_analyses_json}

Return ONLY a valid JSON object:
{{
  "video_summary": "One paragraph summary of the entire video",
  "narrative_arc": "type of arc (repair_process, cooking, tutorial, event, story, other)",
  "domain": "detected domain (workshop, kitchen, outdoor, etc.)",
  "scenes_enriched": [
    {{
      "scene_id": "scene_001",
      "narrative_role": "introduction | setup | process | climax | result | conclusion",
      "connects_to_next": "how this scene leads into the next",
      "narration_importance": 0.0,
      "narration_cue": "What should the narrator say about this scene"
    }}
  ]
}}"""


DOMAIN_HINTS = {
    "workshop": "This is a mechanical workshop/repair video. Look for tools, machinery, metal parts, welding, lathe work, grinding, and repair processes.",
    "cooking": "This is a cooking/food preparation video. Look for ingredients, kitchen tools, cooking techniques, and food transformation.",
    "repair": "This is a repair/restoration video. Look for broken parts, repair tools, disassembly, cleaning, reassembly, and before/after states.",
    "construction": "This is a construction/building video. Look for building materials, heavy equipment, structural work, and construction progress.",
    "tutorial": "This is an instructional/tutorial video. Look for step-by-step demonstrations, tools being explained, and learning sequences.",
    "general": "Analyze the video content as accurately as possible based on what you observe.",
}


def build_scene_prompt(
    scene_id: str,
    time_range: str,
    n_frames: int,
    domain: str = "general",
    context_hint: str = "",
    subtitle_text: str = "",
) -> str:
    """Build a scene analysis prompt with all context."""
    domain_text = DOMAIN_HINTS.get(domain, DOMAIN_HINTS["general"])

    subtitle_section = ""
    if subtitle_text:
        subtitle_section = (
            f"\nADDITIONAL CONTEXT — Subtitle/dialog text from this scene:\n"
            f"```\n{subtitle_text}\n```\n"
            f"Use this text to better understand what is happening, "
            f"but still describe what you SEE in the frames."
        )

    hint_section = ""
    if context_hint:
        hint_section = f"\nDOMAIN CONTEXT: {domain_text}\nADDITIONAL HINT: {context_hint}"
    elif domain != "general":
        hint_section = f"\nDOMAIN CONTEXT: {domain_text}"

    return SCENE_ANALYSIS_PROMPT.format(
        domain=domain,
        n_frames=n_frames,
        time_range=time_range,
        scene_id=scene_id,
        subtitle_context=subtitle_section,
        context_hint=hint_section,
    )


def build_cheap_scene_prompt(
    scene_id: str,
    time_range: str,
    n_frames: int,
    domain: str = "general",
) -> str:
    """Build a cheap scene screening prompt."""
    return CHEAP_SCENE_ANALYSIS_PROMPT.format(
        domain=domain,
        n_frames=n_frames,
        time_range=time_range,
        scene_id=scene_id,
    )


def build_temporal_prompt(
    scene_analyses: list[dict],
    domain: str = "general",
    subtitle_context: str = "",
) -> str:
    """Build the temporal flow analysis prompt.

    To avoid context-window overflows, produce a compact summary of each
    per-scene analysis rather than embedding full verbose JSON. Keep the
    essential fields (scene_id, short action/description, confidence, and
    a few visible objects) and truncate long strings.
    """
    import json

    subtitle_section = ""
    if subtitle_context:
        # Truncate if too long
        if len(subtitle_context) > 3000:
            subtitle_context = subtitle_context[:3000] + "\n... (truncated)"
        subtitle_section = (
            f"\nFULL VIDEO SUBTITLE CONTEXT:\n```\n{subtitle_context}\n```"
        )

    condensed_scenes = []
    for s in scene_analyses:
        c = {
            "id": s.get("scene_id"),
            "action": s.get("action"),
            "changes": s.get("changes"),
            "context": s.get("likely_context"),
        }
        if s.get("words"):
            c["words"] = s.get("words")
        condensed_scenes.append(c)

    return TEMPORAL_FLOW_PROMPT.format(
        domain=domain,
        n_scenes=len(scene_analyses),
        subtitle_context=subtitle_section,
        scene_analyses_json=json.dumps(condensed_scenes, indent=2, ensure_ascii=False),
    )
