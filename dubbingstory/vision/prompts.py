"""Prompt templates for factual visual analysis and temporal understanding."""


SCENE_ANALYSIS_PROMPT = """You are analyzing one video scene from a {domain} video.
You are given {n_frames} keyframes spanning {time_range}.

Your output feeds a separate story planner. Be FACTUAL and COMPACT.

CRITICAL RULES:
- Never invent facts that are not visible or stated in the supplied subtitle/dialogue.
- Subtitle/dialogue can clarify meaning, goals, or object purpose, but do not pretend
  those facts were visually observed.
- If uncertain, use "unknown" or a short hedged phrase.
- Keep every string to one short sentence; visible_objects/text_visible max 8 items.
- Do not write narration, hype, hooks, or audience reactions.

{subtitle_context}
{context_hint}

Extract:
1. visible_objects: important visible objects/tools/materials only
2. people: stable visual description useful for continuity
3. action: concrete action happening now
4. changes: state change from first to last frame
5. environment: setting/workspace
6. likely_context: supported step in the larger process
7. character_goal: immediate goal only when supported by evidence
8. state_before / state_after: concise state transition
9. cause / effect: only when evidence supports the relationship
10. unresolved_question: what remains unclear/unfinished after this scene
11. text_visible: useful visible text only
12. confidence: 0.0 to 1.0

Return ONLY valid JSON:
{{
  "scene_id": "{scene_id}",
  "time_range": "{time_range}",
  "visible_objects": ["object1"],
  "people": "brief stable description",
  "action": "concrete action",
  "changes": "state change",
  "environment": "setting",
  "likely_context": "supported process step",
  "character_goal": "goal or unknown",
  "state_before": "before or unknown",
  "state_after": "after or unknown",
  "cause": "cause or unknown",
  "effect": "effect or unknown",
  "unresolved_question": "question or empty",
  "text_visible": [],
  "confidence": 0.0
}}"""


CHEAP_SCENE_ANALYSIS_PROMPT = """You are screening one scene from a {domain} video.
You are given {n_frames} keyframes spanning {time_range}.

{subtitle_context}

Briefly identify the action and score how useful this scene may be for understanding
the complete process/story. Transcript/dialogue can raise story relevance even when
the visuals are ordinary. Do not reward visual spectacle alone.

Return ONLY valid JSON:
{{
  "scene_id": "{scene_id}",
  "action": "brief action",
  "visual_change": 0.0,
  "story_relevance": 0.0,
  "salience": 0.0,
  "confidence": 0.0
}}"""


TEMPORAL_FLOW_PROMPT = """You are building a temporal understanding of a {domain} video.
Below are factual per-scene analyses from {n_scenes} scenes.

Do not write final narration. Determine:
1. what progresses from start to finish,
2. how one step causes/enables the next,
3. which scenes are setup/process/bridge/problem/climax/result,
4. which scenes are necessary for continuity even if visually ordinary.

{subtitle_context}

Scene analyses:
{scene_analyses_json}

Return ONLY valid JSON:
{{
  "video_summary": "factual paragraph describing the complete flow",
  "narrative_arc": "repair_process|cooking|tutorial|event|story|other",
  "domain": "detected domain",
  "scenes_enriched": [
    {{
      "scene_id": "scene_001",
      "narrative_role": "introduction|setup|problem|process|bridge|climax|result|conclusion",
      "connects_to_next": "causal/progression link",
      "narration_importance": 0.0,
      "causal_importance": 0.0,
      "bridge_importance": 0.0,
      "narration_cue": "fact/relationship a later writer should explain"
    }}
  ]
}}"""


DOMAIN_HINTS = {
    "workshop": "Mechanical workshop/repair context: tools, machining, measurements, joining, assembly, testing.",
    "cooking": "Cooking context: ingredients, tools, techniques, transformations, doneness, plating.",
    "repair": "Repair/restoration context: damage, diagnosis, disassembly, fabrication, cleaning, reassembly, testing.",
    "construction": "Construction context: materials, equipment, structural steps, dependencies, progress.",
    "tutorial": "Instructional context: step-by-step demonstration and why each step is needed.",
    "general": "Analyze only what the supplied evidence supports.",
}


def build_scene_prompt(
    scene_id: str,
    time_range: str,
    n_frames: int,
    domain: str = "general",
    context_hint: str = "",
    subtitle_text: str = "",
) -> str:
    domain_text = DOMAIN_HINTS.get(domain, DOMAIN_HINTS["general"])

    subtitle_section = ""
    if subtitle_text:
        subtitle_section = (
            "ADDITIONAL EVIDENCE — subtitle/dialogue overlapping this scene:\n"
            f"```\n{subtitle_text[:1200]}\n```\n"
            "Use it as semantic evidence, not as permission to invent visuals."
        )

    hint_section = f"DOMAIN CONTEXT: {domain_text}"
    if context_hint:
        hint_section += f"\nVIDEO CONTEXT HINT: {context_hint[:1200]}"

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
    subtitle_text: str = "",
) -> str:
    subtitle_context = "No transcript/dialogue available for this scene."
    if subtitle_text:
        subtitle_context = (
            "Transcript/dialogue evidence:\n"
            + subtitle_text[:600].replace("\n", " ")
        )
    return CHEAP_SCENE_ANALYSIS_PROMPT.format(
        domain=domain,
        n_frames=n_frames,
        time_range=time_range,
        scene_id=scene_id,
        subtitle_context=subtitle_context,
    )


def build_temporal_prompt(
    scene_analyses: list[dict],
    domain: str = "general",
    subtitle_context: str = "",
) -> str:
    import json

    subtitle_section = ""
    if subtitle_context:
        if len(subtitle_context) > 5000:
            subtitle_context = subtitle_context[:5000] + "\n... (truncated)"
        subtitle_section = f"FULL/AVAILABLE TRANSCRIPT CONTEXT:\n```\n{subtitle_context}\n```"

    condensed_scenes = []
    for scene in scene_analyses:
        condensed_scenes.append(
            {
                "id": scene.get("scene_id"),
                "action": scene.get("action", ""),
                "changes": scene.get("changes", ""),
                "context": scene.get("likely_context", ""),
                "goal": scene.get("character_goal", ""),
                "before": scene.get("state_before", ""),
                "after": scene.get("state_after", ""),
                "cause": scene.get("cause", ""),
                "effect": scene.get("effect", ""),
                "transcript": scene.get("transcript_text", ""),
                "confidence": scene.get("confidence", 0),
            }
        )

    return TEMPORAL_FLOW_PROMPT.format(
        domain=domain,
        n_scenes=len(scene_analyses),
        subtitle_context=subtitle_section,
        scene_analyses_json=json.dumps(condensed_scenes, ensure_ascii=False, indent=2),
    )
