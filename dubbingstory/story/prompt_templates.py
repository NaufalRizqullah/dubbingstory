"""Prompt builders for continuity-aware narration."""

from __future__ import annotations

import json

from dubbingstory.story.action_anchors import compact_visual_events, speech_part_bindings


V2_NARRATION_PROMPT = """You are a professional video storyteller.

The viewer must understand the STORY, not receive a mechanical description of
frames. Explain what is happening, why it matters, what changes, and how the
current beat connects to what came before. Use only supplied evidence.

MODE: {mode}
OUTPUT_LANGUAGE_CODE: {language_code}
OUTPUT_LANGUAGE_NAME: {language_label}
STYLE NAME: {style_name}
STYLE TONE: {tone}
LANGUAGE STYLE:
{language_style}

STYLE EXAMPLES (rhythm/structure reference only; never copy unsupported facts):
{examples}

GLOBAL STORY PLAN:
{story_plan_json}

STORY MEMORY / FACTS THAT MUST STAY CONSISTENT:
{story_memory_json}

STYLE RULES (secondary to factuality and continuity):
{rules}

SCENES + NARRATION PLAN, in final playback order:
{scenes_json}

HARD RULES:
0. LANGUAGE CONTRACT IS NON-NEGOTIABLE. The required output language is
   {language_label} with code `{language_code}`. Every spoken value (`text` or each
   `speech_parts` string) MUST be written
   entirely in that language. Source evidence, titles, story-plan fields, or retry
   context may be English; they are INPUT ONLY and MUST NOT change the output language.
   If the required language is Bahasa Indonesia (`id`), translate English evidence
   into natural Bahasa Indonesia instead of copying English sentences. Proper nouns,
   brand names, model numbers, and literal on-screen text may remain unchanged.
   On retries, NEVER infer the language from a previous response; always obey
   OUTPUT_LANGUAGE_CODE/OUTPUT_LANGUAGE_NAME above. Before returning JSON, verify
   every spoken narration field follows this language contract.
1. Return one narration item for every supplied NARRATION BEAT scene_id, in the same order.
   A beat may cover multiple visual scene IDs; narrate them as one thought.
2. Tell one continuous story. A later segment must sound like a continuation,
   not a fresh reaction to a new image.
3. Prioritize: PROCESS/CHANGE -> REASON -> CONSEQUENCE -> NEXT STEP.
4. Transcript/dialogue is stronger evidence for meaning and motivation than a
   visual guess. Visual facts are stronger evidence for visible actions/objects.
5. Never invent component names, character identities, relationships, causes,
   or outcomes. If unsupported, describe only what is known.
6. Avoid generic filler and repeated hype such as "Lihat ini!", "Luar biasa!",
   "Amazing!", "Gila!", "Kalian pasti gak nyangka", "Wow", or equivalents.
7. Exclamation marks should be rare (normally no more than 1 in 5 sentences).
8. Do not say "in this scene", "pada scene ini", "terlihat pada frame", or
   other production-language unless uncertainty genuinely requires hedging.
9. Natural transitions are welcome only when they add meaning: cause, contrast,
   progression, payoff, time, or bridge. Do not force a transition phrase into
   every segment.
10. Respect each segment's target_words and NEVER exceed max_words. Also avoid
    under-writing below min_words unless the evidence is genuinely insufficient.
11. Opening: establish the actual goal/problem, not a generic clickbait promise.
12. Closing: resolve/pay off what the narration has been tracking.
13. Keep Indonesian natural and conversational when language is Indonesian;
    avoid translating English storytelling idioms literally.
14. Do not repeat a fact listed in avoid_repeating unless needed for clarity.
15. If confidence is below {hedging_threshold}, hedge only the uncertain fact.
16. Add `claims`: a short list of atomic factual claims actually stated by the narration.
    Every claim MUST be directly supported by the FIXED evidence for that beat. Do not
    add facts to `claims` that are not present in the assembled speech, and do not use `claims` to
    smuggle in unsupported interpretation.
17. OPENING BEAT: Start the narration with the core PROBLEM, GOAL, or TRANSFORMATION
    that drives the entire video. Do not start with "In this video..." or generic filler.
18. PACING VARIETY: Alternate sentence lengths — short punchy statements after
    longer explanatory sentences. This creates natural rhythm for TTS delivery.
19. PROCESS COMPRESSION: When 3+ consecutive beats show the same type of process
    (e.g., machining steps), compress them into one flowing thought that highlights
    what's different about each step rather than repeating "then they did X".
{action_sync_rules}

{output_contract}
"""


REWRITE_PROMPT = """You are revising an existing video narration after QA.

OUTPUT_LANGUAGE_CODE: {language_code}
OUTPUT_LANGUAGE_NAME: {language_label}
STYLE NAME: {style_name}
STYLE TONE: {tone}
LANGUAGE STYLE:
{language_style}

STYLE RULES:
{rules}

STYLE EXAMPLES (rhythm/structure reference only; never copy unsupported facts):
{examples}

LANGUAGE CONTRACT (NON-NEGOTIABLE):
- Rewrite every spoken value (`text` or each `speech_parts` string) entirely in {language_label} (`{language_code}`).
- CURRENT NARRATION may contain the wrong language because of a failed prior attempt;
  do not preserve that mistake. Translate/rewrite it into the required output language.
- Story plan, QA details, source evidence, and retry context are INPUT ONLY and must
  never override OUTPUT_LANGUAGE_CODE/OUTPUT_LANGUAGE_NAME.
- Proper nouns, model numbers, brands, and literal on-screen text may remain unchanged.
- Before returning JSON, verify every spoken field follows the required language.

QA ISSUES:
{issues_json}

GLOBAL STORY PLAN:
{story_plan_json}

NARRATION PLAN:
{narration_plan_json}

CURRENT NARRATION:
{current_json}

Rewrite the narration to fix the QA issues while preserving factual accuracy,
scene order, scene_id values, and the per-segment min/target/max word budgets.
Reduce generic reactions, repeated sentence openings, and disconnected scene-by-
scene commentary. Make cause/effect and process continuity explicit when the
provided evidence supports it. Do not invent missing facts.

For every item, also return `claims`: atomic factual claims stated in the revised text,
limited to what the fixed narration plan/evidence supports.

{action_sync_rules}

{output_contract}
"""


# Compatibility aliases retained for callers outside script_writer.
NARRATION_PROMPT = V2_NARRATION_PROMPT
SUMMARY_NARRATION_PROMPT = V2_NARRATION_PROMPT


def _language_label(language: str) -> str:
    return {"id": "Bahasa Indonesia", "en": "English"}.get(language, language)


def _rules(style_config: dict) -> str:
    return "\n".join(f"- {rule}" for rule in style_config.get("rules", []) or []) or "- Write naturally and accurately"


def _language_style(style_config: dict, language: str) -> str:
    value = style_config.get("language_style", {}) or {}
    if isinstance(value, dict):
        return str(value.get(language, "") or value.get("en", "") or "Write naturally and accurately").strip()
    return str(value or "Write naturally and accurately").strip()


def _examples(style_config: dict, language: str) -> str:
    values = style_config.get("examples", {}) or {}
    if isinstance(values, dict):
        values = values.get(language, []) or []
    if not isinstance(values, list):
        values = [values]
    rendered = [str(value).strip() for value in values if str(value).strip()]
    return "\n".join(f"- {value}" for value in rendered[:4]) or "- No fixed example; follow the style rules."


def _context_excerpt(value, limit: int = 600) -> str:
    """Bound secondary continuity context, never the current beat's evidence."""
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " … [context excerpt]"


def batch_story_context(story_plan: dict, story_memory: dict, entries: list[dict]) -> tuple[dict, dict]:
    """Scope a global plan to one group; original artifacts remain complete.

    These fields provide continuity only. Complete selected-beat evidence and
    required anchors are supplied separately, and may not be discarded to fit.
    """
    ids = {str(sid) for entry in entries
           for sid in entry.get("covers_scene_ids", [entry.get("scene_id")])}
    ids.update(str(entry.get("scene_id")) for entry in entries)
    roles = [item for item in story_plan.get("scene_roles", []) or []
             if str(item.get("scene_id")) in ids]
    plan = {key: _context_excerpt(story_plan.get(key), limit)
            for key, limit in (("premise", 600), ("main_goal", 300), ("central_conflict", 300))}
    plan["scene_roles"] = [{key: item.get(key) for key in (
        "scene_id", "role", "story_importance", "plot_thread", "reason")}
        for item in roles]
    plan["context_scope"] = "Current group only; other scenes remain in the complete story artifact."
    memory = {}
    omitted = {}
    for key in ("known_facts", "continuity_rules", "open_questions"):
        values = story_memory.get(key, []) or []
        selected = values if len(values) <= 8 else values[:4] + values[-4:]
        memory[key] = [_context_excerpt(value, 240) for value in selected]
        omitted[key] = max(0, len(values) - len(selected))
    for key in ("characters", "important_objects"):
        values = story_memory.get(key, story_plan.get(key, [])) or []
        memory[key] = [{field: _context_excerpt(item.get(field), 160)
                        for field in ("id", "label", "goal", "purpose", "state") if item.get(field)}
                       for item in values[:4] if isinstance(item, dict)]
        omitted[key] = max(0, len(values) - 4)
    memory["context_entries_omitted"] = omitted
    return plan, memory


def narration_batch_plan(plan: dict, entries: list[dict]) -> dict:
    """Keep fixed beat timings while updating group-level QA totals."""
    return {**plan, "segments": entries, "narration_beat_count": len(entries),
            "total_target_words": sum(int(item.get("target_words", 0)) for item in entries),
            "total_duration": sum(max(0.0, float(item.get("timeline_end", 0))
                                       - float(item.get("timeline_start", 0))) for item in entries)}


def batch_continuity_note(entries: list[dict], all_entries: list[dict], previous: list[dict],
                          *, previous_field: str = "text") -> str:
    positions = {str(entry["scene_id"]): index for index, entry in enumerate(all_entries)}
    start = positions[str(entries[0]["scene_id"])]
    end = positions[str(entries[-1]["scene_id"])] + 1
    next_entry = all_entries[end] if end < len(all_entries) else None
    context = {
        "beat_ids": [str(entry["scene_id"]) for entry in entries],
        "is_video_opening": start == 0, "is_video_ending": end == len(all_entries),
        "previous_narration" if previous_field == "text" else "previous_planning_intents": [
            {"scene_id": item["scene_id"], previous_field: _context_excerpt(item.get(previous_field), 700)}
            for item in previous[-2:]],
        "next_beat": ({"scene_id": next_entry["scene_id"], "story_role": next_entry.get("story_role"),
                       "intent": _context_excerpt(next_entry.get("intent"), 300)} if next_entry else None),
    }
    return ("\nBATCH CONTINUITY (context only, never additional output rows):\n"
            + json.dumps(context, ensure_ascii=False)
            + "\nWrite only beat_ids above. Opening/closing rules apply only when is_video_opening/ending is true. "
              "Continue the previous narration or planning intent without repeating facts or starting a new hook. "
              "Leave next_beat actions for that beat; do not invent missing context.\n")


def _output_contract(narration_plan: dict) -> str:
    """Use the same complete field contract for initial and QA generation."""
    entries = narration_plan.get("segments", []) or []
    anchored = bool(narration_plan.get("action_sync_enabled")) or any(
        entry.get("action_sync_enabled") for entry in entries
    )
    fields = (
        "Return ONLY a valid JSON array, with exactly one object per narration beat.\n"
        "Every object must contain these fields:\n"
        '- "scene_id": string, copied EXACTLY from the narration beat (not a visual event ID).\n'
        '- "claims": array of strings stating only evidence-supported facts actually spoken; [] is valid.\n'
        '- "importance": number between 0 and 1.\n'
    )
    if anchored:
        fields += (
            '- "speech_parts": object of spoken strings for each action-sync beat; use EXACTLY the keys below.\n'
            '  Write lead_in, action_1, after_action_1, then action_2/after_action_2 if present.\n'
            '  The application joins these strings with spaces in that order to create one connected paragraph.\n'
            '  action_N is a short 1-6 word phrase describing its bound visual action in the OUTPUT language.\n'
            '  It is part of the speech, not a label or a copied observation. Do not repeat it in another part.\n'
            '  lead_in and after_action_N provide connecting prose and may be empty strings.\n'
            '  With no selected events use only lead_in containing the whole nonempty narration.\n'
            '  Do not return separate text, action_anchors, event IDs, or timestamps on action-sync beats.\n'
            '  For a beat without action sync, return text instead of speech_parts.\n'
            'Fixed speech keys -> visual event bindings (null = connecting prose, NOT an output value):\n'
            + json.dumps({str(entry["scene_id"]): speech_part_bindings(entry)
                          for entry in entries if entry.get("action_sync_enabled")}, ensure_ascii=False)
            + '\nExample Indonesian speech_parts for a SINGLE observed cutting action:\n'
            '  {"lead_in": "Setelah ditandai,", "action_1": "besi dipotong perlahan", '
            '"after_action_1": "hingga bagian ujungnya terpisah."}\n'
            '  This illustrates the structure only. Never copy this cutting action unless supported by YOUR evidence.\n'
        )
    else:
        fields += '- "text": nonempty spoken narration in the requested output language.\n'
    fields += (
        "Do not return timestamps, evidence, plan fields, markdown or explanatory prose.\n"
        "Required narration beat IDs, in playback order: "
        + json.dumps([str(entry.get("scene_id", "")) for entry in entries], ensure_ascii=False)
    )
    return fields


def _beat_payload(scene_map: dict[str, dict], plan_entry: dict) -> dict:
    covered_ids = [str(sid) for sid in plan_entry.get("covers_scene_ids", []) if sid]
    if not covered_ids:
        covered_ids = [str(plan_entry.get("scene_id", ""))]
    covered = [scene_map.get(scene_id, {}) for scene_id in covered_ids]

    evidence = []
    for scene in covered:
        analysis = scene.get("analysis", {}) or {}
        evidence.append(
            {
                "scene_id": scene.get("scene_id"),
                "transcript": analysis.get("transcript_text", ""),
                "action": analysis.get("action", ""),
                "changes": analysis.get("changes", ""),
                "people": analysis.get("people", ""),
                "visible_objects": (analysis.get("visible_objects", []) or [])[:10],
                "environment": analysis.get("environment", ""),
                "likely_context": analysis.get("likely_context", ""),
                "character_goal": analysis.get("character_goal", ""),
                "state_before": analysis.get("state_before", ""),
                "state_after": analysis.get("state_after", ""),
                "cause": analysis.get("cause", ""),
                "effect": analysis.get("effect", ""),
                "confidence": analysis.get("confidence", 0),
            }
        )

    return {
        "scene_id": plan_entry.get("scene_id"),
        "covers_scene_ids": covered_ids,
        "duration_seconds": round(float(plan_entry.get("available_duration", 0) or 0), 2),
        "timeline_start": plan_entry.get("timeline_start"),
        "timeline_end": plan_entry.get("timeline_end"),
        "story_role": plan_entry.get("story_role", "process"),
        "intent": plan_entry.get("intent", ""),
        "continuity_from_previous": plan_entry.get("continuity_from_previous", ""),
        "transition_strategy": plan_entry.get("transition_strategy", ""),
        "must_explain": plan_entry.get("must_explain", []),
        "forbidden_claims": plan_entry.get("forbidden_claims", []),
        "avoid_repeating": plan_entry.get("avoid_repeating", []),
        "min_words": plan_entry.get("min_words", 3),
        "target_words": plan_entry.get("target_words", 8),
        "max_words": plan_entry.get("max_words", 12),
        "evidence_by_visual_scene": evidence,
        **({**compact_visual_events(plan_entry.get("visual_events", []),
                                   required_ids=plan_entry.get("anchor_event_ids", [])),
            "anchor_event_ids": plan_entry.get("anchor_event_ids", [])}
           if plan_entry.get("action_sync_enabled") else {}),
    }


ACTION_SYNC_RULES = """ACTION TIMING CONTRACT (full-video narration only):
- visual_events are timestamped observations in source-video seconds. Read them
  in order to understand what changes during a beat. They are evidence, not an
  instruction to narrate every frame or to restart a sentence every second.
  Prompt evidence may be compacted: visual_events_omitted states how many other
  observations remain in the saved timeline. Do not invent events in those gaps.
- Write the whole narration as a connected process with a relevant opening and
  a supported result. Keep explanation and transitions natural. Do not infer
  technical causes, materials, temperatures, identities or purposes without evidence.
- Every beat has anchor_event_ids, containing at most two important visible actions.
  Write each selected action directly in its bound speech_parts.action_N field.
  Choose 1-6 spoken words containing the action, unique within the complete beat.
  A short verb phrase is enough; do not copy a whole visual observation sentence.
  The field bindings fix the source chronology. Do not generate or change event IDs/times.
- Speak only evidence-supported actions: an observed motion does not establish
  technical purpose, cooling, required consistency or an unseen final product.
  Each action_N must express its bound event, not a generic transition word.
- During QA rewriting copy the existing action_anchors phrase for each event into
  its bound action_N field EXACTLY. Rewrite only surrounding prose, not that phrase.
  Keep the combined speech within the beat's word budget, not a budget per field.
"""


def build_v2_narration_prompt(
    *,
    scenes: list[dict],
    narration_plan: dict,
    story_plan: dict,
    story_memory: dict,
    language: str,
    style: str,
    style_config: dict,
    hedging_threshold: float,
    mode: str,
) -> str:
    scene_map = {str(scene.get("scene_id")): scene for scene in scenes}
    payload = [
        _beat_payload(scene_map, plan_entry)
        for plan_entry in narration_plan.get("segments", []) or []
    ]
    anchored = bool(narration_plan.get("action_sync_enabled"))
    plan_json = json.dumps(story_plan, ensure_ascii=False, indent=None if anchored else 2)
    memory_json = json.dumps(story_memory, ensure_ascii=False, indent=None if anchored else 2)

    return V2_NARRATION_PROMPT.format(
        mode=mode,
        language_code=language,
        language_label=_language_label(language),
        style_name=style_config.get("name", style),
        tone=style_config.get("tone", "natural storytelling"),
        language_style=_language_style(style_config, language),
        examples=_examples(style_config, language),
        story_plan_json=plan_json if anchored else plan_json[:26000],
        story_memory_json=memory_json if anchored else memory_json[:14000],
        rules=_rules(style_config),
        scenes_json=json.dumps(payload, ensure_ascii=False, indent=None if anchored else 2),
        hedging_threshold=hedging_threshold,
        action_sync_rules=ACTION_SYNC_RULES if narration_plan.get("action_sync_enabled") else "",
        output_contract=_output_contract(narration_plan),
    )


def build_rewrite_prompt(
    *,
    current_segments: list[dict],
    qa: dict,
    narration_plan: dict,
    story_plan: dict,
    language: str,
    style: str = "viral_fb",
    style_config: dict | None = None,
) -> str:
    style_config = style_config or {}
    anchored = bool(narration_plan.get("action_sync_enabled"))
    prompt_plan = narration_plan
    if anchored:
        prompt_plan = {**narration_plan, "segments": [
            {**entry, **compact_visual_events(entry.get("visual_events", []),
                                             required_ids=entry.get("anchor_event_ids", []))}
            for entry in narration_plan.get("segments", [])
        ]}
    plan_json = json.dumps(prompt_plan, ensure_ascii=False, indent=None if anchored else 2)
    story_json = json.dumps(story_plan, ensure_ascii=False, indent=None if anchored else 2)
    return REWRITE_PROMPT.format(
        language_code=language,
        language_label=_language_label(language),
        style_name=style_config.get("name", style),
        tone=style_config.get("tone", "natural storytelling"),
        language_style=_language_style(style_config, language),
        rules=_rules(style_config),
        examples=_examples(style_config, language),
        issues_json=json.dumps(qa.get("issues", []), ensure_ascii=False, indent=2),
        story_plan_json=story_json if anchored else story_json[:22000],
        narration_plan_json=plan_json if anchored else plan_json[:26000],
        current_json=json.dumps(current_segments, ensure_ascii=False, indent=2),
        action_sync_rules=ACTION_SYNC_RULES if narration_plan.get("action_sync_enabled") else "",
        output_contract=_output_contract(narration_plan),
    )


def _compat_plan(storyboard: dict, scenes: list[dict], mode: str) -> dict:
    """Small compatibility plan when this module is called directly."""
    cumulative = 0.0
    entries = []
    for index, scene in enumerate(scenes):
        duration = float(scene.get("duration", 0) or 0)
        start = cumulative if mode == "summary" else float(scene.get("start_time", 0) or 0)
        if mode == "summary":
            cumulative += duration
        analysis = scene.get("analysis", {}) or {}
        narrative = scene.get("narrative", {}) or {}
        target_words = max(4, round(duration * 160 / 60 * 0.8))
        entries.append(
            {
                "scene_id": scene.get("scene_id"),
                "available_duration": duration,
                "timeline_start": start,
                "timeline_end": start + duration,
                "story_role": narrative.get("role", "process"),
                "intent": narrative.get("cue", "") or analysis.get("likely_context", "") or analysis.get("action", ""),
                "must_explain": [analysis.get("action", "")] if analysis.get("action") else [],
                "avoid_repeating": [],
                "continuity_from_previous": narrative.get("connects_to_next", ""),
                "transition_strategy": "progression" if index else "none",
                "min_words": max(3, round(target_words * 0.72)),
                "target_words": target_words,
                "max_words": max(target_words + 2, round(target_words * 1.18)),
            }
        )
    return {"mode": mode, "segments": entries}


def build_narration_prompt(
    storyboard: dict,
    language: str,
    style: str,
    style_config: dict,
    hedging_threshold: float = 0.5,
) -> str:
    scenes = storyboard.get("scenes", []) or []
    return build_v2_narration_prompt(
        scenes=scenes,
        narration_plan=_compat_plan(storyboard, scenes, "full"),
        story_plan={"premise": storyboard.get("video_summary", ""), "scene_roles": []},
        story_memory={},
        language=language,
        style=style,
        style_config=style_config,
        hedging_threshold=hedging_threshold,
        mode="full",
    )


def build_summary_narration_prompt(
    storyboard: dict,
    selected_scenes: list[dict],
    language: str,
    style: str,
    style_config: dict,
    hedging_threshold: float = 0.5,
) -> str:
    return build_v2_narration_prompt(
        scenes=selected_scenes,
        narration_plan=_compat_plan(storyboard, selected_scenes, "summary"),
        story_plan={"premise": storyboard.get("video_summary", ""), "scene_roles": []},
        story_memory={},
        language=language,
        style=style,
        style_config=style_config,
        hedging_threshold=hedging_threshold,
        mode="summary",
    )
