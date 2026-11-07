"""Continuity-aware narration script generator for DubbingStory v2."""

from __future__ import annotations

import json
from typing import Any, Callable

from dubbingstory.llm.batching import generate_budgeted_batches, context_batching_enabled

from dubbingstory.story.action_anchors import (
    assemble_speech_parts, canonical_action_anchors, normalized_tokens,
    preserve_action_phrases, speech_part_bindings,
)

from dubbingstory.story.gemini_json import (
    build_story_client, raise_if_strict_text_error, strict_text_routing,
    validate_scene_items, validate_string_list,
)
from dubbingstory.story.language_guard import payload_language_mismatch_reason
from dubbingstory.story.narration_planner import build_narration_plan
from dubbingstory.story.narration_qa import evaluate_narration
from dubbingstory.story.prompt_templates import (
    batch_continuity_note, batch_story_context, narration_batch_plan,
    build_rewrite_prompt, build_v2_narration_prompt,
)
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
    return build_story_client(cfg, model_attr="narration_model")


def _call_json(
    analyzer,
    model: str,
    prompt: str,
    *,
    temperature: float = 0.45,
    language: str | None = None,
    task_name: str = "narration",
    payload_validator: Callable[[Any], str | None] | None = None,
    max_output_tokens: int | None = None,
) -> Any:
    """Generate JSON with retries while preserving the selected language.

    Narration used to bypass ``GeminiVideoAnalyzer`` retry logic and call the
    SDK directly. That meant a 503 immediately fell through to fallback, and
    custom retry code could accidentally rebuild a prompt without ``language``.
    The new path keeps one immutable prompt for all attempts and rejects an
    obviously wrong-language response before it can reach TTS.
    """

    normalize = getattr(payload_validator, "normalize_payload", lambda payload: payload)

    def response_validator(payload: Any) -> str | None:
        try:
            payload = normalize(payload)
        except (KeyError, TypeError, ValueError) as exc:
            return str(exc)
        if payload_validator:
            error = payload_validator(payload)
            if error:
                return error
        return payload_language_mismatch_reason(payload, language) if language else None

    # Keep the public analyzer call signature unchanged. Routed clients can use
    # the provider schema while custom analyzers still receive only a callable.
    response_schema = getattr(payload_validator, "response_schema", None)
    if response_schema is not None:
        response_validator.response_schema = response_schema

    if hasattr(analyzer, "generate_json_with_retry"):
        payload = analyzer.generate_json_with_retry(
            prompt,
            temperature=temperature,
            response_validator=response_validator if language or payload_validator else None,
            task_name=f"{task_name}-{language or 'generic'}",
            model=model,
            **({"max_output_tokens": max_output_tokens}
               if max_output_tokens is not None and hasattr(analyzer, "preflight") else {}),
        )
        validation_error = response_validator(payload)
        if validation_error:
            raise ValueError(validation_error)
        return normalize(payload)

    # Backward-compatible one-shot path for custom analyzers. Keep the same
    # semantic validation even though retry support is unavailable.
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
    payload = json.loads(text)
    validation_error = response_validator(payload)
    if validation_error:
        raise ValueError(validation_error)
    return normalize(payload)


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


def _narration_response_schema(narration_plan: dict, *, preserved_anchors=None) -> dict:
    """Constrain generated JSON; semantic evidence checks remain authoritative.

    Anchored providers generate fixed speech slots; canonical text/anchors remain
    accepted for historical/custom clients and are still the saved script format.
    """
    plan_map = _plan_map(narration_plan)
    scene_id_schema = {"type": "string"}
    if plan_map:
        scene_id_schema["enum"] = list(plan_map)
    properties = {
        "scene_id": scene_id_schema,
        "text": {"type": "string", "minLength": 1},
        "claims": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "importance": {"type": "number", "minimum": 0, "maximum": 1},
    }
    required = list(properties)
    if any(entry.get("action_sync_enabled") for entry in plan_map.values()):
        variants = []
        for scene_id, entry in plan_map.items():
            fields = {**properties, "scene_id": {"type": "string", "enum": [scene_id]}}
            if entry.get("action_sync_enabled"):
                parts = {}
                preserved = {a["event_id"]: a["phrase"] for a in (preserved_anchors or {}).get(scene_id, [])}
                for name, event_id in speech_part_bindings(entry).items():
                    if event_id:
                        parts[name] = {
                            "type": "string", "minLength": 1,
                            "pattern": r"^\S+(?:\s+\S+){0,5}$",
                            "description": f"1-6 spoken words describing selected visual event {event_id}; do not repeat elsewhere.",
                        }
                        if event_id in preserved:
                            # Legacy phrases can have standalone punctuation;
                            # the canonical Unicode word count is authoritative.
                            parts[name].pop("pattern")
                            parts[name]["enum"] = [preserved[event_id]]
                    else:
                        parts[name] = {"type": "string", "description": "Connecting spoken prose; empty string allowed."}
                speech_schema = {"type": "object", "properties": parts,
                                 "required": list(parts), "additionalProperties": False}
                fields = {("speech_parts" if key == "text" else key):
                          (speech_schema if key == "text" else value) for key, value in fields.items()}
            variants.append({"type": "object", "properties": fields,
                             "required": list(fields), "additionalProperties": False})
        return {"type": "array", "minItems": len(plan_map), "maxItems": len(plan_map),
                "items": variants[0] if len(variants) == 1 else {"anyOf": variants}}
    return {
        "type": "array", "minItems": len(plan_map), "maxItems": len(plan_map),
        "items": {"type": "object", "properties": properties,
                  "required": required, "additionalProperties": False},
    }


def _narration_response_validator(narration_plan: dict, *, preserved_anchors=None):
    plan_map = _plan_map(narration_plan)
    expected_ids = set(plan_map)

    def normalize_payload(payload):
        # Accept the historical envelopes, but never silently discard bad rows.
        if isinstance(payload, dict):
            payload = next((payload[key] for key in ("segments", "narration", "items")
                            if isinstance(payload.get(key), list)), None)
        if not isinstance(payload, list):
            return payload
        result = []
        errors = []
        for item in payload:
            if not isinstance(item, dict) or "speech_parts" not in item:
                result.append(item)
                continue
            scene_id = item.get("scene_id")
            planned = plan_map.get(scene_id) if isinstance(scene_id, str) else None
            try:
                if not planned or not planned.get("action_sync_enabled"):
                    raise ValueError("speech_parts requires a selected narration beat with action sync")
                if "text" in item or "action_anchors" in item:
                    raise ValueError("return speech_parts only; do not also generate text/action_anchors")
                text, anchors = assemble_speech_parts(item["speech_parts"], planned)
                result.append({**{k: v for k, v in item.items() if k != "speech_parts"},
                               "text": text, "action_anchors": anchors})
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{scene_id}: {exc}")
        if errors:
            raise ValueError("; ".join(errors[:4]))
        return result

    def validator(payload: Any) -> str | None:
        try:
            payload = normalize_payload(payload)
        except (KeyError, TypeError, ValueError) as exc:
            return str(exc)
        error = validate_scene_items(payload, expected_ids)
        if error:
            expected = ", ".join(list(plan_map)[:20])
            return f"{error}; use each narration beat scene_id exactly once: {expected}"
        errors = []
        for item in payload:
            scene_id = str(item["scene_id"])
            if not isinstance(item.get("text"), str) or not item["text"].strip():
                return f"{scene_id}.text must be a nonempty narration string"
            if not validate_string_list(item.get("claims")):
                return f"{scene_id}.claims must be an array of nonempty strings (or an empty array)"
            importance = item.get("importance", 0.5)
            if type(importance) not in (int, float) or not 0 <= importance <= 1:
                return f"{scene_id}.importance must be a number from zero to one"
            planned = plan_map[scene_id]
            if planned.get("action_sync_enabled"):
                try:
                    canonical_action_anchors(item.get("action_anchors"), item["text"], planned)
                    if preserved_anchors is not None:
                        preserve_action_phrases(item["text"], preserved_anchors.get(scene_id, []))
                except (KeyError, TypeError, ValueError) as exc:
                    required = json.dumps(planned.get("anchor_event_ids", []), ensure_ascii=False)
                    errors.append(f"{scene_id}.action_anchors: {exc}; required event IDs: {required}")
        return "; ".join(errors[:4]) if errors else None

    validator.normalize_payload = normalize_payload
    validator.response_schema = _narration_response_schema(narration_plan, preserved_anchors=preserved_anchors)
    return validator


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
        raw_claims = [str(v).strip() for v in raw.get("claims", []) or [] if str(v).strip()][:8]
        covered_ids = [str(sid) for sid in planned.get("covers_scene_ids", [scene_id]) if sid]
        if not text:
            if planned.get("action_sync_enabled") and planned.get("anchor_event_ids"):
                raise ValueError(f"{scene_id}: anchored narration cannot use an ungrounded fallback")
            covered_scenes = [scene_map.get(sid, {}) for sid in covered_ids]
            text = _fallback_beat_text(covered_scenes, language, planned)

        word_count = len(text.split())
        target_wpm = float(narration_plan.get("target_wpm", 160) or 160)
        estimated = word_count / target_wpm * 60.0 if target_wpm > 0 else 0.0
        validated.append(
            {
                "scene_id": scene_id,
                "covers_scene_ids": covered_ids,
                "text": text,
                "claims": raw_claims or [
                    str(v).strip() for v in planned.get("must_explain", []) or [] if str(v).strip()
                ][:8],
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
        if planned.get("action_sync_enabled"):
            anchors = canonical_action_anchors(raw.get("action_anchors"), text, planned)
            validated[-1].update(
                action_sync_enabled=True,
                visual_events=planned.get("visual_events", []),
                anchor_event_ids=planned.get("anchor_event_ids", []),
                action_anchors=anchors,
                action_sync_status="pending" if anchors else "no_eligible_events",
            )
    return validated


def _fallback_signal_text(scenes: list[dict], planned: dict | None = None) -> str:
    """Collect evidence only for fallback classification, never verbatim output."""
    values: list[str] = []
    for scene in scenes:
        analysis = scene.get("analysis", {}) or {}
        for key in ("action", "likely_context", "changes", "environment"):
            value = str(analysis.get(key, "") or "").strip()
            if value and "unable to analyze" not in value.lower():
                values.append(value.lower())
    if planned:
        values.append(str(planned.get("intent", "") or "").lower())
        values.extend(str(item or "").lower() for item in planned.get("must_explain", []) or [])
    return " ".join(values)


def _fallback_category(scenes: list[dict], planned: dict | None = None) -> str:
    signal = _fallback_signal_text(scenes, planned)
    keyword_groups = [
        ("welding", ("weld", "welding", "arc", "torch")),
        ("drilling", ("drill", "drilling", "hole", "boring")),
        ("machining", ("lathe", "turning", "machine", "machining", "cutting", "shave", "grind")),
        ("measurement", ("measure", "measurement", "inspect", "inspection", "caliper", "gauge", "check")),
        ("assembly", ("assemble", "assembly", "install", "tighten", "wrench", "fit", "mount")),
        ("repair", ("repair", "crack", "damaged", "damage", "fix", "restore")),
    ]
    for category, keywords in keyword_groups:
        if any(keyword in signal for keyword in keywords):
            return category
    return "generic"


def _fallback_beat_text(
    scenes: list[dict],
    language: str = "id",
    planned: dict | None = None,
) -> str:
    """Return a language-safe deterministic fallback for one narration beat.

    English visual evidence is used only to choose a broad action category. It is
    never copied verbatim into an Indonesian fallback. This prevents both the old
    repeated hard-coded sentence and accidental English leakage after retries fail.
    """
    planned = planned or {}
    category = _fallback_category(scenes, planned)
    role = str(planned.get("story_role", "process") or "process").lower()
    scene_id = str(planned.get("scene_id", "") or "")
    variant = sum(ord(ch) for ch in scene_id) % 2

    id_by_category = {
        "measurement": [
            "Pekerjaan dimulai dengan memeriksa kondisi dan ukuran komponen sebelum tahap berikutnya dilakukan.",
            "Komponen diperiksa lebih dulu agar posisi dan ukurannya sesuai sebelum proses dilanjutkan.",
        ],
        "welding": [
            "Bagian yang bermasalah kemudian diperbaiki melalui proses penyambungan sebelum dirapikan kembali.",
            "Perbaikan berlanjut dengan menyambung area yang perlu diperkuat agar dapat diproses lebih lanjut.",
        ],
        "drilling": [
            "Tahap berikutnya membentuk atau merapikan lubang pada komponen dengan pengerjaan yang terkontrol.",
            "Komponen kemudian diproses pada bagian lubangnya agar sesuai dengan kebutuhan pemasangan.",
        ],
        "machining": [
            "Material dikikis sedikit demi sedikit untuk membentuk permukaan komponen sesuai kebutuhan pengerjaan.",
            "Komponen diproses bertahap pada mesin untuk merapikan bentuk dan permukaannya.",
        ],
        "assembly": [
            "Setelah pengerjaan utama, komponen dipasang dan disetel kembali untuk memastikan posisinya sesuai.",
            "Tahap ini beralih ke pemasangan dan penyetelan agar komponen siap untuk pemeriksaan berikutnya.",
        ],
        "repair": [
            "Fokus pengerjaan tetap pada memperbaiki bagian yang rusak sebelum hasilnya diperiksa kembali.",
            "Bagian yang bermasalah ditangani bertahap agar komponen dapat kembali ke kondisi yang lebih baik.",
        ],
        "generic": [
            "Pengerjaan berlanjut ke tahap berikutnya dengan memeriksa perubahan pada komponen secara bertahap.",
            "Proses diteruskan sambil memastikan hasil pada komponen tetap sesuai sebelum langkah selanjutnya.",
        ],
    }
    en_by_category = {
        "measurement": [
            "The work begins by checking the component's condition and dimensions before the next step.",
            "The component is inspected first so its position and dimensions can be verified before processing continues.",
        ],
        "welding": [
            "The damaged area is then joined and reinforced before the surface is refined again.",
            "The repair continues by joining the area that needs reinforcement before further processing.",
        ],
        "drilling": [
            "The next step forms or refines the hole in the component with controlled machining.",
            "The component is then worked around the hole so it can meet the fitting requirement.",
        ],
        "machining": [
            "Material is removed gradually to shape the component surface for the required fit.",
            "The component is machined step by step to refine its shape and surface.",
        ],
        "assembly": [
            "After the main work, the component is fitted and adjusted again to verify its position.",
            "The process shifts to fitting and adjustment so the component is ready for the next check.",
        ],
        "repair": [
            "The work remains focused on correcting the damaged area before the result is checked again.",
            "The problem area is repaired in stages so the component can return to a usable condition.",
        ],
        "generic": [
            "The work continues to the next step while the changes to the component are checked progressively.",
            "The process moves forward while the result is verified before the following operation.",
        ],
    }

    templates = id_by_category if language == "id" else en_by_category
    text = templates.get(category, templates["generic"])[variant]

    # Role-specific framing adds variation without inventing scene facts.
    if language == "id":
        if role in {"problem", "complication"} and category == "generic":
            return "Pada tahap ini masih ada bagian yang perlu ditangani sebelum proses dapat dilanjutkan dengan aman."
        if role in {"resolution", "result", "conclusion"} and category == "generic":
            return "Setelah rangkaian pengerjaan, hasilnya diperiksa kembali untuk memastikan komponen siap digunakan."
    else:
        if role in {"problem", "complication"} and category == "generic":
            return "At this stage, an area still needs attention before the process can continue safely."
        if role in {"resolution", "result", "conclusion"} and category == "generic":
            return "After the sequence of work, the result is checked again to confirm the component is ready to use."
    return text


def _fallback_text(scene: dict, language: str = "id") -> str:
    return _fallback_beat_text([scene], language=language)


def _fallback_narration(scenes: list[dict], narration_plan: dict, language: str) -> list[dict]:
    scene_map = {str(scene.get("scene_id")): scene for scene in scenes}
    raw = []
    for planned in narration_plan.get("segments", []) or []:
        covered_ids = [str(sid) for sid in planned.get("covers_scene_ids", [planned.get("scene_id")]) if sid]
        covered = [scene_map.get(sid, {}) for sid in covered_ids]
        raw.append(
            {
                "scene_id": planned.get("scene_id"),
                "text": _fallback_beat_text(covered, language, planned),
                "claims": [str(v).strip() for v in planned.get("must_explain", []) or [] if str(v).strip()][:8],
                "importance": planned.get("story_importance", 0.5),
            }
        )
    return _validate_narration(raw, scenes, narration_plan, language)


def _batch_limits(cfg) -> tuple[int, int]:
    size = int(getattr(cfg, "llm_batching_max_beats", 4))
    tokens = int(getattr(cfg, "llm_batching_output_tokens", 2048))
    if size <= 0 or tokens <= 0:
        raise ValueError("Narration batch size and output token budget must be positive")
    return size, tokens


def _generate_batched_narration(*, scenes, narration_plan, story_plan, story_memory,
                                language, style, style_config, hedging_threshold,
                                analyzer, model, cfg) -> list[dict]:
    entries = narration_plan.get("segments", []) or []
    max_items, tokens = _batch_limits(cfg)
    merged = []

    def prompt_for(group):
        group_plan = narration_batch_plan(narration_plan, group)
        scoped_story, scoped_memory = batch_story_context(story_plan, story_memory, group)
        return build_v2_narration_prompt(
            scenes=scenes, narration_plan=group_plan, story_plan=scoped_story,
            story_memory=scoped_memory, language=language, style=style,
            style_config=style_config, hedging_threshold=hedging_threshold, mode="full",
        ) + batch_continuity_note(group, entries, merged)

    def generate(group):
        group_plan = narration_batch_plan(narration_plan, group)
        raw = _extract_segment_list(_call_json(
            analyzer, model, prompt_for(group), language=language,
            task_name=f"full-narration-{language}-{group[0]['scene_id']}-{group[-1]['scene_id']}",
            payload_validator=_narration_response_validator(group_plan), max_output_tokens=tokens,
        ))
        return _validate_narration(raw, scenes, group_plan, language)

    for group, validated in generate_budgeted_batches(
        entries, prompt_for, analyzer, generate, task_name=f"full-narration-{language}",
        max_items=max_items, max_output_tokens=tokens,
    ):
        merged.extend(validated)
        print(f"   Narration batches: {len(merged)}/{len(entries)} beats validated")
    # No caller receives any partial narration when a later group fails.
    error = _narration_response_validator(narration_plan)(merged)
    if error:
        raise ValueError(error)
    return merged


def _batched_qa_rewrite(narration, *, qa, narration_plan, story_plan, language,
                        style, style_config, analyzer, model, cfg, scenes):
    entries = narration_plan.get("segments", []) or []
    original = {item["scene_id"]: item for item in narration}
    max_items, tokens = _batch_limits(cfg)
    merged = []

    def prompt_for(group):
        scoped_story, _ = batch_story_context(story_plan, {}, group)
        current = [{key: original[item["scene_id"]].get(key) for key in (
            "scene_id", "text", "claims", "importance", "action_anchors")
            if key in original[item["scene_id"]]} for item in group]
        return build_rewrite_prompt(
            current_segments=current, qa=qa, narration_plan=narration_batch_plan(narration_plan, group),
            story_plan=scoped_story, language=language, style=style, style_config=style_config,
        ) + batch_continuity_note(group, entries, merged)

    def generate(group):
        group_plan = narration_batch_plan(narration_plan, group)
        validator = _narration_response_validator(group_plan, preserved_anchors={
            item["scene_id"]: original[item["scene_id"]].get("action_anchors", []) for item in group
        })
        raw = _extract_segment_list(_call_json(
            analyzer, model, prompt_for(group), temperature=0.25, language=language,
            task_name=f"narration-rewrite-{language}-{group[0]['scene_id']}-{group[-1]['scene_id']}", payload_validator=validator,
            max_output_tokens=tokens,
        ))
        revised = _validate_narration(raw, scenes, group_plan, language)
        for item in revised:
            if item.get("action_sync_enabled"):
                anchors = original[item["scene_id"]].get("action_anchors", [])
                item["action_anchors"] = canonical_action_anchors(anchors, item["text"], item)
        return revised

    for group, revised in generate_budgeted_batches(
        entries, prompt_for, analyzer, generate, task_name=f"narration-rewrite-{language}",
        max_items=max_items, max_output_tokens=tokens,
    ):
        merged.extend(revised)
    error = _narration_response_validator(narration_plan)(merged)
    if error:
        raise ValueError(error)
    return merged


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
    style: str,
    style_config: dict,
) -> tuple[list[dict], dict]:
    qa = evaluate_narration(narration, narration_plan, language=language)
    if not qa.get("needs_rewrite") or not bool(getattr(cfg, "narration_auto_rewrite", True)):
        return narration, qa

    try:
        if context_batching_enabled(cfg) and narration_plan.get("mode", "full") == "full":
            revised = _batched_qa_rewrite(
                narration, qa=qa, narration_plan=narration_plan, story_plan=story_plan,
                language=language, style=style, style_config=style_config,
                analyzer=analyzer, model=model, cfg=cfg, scenes=scenes,
            )
            revised_qa = evaluate_narration(revised, narration_plan, language=language)
            before = (len(qa.get("issues", [])), abs(1.0 - float(qa.get("target_word_ratio", 1.0))))
            after = (len(revised_qa.get("issues", [])), abs(1.0 - float(revised_qa.get("target_word_ratio", 1.0))))
            if after <= before:
                print(f"   {language.upper()} batched QA rewrite applied: {qa.get('issues', [])} -> {revised_qa.get('issues', [])}")
                return revised, revised_qa
            return narration, qa
        prompt = build_rewrite_prompt(
            current_segments=[
                {
                    "scene_id": item["scene_id"],
                    "text": item["text"],
                    "claims": item.get("claims", []),
                    "importance": item.get("importance", 0.5),
                    **({"action_anchors": item.get("action_anchors", [])}
                       if item.get("action_sync_enabled") else {}),
                }
                for item in narration
            ],
            qa=qa,
            narration_plan=narration_plan,
            story_plan=story_plan,
            language=language,
            style=style,
            style_config=style_config,
        )
        revised_raw = _extract_segment_list(
            _call_json(
                analyzer,
                model,
                prompt,
                temperature=0.25,
                language=language,
                task_name="narration-rewrite",
                payload_validator=_narration_response_validator(narration_plan, preserved_anchors={
                    item["scene_id"]: item.get("action_anchors", []) for item in narration
                })
                if strict_text_routing(cfg) or narration_plan.get("action_sync_enabled") else None,
            )
        )
        revised = _validate_narration(revised_raw, scenes, narration_plan, language)
        if narration_plan.get("action_sync_enabled"):
            original_by_id = {item["scene_id"]: item for item in narration}
            for item in revised:
                original_anchors = original_by_id[item["scene_id"]].get("action_anchors", [])
                preserve_action_phrases(item["text"], original_anchors)
                # An improving rewrite may move surrounding prose, never swap
                # the selected phrase for another word referring to the event.
                item["action_anchors"] = canonical_action_anchors(original_anchors, item["text"], item)
        revised_qa = evaluate_narration(revised, narration_plan, language=language)

        # Keep rewrite only when it improves issue count or word-budget accuracy.
        before = (len(qa.get("issues", [])), abs(1.0 - float(qa.get("target_word_ratio", 1.0))))
        after = (len(revised_qa.get("issues", [])), abs(1.0 - float(revised_qa.get("target_word_ratio", 1.0))))
        if after <= before:
            print(f"   🧪 {language.upper()} QA rewrite applied: {qa.get('issues', [])} -> {revised_qa.get('issues', [])}")
            return revised, revised_qa
    except Exception as exc:
        raise_if_strict_text_error(cfg, exc, "narration-rewrite")
        print(f"   ⚠️ {language.upper()} QA rewrite failed: {exc}")
    return narration, qa


def shorten_for_measured_duration(
    segment: dict,
    *,
    language: str,
    measured_duration: float,
    available_duration: float,
    cfg,
    evidence: list[dict] | None = None,
    previous_text: str = "",
    next_text: str = "",
) -> dict:
    """Shorten one beat after measuring TTS; never invent a replacement fallback.

    The caller bounds retries and measures the replacement audio again. Existing
    language validation, narration client and selected style remain authoritative.
    """
    original = str(segment.get("text", "")).strip()
    if not original or available_duration <= 0 or measured_duration <= 0:
        raise ValueError("A nonempty beat and positive measured/window durations are required")
    original_word_count = len(original.split())
    target_words = max(1, int(original_word_count * available_duration / measured_duration * 0.88))
    style = str(getattr(cfg, "narration_style", "natural") or "natural")
    original_claims = [str(value) for value in segment.get("claims", []) or []]
    evidence_json = json.dumps(evidence or [], ensure_ascii=False)
    anchors = segment.get("action_anchors", []) or []
    if segment.get("action_sync_enabled"):
        anchors = canonical_action_anchors(anchors, original, segment)
    required_phrase_words = sum(len(anchor["phrase"].split()) for anchor in anchors)
    original_spoken_count = len(normalized_tokens(original))
    required_spoken_count = sum(len(normalized_tokens(anchor["phrase"])) for anchor in anchors)
    if anchors and required_spoken_count >= original_spoken_count:
        raise ValueError(
            f"{segment.get('scene_id', 'beat')}: duration rewrite has no removable words; "
            f"all {original_spoken_count} spoken words are required action phrases "
            f"({measured_duration:.3f}s measured, {available_duration:.3f}s available). "
            "Revise the anchored narration "
            "plan or voice delivery; no model request was made."
        )
    # The ratio is a text-length heuristic, not an audio-duration proof. Do not
    # ask a model to fit mandatory phrases into an impossible word budget. The
    # caller still synthesizes and measures the result against the actual window.
    if anchors:
        target_words = max(target_words, required_phrase_words)
    analyzer, model = _client(cfg)
    spoken_anchors = [{"event_id": anchor["event_id"], "phrase": anchor["phrase"]}
                      for anchor in anchors]
    anchor_context = (
        "Preserve these exact spoken action phrases, each once and in the same order. "
        "They must still describe their original visual events. Remove secondary prose first.\n"
        f"Required action anchors: {json.dumps(spoken_anchors, ensure_ascii=False)}\n"
    ) if anchors else ""
    prompt = (
        "Shorten this already approved narration beat to fit its measured TTS window. "
        "Preserve chronological meaning, the visible action, uncertainty, and the selected style. "
        "Remove secondary detail rather than adding events, motives, outcomes or claims. "
        "Do not introduce facts absent from the original narration and supplied visual evidence. "
        "Keep a relevant opening hook only when this beat already has one. "
        "Return JSON with text and retained_claim_indices (zero-based indices into original_claims). "
        "Return only a complete, natural spoken sentence or phrase, no ellipsis or stage directions.\n"
        f"Language: {language}\nMaximum words: {target_words}\n"
        f"Measured audio: {measured_duration:.3f}s; available: {available_duration:.3f}s\n"
        f"Style: {json.dumps(_style_config(cfg, style), ensure_ascii=False)}\n"
        f"Original: {json.dumps(original, ensure_ascii=False)}\n"
        f"Previous narration (context only; do not repeat): {json.dumps(previous_text, ensure_ascii=False)}\n"
        f"Next narration (context only; do not steal its actions): {json.dumps(next_text, ensure_ascii=False)}\n"
        f"{anchor_context}"
        f"original_claims: {json.dumps(original_claims, ensure_ascii=False)}\n"
        f"Visual evidence: {evidence_json if strict_text_routing(cfg) else evidence_json[:16000]}"
    )
    def duration_validator(payload: Any) -> str | None:
        if not isinstance(payload, dict):
            return "Duration rewrite must return an object"
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            return "Duration rewrite text must be a nonempty narration string"
        if text.strip() == original:
            return f"Duration rewrite repeated the original; shorten it to at most {target_words} words"
        if len(text.split()) > target_words:
            return f"Duration rewrite text has {len(text.split())} words; maximum is {target_words}"
        indices = payload.get("retained_claim_indices")
        if not isinstance(indices, list) or any(type(i) is not int or not 0 <= i < len(original_claims) for i in indices):
            return (
                f"retained_claim_indices must contain integers from 0 to {len(original_claims) - 1}"
                if original_claims else "retained_claim_indices must be [] because original_claims is empty"
            )
        try:
            preserve_action_phrases(text, anchors)
        except ValueError as exc:
            return str(exc)
        if anchors and len(normalized_tokens(text)) >= original_spoken_count:
            return "Duration rewrite must remove spoken words outside the required action phrases"
        return None

    duration_validator.response_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "minLength": 1},
            "retained_claim_indices": {
                "type": "array", "maxItems": len(original_claims),
                "items": {"type": "integer", "minimum": 0,
                          "maximum": max(0, len(original_claims) - 1)},
            },
        },
        "required": ["text", "retained_claim_indices"], "additionalProperties": False,
    }
    payload = _call_json(analyzer, model, prompt, temperature=0.15,
                         language=language, task_name="measured-duration-rewrite",
                         payload_validator=duration_validator)
    # Retain the gate for custom clients that do not apply supplied validators.
    error = duration_validator(payload)
    if error:
        raise ValueError(error)
    text = payload["text"].strip()
    indices = payload["retained_claim_indices"]
    revised = dict(segment)
    revised.update(text=text, word_count=len(text.split()),
                   claims=[original_claims[i] for i in dict.fromkeys(indices)],
                   duration_fit_rewritten=True)
    if segment.get("action_sync_enabled"):
        revised["action_anchors"] = canonical_action_anchors(anchors, text, revised)
    return revised


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

    if isinstance(languages, str):
        languages = [languages]

    for raw_language in languages:
        language = str(raw_language or "").strip().lower()
        if language not in {"id", "en"}:
            raise ValueError(f"Unsupported narration language: {raw_language!r}")
        print(f"\n   ✍️ Generating {language.upper()} {mode.upper()} narration (story-aware)...")
        try:
            if mode == "full" and context_batching_enabled(cfg):
                narration = _generate_batched_narration(
                    scenes=scenes, narration_plan=narration_plan, story_plan=story_plan,
                    story_memory=story_memory, language=language, style=style,
                    style_config=style_config, hedging_threshold=hedging_threshold,
                    analyzer=analyzer, model=model, cfg=cfg,
                )
            else:
                prompt = build_v2_narration_prompt(
                    scenes=scenes, narration_plan=narration_plan, story_plan=story_plan,
                    story_memory=story_memory, language=language, style=style,
                    style_config=style_config, hedging_threshold=hedging_threshold, mode=mode,
                )
                raw = _extract_segment_list(
                    _call_json(
                        analyzer, model, prompt, temperature=0.45, language=language,
                        task_name=f"{mode}-narration",
                        payload_validator=_narration_response_validator(narration_plan)
                        if strict_text_routing(cfg) or narration_plan.get("action_sync_enabled") else None,
                    )
                )
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
                style=style,
                style_config=style_config,
            )
            final_language_error = payload_language_mismatch_reason(narration, language)
            if final_language_error:
                raise ValueError(
                    f"Final narration language gate rejected {language}: {final_language_error}"
                )
            result[language] = narration
            print(
                f"   ✅ {language.upper()}: {len(narration)} beats, "
                f"{sum(s['word_count'] for s in narration)} words, "
                f"QA issues={len(qa.get('issues', []))}"
            )
        except Exception as exc:
            if narration_plan.get("action_sync_enabled"):
                raise RuntimeError(f"{language} action-grounded narration failed: {exc}") from exc
            if mode == "full" and context_batching_enabled(cfg):
                # A completed prefix is not a complete video narration. The
                # opt-in batch path must not publish a template after failure.
                from dubbingstory.llm.text_client import TextGenerationError
                if isinstance(exc, TextGenerationError):
                    raise
                raise TextGenerationError(f"{mode}-narration: batched narration failed validation") from None
            raise_if_strict_text_error(cfg, exc, f"{mode}-narration")
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
