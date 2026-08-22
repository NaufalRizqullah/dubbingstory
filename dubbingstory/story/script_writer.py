"""Continuity-aware narration script generator for DubbingStory v2."""

from __future__ import annotations

import json
from typing import Any

from dubbingstory.story.language_guard import payload_language_mismatch_reason
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


def _call_json(
    analyzer,
    model: str,
    prompt: str,
    *,
    temperature: float = 0.45,
    language: str | None = None,
    task_name: str = "narration",
) -> Any:
    """Generate JSON with retries while preserving the selected language.

    Narration used to bypass ``GeminiVideoAnalyzer`` retry logic and call the
    SDK directly. That meant a 503 immediately fell through to fallback, and
    custom retry code could accidentally rebuild a prompt without ``language``.
    The new path keeps one immutable prompt for all attempts and rejects an
    obviously wrong-language response before it can reach TTS.
    """

    def response_validator(payload: Any) -> str | None:
        if not language:
            return None
        return payload_language_mismatch_reason(payload, language)

    if hasattr(analyzer, "generate_json_with_retry"):
        return analyzer.generate_json_with_retry(
            prompt,
            temperature=temperature,
            response_validator=response_validator if language else None,
            task_name=f"{task_name}-{language or 'generic'}",
            model=model,
        )

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
    return payload


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
            text = _fallback_beat_text(covered_scenes, language, planned)

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
        revised_raw = _extract_segment_list(
            _call_json(
                analyzer,
                model,
                prompt,
                temperature=0.25,
                language=language,
                task_name="narration-rewrite",
            )
        )
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

    if isinstance(languages, str):
        languages = [languages]

    for raw_language in languages:
        language = str(raw_language or "").strip().lower()
        if language not in {"id", "en"}:
            raise ValueError(f"Unsupported narration language: {raw_language!r}")
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
            raw = _extract_segment_list(
                _call_json(
                    analyzer,
                    model,
                    prompt,
                    temperature=0.45,
                    language=language,
                    task_name=f"{mode}-narration",
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
