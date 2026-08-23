"""Post-write evidence verification and typed beat repair for V2.1."""

from __future__ import annotations

import json
from typing import Any

from dubbingstory.story.gemini_json import generate_story_json
from dubbingstory.story.language_guard import payload_language_mismatch_reason
from dubbingstory.story.semantic_matcher import SimilarityEngine, make_similarity_engine


VERIFIER_VERSION = "v2.1.0"

REPAIR_PROMPT = """You are repairing ONE narration beat after factual evidence QA.
Do not change scene assignment. Do not write any other beat.

OUTPUT_LANGUAGE_CODE: {language_code}
OUTPUT_LANGUAGE_NAME: {language_name}

BEAT / WORD BUDGET:
{plan_json}

FIXED VISUAL EVIDENCE:
{evidence_json}

CURRENT BEAT:
{current_json}

UNSUPPORTED OR WEAK CLAIMS:
{unsupported_json}

Rules:
- Keep scene_id exactly unchanged.
- Rewrite only claims that exceed the fixed evidence boundary.
- Preserve supported useful information.
- If evidence is insufficient, hedge or make the claim more generic; never invent a cause/outcome.
- Respect min_words / target_words / max_words.
- Every text value must be written in the required output language.

Return ONLY one JSON object:
{{"scene_id":"...","text":"...","claims":["atomic supported claim"]}}
"""


def _language_name(code: str) -> str:
    return {"id": "Bahasa Indonesia", "en": "English"}.get(code, code)


def _evidence_text(scene_ids: list[str], signature_map: dict[str, dict]) -> str:
    parts = []
    for sid in scene_ids:
        sig = signature_map.get(sid, {}) or {}
        text = str(sig.get("signature", "") or "").strip()
        if text:
            parts.append(f"[{sid}]\n{text}")
    return "\n\n".join(parts)


def verify_narration(
    *,
    narration: list[dict],
    narration_plan: dict,
    signatures: list[dict],
    match_report: dict | None,
    cfg,
    language: str,
    project_dir: str | None = None,
    engine: SimilarityEngine | None = None,
) -> dict:
    """Verify narration claims against the scene signatures assigned to each beat."""
    engine = engine or make_similarity_engine(cfg, project_dir=project_dir)
    signature_map = {str(s.get("scene_id")): s for s in signatures if s.get("scene_id")}
    plan_map = {
        str(p.get("scene_id")): p
        for p in (narration_plan or {}).get("segments", []) or []
        if p.get("scene_id")
    }
    prewrite_map = {
        str(b.get("beat_id")): b
        for b in (match_report or {}).get("beats", []) or []
        if b.get("beat_id")
    }

    configured_threshold = float(getattr(cfg, "semantic_verifier_claim_threshold", 0.48) or 0.48)
    degraded = engine.backend == "lexical"
    # Lexical similarity is not a safe cross-language factual judge. In degraded
    # mode we still emit telemetry but do not trigger automatic factual repair.
    threshold = configured_threshold if not degraded else 0.12

    beat_reports: list[dict[str, Any]] = []
    all_unsupported: list[dict[str, Any]] = []
    similarity_values: list[float] = []

    for segment in narration:
        beat_id = str(segment.get("scene_id", ""))
        plan = plan_map.get(beat_id, {}) or {}
        scene_ids = [str(s) for s in segment.get("covers_scene_ids", plan.get("covers_scene_ids", [])) or [] if s]
        evidence = _evidence_text(scene_ids, signature_map)
        claims = [str(v).strip() for v in segment.get("claims", []) or [] if str(v).strip()]
        if not claims:
            claims = [str(segment.get("text", "") or "").strip()] if str(segment.get("text", "") or "").strip() else []

        claim_reports = []
        unsupported = []
        for claim in claims:
            score = engine.similarity(claim, evidence) if evidence else 0.0
            similarity_values.append(score)
            supported = bool(evidence) and score >= threshold
            item = {"claim": claim, "similarity": round(score, 4), "supported": supported}
            claim_reports.append(item)
            if not supported:
                unsupported.append(item)
                all_unsupported.append({"beat_id": beat_id, **item})

        prewrite = prewrite_map.get(beat_id, {}) or {}
        if degraded:
            status = "degraded"
        elif unsupported:
            status = "repair"
        else:
            status = "accepted"

        beat_reports.append(
            {
                "beat_id": beat_id,
                "scene_ids": scene_ids,
                "prewrite_score": prewrite.get("prewrite_score"),
                "claims": claim_reports,
                "claim_coverage": round(
                    (len(claim_reports) - len(unsupported)) / max(1, len(claim_reports)), 3
                ),
                "unsupported_claims": [item["claim"] for item in unsupported],
                "status": status,
            }
        )

    coverage = (
        sum(1 for item in all_unsupported if not item.get("supported"))
        / max(1, sum(len(b.get("claims", [])) for b in beat_reports))
    )
    return {
        "version": VERIFIER_VERSION,
        "language": language,
        "backend": engine.backend,
        "degraded": degraded,
        "claim_threshold": threshold,
        "beats": beat_reports,
        "unsupported_claims": all_unsupported,
        "metrics": {
            "unsupported_claim_rate": round(coverage, 3),
            "mean_claim_similarity": round(sum(similarity_values) / max(1, len(similarity_values)), 4),
            "repair_beat_count": sum(1 for b in beat_reports if b.get("status") == "repair"),
        },
    }


def repair_unsupported_beats(
    *,
    narration: list[dict],
    verification: dict,
    narration_plan: dict,
    signatures: list[dict],
    cfg,
    language: str,
) -> tuple[list[dict], list[dict]]:
    """Rewrite only verifier-failed beats while keeping their scene IDs fixed."""
    if verification.get("degraded"):
        return narration, []
    if not bool(getattr(cfg, "semantic_verifier_auto_repair", True)):
        return narration, []
    if not getattr(cfg, "api_key_gemini", ""):
        return narration, []

    failed = {
        str(item.get("beat_id")): item
        for item in verification.get("beats", []) or []
        if item.get("status") == "repair"
    }
    if not failed:
        return narration, []

    plan_map = {
        str(p.get("scene_id")): p
        for p in (narration_plan or {}).get("segments", []) or []
        if p.get("scene_id")
    }
    signature_map = {str(s.get("scene_id")): s for s in signatures if s.get("scene_id")}
    revised_raw = [dict(item) for item in narration]
    repair_log: list[dict] = []

    for index, segment in enumerate(revised_raw):
        beat_id = str(segment.get("scene_id", ""))
        issue = failed.get(beat_id)
        if not issue:
            continue
        plan = plan_map.get(beat_id, {}) or {}
        scene_ids = [str(s) for s in segment.get("covers_scene_ids", plan.get("covers_scene_ids", [])) or [] if s]
        prompt = REPAIR_PROMPT.format(
            language_code=language,
            language_name=_language_name(language),
            plan_json=json.dumps(plan, ensure_ascii=False, indent=2),
            evidence_json=json.dumps(
                [signature_map.get(sid, {}) for sid in scene_ids if sid in signature_map],
                ensure_ascii=False,
                indent=2,
            )[:18000],
            current_json=json.dumps(
                {"scene_id": beat_id, "text": segment.get("text", ""), "claims": segment.get("claims", [])},
                ensure_ascii=False,
                indent=2,
            ),
            unsupported_json=json.dumps(issue.get("unsupported_claims", []), ensure_ascii=False, indent=2),
        )

        def validator(payload: Any) -> str | None:
            mismatch = payload_language_mismatch_reason(payload, language)
            if mismatch:
                return mismatch
            if not isinstance(payload, dict):
                return "repair response must be an object"
            if str(payload.get("scene_id", "")) != beat_id:
                return f"scene_id must remain {beat_id}"
            if not str(payload.get("text", "") or "").strip():
                return "repair text is empty"
            return None

        try:
            payload = generate_story_json(
                cfg,
                prompt,
                task_name=f"v21-beat-repair-{beat_id}",
                temperature=0.2,
                response_validator=validator,
                model_attr="narration_model",
            )
            revised_raw[index]["text"] = str(payload.get("text", "") or "").strip()
            revised_raw[index]["claims"] = [
                str(v).strip() for v in payload.get("claims", []) or [] if str(v).strip()
            ][:8]
            repair_log.append({"beat_id": beat_id, "action": "rewrite_beat", "status": "applied"})
        except Exception as exc:
            repair_log.append({"beat_id": beat_id, "action": "rewrite_beat", "status": "failed", "error": str(exc)})

    # Re-normalize word counts/timing with the existing writer validation logic.
    try:
        from dubbingstory.story.script_writer import _validate_narration

        scene_stubs: list[dict] = []
        seen: set[str] = set()
        for plan in (narration_plan or {}).get("segments", []) or []:
            for sid in plan.get("covers_scene_ids", []) or []:
                sid = str(sid)
                if sid and sid not in seen:
                    scene_stubs.append({"scene_id": sid})
                    seen.add(sid)
        revised_raw = _validate_narration(revised_raw, scene_stubs, narration_plan, language)
    except Exception:
        pass
    return revised_raw, repair_log
