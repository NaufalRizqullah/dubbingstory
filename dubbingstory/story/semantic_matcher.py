"""Pre-write beat -> footage matching for DubbingStory V2.1.

The matcher retrieves only evidence-bearing scene windows, reranks a Top-K set
with story/timeline/continuity signals, penalizes recent scene/parent reuse and
keeps chronological order by default.  Sentence-transformers is optional; a
pure-Python lexical fallback keeps the pipeline usable in degraded mode.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any


MATCHER_VERSION = "v2.1.0"
_TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(str(text or "").lower()) if len(t) > 2}


def lexical_similarity(left: str, right: str) -> float:
    a = _tokens(left)
    b = _tokens(right)
    if not a or not b:
        return 0.0
    unigram = len(a & b) / len(a | b)
    # Character bigrams add some tolerance for morphology/word variants.
    la = str(left or "").lower().replace(" ", "_")
    lb = str(right or "").lower().replace(" ", "_")
    ba = {la[i : i + 2] for i in range(max(0, len(la) - 1))}
    bb = {lb[i : i + 2] for i in range(max(0, len(lb) - 1))}
    bigram = (len(ba & bb) / len(ba | bb)) if ba and bb else 0.0
    return _clamp01(unigram * 0.72 + bigram * 0.28)


@dataclass
class SimilarityEngine:
    model_name: str
    device: str = "cpu"
    project_dir: str | None = None

    def __post_init__(self) -> None:
        self.backend = "lexical"
        self._model = None
        self._cache: dict[str, list[float]] = {}
        self._load_embedding_model()

    def _load_embedding_model(self) -> None:
        if not self.model_name or self.model_name.lower() in {"none", "lexical"}:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.backend = f"sentence-transformers:{self.model_name}"
        except Exception as exc:
            print(f"   ⚠️ Semantic embeddings unavailable; lexical fallback active: {exc}")
            self._model = None
            self.backend = "lexical"

    def _embedding(self, text: str) -> list[float] | None:
        text = str(text or "").strip()
        if not text or self._model is None:
            return None
        if text in self._cache:
            return self._cache[text]
        vec = self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        values = [float(v) for v in vec.tolist()]
        self._cache[text] = values
        return values

    def similarity(self, left: str, right: str) -> float:
        if self._model is None:
            return lexical_similarity(left, right)
        a = self._embedding(left)
        b = self._embedding(right)
        if not a or not b:
            return lexical_similarity(left, right)
        return _clamp01(sum(x * y for x, y in zip(a, b)))


def make_similarity_engine(cfg, project_dir: str | None = None) -> SimilarityEngine:
    return SimilarityEngine(
        model_name=str(
            getattr(
                cfg,
                "semantic_match_embedding_model",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            )
            or "lexical"
        ),
        device=str(getattr(cfg, "semantic_match_embedding_device", "cpu") or "cpu"),
        project_dir=project_dir,
    )


def beat_query_text(beat: dict) -> str:
    parts = [
        str(beat.get("intent", "") or ""),
        " ".join(str(v) for v in beat.get("required_facts", []) or []),
        str(beat.get("story_role", "") or ""),
    ]
    return "\n".join(part for part in parts if part.strip()).strip()


def _story_role_map(story_plan: dict | None) -> dict[str, dict]:
    return {
        str(item.get("scene_id")): item
        for item in (story_plan or {}).get("scene_roles", []) or []
        if isinstance(item, dict) and item.get("scene_id")
    }


def _video_end(signatures: list[dict]) -> float:
    return max((float(s.get("end_time", 0) or 0) for s in signatures), default=1.0) or 1.0


def _scene_ratio(sig: dict, video_end: float) -> float:
    mid = (float(sig.get("start_time", 0) or 0) + float(sig.get("end_time", 0) or 0)) / 2.0
    return _clamp01(mid / max(0.001, video_end))


def _temporal_score(beat: dict, sig: dict, video_end: float, window: float) -> float:
    target = _clamp01(beat.get("approx_ratio", 0.5), 0.5)
    distance = abs(_scene_ratio(sig, video_end) - target)
    return _clamp01(1.0 - distance / max(0.05, window * 2.0))


def _state_text(sig: dict) -> str:
    state = sig.get("state_change", {}) or {}
    return " ".join(
        [
            str(state.get("before", "") or ""),
            str(state.get("after", "") or ""),
            " ".join(str(v) for v in sig.get("new_information_only", []) or []),
        ]
    ).strip()


def _continuity_score(sig: dict, previous_selected: dict | None) -> float:
    if not previous_selected:
        return 0.65
    prev_end = float(previous_selected.get("end_time", 0) or 0)
    start = float(sig.get("start_time", 0) or 0)
    gap = start - prev_end
    if gap < -0.05:
        return 0.0
    if str(sig.get("parent_scene_id", "")) and sig.get("parent_scene_id") == previous_selected.get("parent_scene_id"):
        return 0.95
    relation = str(sig.get("relationship", "") or "")
    base = {
        "progression": 0.9,
        "continuation": 0.88,
        "result": 0.86,
        "transition": 0.74,
        "bridge": 0.72,
        "new_topic": 0.55,
        "uncertain": 0.45,
    }.get(relation, 0.65)
    if gap <= 20:
        base += 0.08
    elif gap > 180:
        base -= 0.08
    return _clamp01(base)


def _candidate_pool(beat: dict, signatures: list[dict], video_end: float, cfg) -> list[dict]:
    quality_min = float(getattr(cfg, "semantic_match_min_evidence_quality", 0.25) or 0.25)
    pool = [
        sig
        for sig in signatures
        if sig.get("deep_evidence", False)
        and float(sig.get("evidence_quality", 0) or 0) >= quality_min
        and str(sig.get("signature", "") or "").strip()
    ]
    if not pool:
        return []

    target = _clamp01(beat.get("approx_ratio", 0.5), 0.5)
    base_window = float(getattr(cfg, "semantic_match_time_window_ratio", 0.18) or 0.18)
    nearby = [sig for sig in pool if abs(_scene_ratio(sig, video_end) - target) <= base_window]
    if nearby:
        return nearby
    expanded = [sig for sig in pool if abs(_scene_ratio(sig, video_end) - target) <= min(0.5, base_window * 2.0)]
    return expanded or pool


def _score_candidates(
    beat: dict,
    pool: list[dict],
    *,
    engine: SimilarityEngine,
    story_plan: dict,
    previous_selected: dict | None,
    used_scene_ids: set[str],
    used_parent_ids: set[str],
    recent_signatures: list[str],
    cfg,
    video_end: float,
) -> list[dict]:
    query = beat_query_text(beat)
    roles = _story_role_map(story_plan)
    window = float(getattr(cfg, "semantic_match_time_window_ratio", 0.18) or 0.18)
    allow_backtrack = float(getattr(cfg, "semantic_match_max_backtrack_ratio", 0.04) or 0.04) * video_end

    w_sem = float(getattr(cfg, "semantic_match_weight_semantic", 0.45) or 0.45)
    w_temp = float(getattr(cfg, "semantic_match_weight_temporal", 0.15) or 0.15)
    w_cont = float(getattr(cfg, "semantic_match_weight_continuity", 0.15) or 0.15)
    w_state = float(getattr(cfg, "semantic_match_weight_state_change", 0.10) or 0.10)
    w_imp = float(getattr(cfg, "semantic_match_weight_importance", 0.05) or 0.05)
    w_conf = float(getattr(cfg, "semantic_match_weight_confidence", 0.10) or 0.10)

    scene_penalty = float(getattr(cfg, "semantic_match_reuse_penalty", 0.18) or 0.18)
    parent_penalty = float(getattr(cfg, "semantic_match_parent_reuse_penalty", 0.10) or 0.10)
    redundancy_weight = float(getattr(cfg, "semantic_match_redundancy_penalty", 0.08) or 0.08)

    scored: list[dict] = []
    for sig in pool:
        sid = str(sig.get("scene_id", ""))
        parent = str(sig.get("parent_scene_id", "") or "")
        start = float(sig.get("start_time", 0) or 0)
        backtrack = 0.0
        if previous_selected:
            prev_start = float(previous_selected.get("start_time", 0) or 0)
            if start + allow_backtrack < prev_start:
                # Hard chronological guard for ordinary recap beats.
                continue
            if start < prev_start:
                backtrack = 0.12

        semantic = engine.similarity(query, str(sig.get("signature", "") or ""))
        temporal = _temporal_score(beat, sig, video_end, window)
        continuity = _continuity_score(sig, previous_selected)
        state = engine.similarity(" ".join(str(v) for v in beat.get("required_facts", []) or []), _state_text(sig))
        importance = _clamp01((roles.get(sid, {}) or {}).get("story_importance", 0.5), 0.5)
        confidence = _clamp01(sig.get("confidence", 0), 0.0)
        reuse = scene_penalty if sid in used_scene_ids else 0.0
        parent_reuse = parent_penalty if parent and parent in used_parent_ids else 0.0
        redundancy = 0.0
        if recent_signatures:
            redundancy = max(
                engine.similarity(str(sig.get("signature", "") or ""), recent)
                for recent in recent_signatures[-3:]
            ) * redundancy_weight

        final = (
            semantic * w_sem
            + temporal * w_temp
            + continuity * w_cont
            + state * w_state
            + importance * w_imp
            + confidence * w_conf
            - reuse
            - parent_reuse
            - redundancy
            - backtrack
        )
        scored.append(
            {
                "scene_id": sid,
                "parent_scene_id": parent or None,
                "start_time": start,
                "end_time": float(sig.get("end_time", 0) or 0),
                "duration": float(sig.get("duration", 0) or 0),
                "semantic": round(semantic, 4),
                "temporal": round(temporal, 4),
                "continuity": round(continuity, 4),
                "state_change": round(state, 4),
                "importance": round(importance, 4),
                "confidence": round(confidence, 4),
                "reuse_penalty": round(reuse, 4),
                "parent_reuse_penalty": round(parent_reuse, 4),
                "redundancy_penalty": round(redundancy, 4),
                "backtrack_penalty": round(backtrack, 4),
                "final": round(final, 4),
                "signature": sig.get("signature", ""),
                "relationship": sig.get("relationship", "uncertain"),
            }
        )
    scored.sort(key=lambda item: item["final"], reverse=True)
    return scored


def match_beats(
    *,
    beats: list[dict],
    signatures: list[dict],
    storyboard: dict,
    story_plan: dict,
    cfg,
    target_duration: float | None = None,
    max_scenes: int | None = None,
    project_dir: str | None = None,
    engine: SimilarityEngine | None = None,
) -> dict:
    """Return a verbose pre-write match report and proposed final scene IDs."""
    engine = engine or make_similarity_engine(cfg, project_dir=project_dir)
    scene_map = {str(s.get("scene_id")): s for s in storyboard.get("scenes", []) or []}
    video_end = _video_end(signatures)
    top_k = max(1, int(getattr(cfg, "semantic_match_top_k", 5) or 5))
    weak_threshold = float(getattr(cfg, "semantic_match_weak_threshold", 0.55) or 0.55)
    hard_limit = None
    if target_duration:
        tolerance = float(getattr(cfg, "summary_duration_tolerance", 1.05) or 1.05)
        hard_limit = float(target_duration) * max(1.0, tolerance)

    used_scene_ids: set[str] = set()
    used_parent_ids: set[str] = set()
    recent_signatures: list[str] = []
    previous_selected: dict | None = None
    beat_reports: list[dict] = []
    selected_ids: list[str] = []
    selected_duration = 0.0

    for beat in beats:
        pool = _candidate_pool(beat, signatures, video_end, cfg)
        scored = _score_candidates(
            beat,
            pool,
            engine=engine,
            story_plan=story_plan,
            previous_selected=previous_selected,
            used_scene_ids=used_scene_ids,
            used_parent_ids=used_parent_ids,
            recent_signatures=recent_signatures,
            cfg=cfg,
            video_end=video_end,
        )
        candidates = scored[:top_k]
        chosen: list[dict] = []

        # Prefer the best candidate that still fits the global duration/scene budget.
        for candidate in candidates:
            if candidate["scene_id"] in used_scene_ids:
                continue
            if max_scenes is not None and len(selected_ids) >= int(max_scenes):
                break
            if hard_limit is not None and selected_duration + candidate["duration"] > hard_limit + 1e-6:
                continue
            chosen = [candidate]
            break

        if not chosen and candidates and (hard_limit is None or not selected_ids):
            # First beat must have an anchor even when a tiny target duration was requested.
            chosen = [candidates[0]]

        # Optional second window for a beat that needs more visual time. Only use
        # a continuity-compatible scene and never force diversity at the cost of relevance.
        if chosen:
            target_beat_duration = float(beat.get("target_duration_sec", 0) or 0)
            max_per_beat = max(1, int(getattr(cfg, "semantic_match_max_scenes_per_beat", 2) or 2))
            primary = chosen[0]
            if max_per_beat > 1 and primary["duration"] < target_beat_duration * 0.7:
                for candidate in candidates[1:]:
                    if candidate["scene_id"] in used_scene_ids or candidate["scene_id"] == primary["scene_id"]:
                        continue
                    compatible = (
                        candidate.get("parent_scene_id") == primary.get("parent_scene_id")
                        or candidate.get("relationship") in {"continuation", "progression", "result"}
                    )
                    if not compatible or candidate["final"] < max(weak_threshold, primary["final"] * 0.78):
                        continue
                    if candidate["start_time"] + 1e-6 < primary["start_time"]:
                        continue
                    if max_scenes is not None and len(selected_ids) + len(chosen) >= int(max_scenes):
                        break
                    if hard_limit is not None and selected_duration + sum(c["duration"] for c in chosen) + candidate["duration"] > hard_limit + 1e-6:
                        continue
                    chosen.append(candidate)
                    break

        for candidate in chosen:
            sid = candidate["scene_id"]
            if sid not in selected_ids:
                selected_ids.append(sid)
                selected_duration += candidate["duration"]
            used_scene_ids.add(sid)
            parent = str(candidate.get("parent_scene_id", "") or "")
            if parent:
                used_parent_ids.add(parent)
            recent_signatures.append(str(candidate.get("signature", "") or ""))
            previous_selected = candidate

        best_score = float(chosen[0]["final"] if chosen else (candidates[0]["final"] if candidates else 0.0))
        status = "accepted" if chosen and best_score >= weak_threshold else ("weak" if candidates else "no_candidate")
        beat_reports.append(
            {
                "beat_id": beat.get("beat_id"),
                "beat_index": beat.get("beat_index"),
                "intent": beat.get("intent", ""),
                "required_facts": beat.get("required_facts", []),
                "approx_ratio": beat.get("approx_ratio", 0),
                "candidates": [
                    {key: value for key, value in c.items() if key != "signature"}
                    for c in candidates
                ],
                "selected": [c["scene_id"] for c in chosen],
                "prewrite_score": round(best_score, 4),
                "status": status,
            }
        )

    selected_scenes = [scene_map[sid] for sid in selected_ids if sid in scene_map]
    weak = [b["beat_id"] for b in beat_reports if b["status"] != "accepted"]
    coverage = len([b for b in beat_reports if b["selected"]]) / max(1, len(beat_reports))

    return {
        "version": MATCHER_VERSION,
        "backend": engine.backend,
        "mode": "prewrite",
        "target_duration": target_duration,
        "selected_duration": round(sum(float(s.get("duration", 0) or 0) for s in selected_scenes), 3),
        "selected_scene_ids": selected_ids,
        "beat_coverage": round(coverage, 3),
        "weak_beat_ids": weak,
        "beats": beat_reports,
        "metrics": {
            "weak_match_rate": round(len(weak) / max(1, len(beat_reports)), 3),
            "mean_prewrite_score": round(
                sum(float(b.get("prewrite_score", 0) or 0) for b in beat_reports) / max(1, len(beat_reports)), 4
            ),
            "selected_parent_reuse": len(selected_ids) - len({str(scene_map[sid].get("parent_scene_id", "") or sid) for sid in selected_ids if sid in scene_map}),
        },
    }


def _screening_text(scene: dict) -> str:
    analysis = scene.get("analysis", {}) or {}
    screen = analysis.get("screening", {}) or {}
    parts = [
        str(screen.get("action", "") or ""),
        str(screen.get("likely_context", "") or ""),
        str(screen.get("story_relevance_reason", "") or ""),
        str(screen.get("transcript_text", "") or ""),
        str(analysis.get("transcript_text", "") or ""),
    ]
    return " ".join(p for p in parts if p.strip()).strip()


def suggest_promotion_candidates(
    *,
    beats: list[dict],
    match_report: dict,
    storyboard: dict,
    cfg,
    project_dir: str | None = None,
    engine: SimilarityEngine | None = None,
) -> dict[str, list[str]]:
    """Suggest screening-only scene windows for on-demand deep promotion."""
    engine = engine or make_similarity_engine(cfg, project_dir=project_dir)
    weak_ids = set(match_report.get("weak_beat_ids", []) or [])
    if not weak_ids:
        return {}
    beat_map = {str(b.get("beat_id")): b for b in beats}
    video_end = max(
        (float(s.get("end_time", 0) or 0) for s in storyboard.get("scenes", []) or []),
        default=1.0,
    ) or 1.0
    per_beat = max(1, int(getattr(cfg, "semantic_match_promotion_candidates_per_beat", 2) or 2))
    window = float(getattr(cfg, "semantic_match_promotion_time_window_ratio", 0.24) or 0.24)

    proposals: dict[str, list[str]] = {}
    already_deep = {
        str(s.get("scene_id"))
        for s in storyboard.get("scenes", []) or []
        if bool((s.get("analysis", {}) or {}).get("selected", True))
        and float((s.get("analysis", {}) or {}).get("confidence", 0) or 0) > 0
    }

    for beat_id in weak_ids:
        beat = beat_map.get(beat_id)
        if not beat:
            continue
        query = beat_query_text(beat)
        target = _clamp01(beat.get("approx_ratio", 0.5), 0.5)
        scored: list[tuple[float, str]] = []
        for scene in storyboard.get("scenes", []) or []:
            sid = str(scene.get("scene_id", ""))
            if not sid or sid in already_deep:
                continue
            text = _screening_text(scene)
            if not text:
                continue
            mid = (float(scene.get("start_time", 0) or 0) + float(scene.get("end_time", 0) or 0)) / 2.0
            ratio = _clamp01(mid / video_end)
            if abs(ratio - target) > window:
                continue
            semantic = engine.similarity(query, text)
            screen_score = _clamp01((scene.get("analysis", {}) or {}).get("screen_score", 0.4), 0.4)
            temporal = _clamp01(1.0 - abs(ratio - target) / max(0.05, window))
            final = semantic * 0.65 + temporal * 0.20 + screen_score * 0.15
            scored.append((final, sid))
        scored.sort(reverse=True)
        if scored:
            proposals[beat_id] = [sid for _, sid in scored[:per_beat]]
    return proposals


def selected_scenes_from_report(match_report: dict, storyboard: dict) -> list[dict]:
    scene_map = {str(s.get("scene_id")): s for s in storyboard.get("scenes", []) or []}
    return [scene_map[sid] for sid in match_report.get("selected_scene_ids", []) or [] if sid in scene_map]
