from types import SimpleNamespace

from dubbingstory.story.beat_dedup import deduplicate_beats
from dubbingstory.story.narration_planner import build_narration_plan
from dubbingstory.story.scene_cards import build_scene_cards
from dubbingstory.story.scene_continuity import build_scene_deltas
from dubbingstory.story.scene_signature import build_scene_signatures
from dubbingstory.story.script_writer import _validate_narration
from dubbingstory.story.semantic_matcher import SimilarityEngine, match_beats
from dubbingstory.story.semantic_verifier import verify_narration


def _cfg(**overrides):
    data = dict(
        narration_target_wpm=160,
        narration_speech_coverage=0.84,
        narration_min_wpm=135,
        narration_max_wpm=175,
        narration_plan_llm_refine=False,
        semantic_match_embedding_model="lexical",
        semantic_match_embedding_device="cpu",
        semantic_match_min_evidence_quality=0.2,
        semantic_match_time_window_ratio=0.5,
        semantic_match_max_backtrack_ratio=0.04,
        semantic_match_top_k=5,
        semantic_match_weak_threshold=0.18,
        semantic_match_max_scenes_per_beat=1,
        semantic_match_weight_semantic=0.45,
        semantic_match_weight_temporal=0.15,
        semantic_match_weight_continuity=0.15,
        semantic_match_weight_state_change=0.10,
        semantic_match_weight_importance=0.05,
        semantic_match_weight_confidence=0.10,
        semantic_match_reuse_penalty=0.18,
        semantic_match_parent_reuse_penalty=0.10,
        semantic_match_redundancy_penalty=0.08,
        summary_duration_tolerance=1.05,
        semantic_verifier_claim_threshold=0.48,
    )
    data.update(overrides)
    return SimpleNamespace(**data)


def _storyboard():
    return {
        "total_duration": 30,
        "scenes": [
            {
                "scene_id": "scene_001", "parent_scene_id": "shot_01", "window_index": 0, "window_count": 2,
                "start_time": 0, "end_time": 10, "duration": 10,
                "analysis": {"selected": True, "people": "mechanic", "visible_objects": ["metal component", "lathe"],
                             "action": "mechanic machines the outer diameter", "changes": "diameter becomes smaller",
                             "state_before": "larger diameter", "state_after": "smaller diameter",
                             "likely_context": "machining a component to fit", "confidence": 0.93},
                "narrative": {"role": "process", "importance": 0.8},
            },
            {
                "scene_id": "scene_002", "parent_scene_id": "shot_01", "window_index": 1, "window_count": 2,
                "start_time": 10, "end_time": 20, "duration": 10,
                "analysis": {"selected": True, "people": "worker", "visible_objects": ["component", "caliper"],
                             "action": "worker measures the component", "changes": "size is checked",
                             "state_before": "machined component", "state_after": "measured component",
                             "likely_context": "checking final dimensions", "confidence": 0.91},
                "narrative": {"role": "result", "importance": 0.9},
            },
            {
                "scene_id": "scene_003", "parent_scene_id": "shot_02", "window_index": 0, "window_count": 1,
                "start_time": 20, "end_time": 30, "duration": 10,
                "analysis": {"selected": True, "people": "mechanic", "visible_objects": ["component", "wrench"],
                             "action": "mechanic installs the component", "changes": "component is mounted",
                             "state_before": "loose component", "state_after": "installed component",
                             "likely_context": "final assembly", "confidence": 0.94},
                "narrative": {"role": "resolution", "importance": 0.95},
            },
        ],
    }


def test_parent_window_delta_and_signatures():
    cards = build_scene_cards(_storyboard())
    assert cards[1]["parent_scene_id"] == "shot_01"
    assert cards[1]["window_index"] == 1
    deltas = build_scene_deltas(cards)
    assert deltas[1]["same_parent_window"] is True
    assert deltas[1]["relationship"] in {"progression", "continuation"}
    signatures = build_scene_signatures(cards, deltas)
    assert signatures[0]["deep_evidence"] is True
    assert "ACTION:" in signatures[0]["signature"]
    assert "CONTINUITY:" in signatures[1]["signature"]


def test_beat_dedup_keeps_more_important_duplicate():
    beats = [
        {"beat_id": "a", "intent": "component diameter is reduced progressively", "importance": 0.4},
        {"beat_id": "b", "intent": "component diameter is reduced progressively", "importance": 0.9},
        {"beat_id": "c", "intent": "component is measured", "importance": 0.7},
    ]
    result = deduplicate_beats(beats, threshold=0.9)
    assert len(result) == 2
    assert result[0]["importance"] == 0.9
    assert [b["beat_id"] for b in result] == ["beat_001", "beat_002"]


def test_semantic_matcher_selects_relevant_chronological_evidence():
    board = _storyboard()
    cards = build_scene_cards(board)
    signatures = build_scene_signatures(cards, build_scene_deltas(cards))
    beats = [
        {"beat_id": "beat_001", "beat_index": 0, "intent": "reduce the component diameter", "required_facts": ["diameter becomes smaller"], "approx_ratio": 0.15, "target_duration_sec": 10},
        {"beat_id": "beat_002", "beat_index": 1, "intent": "measure the final size", "required_facts": ["component is measured"], "approx_ratio": 0.50, "target_duration_sec": 10},
    ]
    story_plan = {"scene_roles": [{"scene_id": "scene_001", "story_importance": 0.8}, {"scene_id": "scene_002", "story_importance": 0.9}]}
    report = match_beats(beats=beats, signatures=signatures, storyboard=board, story_plan=story_plan, cfg=_cfg(), target_duration=20, max_scenes=2, engine=SimilarityEngine("lexical"))
    assert report["beat_coverage"] == 1.0
    assert report["selected_scene_ids"][0] == "scene_001"
    assert "scene_002" in report["selected_scene_ids"]


def test_narration_plan_uses_story_beat_ids_and_fixed_scene_assignment():
    board = _storyboard()
    beats = [{"beat_id": "beat_001", "story_role": "development", "intent": "show machining", "required_facts": ["diameter becomes smaller"], "forbidden_claims": ["already installed"], "importance": 0.8, "approx_ratio": 0.2}]
    report = {"beats": [{"beat_id": "beat_001", "selected": ["scene_001"]}]}
    plan = build_narration_plan(board, [board["scenes"][0]], story_plan={}, story_memory={}, cfg=_cfg(), mode="summary", story_beats=beats, match_report=report)
    item = plan["segments"][0]
    assert plan["semantic_match_active"] is True
    assert item["scene_id"] == "beat_001"
    assert item["covers_scene_ids"] == ["scene_001"]
    assert item["forbidden_claims"] == ["already installed"]


def test_writer_preserves_claims_and_verifier_emits_degraded_telemetry():
    board = _storyboard()
    plan = {
        "target_wpm": 160, "total_duration": 10,
        "segments": [{"scene_id": "beat_001", "covers_scene_ids": ["scene_001"], "timeline_start": 0, "timeline_end": 10,
                      "target_words": 8, "min_words": 4, "max_words": 18, "story_role": "development", "story_importance": 0.8,
                      "must_explain": ["diameter becomes smaller"]}],
    }
    raw = [{"scene_id": "beat_001", "text": "Diameter komponen diperkecil secara bertahap.", "claims": ["diameter becomes smaller"], "importance": 0.8}]
    validated = _validate_narration(raw, board["scenes"], plan, "id")
    assert validated[0]["claims"] == ["diameter becomes smaller"]
    signatures = build_scene_signatures(build_scene_cards(board), build_scene_deltas(build_scene_cards(board)))
    verification = verify_narration(narration=validated, narration_plan=plan, signatures=signatures, match_report={"beats": []}, cfg=_cfg(), language="id", engine=SimilarityEngine("lexical"))
    assert verification["degraded"] is True
    assert verification["metrics"]["repair_beat_count"] == 0
