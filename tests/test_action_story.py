"""Grounded narration contracts; run these mocked tests in Kaggle, not locally."""

import copy
import json
from types import SimpleNamespace

import pytest

from dubbingstory.story import script_writer
from dubbingstory.story.action_anchors import (
    canonical_action_anchors, compact_visual_events, phrase_word_span, scene_visual_events,
)
from dubbingstory.story.narration_planner import build_narration_plan
from dubbingstory.story.prompt_templates import build_rewrite_prompt, build_v2_narration_prompt
from dubbingstory.story.scene_cards import build_scene_cards, compact_scene_cards


def event(name="cut", start=4.0, end=5.0, confidence=0.9):
    return {"event_id": name, "scene_id": "scene_a", "start_time": start, "end_time": end,
            "action": "The worker cuts the metal", "state_change": "The metal separates",
            "confidence": confidence, "frame_ids": [f"frame_{name}"],
            "timing_basis": "sample_window"}


def source_scene(events=None):
    return {"scene_id": "scene_a", "start_time": 0.0, "end_time": 12.0, "duration": 12.0,
            "analysis": {"action": "The worker cuts the metal", "events": [event()] if events is None else events}}


def config(**values):
    defaults = {"narration_action_sync": True, "narration_plan_llm_refine": False,
                "narration_group_scenes": False, "narration_source_duration": 12.0,
                "tts_narration_fit": "rewrite", "tts_min_gap": 0.2}
    return SimpleNamespace(**{**defaults, **values})


def plan(scenes=None, mode="full", **values):
    scenes = [source_scene()] if scenes is None else scenes
    return build_narration_plan({"scenes": scenes}, scenes, story_plan={}, story_memory={},
                                cfg=config(**values), mode=mode)


def spoken_segment(text="Besi dipotong perlahan sebelum bagian berikutnya dikerjakan."):
    entry = plan()["segments"][0]
    raw = [{"scene_id": "scene_a", "text": text, "claims": ["Besi dipotong"],
            "action_anchors": [{"event_id": "cut", "phrase": "dipotong"}]}]
    return script_writer._validate_narration(raw, [source_scene()], {"segments": [entry]}, "id")[0]


def test_full_plan_and_writer_keep_source_event_evidence():
    result = plan()
    assert result["action_sync_enabled"] is True
    assert result["segments"][0]["anchor_event_ids"] == ["cut"]
    segment = spoken_segment()
    assert segment["action_sync_enabled"] is True
    assert segment["visual_events"][0]["frame_ids"] == ["frame_cut"]
    assert segment["action_anchors"][0]["start_time"] == 4.0
    assert segment["action_anchors"][0]["timing_basis"] == "sample_window"


def test_action_mode_requires_timestamped_analysis_but_legacy_does_not():
    scene = source_scene()
    del scene["analysis"]["events"]
    with pytest.raises(ValueError, match="rerun.*analyze"):
        plan([scene])
    assert "action_sync_enabled" not in plan([scene], narration_action_sync=False)


@pytest.mark.parametrize("events", [[], [event(confidence=0.1)]])
def test_uncertain_or_empty_observations_do_not_invent_an_anchor(events):
    entry = plan([source_scene(events)])["segments"][0]
    assert entry["anchor_event_ids"] == []
    assert entry["action_sync_status"] == "no_eligible_events"
    assert canonical_action_anchors([], "Pengerjaan terlihat berlanjut.", entry) == []


def test_summary_ignores_action_sync_and_preserves_cumulative_timeline():
    scene = source_scene()
    scene.update(start_time=60.0, end_time=72.0)
    result = plan([scene], mode="summary")
    assert "action_sync_enabled" not in result
    assert "visual_events" not in result["segments"][0]
    assert result["segments"][0]["timeline_start"] == 0.0
    assert result["total_duration"] == 12.0


def test_direct_action_plan_uses_strict_duration_budget_even_without_cli():
    short = source_scene([])
    short.update(end_time=0.5, duration=0.5)
    entry = plan([short], tts_narration_fit="legacy")["segments"][0]
    assert entry["max_words"] <= 2


def test_model_cannot_override_observed_event_timestamps():
    entry = plan()["segments"][0]
    checked = canonical_action_anchors(
        [{"event_id": "cut", "phrase": "dipotong", "start_time": 0, "end_time": 100}],
        "Besi dipotong perlahan.", entry,
    )
    assert (checked[0]["start_time"], checked[0]["end_time"]) == (4, 5)


@pytest.mark.parametrize("raw,text", [
    ([], "Besi dipotong."),
    ([{"event_id": "invented", "phrase": "dipotong"}], "Besi dipotong."),
    ([{"event_id": "cut", "phrase": "dibor"}], "Besi dipotong."),
    ([{"event_id": "cut", "phrase": "dipotong"}], "Besi dipotong lalu dipotong kembali."),
    ([{"event_id": "cut", "phrase": "dipotong"}] * 2, "Besi dipotong."),
])
def test_invalid_missing_or_ambiguous_spoken_anchor_is_rejected(raw, text):
    with pytest.raises(ValueError):
        canonical_action_anchors(raw, text, plan()["segments"][0])


@pytest.mark.parametrize("field,value", [
    ("start_time", float("nan")), ("end_time", float("inf")),
    ("confidence", 2), ("frame_ids", []), ("scene_id", "other"),
])
def test_saved_script_event_is_revalidated_before_use(field, value):
    entry = plan()["segments"][0]
    entry["visual_events"][0][field] = value
    with pytest.raises(ValueError):
        canonical_action_anchors([{"event_id": "cut", "phrase": "dipotong"}], "Besi dipotong.", entry)


@pytest.mark.parametrize("bad", [{}, None, "events", 0])
def test_invalid_event_collection_is_not_treated_as_empty_evidence(bad):
    scene = source_scene()
    scene["analysis"]["events"] = bad
    with pytest.raises(ValueError, match="list"):
        scene_visual_events(scene)


def test_phrase_span_handles_unicode_punctuation_and_provider_token_groups():
    words = [{"word": "Lalu,"}, {"word": "BESI dipotong."}, {"word": "Rapi."}]
    assert phrase_word_span(words, "besi dipotong") == (1, 2)
    assert phrase_word_span([{"text": "café"}, {"text": "dirapikan"}], "cafe\u0301 dirapikan") == (0, 2)


def dense_events():
    return [event(f"e{i:02}", i * 0.5, i * 0.5 + 0.4) for i in range(24)]


def test_prompt_compaction_keeps_anchors_beginning_and_ending_without_mutation():
    events = dense_events()
    original = copy.deepcopy(events)
    compact = compact_visual_events(events, required_ids=["e19"], max_events=8)
    ids = [item["event_id"] for item in compact["visual_events"]]
    assert len(ids) == 8 and {"e00", "e19", "e23"} <= set(ids)
    assert compact["visual_event_count"] == 24
    assert compact["visual_events_omitted"] == 16
    assert events == original
    assert ids == sorted(ids)


def test_scene_card_retains_full_timeline_but_global_prompt_is_bounded():
    cards = build_scene_cards({"scenes": [source_scene(dense_events())]})
    compact = compact_scene_cards(cards)
    assert len(cards[0]["visual_events"]) == 24
    assert len(compact[0]["visual_events"]) == 6
    assert compact[0]["visual_events_omitted"] == 18


def test_writer_prompt_includes_late_selected_event_and_full_story_ending():
    result = plan([source_scene(dense_events())])
    result["segments"][0]["anchor_event_ids"] = ["e19"]
    prompt = build_v2_narration_prompt(
        scenes=[source_scene(dense_events())], narration_plan=result,
        story_plan={"context": "x" * 27000, "ending": "END_OF_STORY"}, story_memory={},
        language="id", style="jozzr_social", style_config={}, hedging_threshold=0.5, mode="full",
    )
    assert "e19" in prompt and "END_OF_STORY" in prompt
    assert '"visual_events_omitted": 16' in prompt


def test_rewrite_plan_remains_parseable_and_keeps_all_beat_ids():
    result = plan([source_scene(dense_events())])
    result["segments"][0]["anchor_event_ids"] = ["e19"]
    prompt = build_rewrite_prompt(current_segments=[], qa={}, narration_plan=result,
                                  story_plan={}, language="id")
    encoded = prompt.split("NARRATION PLAN:\n", 1)[1].split("\n\nCURRENT NARRATION:", 1)[0]
    restored = json.loads(encoded)
    assert restored["segments"][0]["scene_id"] == "scene_a"
    assert restored["segments"][0]["anchor_event_ids"] == ["e19"]
    assert any(item["event_id"] == "e19" for item in restored["segments"][0]["visual_events"])


def test_duration_rewrite_retains_anchor_and_receives_neighboring_story(monkeypatch):
    original = spoken_segment()
    captured = {}
    monkeypatch.setattr(script_writer, "_client", lambda cfg: (object(), "fake"))

    def generate(analyzer, model, prompt, **kwargs):
        captured["prompt"] = prompt
        payload = {"text": "Besi dipotong perlahan.", "retained_claim_indices": [0]}
        assert kwargs["payload_validator"](payload) is None
        return payload

    monkeypatch.setattr(script_writer, "_call_json", generate)
    result = script_writer.shorten_for_measured_duration(
        original, language="id", measured_duration=8, available_duration=5, cfg=config(),
        previous_text="Komponen sudah diperiksa.", next_text="Hasilnya kemudian dirapikan.",
    )
    assert "Komponen sudah diperiksa." in captured["prompt"]
    assert "Hasilnya kemudian dirapikan." in captured["prompt"]
    assert result["text"] == "Besi dipotong perlahan."
    assert result["action_anchors"] == original["action_anchors"]
    assert result["claims"] == original["claims"]


@pytest.mark.parametrize("text", ["Besi diperbaiki perlahan.", "Besi dipotong dipotong."])
def test_duration_rewrite_cannot_drop_or_duplicate_required_phrase(monkeypatch, text):
    monkeypatch.setattr(script_writer, "_client", lambda cfg: (object(), "fake"))
    monkeypatch.setattr(script_writer, "_call_json", lambda *args, **kwargs: {
        "text": text, "retained_claim_indices": [0],
    })
    with pytest.raises(ValueError, match="anchor phrase"):
        script_writer.shorten_for_measured_duration(
            spoken_segment(), language="id", measured_duration=8, available_duration=5, cfg=config(),
        )


def test_response_validator_rejects_anchor_to_unseen_event_before_tts():
    payload = [{"scene_id": "scene_a", "text": "Besi dipotong perlahan.", "claims": [],
                "action_anchors": [{"event_id": "unseen", "phrase": "dipotong"}]}]
    assert "Unknown" in script_writer._narration_response_validator(plan())(payload)


def test_resume_cannot_use_story_plan_from_previous_dense_analysis(tmp_path):
    from dubbingstory.story.story_planner import load_story_artifacts

    for filename in ("story_plan.json", "story_memory.json"):
        (tmp_path / filename).write_text(json.dumps({"visual_timeline_sha256": "old", "stale": True}), encoding="utf-8")
    storyboard = {"scenes": [source_scene()], "visual_timeline": {"artifact_sha256": "new"}}
    story, memory = load_story_artifacts(str(tmp_path), storyboard)
    assert "stale" not in story and "stale" not in memory


def test_current_dense_story_plan_can_be_resumed(tmp_path):
    from dubbingstory.story.story_planner import load_story_artifacts

    payload = {"visual_timeline_sha256": "current", "marker": "current-story"}
    for filename in ("story_plan.json", "story_memory.json"):
        (tmp_path / filename).write_text(json.dumps(payload), encoding="utf-8")
    story, memory = load_story_artifacts(str(tmp_path), {
        "scenes": [source_scene()], "visual_timeline": {"artifact_sha256": "current"},
    })
    assert story == memory == payload


def test_qa_rewrite_retains_original_when_it_changes_anchor_phrase(monkeypatch):
    original = spoken_segment()
    monkeypatch.setattr(script_writer, "evaluate_narration", lambda *args, **kwargs: {
        "needs_rewrite": True, "issues": ["repetition"], "target_word_ratio": 1.0,
    })
    monkeypatch.setattr(script_writer, "_call_json", lambda *args, **kwargs: [{
        "scene_id": "scene_a", "text": "Besi dibelah perlahan.", "claims": ["Besi dibelah"],
        "action_anchors": [{"event_id": "cut", "phrase": "dibelah"}],
    }])
    monkeypatch.setattr(script_writer, "raise_if_strict_text_error", lambda *args: None)
    revised, _ = script_writer._maybe_rewrite(
        [original], language="id", narration_plan=plan(), story_plan={}, analyzer=object(), model="fake",
        cfg=config(), scenes=[source_scene()], style="jozzr_social", style_config={},
    )
    assert revised == [original]
