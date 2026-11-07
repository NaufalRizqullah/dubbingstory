"""Bounded full-video text contracts; author-only until executed in Kaggle."""

import copy
import json
from types import SimpleNamespace

import pytest

from dubbingstory.llm.text_client import TextGenerationError, LocalGenerationTimeout
from dubbingstory.story import narration_planner, script_writer
from dubbingstory.story.prompt_templates import batch_story_context


def fixture(count=20, *, anchored=True):
    scenes, entries, roles = [], [], []
    for index in range(count):
        sid, bid, eid = f"visual_{index:03}", f"beat_{index:03}", f"event_{index:03}"
        start = float(index * 5)
        event = {"event_id": eid, "scene_id": sid, "start_time": start + 1,
                 "end_time": start + 2, "action": "The metal is cut", "state_change": "It separates",
                 "confidence": 0.9, "frame_ids": [f"frame_{index:03}"]}
        scenes.append({"scene_id": sid, "start_time": start, "end_time": start + 5,
                       "duration": 5, "analysis": {"action": "The metal is cut", "events": [event]}})
        entry = {"scene_id": bid, "covers_scene_ids": [sid], "timeline_start": start,
                 "timeline_end": start + 5, "available_duration": 5, "target_words": 9,
                 "min_words": 3, "max_words": 14, "story_role": "process", "intent": "Explain the visible cut",
                 "must_explain": ["The metal is cut"], "story_importance": 0.7}
        if anchored:
            entry.update(action_sync_enabled=True, visual_events=[event], anchor_event_ids=[eid])
        entries.append(entry)
        roles.append({"scene_id": sid, "role": "process", "reason": f"ROLE_{index:03}", "story_importance": 0.7})
    board = {"scenes": scenes, "total_duration": count * 5, "video_summary": "Metal work progresses."}
    plan = {"mode": "full", "segments": entries, "target_wpm": 160,
            "total_duration": count * 5, "total_target_words": count * 9,
            "action_sync_enabled": anchored}
    story = {"premise": "Metal work progresses", "scene_roles": roles}
    return board, plan, story


def config(**values):
    return SimpleNamespace(**{
        "llm_provider": "local", "llm_batching_enabled": True,
        "llm_batching_max_beats": 4, "llm_batching_output_tokens": 2048,
        "narration_auto_rewrite": False, **values,
    })


def continuity(prompt):
    suffix = prompt.split("BATCH CONTINUITY (context only, never additional output rows):\n", 1)[1]
    return json.JSONDecoder().raw_decode(suffix)[0]


def spoken(entry, *, changed=False):
    row = {"scene_id": entry["scene_id"],
           "text": "Besi dipotong perlahan sebelum tahap berikutnya mulai dikerjakan.",
           "claims": ["The metal is cut"], "importance": 0.7}
    if changed:
        row["text"] = "Kini besi dipotong perlahan sebelum proses berikutnya dikerjakan."
    if entry.get("action_sync_enabled"):
        row["action_anchors"] = [{"event_id": entry["anchor_event_ids"][0], "phrase": "Besi dipotong perlahan"}]
    return row


class Narrator:
    def __init__(self, plan, *, fail_on=None, changed=False):
        self.entries = {item["scene_id"]: item for item in plan["segments"]}
        self.calls, self.preflights = [], []
        self.fail_on, self.changed = fail_on, changed

    def preflight(self, prompt, **kwargs):
        self.preflights.append((prompt, kwargs))
        return {"input_tokens": 1000}

    def generate_json_with_retry(self, prompt, *, response_validator, max_output_tokens, **kwargs):
        context = continuity(prompt)
        self.calls.append((prompt, context))
        assert max_output_tokens == 2048
        if len(self.calls) == self.fail_on:
            raise TextGenerationError("simulated route exhaustion")
        payload = [spoken(self.entries[bid], changed=self.changed) for bid in reversed(context["beat_ids"])]
        assert response_validator(payload) is None
        return payload


def test_twenty_beats_use_bounded_prompts_and_keep_source_timing_and_continuity(monkeypatch):
    board, plan, story = fixture()
    client = Narrator(plan)
    monkeypatch.setattr(script_writer, "_client", lambda cfg: (client, "mock-model"))
    result = script_writer.generate_narration(
        board, languages=["id"], cfg=config(), story_plan=story, narration_plan=plan,
    )["id"]
    assert len(client.calls) == 5
    assert [item["scene_id"] for item in result] == [item["scene_id"] for item in plan["segments"]]
    for index, (prompt, context) in enumerate(client.calls):
        assert len(context["beat_ids"]) == 4
        assert len(context["previous_narration"]) == (0 if index == 0 else 2)
        assert context["is_video_opening"] == (index == 0)
        assert context["is_video_ending"] == (index == 4)
        if index:
            assert context["previous_narration"][-1]["scene_id"] == f"beat_{index * 4 - 1:03}"
        for outside in set(range(20)) - set(range(index * 4, index * 4 + 4)):
            assert f"ROLE_{outside:03}" not in prompt
            assert f'"frame_{outside:03}"' not in prompt
    for actual, expected in zip(result, plan["segments"]):
        assert actual["timeline_start"] == expected["timeline_start"]
        assert actual["timeline_end"] == expected["timeline_end"]
        assert actual["covers_scene_ids"] == expected["covers_scene_ids"]
        assert actual["action_anchors"][0]["start_time"] == expected["visual_events"][0]["start_time"]


def test_late_batch_failure_never_returns_partial_or_template_narration(monkeypatch):
    board, plan, story = fixture(9, anchored=False)
    client = Narrator(plan, fail_on=2)
    monkeypatch.setattr(script_writer, "_client", lambda cfg: (client, "mock"))
    monkeypatch.setattr(script_writer, "_fallback_narration", lambda *args: pytest.fail("No template fallback"))
    with pytest.raises(TextGenerationError, match="exhaustion"):
        script_writer.generate_narration(board, languages=["id"], cfg=config(), story_plan=story, narration_plan=plan)
    assert len(client.calls) == 2


@pytest.mark.parametrize("rewrite", [False, True])
def test_slow_narration_batches_split_and_preserve_anchors_and_continuity(monkeypatch, rewrite):
    board, plan, story = fixture(4)
    attempts = []

    class SlowNarrator(Narrator):
        def generate_json_with_retry(self, prompt, **kwargs):
            context = continuity(prompt)
            attempts.append(context)
            if len(context["beat_ids"]) > 2:
                raise LocalGenerationTimeout("slow completion")
            return super().generate_json_with_retry(prompt, **kwargs)

    client = SlowNarrator(plan)
    monkeypatch.setattr(script_writer, "_client", lambda cfg: (client, "mock"))
    if rewrite:
        original = script_writer._validate_narration(
            [spoken(item) for item in plan["segments"]], board["scenes"], plan, "id")
        result = script_writer._batched_qa_rewrite(
            original, qa={"issues": ["repetition"]}, narration_plan=plan, story_plan=story,
            language="id", style="jozzr_social", style_config={}, analyzer=client,
            model="mock", cfg=config(), scenes=board["scenes"])
    else:
        result = script_writer.generate_narration(
            board, languages=["id"], cfg=config(), story_plan=story, narration_plan=plan)["id"]
    assert [len(context["beat_ids"]) for context in attempts] == [4, 2, 2]
    assert attempts[-1]["previous_narration"][-1]["scene_id"] == "beat_001"
    assert [row["scene_id"] for row in result] == [row["scene_id"] for row in plan["segments"]]
    assert [row["timeline_start"] for row in result] == [row["timeline_start"] for row in plan["segments"]]
    assert all(row["action_anchors"] for row in result)


def test_single_beat_timeout_does_not_return_partial_narration(monkeypatch):
    board, plan, story = fixture(2)
    calls = []

    class SlowNarrator(Narrator):
        def generate_json_with_retry(self, prompt, **kwargs):
            ids = continuity(prompt)["beat_ids"]
            calls.append(ids)
            if "beat_001" in ids:
                raise LocalGenerationTimeout("single beat still slow")
            return super().generate_json_with_retry(prompt, **kwargs)

    monkeypatch.setattr(script_writer, "_client", lambda cfg: (SlowNarrator(plan), "mock"))
    with pytest.raises(RuntimeError, match="single beat") as exc:
        script_writer.generate_narration(board, languages=["id"], cfg=config(), story_plan=story, narration_plan=plan)
    assert isinstance(exc.value.__cause__, LocalGenerationTimeout)
    assert calls == [["beat_000", "beat_001"], ["beat_000"], ["beat_001"]]


@pytest.mark.parametrize("fault", ["missing", "duplicate", "unknown_event", "wrong_language"])
def test_batch_rejects_invalid_custom_client_output(monkeypatch, fault):
    board, plan, story = fixture(2)

    class Broken(Narrator):
        def generate_json_with_retry(self, prompt, **kwargs):
            payload = [spoken(item) for item in plan["segments"]]
            if fault == "missing":
                payload.pop()
            elif fault == "duplicate":
                payload[1] = copy.deepcopy(payload[0])
            elif fault == "unknown_event":
                payload[0]["action_anchors"][0]["event_id"] = "unseen"
            else:
                for row in payload:
                    row["text"] = "The metal is cut before the next stage of the process begins."
                    row["action_anchors"][0]["phrase"] = "metal is cut"
            return payload  # Custom client deliberately ignores supplied validator.

    monkeypatch.setattr(script_writer, "_client", lambda cfg: (Broken(plan), "mock"))
    with pytest.raises(RuntimeError, match="action-grounded narration failed"):
        script_writer.generate_narration(board, languages=["id"], cfg=config(), story_plan=story, narration_plan=plan)


def test_summary_uses_existing_single_request_even_when_full_batching_enabled(monkeypatch):
    board, plan, story = fixture(8, anchored=False)
    plan["mode"] = "summary"
    seen = []

    class SummaryClient:
        def generate_json_with_retry(self, prompt, **kwargs):
            seen.append(prompt)
            assert "BATCH CONTINUITY" not in prompt
            return [spoken(entry) for entry in plan["segments"]]

    monkeypatch.setattr(script_writer, "_client", lambda cfg: (SummaryClient(), "mock"))
    result = script_writer.generate_summary_narration(
        board, board["scenes"], languages=["id"], cfg=config(), story_plan=story, narration_plan=plan,
    )
    assert len(seen) == 1 and len(result["id"]) == 8


class Refiner:
    def __init__(self, *, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def preflight(self, prompt, **kwargs):
        return {"input_tokens": 1000}

    def generate_json_with_retry(self, prompt, *, response_validator, **kwargs):
        context = continuity(prompt)
        self.calls.append(context)
        if len(self.calls) == self.fail_on:
            raise TextGenerationError("simulated refinement exhaustion")
        payload = [{"scene_id": sid, "intent": f"Intent for {sid}",
                    "must_explain": ["The metal is cut"], "continuity_from_previous": "Continue observed process",
                    "transition_strategy": "progression", "avoid_repeating": []}
                   for sid in reversed(context["beat_ids"])]
        assert response_validator(payload) is None
        return payload


def test_refinement_timeout_splits_with_latest_intents(monkeypatch):
    board, plan, story = fixture(4)
    before = copy.deepcopy(plan["segments"])

    class SlowRefiner(Refiner):
        def generate_json_with_retry(self, prompt, **kwargs):
            if len(continuity(prompt)["beat_ids"]) > 2:
                raise LocalGenerationTimeout("slow planning")
            return super().generate_json_with_retry(prompt, **kwargs)

    client = SlowRefiner()
    monkeypatch.setattr(narration_planner, "build_story_client", lambda *args, **kwargs: (client, "mock"))
    narration_planner._refine_plan_with_gemini(
        plan["segments"], scenes=board["scenes"], story_plan=story, story_memory={}, cfg=config(), mode="full")
    assert [len(item["beat_ids"]) for item in client.calls] == [2, 2]
    assert client.calls[1]["previous_planning_intents"][-1]["intent"] == "Intent for beat_001"
    for old, revised in zip(before, plan["segments"]):
        assert revised["intent"] == f"Intent for {old['scene_id']}"
        assert revised["visual_events"] == old["visual_events"]
        assert revised["timeline_start"] == old["timeline_start"]


def test_refinement_batches_keep_complete_coverage_without_retiming(monkeypatch):
    board, plan, story = fixture(10)
    before = copy.deepcopy(plan["segments"])
    client = Refiner()
    monkeypatch.setattr(narration_planner, "build_story_client", lambda *args, **kwargs: (client, "mock"))
    narration_planner._refine_plan_with_gemini(
        plan["segments"], scenes=board["scenes"], story_plan=story, story_memory={}, cfg=config(), mode="full",
    )
    assert [len(item["beat_ids"]) for item in client.calls] == [4, 4, 2]
    assert client.calls[1]["previous_planning_intents"][-1]["intent"] == "Intent for beat_003"
    for old, revised in zip(before, plan["segments"]):
        assert revised["intent"] == f"Intent for {old['scene_id']}"
        for key in ("timeline_start", "timeline_end", "covers_scene_ids", "visual_events", "anchor_event_ids", "max_words"):
            assert revised[key] == old[key]


def test_refinement_failure_does_not_publish_partial_entry_changes(monkeypatch):
    board, plan, story = fixture(9)
    before = copy.deepcopy(plan["segments"])
    monkeypatch.setattr(narration_planner, "build_story_client", lambda *args, **kwargs: (Refiner(fail_on=2), "mock"))
    with pytest.raises(TextGenerationError, match="refinement exhaustion"):
        narration_planner._refine_plan_with_gemini(
            plan["segments"], scenes=board["scenes"], story_plan=story, story_memory={}, cfg=config(), mode="full",
        )
    assert plan["segments"] == before


def test_qa_rewrite_compares_all_batches_globally_and_preserves_anchors(monkeypatch):
    board, plan, story = fixture(9)
    original = script_writer._validate_narration([spoken(entry) for entry in plan["segments"]], board["scenes"], plan)
    client = Narrator(plan, changed=True)
    evaluated = []

    def evaluate(items, supplied_plan, **kwargs):
        evaluated.append((len(items), supplied_plan is plan))
        return {"needs_rewrite": len(evaluated) == 1, "issues": ["monotonous_pacing"] if len(evaluated) == 1 else [],
                "target_word_ratio": 1.0}

    monkeypatch.setattr(script_writer, "evaluate_narration", evaluate)
    revised, qa = script_writer._maybe_rewrite(
        original, language="id", narration_plan=plan, story_plan=story, analyzer=client,
        model="mock", cfg=config(narration_auto_rewrite=True), scenes=board["scenes"], style="natural", style_config={},
    )
    assert evaluated == [(9, True), (9, True)]
    assert len(client.calls) == 3 and not qa["issues"]
    assert [item["action_anchors"] for item in revised] == [item["action_anchors"] for item in original]
    assert all(item["text"].startswith("Kini") for item in revised)


def test_secondary_global_context_is_bounded_while_selected_roles_remain():
    _, plan, story = fixture(50)
    story["premise"] = "Long premise " * 5000
    memory = {"known_facts": [f"Fact {index} " * 100 for index in range(1000)]}
    scoped, facts = batch_story_context(story, memory, plan["segments"][20:24])
    assert len(scoped["scene_roles"]) == 4
    assert len(scoped["premise"]) < 650
    assert len(facts["known_facts"]) == 8
    assert facts["context_entries_omitted"]["known_facts"] == 992
    assert len(memory["known_facts"]) == 1000  # Full source artifact is not truncated.
