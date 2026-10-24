from types import SimpleNamespace

import pytest

from dubbingstory.story.narration_planner import build_narration_plan


def plan(scenes, mode="full", **overrides):
    values = dict(tts_narration_fit="rewrite", tts_min_gap=0.2,
                  narration_source_duration=30.0, narration_group_scenes=False,
                  narration_plan_llm_refine=False)
    values.update(overrides)
    return build_narration_plan({"scenes": scenes}, scenes, story_plan={}, story_memory={},
                                cfg=SimpleNamespace(**values), mode=mode)


def scene(name, start, end, duration=None):
    return {"scene_id": name, "start_time": start, "end_time": end,
            "duration": end - start if duration is None else duration}


def test_source_duration_retains_tail_without_narration():
    result = plan([scene("first", 0, 8), scene("second", 8, 20)])
    assert result["total_duration"] == 30
    assert result["segments"][-1]["timeline_end"] == 20


def test_strict_plan_uses_source_boundaries_instead_of_stale_duration():
    result = plan([scene("first", 0, 8, duration=20), scene("second", 8, 30.2)])
    assert result["segments"][0]["timeline_end"] == 8
    assert result["segments"][1]["timeline_end"] == 30


def test_subsecond_beat_does_not_require_five_words():
    beat = plan([scene("short", 0, 0.5)])["segments"][0]
    assert 0 <= beat["min_words"] <= beat["target_words"] <= beat["max_words"] <= 2


def test_overlapping_source_beats_fail_instead_of_drifting_forward():
    with pytest.raises(ValueError, match="overlapping"):
        plan([scene("first", 0, 8), scene("second", 7, 12)])


def test_summary_uses_its_own_cumulative_timeline():
    result = plan([scene("first", 60, 64), scene("second", 120, 125)], mode="summary")
    assert result["total_duration"] == 9
    assert [(s["timeline_start"], s["timeline_end"]) for s in result["segments"]] == [(0, 4), (4, 9)]
