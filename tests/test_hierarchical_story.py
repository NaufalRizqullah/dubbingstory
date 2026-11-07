"""Bounded story planning regressions for Kaggle; providers are entirely mocked."""

import copy
import json
from types import SimpleNamespace

import pytest

from dubbingstory.llm.text_client import ContextBudgetError, TextGenerationError
from dubbingstory.story import story_planner


def board(count=20):
    return {
        "video_summary": "A worker makes and finishes a component.",
        "scenes": [{"scene_id": f"scene_{index + 1:03d}", "start_time": index * 5.0,
                    "end_time": (index + 1) * 5.0, "duration": 5.0,
                    "analysis": {"action": f"Visible action {index + 1}", "confidence": .9}}
                   for index in range(count)],
    }


def config(**overrides):
    settings = dict(llm_provider="local", llm_batching_enabled=True,
                    llm_batching_max_scenes=4, llm_batching_output_tokens=2048,
                    llm_local_model="fixture-local", story_model="fixture-cloud")
    settings.update(overrides)
    return SimpleNamespace(**settings)


def response(scene_ids, *, roles):
    payload = {
        "premise": f"Work from {scene_ids[0]} through {scene_ids[-1]}",
        "main_goal": "Finish the visible component", "central_conflict": "",
        "characters": [{"id": "worker", "label": "Visible worker", "goal": "", "state": "",
                        "last_seen_scene": scene_ids[-1]}],
        "important_objects": [], "unresolved_threads": [],
        "story_memory": {"known_facts": [f"Work reaches {scene_ids[-1]}"],
                         "continuity_rules": [], "open_questions": []},
    }
    if roles:
        payload["scene_roles"] = [
            {"scene_id": sid, "role": "process", "story_importance": .5,
             "causal_importance": .5, "bridge_importance": .5, "must_keep": False,
             "plot_thread": "component", "reason": "Visible process step"}
            for sid in reversed(scene_ids)
        ]
        payload["story_arc"] = {role: list(scene_ids) if role == "process" else []
                                for role in story_planner._STORY_ROLES}
    return payload


def prompt_group(prompt):
    marker = "SCENE CARDS (chronological):\n"
    if marker in prompt:
        group, _ = json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1].lstrip())
        return "map", group
    return "reduce", json.loads(prompt.rsplit("\n", 1)[-1])


class StoryClient:
    def __init__(self, *, map_capacity=4, reduce_capacity=4, fail_task=None):
        self.map_capacity, self.reduce_capacity = map_capacity, reduce_capacity
        self.fail_task = fail_task
        self.preflights, self.calls = [], []

    def preflight(self, prompt, *, task_name, max_output_tokens):
        phase, group = prompt_group(prompt)
        self.preflights.append((phase, len(group)))
        capacity = self.map_capacity if phase == "map" else self.reduce_capacity
        if len(group) > capacity:
            raise ContextBudgetError({"input_tokens": 9000, "configured_context": 8192,
                                      "server_context": 8192, "effective_context": 8192,
                                      "output_tokens": max_output_tokens, "safety_margin": 256,
                                      "available_output_tokens": 0, "deficit_tokens": 3112})
        return {"input_tokens": 500}

    def generate_json_with_retry(self, prompt, *, task_name, response_validator,
                                 max_output_tokens, **kwargs):
        phase, group = prompt_group(prompt)
        assert max_output_tokens == 2048
        assert len(group) <= (self.map_capacity if phase == "map" else self.reduce_capacity)
        self.calls.append((task_name, phase, group))
        if task_name == self.fail_task:
            self.fail_task = None
            raise TextGenerationError("simulated provider failure")
        ids = ([item["scene_id"] for item in group] if phase == "map"
               else [group[0]["first_scene_id"], group[-1]["last_scene_id"]])
        payload = response(ids, roles=phase == "map")
        assert response_validator(payload) is None
        return payload


def install_client(monkeypatch, client):
    monkeypatch.setattr(story_planner, "build_story_client", lambda *args, **kwargs: (client, "fixture-cloud"))


def test_generation_timeouts_split_map_and_reduce_without_losing_scene_coverage(monkeypatch):
    from dubbingstory.llm.text_client import LocalGenerationTimeout

    class SlowClient(StoryClient):
        def generate_json_with_retry(self, prompt, **kwargs):
            phase, group = prompt_group(prompt)
            if len(group) > 2:
                raise LocalGenerationTimeout("slow completion")
            return super().generate_json_with_retry(prompt, **kwargs)

    client = SlowClient()
    install_client(monkeypatch, client)
    source = board(16)
    _, plan, _ = story_planner.generate_story_artifacts(source, subtitle_context=None, cfg=config())
    assert [row["scene_id"] for row in plan["scene_roles"]] == [row["scene_id"] for row in source["scenes"]]
    assert all(len(group) <= 2 for _, _, group in client.calls)


def test_twenty_scenes_keep_all_roles_and_ending_without_a_giant_final_prompt(tmp_path, monkeypatch):
    client = StoryClient()
    install_client(monkeypatch, client)
    source = board()
    original = copy.deepcopy(source)
    _, plan, memory = story_planner.generate_story_artifacts(
        source, subtitle_context=None, cfg=config(), project_dir=str(tmp_path))
    ids = [scene["scene_id"] for scene in source["scenes"]]
    assert [item["scene_id"] for item in plan["scene_roles"]] == ids
    assert plan["story_arc"]["process"] == ids
    assert "scene_020" in plan["premise"] and "scene_020" in memory["known_facts"][0]
    mapped = [item["scene_id"] for _, phase, group in client.calls if phase == "map" for item in group]
    assert mapped == ids
    reductions = [group for _, phase, group in client.calls if phase == "reduce"]
    assert len(reductions) >= 2
    assert all(len(group) <= 4 for group in reductions)
    assert all("scene_ids" not in item for group in reductions for item in group)
    assert source == original
    assert json.loads((tmp_path / "story_plan.json").read_text())["scene_roles"] == plan["scene_roles"]


def test_capacity_split_keeps_every_scene_and_reserves_generation_output(tmp_path, monkeypatch):
    client = StoryClient(map_capacity=2, reduce_capacity=2)
    install_client(monkeypatch, client)
    _, plan, _ = story_planner.generate_story_artifacts(
        board(9), subtitle_context=None, cfg=config(), project_dir=str(tmp_path))
    assert len(plan["scene_roles"]) == 9
    assert any(phase == "map" and size == 4 for phase, size in client.preflights)
    assert all(len(group) <= 2 for _, _, group in client.calls)


def test_single_overbudget_scene_fails_before_generation_or_publication(tmp_path, monkeypatch):
    client = StoryClient(map_capacity=0)
    install_client(monkeypatch, client)
    with pytest.raises(ContextBudgetError):
        story_planner.generate_story_artifacts(board(1), subtitle_context=None,
                                               cfg=config(), project_dir=str(tmp_path))
    assert client.calls == []
    assert not (tmp_path / "story_plan.json").exists()
    assert not (tmp_path / "story_memory.json").exists()


def test_completed_map_batches_are_reused_after_later_failure(tmp_path, monkeypatch):
    client = StoryClient(fail_task="story-map-scene_005-scene_008")
    install_client(monkeypatch, client)
    with pytest.raises(TextGenerationError, match="simulated provider"):
        story_planner.generate_story_artifacts(board(12), subtitle_context=None,
                                               cfg=config(), project_dir=str(tmp_path))
    assert not (tmp_path / "story_plan.json").exists()
    assert not (tmp_path / "story_memory.json").exists()
    assert len(list((tmp_path / "story_cache").glob("*.json"))) == 1
    _, plan, _ = story_planner.generate_story_artifacts(board(12), subtitle_context=None,
                                                       cfg=config(), project_dir=str(tmp_path))
    tasks = [task for task, _, _ in client.calls]
    assert tasks.count("story-map-scene_001-scene_004") == 1
    assert tasks.count("story-map-scene_005-scene_008") == 2
    assert len(plan["scene_roles"]) == 12


def test_summary_reduction_cannot_spin_when_only_singletons_fit(tmp_path, monkeypatch):
    client = StoryClient(reduce_capacity=1)
    install_client(monkeypatch, client)
    with pytest.raises(TextGenerationError, match="cannot process two summaries"):
        story_planner.generate_story_artifacts(board(8), subtitle_context=None,
                                               cfg=config(), project_dir=str(tmp_path))
    assert len(client.calls) == 2  # Two map calls, no wasteful singleton reductions.
    assert not (tmp_path / "story_plan.json").exists()


@pytest.mark.parametrize("fault", ["duplicate", "unknown_character", "unbounded", "nonfinite"])
def test_batch_contract_rejects_invalid_output_before_normalization(fault):
    ids = ["scene_001", "scene_002"]
    payload = response(ids, roles=True)
    if fault == "duplicate":
        payload["scene_roles"] = [payload["scene_roles"][0]] * 2
    elif fault == "unknown_character":
        payload["characters"][0]["last_seen_scene"] = "scene_unknown"
    elif fault == "unbounded":
        payload["story_memory"]["known_facts"] = ["fact"] * 7
    else:
        payload["scene_roles"][0]["story_importance"] = float("nan")
    assert story_planner._batch_response_validator(ids, roles=True)(payload)


def test_old_custom_client_interface_remains_usable(monkeypatch):
    class LegacyClient:
        def generate_json_with_retry(self, prompt, *, task_name, temperature, model, response_validator):
            phase, group = prompt_group(prompt)
            assert phase == "map"
            return response([item["scene_id"] for item in group], roles=True)

    install_client(monkeypatch, LegacyClient())
    _, plan, _ = story_planner.generate_story_artifacts(board(2), subtitle_context=None, cfg=config())
    assert len(plan["scene_roles"]) == 2


def test_source_change_invalidates_cache_even_if_storyboard_is_identical(tmp_path, monkeypatch):
    client = StoryClient()
    install_client(monkeypatch, client)
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"first source fixture, never decoded")
    args = dict(subtitle_context=None, cfg=config(), project_dir=str(tmp_path))
    story_planner.generate_story_artifacts(board(2), **args)
    story_planner.generate_story_artifacts(board(2), **args)
    assert len(client.calls) == 1
    source_path.write_bytes(b"changed source fixture, never decoded")
    story_planner.generate_story_artifacts(board(2), **args)
    assert len(client.calls) == 2


def test_disabled_batching_preserves_existing_single_request_route(monkeypatch):
    sentinel = ({"premise": "legacy"}, {"known_facts": []})
    calls = []
    monkeypatch.setattr(story_planner, "_single_story_artifacts",
                        lambda *args: calls.append(args) or sentinel)
    _, plan, memory = story_planner.generate_story_artifacts(
        board(), subtitle_context=None, cfg=config(llm_batching_enabled=False))
    assert (plan, memory) == sentinel and len(calls) == 1
