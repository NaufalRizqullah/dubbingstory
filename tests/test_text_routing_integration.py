"""Offline routing/validation regressions; author-only until run in Kaggle."""

import copy
import json
from types import SimpleNamespace

import pytest

from dubbingstory.llm.text_client import TextGenerationError
from dubbingstory.story import (
    beat_planner, gemini_json, narration_planner, script_writer,
    semantic_verifier, story_planner,
)


SCENE = {"scene_id": "scene_001", "start_time": 0.0, "end_time": 8.0,
         "duration": 8.0, "analysis": {"action": "A component is inspected"}}
BOARD = {"scenes": [SCENE], "video_summary": "A component is inspected.", "total_duration": 8.0}
PLAN = {"target_wpm": 160, "segments": [{
    "scene_id": "scene_001", "covers_scene_ids": ["scene_001"],
    "story_role": "process", "target_words": 12, "min_words": 3, "max_words": 20,
    "timeline_start": 0.0, "timeline_end": 8.0,
}]}


def local_config():
    return SimpleNamespace(llm_provider="local", api_key_gemini="", narration_auto_rewrite=False)


@pytest.mark.parametrize("settings,attribute,expected", [
    ({"narration_model": "narrator", "story_model": "planner"}, "narration_model", "narrator"),
    ({"story_model": "planner"}, "narration_model", "planner"),
    ({"vision_gemini_model": "visual"}, None, "visual"),
    ({}, None, "gemini-2.5-flash"),
])
def test_story_client_preserves_model_precedence_without_requiring_cloud_key(monkeypatch, settings, attribute, expected):
    cfg = SimpleNamespace(llm_provider="local", **settings)
    received = []
    client = object()
    monkeypatch.setattr(gemini_json, "TextLLMClient", lambda supplied, model: received.append((supplied, model)) or client)
    actual, model = gemini_json.build_story_client(cfg, model_attr=attribute)
    assert actual is client
    assert model == expected
    assert received == [(cfg, expected)]


@pytest.mark.parametrize("payload", [
    [],
    [{"scene_id": "scene_other", "text": "Komponen diperiksa.", "claims": []}],
    [{"scene_id": "scene_001", "text": "", "claims": []}],
    [{"scene_id": "scene_001", "text": "Komponen diperiksa.", "claims": "invalid"}],
    [{"scene_id": "scene_001", "text": "Komponen diperiksa.", "claims": [], "importance": float("nan")}],
    [{"scene_id": "scene_001", "text": "Komponen diperiksa.", "claims": []}] * 2,
])
def test_narration_schema_rejects_missing_duplicate_or_invalid_beats(payload):
    assert script_writer._narration_response_validator(PLAN)(payload)


def test_narration_validation_composes_schema_and_language_without_replacing_prompt():
    prompt = "OUTPUT_LANGUAGE_CODE: id\nOriginal visual evidence."
    valid = [{"scene_id": "scene_001", "text": "Komponen kemudian diperiksa sebelum proses dilanjutkan.", "claims": []}]

    class Analyzer:
        def generate_json_with_retry(self, supplied, *, response_validator, **kwargs):
            assert supplied == prompt
            assert response_validator([])
            assert response_validator([dict(valid[0], text="The component is then checked before the process continues.")])
            assert response_validator(valid) is None
            return valid

    result = script_writer._call_json(
        Analyzer(), "model", prompt, language="id",
        payload_validator=script_writer._narration_response_validator(PLAN),
    )
    assert result == valid


def test_duration_rewrite_validates_word_budget_and_claim_references_inside_retry(monkeypatch):
    original = {"scene_id": "scene_001", "text": "Komponen kemudian diperiksa dengan alat sebelum proses berikutnya dimulai.",
                "claims": ["Komponen diperiksa", "Menggunakan alat"]}
    evidence = [{"detail": "x" * 17000, "tail": "LAST_EVIDENCE"}]

    class Analyzer:
        def generate_json_with_retry(self, prompt, *, response_validator, **kwargs):
            assert "LAST_EVIDENCE" in prompt
            assert "Language: id" in prompt
            assert response_validator({"text": original["text"], "retained_claim_indices": [0]})
            assert response_validator({"text": "Komponen diperiksa.", "retained_claim_indices": [True]})
            assert response_validator({"text": "Komponen diperiksa.", "retained_claim_indices": [99]})
            result = {"text": "Komponen diperiksa.", "retained_claim_indices": [0]}
            assert response_validator(result) is None
            return result

    monkeypatch.setattr(script_writer, "_client", lambda cfg: (Analyzer(), "model"))
    result = script_writer.shorten_for_measured_duration(
        original, language="id", measured_duration=10, available_duration=5,
        cfg=local_config(), evidence=evidence,
    )
    assert result["text"] == "Komponen diperiksa."
    assert result["claims"] == ["Komponen diperiksa"]
    assert result["duration_fit_rewritten"] is True
    assert original["text"].startswith("Komponen kemudian")


@pytest.mark.parametrize("strict", [False, True])
def test_failed_narration_uses_legacy_fallback_only_when_new_routing_disabled(monkeypatch, strict):
    class FailedAnalyzer:
        def generate_json_with_retry(self, *args, **kwargs):
            raise RuntimeError("simulated legacy generation failure")

    monkeypatch.setattr(script_writer, "_client", lambda cfg: (FailedAnalyzer(), "model"))
    monkeypatch.setattr(script_writer, "build_v2_narration_prompt", lambda **kwargs: "evidence")
    sentinel = [{"text": "legacy fallback"}]
    monkeypatch.setattr(script_writer, "_fallback_narration", lambda *args: sentinel)
    cfg = local_config() if strict else SimpleNamespace()
    arguments = dict(storyboard=BOARD, scenes=BOARD["scenes"], languages=["id"],
                     style="natural", cfg=cfg, mode="full", story_plan={"premise": "Observed"},
                     story_memory={}, narration_plan=PLAN)
    if strict:
        with pytest.raises(TextGenerationError, match="full-narration"):
            script_writer._generate(**arguments)
    else:
        assert script_writer._generate(**arguments) == {"id": sentinel}


def _invoke_story_task(task, cfg, monkeypatch):
    if task == "story":
        return story_planner.generate_story_artifacts(BOARD, subtitle_context=None, cfg=cfg)
    if task == "refine":
        return narration_planner._refine_plan_with_gemini(
            copy.deepcopy(PLAN["segments"]), scenes=BOARD["scenes"], story_plan={}, story_memory={}, cfg=cfg, mode="full",
        )
    if task == "beats":
        return beat_planner.plan_story_beats(story_plan={}, story_memory={}, signatures=[], target_duration=20, cfg=cfg)
    return semantic_verifier.repair_unsupported_beats(
        narration=[{"scene_id": "scene_001", "text": "Komponen diperiksa.", "claims": []}],
        verification={"beats": [{"beat_id": "scene_001", "status": "repair"}]},
        narration_plan=PLAN, signatures=[], cfg=cfg, language="id",
    )


@pytest.mark.parametrize("task,module", [
    ("story", story_planner), ("refine", narration_planner),
    ("beats", beat_planner), ("repair", semantic_verifier),
])
def test_local_without_cloud_key_reaches_route_and_propagates_exhaustion(monkeypatch, task, module):
    calls = []
    failure = TextGenerationError("both configured routes exhausted")

    def fail(*args, **kwargs):
        calls.append(kwargs["response_validator"])
        raise failure

    monkeypatch.setattr(module, "generate_story_json", fail)
    with pytest.raises(TextGenerationError) as caught:
        _invoke_story_task(task, local_config(), monkeypatch)
    assert caught.value is failure
    assert len(calls) == 1
    assert callable(calls[0])


def test_strict_wrapped_errors_hide_raw_provider_details():
    with pytest.raises(TextGenerationError) as caught:
        gemini_json.raise_if_strict_text_error(local_config(), ValueError("private provider body"), "narration")
    assert "private provider body" not in str(caught.value)
    assert caught.value.__suppress_context__


def test_story_schema_rejects_unrecognized_scene_references():
    plan, memory = story_planner.fallback_story_artifacts(BOARD)
    payload = dict(plan, story_memory=memory)
    validate = story_planner._story_response_validator(BOARD)
    assert validate(payload) is None
    payload["story_arc"]["process"] = ["scene_not_in_video"]
    assert validate(payload)


def test_refinement_and_beat_schemas_reject_structurally_valid_but_unusable_json():
    validate = narration_planner._refinement_response_validator(PLAN["segments"])
    assert validate([])
    assert validate([{"scene_id": "scene_001", "intent": ""}])
    assert beat_planner._beat_response_validator([])
    assert beat_planner._beat_response_validator([{
        "beat_id": "b1", "intent": "Observed action", "required_facts": [], "forbidden_claims": [],
        "approx_ratio": 0.5, "target_duration_sec": float("inf"),
    }])


def test_analyze_preserves_storyboard_but_does_not_hide_text_route_failure(monkeypatch, tmp_path):
    from dubbingstory import cli
    from dubbingstory.vision import scene_understanding

    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"mock frame; never decoded")
    (tmp_path / "segment_manifest.json").write_text(json.dumps({
        "status": "segmented", "scenes": BOARD["scenes"],
        "keyframes": {scene["scene_id"]: [str(frame)] for scene in BOARD["scenes"]},
    }), encoding="utf-8")
    monkeypatch.setattr(cli, "_resolve_project_name", lambda args: "sample")
    monkeypatch.setattr(cli, "_setup_project_dir", lambda cfg, project: str(tmp_path))
    monkeypatch.setattr(scene_understanding, "run_analysis", lambda **kwargs: BOARD)

    def fail(*args, **kwargs):
        raise TextGenerationError("local route exhausted")

    monkeypatch.setattr(story_planner, "generate_story_artifacts", fail)
    with pytest.raises(TextGenerationError, match="local route exhausted"):
        cli.cmd_analyze(SimpleNamespace(), local_config())
    assert json.loads((tmp_path / "storyboard.json").read_text(encoding="utf-8")) == BOARD
