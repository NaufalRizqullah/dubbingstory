"""Mocked narration contracts; run in Kaggle, without model/API downloads."""

import copy
from types import SimpleNamespace

import pytest

from dubbingstory.story import script_writer
from dubbingstory.story.prompt_templates import build_rewrite_prompt, build_v2_narration_prompt


def narration_plan(*, anchored=True):
    entry = {
        "scene_id": "beat_original", "covers_scene_ids": ["visual_scene"],
        "timeline_start": 0.0, "timeline_end": 10.0,
        "target_words": 12, "min_words": 3, "max_words": 18,
    }
    result = {"segments": [entry]}
    if anchored:
        result["action_sync_enabled"] = True
        entry.update(action_sync_enabled=True, anchor_event_ids=["cut_observed"], visual_events=[{
            "event_id": "cut_observed", "scene_id": "visual_scene",
            "start_time": 2.0, "end_time": 3.0, "confidence": 0.9,
            "frame_ids": ["frame_observed"], "timing_basis": "sample_window",
            "action": "The worker cuts the metal", "state_change": "The metal separates",
        }])
    return result


def narration_payload():
    return [{
        "scene_id": "beat_original",
        "text": "Besi dipotong perlahan sebelum proses berikutnya dimulai.",
        "claims": ["The worker cuts the metal with the tool before the next process."],
        "importance": 0.8,
        "action_anchors": [{"event_id": "cut_observed", "phrase": "Besi dipotong perlahan"}],
    }]


def test_provider_schema_requires_selected_anchor_shape_and_original_ids():
    validator = script_writer._narration_response_validator(narration_plan())
    schema = validator.response_schema
    assert schema["type"] == "array"
    assert schema["minItems"] == schema["maxItems"] == 1
    item = schema["items"]
    assert set(item["required"]) == {"scene_id", "text", "claims", "importance", "action_anchors"}
    assert item["properties"]["scene_id"]["enum"] == ["beat_original"]
    anchor = item["properties"]["action_anchors"]["items"]
    assert anchor["properties"]["event_id"]["enum"] == ["cut_observed"]
    assert set(anchor["required"]) == {"event_id", "phrase"}
    assert anchor["additionalProperties"] is False
    assert validator(narration_payload()) is None


def test_unanchored_schema_keeps_legacy_fields_and_empty_plan_has_no_invalid_enum():
    schema = script_writer._narration_response_validator(narration_plan(anchored=False)).response_schema
    assert "action_anchors" not in schema["items"]["required"]
    empty_schema = script_writer._narration_response_validator({"segments": []}).response_schema
    assert empty_schema["maxItems"] == 0
    assert "enum" not in empty_schema["items"]["properties"]["scene_id"]


def test_no_eligible_events_schema_requires_empty_anchor_array():
    plan = narration_plan()
    plan["segments"][0]["anchor_event_ids"] = []
    schema = script_writer._narration_response_validator(plan).response_schema
    assert schema["items"]["properties"]["action_anchors"]["maxItems"] == 0


@pytest.mark.parametrize("envelope", [None, "segments", "narration", "items"])
def test_historical_envelopes_remain_accepted(envelope):
    payload = narration_payload()
    if envelope:
        payload = {envelope: payload}
    assert script_writer._narration_response_validator(narration_plan())(payload) is None


def test_missing_anchors_and_wrong_scene_ids_have_actionable_feedback():
    validator = script_writer._narration_response_validator(narration_plan())
    payload = narration_payload()
    del payload[0]["action_anchors"]
    reason = validator(payload)
    assert "beat_original.action_anchors" in reason
    assert "cut_observed" in reason
    payload[0]["scene_id"] = "visual_scene"
    assert "beat_original" in validator(payload)


def test_schema_propagates_without_new_analyzer_kwargs_and_language_checks_only_spoken_text():
    validator = script_writer._narration_response_validator(narration_plan())
    payload = narration_payload()

    class Analyzer:
        def generate_json_with_retry(self, prompt, *, temperature, response_validator, task_name, model):
            assert response_validator.response_schema == validator.response_schema
            assert response_validator(payload) is None  # English claims are input metadata, not speech.
            wrong_language = copy.deepcopy(payload)
            wrong_language[0]["text"] = "The metal is cut with the tool before the next process begins."
            wrong_language[0]["action_anchors"][0]["phrase"] = "metal is cut"
            assert "language mismatch" in response_validator(wrong_language)
            return payload

    result = script_writer._call_json(
        Analyzer(), "mock", "Fixed evidence", language="id", payload_validator=validator,
    )
    assert result == payload


@pytest.mark.parametrize("rewrite", [False, True])
def test_initial_and_qa_prompts_include_complete_anchor_output_contract(rewrite):
    plan = narration_plan()
    if rewrite:
        prompt = build_rewrite_prompt(
            current_segments=narration_payload(), qa={}, narration_plan=plan,
            story_plan={}, language="id",
        )
    else:
        prompt = build_v2_narration_prompt(
            scenes=[], narration_plan=plan, story_plan={}, story_memory={}, language="id",
            style="natural", style_config={}, hedging_threshold=0.5, mode="full",
        )
    output_contract = prompt.rsplit("Return ONLY a valid JSON array", 1)[1]
    assert '"action_anchors"' in output_contract
    assert '"event_id"' in output_contract and '"phrase"' in output_contract
    assert "beat_original" in output_contract
    assert "return [] only when this beat has no selected IDs" in output_contract
    assert "Do not return timestamps" in output_contract


def test_duration_rewrite_floors_heuristic_at_required_phrase_words_without_changing_times(monkeypatch):
    original = dict(narration_plan()["segments"][0], **narration_payload()[0])
    original["text"] = "Lalu besi dipotong perlahan sebelum tahap berikutnya dimulai."
    cfg = SimpleNamespace(llm_provider="local")

    class Analyzer:
        def generate_json_with_retry(self, prompt, *, response_validator, **kwargs):
            assert "Maximum words: 3" in prompt
            schema = response_validator.response_schema
            assert schema["type"] == "object"
            assert set(schema["required"]) == {"text", "retained_claim_indices"}
            indices = schema["properties"]["retained_claim_indices"]
            assert indices["items"]["maximum"] == 0
            result = {"text": "Besi dipotong perlahan.", "retained_claim_indices": [0]}
            assert response_validator(result) is None
            too_long = dict(result, text="Besi dipotong perlahan lalu dirapikan.")
            assert "5 words; maximum is 3" in response_validator(too_long)
            return result

    monkeypatch.setattr(script_writer, "_client", lambda cfg: (Analyzer(), "mock"))
    result = script_writer.shorten_for_measured_duration(
        original, language="id", measured_duration=12, available_duration=2, cfg=cfg,
    )
    assert result["word_count"] == 3
    assert result["claims"] == original["claims"]
    assert result["action_anchors"][0]["start_time"] == 2.0
    assert result["action_anchors"][0]["end_time"] == 3.0
    assert result["timeline_start"] == original["timeline_start"]
    assert result["timeline_end"] == original["timeline_end"]


def test_duration_rewrite_with_no_removable_words_stops_before_provider_calls(monkeypatch):
    original = dict(narration_plan()["segments"][0], **narration_payload()[0])
    original["text"] = "Besi dipotong perlahan."

    def unexpected_client(cfg):
        pytest.fail("An impossible phrase-preserving rewrite must not contact a model")

    monkeypatch.setattr(script_writer, "_client", unexpected_client)
    with pytest.raises(ValueError, match="no removable words"):
        script_writer.shorten_for_measured_duration(
            original, language="id", measured_duration=8, available_duration=1,
            cfg=SimpleNamespace(llm_provider="local"),
        )


@pytest.mark.parametrize("text", ["Besi-dipotong perlahan", "Besi-dipotong-perlahan"])
def test_duration_rewrite_removability_uses_spoken_tokens_not_whitespace(monkeypatch, text):
    original = dict(narration_plan()["segments"][0], **narration_payload()[0])
    original.update(text=text, action_anchors=[{"event_id": "cut_observed", "phrase": "Besi dipotong"}])

    class Analyzer:
        def generate_json_with_retry(self, prompt, *, response_validator, **kwargs):
            assert "Maximum words: 2" in prompt
            # Punctuation changes alone must not satisfy the shortening gate.
            unchanged_speech = {"text": "Besi-dipotong-perlahan.", "retained_claim_indices": [0]}
            assert "remove spoken words" in response_validator(unchanged_speech)
            shorter_speech = {"text": "Besi dipotong.", "retained_claim_indices": [0]}
            assert response_validator(shorter_speech) is None
            return shorter_speech

    monkeypatch.setattr(script_writer, "_client", lambda cfg: (Analyzer(), "mock"))
    result = script_writer.shorten_for_measured_duration(
        original, language="id", measured_duration=8, available_duration=1,
        cfg=SimpleNamespace(llm_provider="local"),
    )
    assert result["text"] == "Besi dipotong."
    assert result["action_anchors"][0]["start_time"] == 2.0


def test_duration_schema_with_no_original_claims_and_specific_invalid_references(monkeypatch):
    original = {"text": "Komponen kemudian diperiksa sebelum proses berikutnya dimulai.", "claims": []}

    class Analyzer:
        def generate_json_with_retry(self, prompt, *, response_validator, **kwargs):
            schema = response_validator.response_schema
            assert schema["properties"]["retained_claim_indices"]["maxItems"] == 0
            result = {"text": "Komponen diperiksa.", "retained_claim_indices": []}
            assert response_validator(result) is None
            assert "original_claims is empty" in response_validator(dict(result, retained_claim_indices=[0]))
            return result

    monkeypatch.setattr(script_writer, "_client", lambda cfg: (Analyzer(), "mock"))
    result = script_writer.shorten_for_measured_duration(
        original, language="id", measured_duration=10, available_duration=5,
        cfg=SimpleNamespace(llm_provider="local"),
    )
    assert result["claims"] == []
