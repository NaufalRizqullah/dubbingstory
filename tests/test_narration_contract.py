"""Mocked narration contracts; run in Kaggle, without model/API downloads."""

import copy
import jsonschema
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
    assert set(item["required"]) == {"scene_id", "speech_parts", "claims", "importance"}
    assert item["properties"]["scene_id"]["enum"] == ["beat_original"]
    parts = item["properties"]["speech_parts"]
    assert set(parts["required"]) == {"lead_in", "action_1", "after_action_1"}
    assert "cut_observed" in parts["properties"]["action_1"]["description"]
    assert parts["properties"]["action_1"]["pattern"]
    assert parts["additionalProperties"] is False
    assert validator(narration_payload()) is None


def test_unanchored_schema_keeps_legacy_fields_and_empty_plan_has_no_invalid_enum():
    schema = script_writer._narration_response_validator(narration_plan(anchored=False)).response_schema
    assert "action_anchors" not in schema["items"]["required"]
    empty_schema = script_writer._narration_response_validator({"segments": []}).response_schema
    assert empty_schema["maxItems"] == 0
    assert "enum" not in empty_schema["items"]["properties"]["scene_id"]


def test_no_eligible_events_schema_requires_only_unanchored_speech():
    plan = narration_plan()
    plan["segments"][0]["anchor_event_ids"] = []
    schema = script_writer._narration_response_validator(plan).response_schema
    assert schema["items"]["properties"]["speech_parts"]["required"] == ["lead_in"]


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
    assert '"speech_parts"' in output_contract
    assert '"action_1": "cut_observed"' in output_contract
    assert "beat_original" in output_contract
    assert "With no selected events use only lead_in" in output_contract
    assert "Do not return timestamps" in output_contract


def speech_payload():
    row = narration_payload()[0]
    row.pop("text")
    row.pop("action_anchors")
    row["speech_parts"] = {"lead_in": "Lalu", "action_1": "besi dipotong perlahan",
                           "after_action_1": "sebelum tahap berikutnya dimulai."}
    return [row]


def test_spoken_parts_form_exact_canonical_text_and_anchors_without_mutating_response():
    validator = script_writer._narration_response_validator(narration_plan())
    payload = speech_payload()
    before = copy.deepcopy(payload)
    jsonschema.validate(payload, validator.response_schema)
    assert validator(payload) is None

    class Analyzer:
        def generate_json_with_retry(self, prompt, *, response_validator, **kwargs):
            assert response_validator(payload) is None
            return payload

    result = script_writer._call_json(Analyzer(), "mock", "evidence", language="id", payload_validator=validator)
    assert result[0]["text"] == "Lalu besi dipotong perlahan sebelum tahap berikutnya dimulai."
    assert result[0]["action_anchors"] == [{"event_id": "cut_observed", "phrase": "besi dipotong perlahan"}]
    assert "speech_parts" not in result[0] and payload == before
    hydrated = script_writer._validate_narration(result, [], narration_plan(), "id")
    assert hydrated[0]["action_anchors"][0]["start_time"] == 2.0
    assert hydrated[0]["action_anchors"][0]["frame_ids"] == ["frame_observed"]


@pytest.mark.parametrize("fault", ["long", "missing", "extra", "mixed", "wrong_id", "repeated"])
def test_invalid_speech_parts_remain_rejected(fault):
    payload = speech_payload()
    row = payload[0]
    if fault == "long":
        row["speech_parts"]["action_1"] = "Tangan memegang alat dan mengaduk adonan kuning"
    elif fault == "missing":
        del row["speech_parts"]["action_1"]
    elif fault == "extra":
        row["speech_parts"]["event_id"] = "made_up"
    elif fault == "mixed":
        row["text"] = "A competing copy of narration"
    elif fault == "wrong_id":
        row["scene_id"] = "visual_scene"
    else:
        row["speech_parts"]["after_action_1"] = "kemudian besi dipotong perlahan lagi."
    validator = script_writer._narration_response_validator(narration_plan())
    assert validator(payload)
    if fault != "repeated":
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(payload, validator.response_schema)


def test_language_guard_checks_assembled_speech_before_accepting_provider_output():
    payload = speech_payload()
    payload[0]["speech_parts"] = {"lead_in": "The worker then", "action_1": "cuts the metal",
                                  "after_action_1": "with the tool before the next step begins."}

    class Analyzer:
        def generate_json_with_retry(self, prompt, *, response_validator, **kwargs):
            assert "language mismatch" in response_validator(payload)
            return payload

    with pytest.raises(ValueError, match="language mismatch"):
        script_writer._call_json(Analyzer(), "mock", "evidence", language="id",
                                  payload_validator=script_writer._narration_response_validator(narration_plan()))


def test_zero_and_two_actions_keep_plan_order_without_model_generated_ids():
    plan = narration_plan()
    entry = plan["segments"][0]
    entry["anchor_event_ids"].append("finish_observed")
    entry["visual_events"].append({**entry["visual_events"][0], "event_id": "finish_observed",
                                     "start_time": 7.0, "end_time": 8.0, "action": "The edge is smoothed"})
    payload = speech_payload()
    # JSON property order is immaterial; fixed plan order controls playback.
    payload[0]["speech_parts"] = {"action_2": "ujungnya diratakan", "after_action_2": "dengan alat.",
                                  "lead_in": "", "action_1": "Besi dipotong", "after_action_1": "lalu"}
    validator = script_writer._narration_response_validator(plan)
    jsonschema.validate(payload, validator.response_schema)
    assert validator(payload) is None
    result = validator.normalize_payload(payload)[0]
    assert result["text"] == "Besi dipotong lalu ujungnya diratakan dengan alat."
    assert [a["event_id"] for a in result["action_anchors"]] == entry["anchor_event_ids"]
    entry["anchor_event_ids"] = []
    payload[0]["speech_parts"] = {"lead_in": "Pekerjaan berlanjut."}
    validator = script_writer._narration_response_validator(plan)
    assert validator(payload) is None
    assert validator.normalize_payload(payload)[0]["action_anchors"] == []


def test_qa_schema_and_semantics_preserve_the_existing_spoken_action():
    original = {"beat_original": [{"event_id": "cut_observed", "phrase": "besi dipotong perlahan"}]}
    validator = script_writer._narration_response_validator(narration_plan(), preserved_anchors=original)
    schema = validator.response_schema
    assert schema["items"]["properties"]["speech_parts"]["properties"]["action_1"]["enum"] == ["besi dipotong perlahan"]
    assert validator(speech_payload()) is None
    payload = speech_payload()
    payload[0]["speech_parts"]["action_1"] = "besi dibelah"
    assert "absent" in validator(payload)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, schema)


def test_per_beat_schema_and_feedback_cover_both_failed_beats():
    plan = narration_plan()
    second = copy.deepcopy(plan["segments"][0])
    second["scene_id"] = "beat_second"
    second["anchor_event_ids"] = []
    plan["segments"].append(second)
    validator = script_writer._narration_response_validator(plan)
    assert len(validator.response_schema["items"]["anyOf"]) == 2
    payload = speech_payload() + [{**speech_payload()[0], "scene_id": "beat_second",
                                   "speech_parts": {"lead_in": "Pekerjaan berlanjut."}}]
    jsonschema.validate(payload, validator.response_schema)
    assert validator(payload) is None
    payload[0]["speech_parts"]["action_1"] = "satu dua tiga empat lima enam tujuh"
    payload[1]["speech_parts"]["action_1"] = "unexpected action"
    reason = validator(payload)
    assert "beat_original" in reason and "beat_second" in reason


def test_qa_accepts_legacy_phrase_punctuation_using_canonical_word_count():
    phrase = "besi — dipotong — perlahan — sekarang"
    validator = script_writer._narration_response_validator(narration_plan(), preserved_anchors={
        "beat_original": [{"event_id": "cut_observed", "phrase": phrase}]})
    payload = speech_payload()
    payload[0]["speech_parts"]["action_1"] = phrase
    jsonschema.validate(payload, validator.response_schema)
    assert validator(payload) is None


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
