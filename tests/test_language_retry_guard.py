from dubbingstory.story.language_guard import payload_language_mismatch_reason
from dubbingstory.story.narration_qa import evaluate_narration
from dubbingstory.story.prompt_templates import build_v2_narration_prompt
from dubbingstory.story.script_writer import _call_json, _fallback_beat_text


class FakeRetryAnalyzer:
    def __init__(self):
        self.calls = []

    def generate_json_with_retry(
        self,
        prompt,
        *,
        temperature,
        response_validator,
        task_name,
        model,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "temperature": temperature,
                "task_name": task_name,
                "model": model,
            }
        )
        wrong = [{"scene_id": "scene_001", "text": "The component is checked before the process continues."}]
        correct = [{"scene_id": "scene_001", "text": "Komponen diperiksa sebelum proses dilanjutkan ke tahap berikutnya."}]
        assert response_validator(wrong) is not None
        assert response_validator(correct) is None
        return correct


def test_call_json_passes_language_validator_to_retry_layer():
    analyzer = FakeRetryAnalyzer()
    payload = _call_json(
        analyzer,
        "gemini-test",
        "OUTPUT_LANGUAGE_CODE: id\nOUTPUT_LANGUAGE_NAME: Bahasa Indonesia",
        language="id",
        task_name="summary-narration",
    )
    assert payload[0]["text"].startswith("Komponen")
    assert analyzer.calls[0]["task_name"] == "summary-narration-id"


def test_language_guard_rejects_obvious_english_for_id():
    english = [{"scene_id": "scene_001", "text": "The component is then checked before the process continues."}]
    indonesian = [{"scene_id": "scene_001", "text": "Komponen kemudian diperiksa sebelum proses dilanjutkan."}]
    assert payload_language_mismatch_reason(english, "id") is not None
    assert payload_language_mismatch_reason(indonesian, "id") is None


def test_qa_flags_wrong_language():
    segments = [{"scene_id": "scene_001", "text": "The component is then checked before the process continues."}]
    plan = {
        "total_duration": 5,
        "segments": [{"scene_id": "scene_001", "target_words": 9}],
    }
    qa = evaluate_narration(segments, plan, language="id")
    assert "language_mismatch:expected_id" in qa["issues"]
    assert qa["language_issue"]


def test_indonesian_fallback_uses_scene_context_category():
    measurement_scene = {
        "scene_id": "scene_002",
        "analysis": {"action": "worker measures a component with a caliper"},
    }
    welding_scene = {
        "scene_id": "scene_035",
        "analysis": {"action": "worker welds a cracked metal surface"},
    }
    measurement = _fallback_beat_text(
        [measurement_scene],
        "id",
        {"scene_id": "scene_002", "story_role": "setup", "intent": "measure component"},
    )
    welding = _fallback_beat_text(
        [welding_scene],
        "id",
        {"scene_id": "scene_035", "story_role": "process", "intent": "repair damaged area"},
    )
    assert measurement != welding
    assert "diperiksa" in measurement.lower() or "ukuran" in measurement.lower()
    assert "perbaikan" in welding.lower() or "menyambung" in welding.lower()


def test_prompt_repeats_language_contract_for_retry_safety():
    scene = {
        "scene_id": "scene_001",
        "analysis": {"action": "worker checks a component", "confidence": 0.8},
    }
    plan = {
        "segments": [
            {
                "scene_id": "scene_001",
                "covers_scene_ids": ["scene_001"],
                "available_duration": 8,
                "timeline_start": 0,
                "timeline_end": 8,
                "story_role": "setup",
                "intent": "introduce the task",
                "must_explain": [],
                "avoid_repeating": [],
                "continuity_from_previous": "",
                "transition_strategy": "none",
                "min_words": 10,
                "target_words": 15,
                "max_words": 20,
            }
        ]
    }
    prompt = build_v2_narration_prompt(
        scenes=[scene],
        narration_plan=plan,
        story_plan={},
        story_memory={},
        language="id",
        style="viral_fb",
        style_config={"name": "test", "tone": "natural", "rules": []},
        hedging_threshold=0.5,
        mode="summary",
    )
    assert "OUTPUT_LANGUAGE_CODE: id" in prompt
    assert "OUTPUT_LANGUAGE_NAME: Bahasa Indonesia" in prompt
    assert "On retries, NEVER infer the language from a previous response" in prompt
