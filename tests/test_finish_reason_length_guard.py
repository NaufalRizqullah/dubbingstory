from types import SimpleNamespace

import pytest

from dubbingstory.vision.openai_vision import (
    OpenAIVisionAnalyzer,
    VisionContextBudgetError,
    VisionOutputTruncatedError,
    VisionRunawayError,
    _qwen3vl_visual_tokens,
    _repetition_score,
    _validate_response_schema,
)
from dubbingstory.vision.schemas import DEEP_SCENE_RESCUE_SCHEMA, DEEP_SCENE_SCHEMA


def _analyzer_without_client():
    a = object.__new__(OpenAIVisionAnalyzer)
    a.model = "Qwen/Qwen3-VL-4B-Instruct"
    a.temperature = 0.7
    a.max_tokens = 640
    a.model_max_context = 10240
    a.image_mode = "file"
    a.structured_outputs = True
    a.allow_schema_fallback = False
    a.repetition_penalty = 1.0
    a.top_p = 0.8
    a.top_k = 20
    a.presence_penalty = 1.5
    a.image_token_cost = 512
    a.truncation_retry_tokens = 896
    a.runaway_rescue_tokens = 448
    a.temporal_truncation_retry_tokens = 6144
    a.runaway_threshold = 0.35
    return a


def _response(text, *, finish_reason="stop", prompt_tokens=100, completion_tokens=20):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def test_qwen3vl_dimension_estimator_matches_32px_grid():
    # 768/32=24 and 432 rounds to 448 -> 14 cells; +8 boundary overhead.
    assert _qwen3vl_visual_tokens(768, 432) == 344


def test_repetition_classifier_distinguishes_normal_and_runaway():
    normal = (
        '{"action":"worker removes the fastener",'
        '"changes":"the cover separates from the housing",'
        '"likely_context":"mechanical disassembly before inspection"}'
    )
    runaway = " ".join(["component is rotating in the machine repeatedly"] * 30)
    assert _repetition_score(normal) < 0.10
    assert _repetition_score(runaway) >= 0.35


def test_deep_schema_is_budgeted_and_rescue_is_smaller():
    normal = DEEP_SCENE_SCHEMA["properties"]
    rescue = DEEP_SCENE_RESCUE_SCHEMA["properties"]
    assert normal["visible_objects"]["maxItems"] == 6
    assert normal["action"]["maxLength"] == 120
    assert normal["text_visible"]["maxItems"] == 3
    assert rescue["visible_objects"]["maxItems"] == 4
    assert rescue["action"]["maxLength"] == 80
    assert rescue["text_visible"]["maxItems"] == 2


def test_client_side_schema_validation_rejects_semantic_overflow():
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string", "maxLength": 3}},
        "required": ["value"],
        "additionalProperties": False,
    }
    with pytest.raises(Exception, match="schema validation failed"):
        _validate_response_schema({"value": "toolong"}, schema)


def test_sampling_baseline_is_sent_to_vllm():
    analyzer = _analyzer_without_client()
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return _response('{"ok":"yes"}')

    analyzer.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "string", "maxLength": 8}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    out = analyzer._call_with_retry(
        [{"role": "user", "content": "return json"}],
        max_tokens_override=100,
        response_schema=schema,
    )
    assert out == {"ok": "yes"}
    sent = calls[0]
    assert sent["temperature"] == 0.7
    assert sent["top_p"] == 0.8
    assert sent["presence_penalty"] == 1.5
    assert sent["extra_body"]["top_k"] == 20
    assert "repetition_penalty" not in sent["extra_body"]  # Qwen baseline is 1.0.


def test_length_is_classified_as_clean_truncation_when_not_repetitive():
    analyzer = _analyzer_without_client()

    class Completions:
        def create(self, **kwargs):
            return _response(
                '{"action":"worker removes a part","changes":"part is now loose"',
                finish_reason="length",
                completion_tokens=100,
            )

    analyzer.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    with pytest.raises(VisionOutputTruncatedError, match="clean_truncation"):
        analyzer._call_with_retry(
            [{"role": "user", "content": "return json"}],
            max_tokens_override=100,
        )


def test_length_is_classified_as_runaway_when_repetitive():
    analyzer = _analyzer_without_client()
    repeated = " ".join(["component is rotating in the machine repeatedly"] * 35)

    class Completions:
        def create(self, **kwargs):
            return _response(repeated, finish_reason="length", completion_tokens=100)

    analyzer.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    with pytest.raises(VisionRunawayError, match="runaway"):
        analyzer._call_with_retry(
            [{"role": "user", "content": "return json"}],
            max_tokens_override=100,
        )


def test_clean_truncation_retries_same_frames_with_larger_budget():
    analyzer = _analyzer_without_client()
    analyzer._build_image_content = lambda paths: [
        {"type": "image_url", "image_url": {"url": f"file://{p}"}} for p in paths
    ]
    calls = []

    def fake_call(messages, **kwargs):
        calls.append((messages, kwargs))
        if len(calls) == 1:
            raise VisionOutputTruncatedError("finish_reason=length clean_truncation")
        return {"action": "worker inspects the part", "confidence": 0.8}

    analyzer._call_with_retry = fake_call
    result = analyzer.analyze_scene_from_frames(["a.jpg", "b.jpg", "c.jpg"], "prompt", trace_id="s1")
    assert result["action"] == "worker inspects the part"
    assert calls[0][1]["max_tokens_override"] == 640
    assert calls[1][1]["max_tokens_override"] == 896
    assert calls[1][1]["trace_id"] == "s1-truncation-retry"
    first_images = calls[0][0][0]["content"][:-1]
    retry_images = calls[1][0][0]["content"][:-1]
    assert retry_images == first_images


def test_runaway_rescue_uses_one_frame_compact_schema_and_448_tokens():
    analyzer = _analyzer_without_client()
    analyzer._build_image_content = lambda paths: [
        {"type": "image_url", "image_url": {"url": f"file://{p}"}} for p in paths
    ]
    calls = []

    def fake_call(messages, **kwargs):
        calls.append((messages, kwargs))
        if len(calls) == 1:
            raise VisionRunawayError("finish_reason=length runaway")
        return {"action": "machine cuts the surface", "confidence": 0.7}

    analyzer._call_with_retry = fake_call
    result = analyzer.analyze_scene_from_frames(["a.jpg", "b.jpg", "c.jpg"], "prompt", trace_id="s2")
    assert result["action"] == "machine cuts the surface"
    assert calls[1][1]["max_tokens_override"] == 448
    assert calls[1][1]["schema_name"] == "deep_scene_analysis_rescue"
    rescue_images = calls[1][0][0]["content"][:-1]
    assert len(rescue_images) == 1
    assert rescue_images[0]["image_url"]["url"].endswith("b.jpg")


def test_context_rescue_reduces_images_but_preserves_output_budget():
    analyzer = _analyzer_without_client()
    analyzer._build_image_content = lambda paths: [
        {"type": "image_url", "image_url": {"url": f"file://{p}"}} for p in paths
    ]
    calls = []

    def fake_call(messages, **kwargs):
        calls.append((messages, kwargs))
        if len(calls) == 1:
            raise VisionContextBudgetError("prompt too large")
        return {"action": "operator measures the component", "confidence": 0.9}

    analyzer._call_with_retry = fake_call
    analyzer.analyze_scene_from_frames(["a.jpg", "b.jpg", "c.jpg"], "prompt", trace_id="s3")
    assert calls[1][1]["max_tokens_override"] == 640
    assert calls[1][1]["trace_id"] == "s3-context-rescue"
    rescue_images = calls[1][0][0]["content"][:-1]
    assert len(rescue_images) == 1
