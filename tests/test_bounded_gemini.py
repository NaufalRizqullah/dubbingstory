"""Bounded Gemini policy tests with fake SDK responses and a virtual clock.

Authored for Kaggle validation; no provider or real sleep is used by these tests.
"""
import json
from types import SimpleNamespace

import pytest

from dubbingstory.vision import gemini_analyzer as gemini


class ServiceError(RuntimeError):
    def __init__(self, status, message="provider unavailable", retry_after=None):
        super().__init__(message)
        self.status_code = status
        self.headers = {} if retry_after is None else {"Retry-After": str(retry_after)}


class VirtualClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeModels:
    def __init__(self, outcomes, clock, durations=()):
        self.outcomes = iter(outcomes)
        self.clock = clock
        self.durations = iter(durations)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        self.clock.now += next(self.durations, 0)
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)


def analyzer_with_fake_sdk(monkeypatch, outcomes, *, durations=(), **options):
    clock = VirtualClock()
    monkeypatch.setattr(gemini.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(gemini.time, "sleep", clock.sleep)
    models = FakeModels(outcomes, clock, durations)
    # Existing integrations sometimes supply their own initialized SDK client.
    analyzer = gemini.GeminiVideoAnalyzer.__new__(gemini.GeminiVideoAnalyzer)
    analyzer.client = SimpleNamespace(models=models)
    analyzer.model = "primary-model"
    analyzer.fallback_model = "fallback-model"
    for name, value in options.items():
        setattr(analyzer, name, value)
    return analyzer, models, clock


def test_default_three_total_attempts_and_no_sleep_after_last(monkeypatch):
    analyzer, models, clock = analyzer_with_fake_sdk(monkeypatch, [ServiceError(429) for _ in range(3)])
    with pytest.raises(gemini.GeminiGenerationError) as error:
        analyzer.generate_json_with_retry("Return JSON")
    assert len(models.calls) == error.value.attempts == 3
    assert [call["model"] for call in models.calls] == ["primary-model", "primary-model", "fallback-model"]
    assert clock.sleeps == [5, 10]
    assert error.value.status_code == 429
    assert error.value.retryable is True
    assert error.value.service_failure is True
    assert error.value.service_failures == 3
    assert error.value.service_streak_reset is False
    assert all(call["config"].http_options.retry_options.attempts == 1 for call in models.calls)


def test_single_attempt_uses_primary_and_reports_actual_count(monkeypatch):
    analyzer, models, clock = analyzer_with_fake_sdk(monkeypatch, [ServiceError(503)], max_attempts=1)
    with pytest.raises(gemini.GeminiGenerationError) as error:
        analyzer.generate_json_with_retry("Return JSON")
    assert error.value.attempts == 1
    assert models.calls[0]["model"] == "primary-model"
    assert clock.sleeps == []


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_nonretryable_http_errors_stop_immediately(monkeypatch, status):
    analyzer, models, clock = analyzer_with_fake_sdk(monkeypatch, [ServiceError(status)])
    with pytest.raises(gemini.GeminiGenerationError) as error:
        analyzer.generate_json_with_retry("Return JSON")
    assert len(models.calls) == error.value.attempts == 1
    assert error.value.retryable is False
    assert error.value.service_failure is False
    assert clock.sleeps == []


def test_timeout_is_capped_by_remaining_budget(monkeypatch):
    analyzer, models, clock = analyzer_with_fake_sdk(
        monkeypatch, [TimeoutError("slow") for _ in range(3)], durations=[30, 30, 15])
    with pytest.raises(gemini.GeminiGenerationError) as error:
        analyzer.generate_json_with_retry("Return JSON")
    assert error.value.attempts == 3
    assert [call["config"].http_options.timeout for call in models.calls] == [30000, 30000, 15000]
    assert clock.now == 90
    assert clock.sleeps == [5, 10]


def test_no_wait_or_new_attempt_when_backoff_would_exhaust_budget(monkeypatch):
    analyzer, models, clock = analyzer_with_fake_sdk(
        monkeypatch, [ServiceError(503)], durations=[8], budget_seconds=12)
    with pytest.raises(gemini.GeminiGenerationError) as error:
        analyzer.generate_json_with_retry("Return JSON")
    assert error.value.reason == "budget_exhausted"
    assert error.value.attempts == 1
    assert error.value.service_failures == 1
    assert clock.sleeps == []


def test_retry_after_longer_than_budget_does_not_stall(monkeypatch):
    analyzer, models, clock = analyzer_with_fake_sdk(monkeypatch, [ServiceError(429, retry_after=300)])
    with pytest.raises(gemini.GeminiGenerationError) as error:
        analyzer.generate_json_with_retry("Return JSON")
    assert error.value.reason == "budget_exhausted"
    assert len(models.calls) == 1
    assert clock.sleeps == []


def test_language_retry_keeps_original_prompt_and_validator(monkeypatch):
    prompt = "OUTPUT_LANGUAGE_CODE: id\nDescribe the measured evidence only."
    wrong = json.dumps({"text": "The part is repaired."})
    correct = json.dumps({"text": "Komponen diperbaiki."})
    analyzer, models, _ = analyzer_with_fake_sdk(monkeypatch, [wrong, correct])
    payload = analyzer.generate_json_with_retry(prompt, response_validator=lambda item:
        "Use Bahasa Indonesia" if item["text"].startswith("The") else None)
    assert payload["text"] == "Komponen diperbaiki."
    assert models.calls[0]["contents"] == [prompt]
    assert models.calls[1]["contents"][0] == prompt
    assert "Use Bahasa Indonesia" in models.calls[1]["contents"][1]


@pytest.mark.parametrize("response", ["", "not JSON", "null", '"scalar"'])
def test_invalid_outputs_are_retryable_but_not_service_outages(monkeypatch, response):
    analyzer, _, _ = analyzer_with_fake_sdk(monkeypatch, [response] * 3)
    with pytest.raises(gemini.GeminiGenerationError) as error:
        analyzer.generate_json_with_retry("Return object")
    assert error.value.attempts == 3
    assert error.value.retryable is True
    assert error.value.service_failure is False
    assert error.value.service_failures == 0
    assert error.value.service_streak_reset is True
    assert error.value.reason == "response_validation"


def test_validation_response_resets_service_failure_streak(monkeypatch):
    analyzer, _, _ = analyzer_with_fake_sdk(monkeypatch, [ServiceError(503), "not JSON", ServiceError(503)])
    with pytest.raises(gemini.GeminiGenerationError) as error:
        analyzer.generate_json_with_retry("Return object")
    assert error.value.service_failures == 1
    assert error.value.service_streak_reset is True


def test_provider_messages_are_not_logged_or_chained(monkeypatch, capsys):
    private = "private provider detail https://example.invalid/?private=value"
    analyzer, _, _ = analyzer_with_fake_sdk(monkeypatch, [ServiceError(401, private)])
    with pytest.raises(gemini.GeminiGenerationError) as error:
        analyzer.generate_json_with_retry("Unlogged user prompt")
    captured = capsys.readouterr().out
    assert private not in captured
    assert private not in str(error.value)
    assert "Unlogged user prompt" not in captured
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__ is True


def test_late_response_is_not_published_after_deadline(monkeypatch):
    analyzer, models, _ = analyzer_with_fake_sdk(monkeypatch, ['{"ok": true}'], durations=[91])
    with pytest.raises(gemini.GeminiGenerationError) as error:
        analyzer.generate_json_with_retry("Return JSON")
    assert len(models.calls) == 1
    assert error.value.reason == "budget_exhausted"


def test_client_configuration_disables_sdk_retries(monkeypatch):
    captured = {}
    def client(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()
    monkeypatch.setattr(gemini.genai, "Client", client)
    analyzer = gemini.GeminiVideoAnalyzer("example-key", timeout_seconds=12, max_attempts=3)
    assert captured["http_options"].timeout == 12000
    assert captured["http_options"].retry_options.attempts == 1
    assert analyzer.max_attempts == 3


def test_monkeypatched_module_defaults_work_with_existing_client(monkeypatch):
    monkeypatch.setattr(gemini, "MAX_ATTEMPTS", 2)
    monkeypatch.setattr(gemini, "INITIAL_WAIT_SECONDS", 0)
    monkeypatch.setattr(gemini, "WAIT_INCREMENT_SECONDS", 0)
    analyzer, models, clock = analyzer_with_fake_sdk(monkeypatch, [ServiceError(503), ServiceError(503)])
    with pytest.raises(gemini.GeminiGenerationError) as error:
        analyzer.generate_json_with_retry("Return JSON")
    assert error.value.attempts == len(models.calls) == 2
    assert clock.sleeps == [0]
