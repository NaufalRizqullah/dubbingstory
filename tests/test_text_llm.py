"""Offline provider fixtures; run in Kaggle, never load a model or call an API."""

import json
from types import SimpleNamespace

import pytest

from dubbingstory.llm import text_client as routing
from dubbingstory.vision.gemini_analyzer import GeminiGenerationError


def config(**overrides):
    values = dict(llm_provider="gemini", llm_fallback_enabled=True,
                  llm_fallback_run_id="run-one", api_key_gemini="fixture-only")
    values.update(overrides)
    return SimpleNamespace(**values)


def service_error():
    return GeminiGenerationError(attempts=3, status_code=429, retryable=True,
                                 service_failure=True, service_failures=3)


def test_service_failure_is_sticky_across_stage_clients_and_resets_for_new_run(tmp_path, monkeypatch):
    path = tmp_path / "routing.json"
    calls = []

    def cloud(self, prompt, **kwargs):
        calls.append("cloud")
        raise service_error()

    def local(self, prompt, **kwargs):
        calls.append("local")
        return {"text": "fixture"}

    monkeypatch.setattr(routing.TextLLMClient, "_generate_gemini", cloud)
    monkeypatch.setattr(routing.TextLLMClient, "_generate_local", local)
    for _ in range(2):
        # Each CLI process constructs a new config/client.
        client = routing.TextLLMClient(config(llm_fallback_state_path=str(path)), "cloud-model")
        assert client.generate_json_with_retry("private evidence") == {"text": "fixture"}
    assert calls == ["cloud", "local", "local"]
    state = json.loads(path.read_text())
    assert state == {"schema_version": 1, "run_id": "run-one", "provider": "local",
                     "reason": "gemini_service_exhausted", "gemini_attempts": 3}
    assert "private" not in path.read_text()
    assert "fixture-only" not in path.read_text()
    fresh = routing.TextLLMClient(config(llm_fallback_state_path=str(path), llm_fallback_run_id="run-two"), "cloud-model")
    fresh.generate_json_with_retry("new evidence")
    assert calls[-2:] == ["cloud", "local"]


def test_output_validation_failure_tries_local_without_latching_service(monkeypatch):
    calls = []

    def cloud(self, prompt, **kwargs):
        calls.append("cloud")
        raise GeminiGenerationError(attempts=3, retryable=True, reason="invalid_json")

    monkeypatch.setattr(routing.TextLLMClient, "_generate_gemini", cloud)
    monkeypatch.setattr(routing.TextLLMClient, "_generate_local", lambda *a, **k: calls.append("local") or [])
    cfg = config()
    for _ in range(2):
        routing.TextLLMClient(cfg, "model").generate_json_with_retry("prompt")
    assert calls == ["cloud", "local", "cloud", "local"]
    assert not getattr(cfg, "_text_llm_state", {})


def test_authentication_failure_does_not_enter_fallback_loop(monkeypatch):
    def cloud(*args, **kwargs):
        raise GeminiGenerationError(attempts=1, status_code=401, retryable=False)

    monkeypatch.setattr(routing.TextLLMClient, "_generate_gemini", cloud)
    monkeypatch.setattr(routing.TextLLMClient, "_generate_local", lambda *a, **k: pytest.fail("unexpected fallback"))
    with pytest.raises(routing.TextGenerationError, match="model/access/configuration"):
        routing.TextLLMClient(config(), "model").generate_json_with_retry("prompt")


def test_disabled_fallback_preserves_cloud_error(monkeypatch):
    def cloud(*args, **kwargs):
        raise service_error()

    monkeypatch.setattr(routing.TextLLMClient, "_generate_gemini", cloud)
    with pytest.raises(GeminiGenerationError):
        routing.TextLLMClient(config(llm_fallback_enabled=False), "model").generate_json_with_retry("prompt")


def test_explicit_local_needs_no_gemini_key(monkeypatch):
    cfg = config(llm_provider="local", api_key_gemini="", llm_fallback_enabled=False)
    assert routing.text_llm_available(cfg)
    assert routing.strict_text_routing(cfg)
    monkeypatch.setattr(routing.TextLLMClient, "_generate_gemini", lambda *a, **k: pytest.fail("cloud called"))
    monkeypatch.setattr(routing.TextLLMClient, "_generate_local", lambda *a, **k: {"ok": True})
    assert routing.TextLLMClient(cfg, "cloud-name").generate_json_with_retry("prompt") == {"ok": True}


def test_corrupt_state_and_missing_run_id_fail_before_cloud(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    path.write_text("not json")
    monkeypatch.setattr(routing.TextLLMClient, "_generate_gemini", lambda *a, **k: pytest.fail("cloud called"))
    with pytest.raises(routing.TextGenerationError, match="routing state"):
        routing.TextLLMClient(config(llm_fallback_state_path=str(path)), "model").generate_json_with_retry("prompt")
    with pytest.raises(routing.TextGenerationError, match="run_id"):
        routing.TextLLMClient(config(llm_fallback_state_path=str(path), llm_fallback_run_id=""), "model").generate_json_with_retry("prompt")


class FakeResponse:
    status_code = 200

    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def json(self):
        return self.body


class FakeSession:
    def __init__(self, outputs, *, count=100, window=10240):
        self.outputs = iter(outputs)
        self.count = count
        self.window = window
        self.calls = []
        self.trust_env = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        assert kwargs["allow_redirects"] is False
        assert 0 < kwargs["timeout"] <= 60
        if url.endswith("/tokenize"):
            return FakeResponse({"count": self.count, "max_model_len": self.window})
        return FakeResponse(next(self.outputs))


def completion(payload, *, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {
        "content": json.dumps({"result": payload})}}]}


def setup_local(monkeypatch, outputs, **session_options):
    session = FakeSession(outputs, **session_options)
    monkeypatch.setattr(routing.requests, "Session", lambda: session)
    monkeypatch.setattr(routing.time, "sleep", lambda _: None)
    client = routing.TextLLMClient(config(llm_provider="local"), "cloud-model")
    return client, session


def test_local_retries_validation_preserves_prompt_and_uses_server_tokenizer(monkeypatch):
    accepted = [{"text": "Komponen diperiksa sebelum proses dilanjutkan."}]
    client, session = setup_local(monkeypatch, [completion({"text": "wrong"}), completion(accepted)])
    prompt = "Original visual evidence. OUTPUT_LANGUAGE_CODE: id"
    result = client.generate_json_with_retry(prompt, response_validator=lambda value: None if value == accepted else "wrong language")
    assert result == accepted
    assert len(session.calls) == 4
    assert session.trust_env is False
    for url, kwargs in session.calls:
        assert kwargs["json"]["model"] == "Qwen/Qwen3-VL-4B-Instruct"
        assert kwargs["json"]["messages"][1]["content"] == prompt
    assert session.calls[0][0] == "http://127.0.0.1:8000/tokenize"
    assert session.calls[1][1]["json"]["response_format"] == {"type": "json_object"}


def test_context_overflow_fails_without_truncating_or_generating(monkeypatch):
    client, session = setup_local(monkeypatch, [], count=9000)
    with pytest.raises(routing.TextGenerationError, match="context_budget_exceeded"):
        client.generate_json_with_retry("all the evidence")
    assert len(session.calls) == 1
    assert session.calls[0][1]["json"]["messages"][1]["content"] == "all the evidence"


def test_local_truncation_is_rejected_after_two_total_attempts(monkeypatch, capsys):
    client, session = setup_local(monkeypatch, [completion({"text": "sensitive-content"}, finish="length")] * 2)
    with pytest.raises(routing.TextGenerationError, match="2 attempt.*incomplete_output"):
        client.generate_json_with_retry("sensitive-evidence")
    assert len(session.calls) == 4
    logged = capsys.readouterr().out
    assert "sensitive-content" not in logged
    assert "sensitive-evidence" not in logged


def test_local_http_access_error_is_not_retried_or_logged_raw(monkeypatch):
    client, session = setup_local(monkeypatch, [])
    response = FakeResponse({"error": "secret error body"})
    response.status_code = 401
    session.post = lambda *a, **k: response
    with pytest.raises(routing.TextGenerationError, match="1 attempt.*local_http_401"):
        client.generate_json_with_retry("prompt")


def test_local_budget_caps_each_request_and_stops_without_extra_retry(monkeypatch):
    client, session = setup_local(monkeypatch, [])
    now = [0.0]
    monkeypatch.setattr(routing.time, "monotonic", lambda: now[0])
    client.cfg.llm_local_budget_seconds = 4
    original_post = session.post

    def slow_tokenizer(*args, **kwargs):
        assert kwargs["timeout"] == 4
        result = original_post(*args, **kwargs)
        now[0] = 4
        return result

    session.post = slow_tokenizer
    with pytest.raises(routing.TextGenerationError, match="1 attempt.*budget_exhausted"):
        client.generate_json_with_retry("prompt")
    assert len(session.calls) == 1


def test_local_route_rejects_nonlocal_or_credentialed_endpoints():
    for base in ("https://example.com/v1", "http://user:password@127.0.0.1:8000/v1",
                 "http://127.0.0.1:8000/v1?token=value"):
        client = routing.TextLLMClient(config(llm_provider="local", llm_local_base_url=base), "model")
        with pytest.raises(routing.TextGenerationError, match="loopback"):
            client.generate_json_with_retry("prompt")


@pytest.mark.parametrize("value", [0, -1, 0.5, True, float("nan"), "invalid"])
def test_local_attempts_must_be_positive_integers(value):
    client = routing.TextLLMClient(config(llm_provider="local", llm_local_max_attempts=value), "model")
    with pytest.raises(routing.TextGenerationError, match="llm_local_max_attempts"):
        client.generate_json_with_retry("prompt")


@pytest.mark.parametrize("fails", [False, True])
def test_text_task_closes_owned_gemini_client(monkeypatch, fails):
    from dubbingstory.vision import gemini_analyzer

    closed = []

    class Analyzer:
        def __init__(self, **kwargs):
            self.client = SimpleNamespace(close=lambda: closed.append(True))

        def generate_json_with_retry(self, prompt, **kwargs):
            if fails:
                raise service_error()
            return {"ok": True}

    monkeypatch.setattr(gemini_analyzer, "GeminiVideoAnalyzer", Analyzer)
    client = routing.TextLLMClient(config(llm_fallback_enabled=False), "model")
    if fails:
        with pytest.raises(GeminiGenerationError):
            client.generate_json_with_retry("prompt")
    else:
        assert client.generate_json_with_retry("prompt") == {"ok": True}
    assert closed == [True]
