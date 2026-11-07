"""Capacity and chronological batching fixtures; intended for Kaggle, no model needed."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from dubbingstory.llm import text_client as routing
from dubbingstory.llm.batching import budgeted_groups, generate_budgeted_batches
from dubbingstory.llm.diagnostics import LLMDiagnostics


def settings(**overrides):
    values = dict(llm_provider="gemini", llm_fallback_enabled=True, llm_preflight_enabled=True,
                  llm_fallback_run_id="fixture-run", llm_local_context_window=10240,
                  llm_local_max_tokens=2048, llm_preflight_safety_margin=256)
    values.update(overrides)
    return SimpleNamespace(**values)


class Response:
    status_code = 200

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def json(self):
        return self.value


class Session:
    def __init__(self, counts, *, server_context=10240, payloads=()):
        self.counts = iter(counts)
        self.server_context = server_context
        self.payloads = iter(payloads)
        self.calls = []
        self.trust_env = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        assert kwargs["allow_redirects"] is False
        if url.endswith("/tokenize"):
            return Response({"count": next(self.counts), "max_model_len": self.server_context})
        return Response({"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"result": next(self.payloads)})}}]})


def setup(monkeypatch, counts, *, cfg=None, **options):
    session = Session(counts, **options)
    monkeypatch.setattr(routing.requests, "Session", lambda: session)
    monkeypatch.setattr(routing.time, "sleep", lambda value: None)
    return routing.TextLLMClient(cfg or settings(), "cloud-model"), session


def test_preflight_uses_actual_chat_template_and_minimum_of_configured_and_server_context(monkeypatch):
    client, session = setup(monkeypatch, [5000], server_context=8192)
    budget = client.preflight("Complete private visual evidence")
    assert budget == {"input_tokens": 5000, "configured_context": 10240, "server_context": 8192,
                      "effective_context": 8192, "output_tokens": 2048, "safety_margin": 256,
                      "available_output_tokens": 2936, "deficit_tokens": 0}
    body = session.calls[0][1]["json"]
    assert body["add_generation_prompt"] is True
    assert body["messages"] == client._messages("Complete private visual evidence")
    assert '"result"' in body["messages"][0]["content"]
    assert session.trust_env is False
    assert not hasattr(client.cfg, "_text_llm_state")


def test_oversized_fallback_is_rejected_before_cloud_without_latching_route(monkeypatch, tmp_path, capsys):
    cfg = settings(llm_fallback_state_path=str(tmp_path / "route.json"))
    client, session = setup(monkeypatch, [9000], cfg=cfg)
    path = tmp_path / "trace.jsonl"
    client.diagnostics = LLMDiagnostics(enabled=True, path=path, run_id="fixture-run")
    monkeypatch.setattr(client, "_generate_gemini", lambda *args, **kwargs: pytest.fail("cloud generation ran"))
    with pytest.raises(routing.ContextBudgetError) as error:
        client.generate_json_with_retry("private-evidence")
    assert error.value.budget["deficit_tokens"] == 1064
    assert len(session.calls) == 1
    assert not (tmp_path / "route.json").exists()
    row = json.loads(path.read_text())
    assert row["token_budget"] == error.value.budget
    assert row["raw_response"] is None and row["attempt"] == 0
    assert row["category"] == "context_budget_exceeded"
    assert "private-evidence" not in path.read_text() + capsys.readouterr().out


def test_fitting_fallback_is_checked_before_cloud_and_output_reservation_is_forwarded(monkeypatch):
    client, session = setup(monkeypatch, [7000])
    cloud_calls = []

    def cloud(prompt, **kwargs):
        assert len(session.calls) == 1 and session.calls[0][0].endswith("/tokenize")
        cloud_calls.append(kwargs)
        return {"accepted": True}

    monkeypatch.setattr(client, "_generate_gemini", cloud)
    assert client.generate_json_with_retry("evidence", max_output_tokens=1024) == {"accepted": True}
    assert cloud_calls[0]["max_output_tokens"] == 1024


@pytest.mark.parametrize("overrides", [{"llm_fallback_enabled": False}, {"llm_preflight_enabled": False}])
def test_pure_cloud_or_legacy_mode_does_not_require_a_local_server(monkeypatch, overrides):
    client = routing.TextLLMClient(settings(**overrides), "cloud")
    monkeypatch.setattr(routing.requests, "Session", lambda: pytest.fail("unexpected local HTTP"))
    monkeypatch.setattr(client, "_generate_gemini", lambda *args, **kwargs: {"ok": True})
    assert client.preflight("evidence") == {}
    assert client.generate_json_with_retry("evidence") == {"ok": True}


def test_repairs_are_retokenized_and_fail_before_second_completion(monkeypatch, tmp_path):
    cfg = settings(llm_provider="local")
    client, session = setup(monkeypatch, [1000, 9000], cfg=cfg, payloads=({"wrong": True},))
    path = tmp_path / "attempts.jsonl"
    client.diagnostics = LLMDiagnostics(enabled=True, path=path)
    with pytest.raises(routing.ContextBudgetError):
        client.generate_json_with_retry("original-evidence", response_validator=lambda value: "missing scene_004")
    assert [url.rsplit("/", 1)[-1] for url, _ in session.calls] == ["tokenize", "completions", "tokenize"]
    assert "missing scene_004" in session.calls[2][1]["json"]["messages"][0]["content"]
    assert session.calls[2][1]["json"]["messages"][1]["content"] == "original-evidence"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["category"] for row in rows] == ["schema_or_language_rejected", "context_budget_exceeded"]
    assert rows[-1]["token_budget"]["input_tokens"] == 9000


def test_local_generation_respects_requested_output_cap_and_keeps_same_messages_as_preflight(monkeypatch):
    client, session = setup(monkeypatch, [1000, 1000], cfg=settings(llm_provider="local"), payloads=({"ok": True},))
    first = client.preflight("evidence", max_output_tokens=777)
    assert client.generate_json_with_retry("evidence", max_output_tokens=777) == {"ok": True}
    assert first["output_tokens"] == 777
    assert session.calls[0][1]["json"]["messages"] == session.calls[-1][1]["json"]["messages"]
    assert session.calls[-1][1]["json"]["max_tokens"] == 777


def test_preflight_transport_is_bounded_and_does_not_change_route(monkeypatch):
    client, session = setup(monkeypatch, [1000], cfg=settings(llm_local_budget_seconds=4))
    clock = [0.0]
    monkeypatch.setattr(routing.time, "monotonic", lambda: clock[0])
    original_post = session.post

    def post(*args, **kwargs):
        assert kwargs["timeout"] == 4
        result = original_post(*args, **kwargs)
        clock[0] = 4
        return result

    session.post = post
    with pytest.raises(routing.TextGenerationError, match="preflight failed.*budget_exhausted"):
        client.preflight("evidence")
    assert len(session.calls) == 1 and not hasattr(client.cfg, "_text_llm_state")


def overflow():
    return routing.ContextBudgetError({"input_tokens": 100, "configured_context": 200,
        "server_context": 200, "effective_context": 200, "output_tokens": 150,
        "safety_margin": 10, "available_output_tokens": 90, "deficit_tokens": 60})


def test_adaptive_groups_preserve_chronology_and_build_with_latest_continuity():
    emitted, prompts = [], []

    def prompt(group):
        return {"items": list(group), "previous": list(emitted)}

    def preflight(value, **kwargs):
        prompts.append(value)
        if len(value["items"]) > 2:
            raise overflow()

    groups = []
    for group in budgeted_groups(range(9), prompt, SimpleNamespace(preflight=preflight), max_items=4):
        groups.append(list(group))
        assert prompts[-1]["previous"] == emitted
        emitted.extend(group)
    assert groups == [[0, 1], [2, 3], [4, 5], [6, 7], [8]]
    assert emitted == list(range(9))


def test_single_item_that_cannot_fit_is_never_dropped_or_truncated():
    seen = []

    def preflight(prompt, **kwargs):
        seen.append(prompt)
        raise overflow()

    with pytest.raises(routing.ContextBudgetError):
        list(budgeted_groups(["complete-scene"], lambda group: group,
                             SimpleNamespace(preflight=preflight)))
    assert seen == [["complete-scene"]]


def test_custom_analyzer_without_preflight_keeps_fixed_bounded_groups():
    assert list(budgeted_groups(range(5), lambda group: pytest.fail("no tokenizer"), object(), max_items=2)) == [
        [0, 1], [2, 3], [4]]


@pytest.mark.parametrize("splitting,attempts", [(True, 1), (False, 2)])
def test_completion_timeout_is_scoped_to_splitting_callers(monkeypatch, tmp_path, splitting, attempts):
    client, session = setup(monkeypatch, [7801, 7801], cfg=settings(
        llm_provider="local", llm_batching_enabled=True,
        llm_local_timeout_seconds=120, llm_local_budget_seconds=240))
    path = tmp_path / "trace.jsonl"
    client.diagnostics = LLMDiagnostics(enabled=True, path=path)
    original_post = session.post
    completions = []

    def post(url, **kwargs):
        if url.endswith("/chat/completions"):
            completions.append(kwargs)
            raise routing.requests.ReadTimeout("private-request-body")
        return original_post(url, **kwargs)

    session.post = post
    expected = routing.LocalGenerationTimeout if splitting else routing.TextGenerationError
    with pytest.raises(expected), (routing.batch_timeout_recovery() if splitting else nullcontext()):
        client.generate_json_with_retry("private-evidence", max_output_tokens=2048)
    assert len(completions) == attempts
    assert not routing._BATCH_TIMEOUT_RECOVERY.get()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert all(row["category"] == "local_read_timeout" for row in rows)
    assert all(row["request_stage"] == "completion" for row in rows)
    assert all(row["request_timeout_seconds"] == 120 for row in rows)
    assert rows[0]["token_budget"]["deficit_tokens"] == 0
    assert "private" not in path.read_text()


@pytest.mark.parametrize("stage,error", [
    ("tokenize", routing.requests.ReadTimeout),
    ("completion", routing.requests.ConnectTimeout),
    ("completion", routing.requests.ConnectionError),
])
def test_non_completion_read_failures_never_request_splitting(monkeypatch, stage, error):
    client, session = setup(monkeypatch, [1000, 1000], cfg=settings(llm_provider="local"))
    original_post = session.post
    failures = []

    def post(url, **kwargs):
        if (stage == "tokenize") == url.endswith("/tokenize"):
            failures.append(url)
            raise error("unavailable")
        return original_post(url, **kwargs)

    session.post = post
    with pytest.raises(routing.TextGenerationError) as exc, routing.batch_timeout_recovery():
        client.generate_json_with_retry("evidence")
    assert not isinstance(exc.value, routing.LocalGenerationTimeout)
    assert len(failures) == 2


@pytest.mark.parametrize("failure", [routing.LocalGenerationTimeout("slow"), overflow()])
def test_generation_failure_splits_without_losing_continuity(failure):
    emitted, attempts = [], []

    def generate(group):
        assert routing._BATCH_TIMEOUT_RECOVERY.get()
        attempts.append((list(group), list(emitted)))
        if len(group) > 2:
            raise failure
        return list(group)

    for group, payload in generate_budgeted_batches(range(5), lambda group: group, object(), generate):
        assert not routing._BATCH_TIMEOUT_RECOVERY.get()
        assert payload == group
        emitted.extend(payload)
    assert emitted == list(range(5))
    assert attempts == [([0, 1, 2, 3], []), ([0, 1], []), ([2, 3], [0, 1]), ([4], [0, 1, 2, 3])]


def test_timeout_singleton_stops_and_unrelated_failure_never_splits():
    for error in (routing.LocalGenerationTimeout("slow"), routing.TextGenerationError("unavailable")):
        groups = []

        def generate(group):
            groups.append(list(group))
            raise error

        with pytest.raises(type(error)):
            list(generate_budgeted_batches([1, 2], lambda group: group, object(), generate))
        assert groups == ([[1, 2], [1]] if isinstance(error, routing.LocalGenerationTimeout) else [[1, 2]])
        assert not routing._BATCH_TIMEOUT_RECOVERY.get()
