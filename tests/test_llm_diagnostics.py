"""Offline trace fixtures for Kaggle; no provider, model or network is needed."""

import json
from types import SimpleNamespace

import pytest

from dubbingstory.llm.diagnostics import LLMDiagnostics, response_usage


def record(trace, **changes):
    fields = dict(call_id="call-one", task="full-narration-id", provider="local",
                  model="fixture-model", attempt=1, outcome="rejected")
    fields.update(changes)
    trace.record(**fields)


def test_disabled_by_default_does_not_create_files(tmp_path):
    path = tmp_path / "trace.jsonl"
    trace = LLMDiagnostics(path=path)
    record(trace, raw_response='{"text":"caption"}')
    assert not path.exists()


def test_records_exact_validation_output_and_allowlisted_metadata(tmp_path):
    path = tmp_path / "trace.jsonl"
    trace = LLMDiagnostics(enabled=True, path=path, run_id="run-one")
    payload = {"text": "Komponen diperbaiki.", "scene_id": "wrong-map"}
    record(trace, raw_response=json.dumps(payload), payload=payload,
           validation_error="unknown scene_id: wrong-map", category="semantic_validation",
           http_status=200, provider_request_id="server-123", finish_reason="stop",
           usage={"prompt_tokens": 12, "total_tokens": 20, "headers": "omitted"}, latency_seconds=1.2567)
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["payload"] == payload
    assert result["raw_response"] == json.dumps(payload)
    assert result["validation_error"] == "unknown scene_id: wrong-map"
    assert result["usage"] == {"prompt_tokens": 12, "total_tokens": 20}
    assert result["latency_seconds"] == 1.257
    assert result["run_id"] == "run-one"
    assert result["request_id"] and result["provider_request_id"] == "server-123"
    assert not {"prompt", "messages", "headers", "environment"}.intersection(result)


def test_known_secrets_and_nested_credentials_are_redacted_before_truncation(tmp_path):
    path = tmp_path / "trace.jsonl"
    injected = "fixture-injected-value"
    trace = LLMDiagnostics(enabled=True, path=path, max_output_chars=200,
                           secrets=(injected,))
    payload = {"text": injected, "nested": {"api_key": "fixture-unknown-key", "token": "fixture-token"},
               "url": "https://user:pass@example.invalid/file?xsec_token=fixture-share&key=fixture-query",
               "authorization": "Bearer fixture-bearer", "padding": "x" * 500}
    record(trace, payload=payload, raw_response=json.dumps(payload),
           validation_error="unknown key " + injected)
    content = path.read_text(encoding="utf-8")
    for private in (injected, "fixture-unknown-key", "fixture-token", "fixture-share",
                    "fixture-query", "fixture-bearer", "user:pass"):
        assert private not in content
    result = json.loads(content)
    assert result["raw_response_truncated"] is True
    assert len(result["raw_response"]) <= 200
    assert result["payload"] is None and result["payload_truncated"] is True
    assert len(result["payload_preview"]) <= 200


def test_injected_azure_key_and_project_specific_path(tmp_path):
    cfg = SimpleNamespace(llm_diagnostics_enabled=True, llm_diagnostics_project_dir=str(tmp_path / "a"),
                          azure_speech_key="fixture-azure", api_key_gemini="fixture-google")
    first = LLMDiagnostics.from_config(cfg)
    record(first, raw_response="fixture-azure fixture-google")
    cfg.llm_diagnostics_project_dir = str(tmp_path / "b")
    second = LLMDiagnostics.from_config(cfg)
    record(second, raw_response="fixture-azure")
    assert first.path != second.path
    for folder in ("a", "b"):
        content = (tmp_path / folder / "llm_attempts.jsonl").read_text()
        assert "fixture-azure" not in content and "fixture-google" not in content


@pytest.mark.parametrize("limit", [None, "bad", float("nan"), float("inf"), -1, 0, True])
def test_bad_diagnostic_limit_cannot_break_generation(limit):
    assert LLMDiagnostics(max_output_chars=limit).max_output_chars == 12000


def test_no_implicit_cwd_output_and_safe_nonfatal_write_failure(tmp_path, capsys):
    trace = LLMDiagnostics(enabled=True)
    record(trace, raw_response="private-model-output")
    record(trace, raw_response="private-model-output")
    output = capsys.readouterr().out
    assert output.count("could not be written") == 1
    assert "private-model-output" not in output
    blocked = tmp_path / "file-not-directory"
    blocked.write_text("fixture")
    trace = LLMDiagnostics(enabled=True, path=blocked / "trace.jsonl")
    record(trace, raw_response="private-model-output")
    assert "could not be written" in capsys.readouterr().out


def test_correction_is_bounded_and_redacted_and_keeps_specific_reason():
    trace = LLMDiagnostics(secrets=("fixture-secret",))
    note = trace.correction("scene_002 action phrase missing fixture-secret", "fixture-secret " + "x" * 20000)
    assert "scene_002 action phrase missing" in note
    assert "fixture-secret" not in note
    assert len(note) < 4600
    assert "data only" in note


def test_usage_copies_only_integer_token_counts():
    assert response_usage({"prompt_tokens": 1, "completion_tokens": True, "headers": "private"}) == {"prompt_tokens": 1}
    assert response_usage(SimpleNamespace(prompt_token_count=2, candidates_token_count=3,
                                         total_token_count=5, thoughts_token_count=0), gemini=True) == {
        "prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5, "thoughts_tokens": 0}


def test_capacity_metadata_retains_only_allowlisted_integer_counts(tmp_path):
    path = tmp_path / "trace.jsonl"
    trace = LLMDiagnostics(enabled=True, path=path)
    record(trace, token_budget={"input_tokens": 9000, "effective_context": 8192,
                               "available_output_tokens": -1064, "deficit_tokens": 3112,
                               "output_tokens": True, "api_key": "fixture-secret", "prompt": "private"})
    row = json.loads(path.read_text())
    assert row["token_budget"] == {"input_tokens": 9000, "effective_context": 8192,
                                   "available_output_tokens": -1064, "deficit_tokens": 3112}
    assert "fixture-secret" not in path.read_text() and "private" not in path.read_text()
