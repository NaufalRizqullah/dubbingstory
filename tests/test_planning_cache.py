"""Content-bound planning cache tests; no providers or media tools are used."""

import json
from types import SimpleNamespace

import pytest

from dubbingstory.llm.text_client import TextGenerationError
from dubbingstory.story.planning_cache import cached_json, payload_hash


def validate(payload):
    return None if isinstance(payload, dict) and payload == {"fact": "observed"} else "wrong fact schema"


def invoke(tmp_path, calls, **changes):
    kwargs = dict(project_dir=tmp_path, cfg=SimpleNamespace(llm_local_model="fixture-model"),
                  task="map-scenes", prompt="Private prompt fixture", response_validator=validate,
                  version=1, evidence_identity={"source": "source-hash", "storyboard": "board-hash"})
    kwargs.update(changes)
    def generate():
        calls.append("generate")
        return {"fact": "observed"}
    return cached_json(**kwargs, generate=generate)


def test_cache_reuses_valid_response_without_storing_prompt_or_configuration(tmp_path):
    calls = []
    invoke(tmp_path, calls)
    invoke(tmp_path, calls)
    assert calls == ["generate"]
    text = next((tmp_path / "story_cache").glob("*.json")).read_text()
    assert "Private prompt fixture" not in text and "fixture-model" not in text
    assert json.loads(text)["payload"] == {"fact": "observed"}


@pytest.mark.parametrize("change", [
    {"prompt": "Changed prompt"}, {"version": 2},
    {"evidence_identity": {"source": "new-source", "storyboard": "board-hash"}},
    {"cfg": SimpleNamespace(llm_local_model="other-model")},
    {"cfg": SimpleNamespace(llm_local_model="fixture-model", llm_local_base_url="http://other/v1")},
])
def test_model_evidence_prompt_or_contract_change_invalidates_cache(tmp_path, change):
    calls = []
    invoke(tmp_path, calls)
    invoke(tmp_path, calls, **change)
    assert calls == ["generate", "generate"]


@pytest.mark.parametrize("fault", ["broken_json", "hash", "schema", "status"])
def test_invalid_cache_is_revalidated_and_replaced(tmp_path, fault):
    calls = []
    invoke(tmp_path, calls)
    path = next((tmp_path / "story_cache").glob("*.json"))
    row = json.loads(path.read_text())
    if fault == "hash":
        row["payload"]["fact"] = "changed"
    elif fault == "schema":
        row["payload"] = {"wrong": "shape"}
        row["payload_sha256"] = payload_hash(row["payload"])
    elif fault == "status":
        row["status"] = "partial"
    path.write_text("{broken" if fault == "broken_json" else json.dumps(row))
    invoke(tmp_path, calls)
    assert calls == ["generate", "generate"]
    assert json.loads(path.read_text())["payload"] == {"fact": "observed"}


def test_rejected_generation_never_publishes_a_cache_entry(tmp_path):
    with pytest.raises(TextGenerationError, match="wrong fact schema"):
        cached_json(tmp_path, SimpleNamespace(), task="one", prompt="evidence", version=1,
                    response_validator=validate, generate=lambda: {"wrong": "shape"})
    assert not list(tmp_path.rglob("*.json"))


def test_run_id_does_not_invalidate_good_batches(tmp_path):
    calls = []
    for run in ("first", "resumed"):
        invoke(tmp_path, calls, cfg=SimpleNamespace(llm_local_model="fixture-model", llm_fallback_run_id=run))
    assert calls == ["generate"]
