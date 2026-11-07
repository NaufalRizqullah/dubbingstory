"""Run bundles must not mix stale logs, video sources or credentials."""

import json
import zipfile

import pytest

from dubbingstory.llm.run_diagnostics import RunDiagnostics


def test_failed_run_bundle_contains_only_registered_run_artifacts(tmp_path):
    (tmp_path / "notebook_analyze.log").write_text("stale project log")
    old = tmp_path / "runs" / "old"
    old.mkdir(parents=True)
    (old / "notebook_analyze.log").write_text("other run")
    run = RunDiagnostics(tmp_path, "current", "project-one", secrets=("fixture-key",))
    run.snapshot_config({"llm": {"local": {"max_tokens": 2048}, "preflight": {"safety_margin": 256}},
                         "api_key": "fixture-key", "text": "fixture-key"})
    run.stage("plan", "running")
    run.log_path("plan").write_text("failure with fixture-key")
    run.trace_path.write_text('\n'.join(json.dumps(row) for row in (
        {"run_id": "old", "task": "old-story"}, {"run_id": "current", "task": "story-planner"},
    )))
    run.stage("plan", "failed")
    bundle = run.finish(success=False, error_type="TextGenerationError")
    with zipfile.ZipFile(bundle) as archive:
        assert set(archive.namelist()) == {"notebook_plan.log", "llm_attempts.jsonl", "notebook_runtime.yaml", "run_manifest.json"}
        output = '\n'.join(archive.read(name).decode() for name in archive.namelist())
        assert "fixture-key" not in output and "old-story" not in output and "stale project log" not in output
        manifest = json.loads(archive.read("run_manifest.json"))
        assert manifest["status"] == "failed" and manifest["project"] == "project-one"
        snapshot = json.loads(archive.read("notebook_runtime.yaml"))
        assert snapshot["llm"]["local"]["max_tokens"] == 2048


@pytest.mark.parametrize("run_id", ["../outside", "a/b", ".", "..", ""])
def test_bundle_path_cannot_escape_project(tmp_path, run_id):
    with pytest.raises(ValueError):
        RunDiagnostics(tmp_path, run_id, "sample")


def test_run_folder_cannot_silently_reuse_previous_session(tmp_path):
    RunDiagnostics(tmp_path, "unique", "sample")
    with pytest.raises(FileExistsError):
        RunDiagnostics(tmp_path, "unique", "sample")
