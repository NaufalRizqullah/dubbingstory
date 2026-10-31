"""Project trace configuration regressions; run with the suite in Kaggle."""

from types import SimpleNamespace

import yaml

from dubbingstory import cli
from dubbingstory.config import DEFAULT_CONFIG, _flatten_dict


def test_diagnostics_defaults_are_opt_in_and_flatten_without_secret_loading():
    defaults = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    cfg = _flatten_dict(defaults)
    assert cfg["llm_diagnostics_enabled"] is False
    assert cfg["llm_diagnostics_path"] == ""
    assert cfg["llm_diagnostics_max_output_chars"] == 12000


def test_project_context_tracks_resume_and_does_not_override_explicit_trace(tmp_path):
    cfg = SimpleNamespace(project_output_dir=str(tmp_path),
                          llm_diagnostics_path=str(tmp_path / "custom.jsonl"),
                          llm_diagnostics_enabled=False)
    first = cli._setup_project_dir(cfg, "first")
    assert cfg.llm_diagnostics_project_dir == first
    second = cli._setup_project_dir(cfg, "second")
    assert cfg.llm_diagnostics_project_dir == second
    assert cfg.llm_diagnostics_path == str(tmp_path / "custom.jsonl")
    assert cfg.llm_diagnostics_enabled is False
    assert not list(tmp_path.rglob("*.jsonl"))
