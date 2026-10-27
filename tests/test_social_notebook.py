"""Notebook contract checks for Kaggle; never run installation or media cells."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "mereska-dubbingstory-tiktok-rednote.ipynb"


def read_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def source(cell_id):
    return "".join(next(cell for cell in read_notebook()["cells"] if cell["id"] == cell_id)["source"])


def literal_setting(name):
    for node in ast.parse(source("settings")).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing notebook setting: {name}")


def helpers():
    # Define only pure helper functions; no top-level install, network or file IO.
    allowed = {"render_options", "expected_output_names", "ingestion_arguments", "pipeline_steps"}
    definitions = [node for node in ast.parse(source("helpers")).body
                   if isinstance(node, ast.FunctionDef) and node.name in allowed]
    namespace = {"VOICE_ID": "id-ID-GadisNeural"}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


def runtime_policy():
    allowed = {"needs_local_server", "needs_gemini_secret"}
    definitions = [node for node in ast.parse(source("settings")).body
                   if isinstance(node, ast.FunctionDef) and node.name in allowed]
    namespace = {}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


def pipeline_harness(tmp_path, *, start_from="segment", vision="qwen", provider="gemini",
                     fallback=True, fail_stage=None):
    """Execute only orchestration with fake CLI/server functions, never media or models."""
    for relative in ("source.mp4", "segment_manifest.json", "storyboard.json",
                     "scripts/script_id.json", "audio/audio_id.manifest.json"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    state_path = tmp_path / "llm_runtime_state.json"
    state_path.write_text('{"previous_run": "keep"}', encoding="utf-8")
    events, saved_runs = [], []
    config = {"llm": {"fallback": {"run_id": "old", "state_path": str(state_path)}}}
    runs = iter(["run-one", "run-two"])
    server = object()

    def start_server():
        events.append("start-server")
        return server, SimpleNamespace(close=lambda: events.append("close-log"))

    def stop_server(process):
        if process is not None:
            assert process is server
            events.append("stop-server")

    def run_cli(arguments, stage):
        events.append((stage, config["llm"]["fallback"]["run_id"]))
        if stage == fail_stage:
            raise RuntimeError("simulated CLI failure")

    namespace = {
        **helpers(), **runtime_policy(), "ACTIVE_PROJECT": "sample", "START_FROM": start_from,
        "PROJECT_DIR": tmp_path, "VISION_BACKEND": vision, "TEXT_LLM_PROVIDER": provider,
        "LOCAL_FALLBACK_ENABLED": fallback, "CONFIG": config,
        "uuid": SimpleNamespace(uuid4=lambda: next(runs)), "run_cli": run_cli,
        "save_config": lambda: saved_runs.append(config["llm"]["fallback"]["run_id"]),
        "start_qwen_server": start_server, "stop_server": stop_server,
    }
    statements = ast.parse(source("pipeline")).body
    start = next(index for index, node in enumerate(statements) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "steps" for target in node.targets))
    orchestration = compile(ast.Module(body=statements[start:], type_ignores=[]), str(NOTEBOOK), "exec")

    def run():
        exec(orchestration, namespace)

    return run, events, saved_runs, state_path


def test_notebook_has_no_saved_execution_or_old_platform_metadata():
    notebook = read_notebook()
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    assert "papermill" not in notebook["metadata"]
    assert "kaggle" not in notebook["metadata"]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
        else:
            assert "\\`" not in "".join(cell["source"])


def test_all_code_cells_parse_as_python_without_execution():
    for cell in read_notebook()["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{NOTEBOOK.name}:{cell['id']}")


def test_native_social_defaults_and_explicit_rights_gate():
    assert literal_setting("VIDEO_INPUT") == "https://www.rednote.com/discovery/item/6ad206ee000000001400240a"
    assert literal_setting("ENABLE_CANVAS") is False
    assert literal_setting("I_HAVE_RIGHTS") is False
    assert literal_setting("DOWNLOAD_QUALITY") == "max"
    assert literal_setting("NARRATION_SOURCE") == "visual"
    assert literal_setting("SUBTITLE_STYLE") == "karaoke"
    assert literal_setting("WORD_ALIGNMENT") == "edge"
    assert literal_setting("NARRATION_FIT") == "rewrite"


def test_text_retry_defaults_and_shared_interpreter_setup():
    assert literal_setting("TEXT_LLM_PROVIDER") == "gemini"
    assert literal_setting("LOCAL_FALLBACK_ENABLED") is True
    assert literal_setting("GEMINI_MAX_ATTEMPTS") == 3
    assert literal_setting("GEMINI_TIMEOUT_SECONDS") == 30
    assert literal_setting("GEMINI_BUDGET_SECONDS") == 90
    assert literal_setting("LOCAL_MAX_ATTEMPTS") == 2
    setup = source("setup")
    assert "RUNTIME_PYTHON = Path(sys.executable)" in setup
    assert 'pip_stage(["install", "--prefer-binary", *packages]' in setup
    assert 'packages.append(f"vllm=={VLLM_VERSION}")' in setup
    assert 'pip_stage(["check"]' in setup
    for forbidden in ("VENV_DIR", '"venv"', '"-e"', '"--upgrade"', '"torch=='):
        assert forbidden not in setup


@pytest.mark.parametrize("start, vision, provider, fallback, expected", [
    ("analyze", "qwen", "gemini", False, True),
    ("narrate", "qwen", "gemini", False, False),
    ("narrate", "qwen", "gemini", True, True),
    ("dub", "qwen", "gemini", True, True),
    ("narrate", "gemini", "gemini", True, True),
    ("dub", "gemini", "local", False, True),
    ("analyze", "gemini", "gemini", False, False),
    ("render", "qwen", "local", True, False),
])
def test_local_model_policy_for_resume_and_text_fallback(start, vision, provider, fallback, expected):
    assert runtime_policy()["needs_local_server"](start, vision, provider, fallback) is expected


@pytest.mark.parametrize("start, vision, provider, expected", [
    ("segment", "qwen", "local", False),
    ("segment", "qwen", "gemini", True),
    ("analyze", "gemini", "local", True),
    ("narrate", "gemini", "local", False),
    ("dub", "gemini", "local", False),
    ("render", "gemini", "gemini", False),
])
def test_local_only_and_render_do_not_require_gemini_secret(start, vision, provider, expected):
    assert runtime_policy()["needs_gemini_secret"](start, vision, provider) is expected
    assert "if not needs_gemini_secret(" in source("secret")


def test_text_configuration_matches_router_contract(tmp_path):
    config_assignment = next(node for node in ast.parse(source("helpers")).body
                             if isinstance(node, ast.Assign) and any(
                                 isinstance(target, ast.Name) and target.id == "CONFIG" for target in node.targets))
    llm_node = next(value for key, value in zip(config_assignment.value.keys, config_assignment.value.values)
                    if isinstance(key, ast.Constant) and key.value == "llm")
    names = {node.id for node in ast.walk(llm_node) if isinstance(node, ast.Name)} - {"str", "PROJECT_DIR", "BASE_URL"}
    values = {name: literal_setting(name) for name in names}
    values.update(PROJECT_DIR=tmp_path, BASE_URL="http://127.0.0.1:8000/v1")
    config = eval(compile(ast.Expression(llm_node), str(NOTEBOOK), "eval"), values)
    assert config["provider"] == "gemini"
    assert config["gemini"] == {"max_attempts": 3, "timeout_seconds": 30, "budget_seconds": 90,
                                "initial_wait_seconds": 5, "wait_increment_seconds": 5}
    assert config["fallback"] == {"enabled": True, "state_path": str(tmp_path / "llm_runtime_state.json"), "run_id": ""}
    assert config["local"] == {"base_url": "http://127.0.0.1:8000/v1", "model": "Qwen/Qwen3-VL-4B-Instruct",
                               "max_attempts": 2, "timeout_seconds": 60, "budget_seconds": 120,
                               "max_tokens": 2048, "context_window": 10240}


def test_pipeline_keeps_model_until_dub_and_shares_one_new_run_id(tmp_path):
    run, events, saved_runs, state_path = pipeline_harness(tmp_path)
    run()
    assert events == [("segment", "run-one"), "start-server", ("analyze", "run-one"),
                      ("narrate", "run-one"), ("dub", "run-one"), "stop-server", "close-log",
                      ("render", "run-one")]
    assert saved_runs == ["run-one"]
    events.clear()
    run()
    assert saved_runs == ["run-one", "run-two"]
    assert all(event[1] == "run-two" for event in events if isinstance(event, tuple))
    assert json.loads(state_path.read_text(encoding="utf-8")) == {"previous_run": "keep"}


@pytest.mark.parametrize("start", ["narrate", "dub"])
@pytest.mark.parametrize("vision", ["qwen", "gemini"])
def test_fresh_resume_starts_local_server_for_text_fallback(tmp_path, start, vision):
    run, events, _, _ = pipeline_harness(tmp_path, start_from=start, vision=vision)
    run()
    assert events[0] == "start-server"
    assert events[1] == (start, "run-one")
    assert events.count("start-server") == 1
    assert events[-3:] == ["stop-server", "close-log", ("render", "run-one")]


def test_render_only_has_no_model_lifecycle(tmp_path):
    run, events, _, _ = pipeline_harness(tmp_path, start_from="render")
    run()
    assert events == [("render", "run-one")]


def test_qwen_vision_only_releases_model_before_cloud_narration(tmp_path):
    run, events, _, _ = pipeline_harness(tmp_path, fallback=False)
    run()
    assert events == [("segment", "run-one"), "start-server", ("analyze", "run-one"),
                      "stop-server", "close-log", ("narrate", "run-one"),
                      ("dub", "run-one"), ("render", "run-one")]


def test_cloud_only_does_not_start_local_model(tmp_path):
    run, events, _, _ = pipeline_harness(tmp_path, vision="gemini", fallback=False)
    run()
    assert events == [(stage, "run-one") for stage in ("segment", "analyze", "narrate", "dub", "render")]


def test_explicit_local_text_stays_available_through_tts(tmp_path):
    run, events, _, _ = pipeline_harness(tmp_path, vision="gemini", provider="local", fallback=False)
    run()
    assert events == [("segment", "run-one"), "start-server", ("analyze", "run-one"),
                      ("narrate", "run-one"), ("dub", "run-one"), "stop-server", "close-log",
                      ("render", "run-one")]


def test_pipeline_always_stops_model_after_cli_error(tmp_path):
    run, events, _, _ = pipeline_harness(tmp_path, start_from="narrate", fail_stage="narrate")
    with pytest.raises(RuntimeError, match="simulated CLI failure"):
        run()
    assert events == ["start-server", ("narrate", "run-one"), "stop-server", "close-log"]


def test_standalone_stages_filter_cached_other_languages():
    config_assignment = next(node for node in ast.parse(source("helpers")).body
                             if isinstance(node, ast.Assign) and any(
                                 isinstance(target, ast.Name) and target.id == "CONFIG" for target in node.targets))
    requested = next(value for key, value in zip(config_assignment.value.keys, config_assignment.value.values)
                     if isinstance(key, ast.Constant) and key.value == "requested_languages")
    assert ast.literal_eval(requested) == ["id"]


@pytest.mark.parametrize("enabled, ratios, layout, names", [
    (False, ["source"], "pad", ["final_id_source.mp4"]),
    (True, ["source", "9:16"], "layered", ["final_id_source.mp4", "final_id_canvas_9x16.mp4"]),
])
def test_canvas_toggle_keeps_source_output(enabled, ratios, layout, names):
    functions = helpers()
    assert functions["render_options"](enabled) == {"ratios": ratios, "vertical_layout": layout}
    assert functions["expected_output_names"](enabled) == names


def test_remote_ingest_keeps_share_query_and_rights_flag():
    url = "https://www.xiaohongshu.com/discovery/item/abc123?share=example"
    arguments = helpers()["ingestion_arguments"](url, True, "sample", "max", True)
    assert arguments[arguments.index("--url") + 1] == url
    assert "--i-have-rights" in arguments
    assert "--input" not in arguments
    assert "--yt-cookies" not in arguments
    assert arguments[arguments.index("--quality") + 1] == "max"


def test_ingest_requires_rights_and_supports_local_fallback():
    build = helpers()["ingestion_arguments"]
    with pytest.raises(ValueError, match="hak penggunaan"):
        build("https://example.invalid/video", True, "sample", "max", False)
    local = build("/kaggle/input/source/video.mp4", False, "sample", "max", True)
    assert local[local.index("--input") + 1] == "/kaggle/input/source/video.mp4"
    assert "--url" not in local


def test_pipeline_uses_full_stages_and_resume_skips_vision():
    build = helpers()["pipeline_steps"]
    stages = build("sample", "segment")
    assert [stage for stage, _ in stages] == ["segment", "analyze", "narrate", "dub", "render"]
    render_arguments = stages[-1][1]
    assert render_arguments[render_arguments.index("--mode") + 1] == "full"
    assert [stage for stage, _ in build("sample", "dub")] == ["dub", "render"]
    assert [stage for stage, _ in build("sample", "render")] == ["render"]


def test_feature_preflight_matches_current_repository():
    assignments = [node for node in ast.parse(source("setup")).body if isinstance(node, ast.Assign)]
    assignment = next(node for node in assignments if any(
        isinstance(target, ast.Name) and target.id == "required_markers" for target in node.targets))
    for relative, markers in ast.literal_eval(assignment.value).items():
        content = (ROOT / relative).read_text(encoding="utf-8")
        assert all(marker in content for marker in markers), relative


def test_no_original_cleanup_or_youtube_downloader_is_copied():
    code = "\n".join("".join(cell["source"]) for cell in read_notebook()["cells"] if cell["cell_type"] == "code")
    for forbidden in ("rm -rf", "shutil.rmtree", "download_subtitles", "ingest.youtube", "cookiesfrombrowser", "load_dotenv"):
        assert forbidden not in code
    assert "normalize_source_url" in code
    assert 'source.provider not in {"tiktok", "xiaohongshu"}' in code
