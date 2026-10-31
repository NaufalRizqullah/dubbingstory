"""Notebook contract checks for Kaggle; never run installation or media cells."""
import ast
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
    allowed = {"render_options", "expected_output_names", "ingestion_arguments", "pipeline_steps", "selected_voice_id"}
    definitions = [node for node in ast.parse(source("helpers")).body
                   if isinstance(node, ast.FunctionDef) and node.name in allowed]
    namespace = {"VOICE_ID": "", "TTS_ENGINE": "edge", "TTS_VOICES": literal_setting("TTS_VOICES")}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


def runtime_policy():
    allowed = {"needs_local_server", "needs_gemini_secret"}
    definitions = [node for node in ast.parse(source("settings")).body
                   if isinstance(node, ast.FunctionDef) and node.name in allowed]
    namespace = {}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


def setup_helpers(names, **namespace):
    """Load selected setup functions only; never execute installation cells."""
    definitions = [node for node in ast.parse(source("setup")).body
                   if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in definitions} == set(names)
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
    keyframe = tmp_path / "keyframes" / "scene_001.jpg"
    keyframe.parent.mkdir(parents=True, exist_ok=True)
    keyframe.write_bytes(b"keyframe fixture")
    (tmp_path / "segment_manifest.json").write_text(json.dumps({
        "status": "segmented", "total_scenes": 1,
        "scenes": [{"scene_id": "scene_001", "start_time": 0.0, "end_time": 2.0}],
        "keyframes": {"scene_001": [str(keyframe)]},
    }), encoding="utf-8")
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
        "REPO_DIR": ROOT, "sys": sys, "json": json,
    }
    statements = ast.parse(source("pipeline")).body
    validation = [node for node in statements if isinstance(node, ast.FunctionDef)
                  and node.name == "validate_segment_for_analysis"]
    exec(compile(ast.Module(body=validation, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
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


def test_personalized_native_social_configuration():
    assert "xhslink.com/" in literal_setting("VIDEO_INPUT")
    assert literal_setting("ENABLE_CANVAS") is False
    assert literal_setting("I_HAVE_RIGHTS") is True
    assert literal_setting("DOWNLOAD_QUALITY") == "max"
    assert literal_setting("NARRATION_SOURCE") == "visual"
    assert literal_setting("SUBTITLE_STYLE") == "karaoke"
    assert literal_setting("WORD_ALIGNMENT") == "auto"
    assert literal_setting("DENSE_VISUAL_ANALYSIS") is True
    assert literal_setting("FRAME_INTERVAL_SECONDS") == 1
    assert literal_setting("AUDIO_STRATEGY") == "dynamic_duck"
    assert literal_setting("ORIGINAL_VOLUME") == .20
    assert literal_setting("ACTION_SYNC") is True
    assert literal_setting("NARRATION_FIT") == "rewrite"


@pytest.mark.parametrize("strategy", ["dynamic_duck", "duck", "replace", "mute_original"])
def test_selected_audio_strategy_reaches_render_configuration(strategy):
    # Evaluate only the render mapping with literal settings and a pure helper;
    # do not execute file/network/setup code from the notebook.
    assignment = next(node for node in ast.parse(source("helpers")).body
                      if isinstance(node, ast.Assign) and any(
                          isinstance(target, ast.Name) and target.id == "CONFIG" for target in node.targets))
    render = next(value for key, value in zip(assignment.value.keys, assignment.value.values)
                  if isinstance(key, ast.Constant) and key.value == "render")
    namespace = helpers()
    for node in ast.walk(render):
        if isinstance(node, ast.Name) and node.id != "render_options":
            namespace[node.id] = strategy if node.id == "AUDIO_STRATEGY" else literal_setting(node.id)
    configured = eval(compile(ast.Expression(body=render), str(NOTEBOOK), "eval"), namespace)
    assert configured["audio_strategy"] == strategy
    assert configured["original_volume"] == .20
    assert configured["ratios"] == ["source"]


@pytest.mark.parametrize("engine", ["edge", "azure", "chirp", "piper", "elevenlabs"])
def test_switching_tts_selects_that_engines_voice_and_cli(engine):
    namespace = helpers()
    namespace["TTS_ENGINE"] = engine
    if engine == "elevenlabs":
        namespace["TTS_VOICES"][engine] = "configured-voice-id"
    stages = namespace["pipeline_steps"]("sample", "dub")
    arguments = stages[0][1]
    assert arguments[arguments.index("--engine") + 1] == engine
    assert arguments[arguments.index("--voice-id") + 1] == namespace["TTS_VOICES"][engine]
    namespace["VOICE_ID"] = "explicit-voice"
    assert namespace["selected_voice_id"]() == "explicit-voice"


@pytest.mark.parametrize("engine,alignment,expected", [
    ("edge", "auto", []), ("azure", "auto", ["azure"]),
    ("elevenlabs", "auto", []), ("chirp", "auto", ["chirp", "alignment"]),
    ("piper", "auto", ["alignment"]), ("edge", "whisperx", ["alignment"]),
])
def test_provider_dependencies_include_alignment_when_captions_are_off(engine, alignment, expected):
    definitions = [node for node in ast.parse(source("settings")).body
                   if isinstance(node, ast.FunctionDef) and node.name == "tts_optional_extras"]
    namespace = {}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    assert namespace["tts_optional_extras"](engine, alignment, True) == expected


def test_new_provider_credentials_are_redacted_in_child_logs():
    definitions = [node for node in ast.parse(source("helpers")).body
                   if isinstance(node, ast.FunctionDef) and node.name == "safe_log"]
    namespace = {"re": re, "CHILD_ENV": {"AZURE_SPEECH_KEY": "fake-azure-secret", "ELEVENLABS_API_KEY": "fake-eleven-secret"}}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    assert namespace["safe_log"]("fake-azure-secret fake-eleven-secret") == "[redacted] [redacted]"


def test_setup_requires_canvas_fix_before_resuming_existing_scripts():
    setup = source("setup")
    assignment = next(node for node in ast.parse(setup).body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "required_markers"
                              for target in node.targets))
    markers = ast.literal_eval(assignment.value)["dubbingstory/tts/voice_manager.py"]
    assert {"strict_timing_validated", "atrim=end_sample", "Narration canvas:"} <= set(markers)
    renderer = (ROOT / "dubbingstory" / "tts" / "voice_manager.py").read_text(encoding="utf-8")
    assert all(marker in renderer for marker in markers)


def test_text_retry_defaults_and_shared_interpreter_setup():
    assert literal_setting("TEXT_LLM_PROVIDER") == "gemini"
    assert literal_setting("LOCAL_FALLBACK_ENABLED") is True
    assert literal_setting("GEMINI_MAX_ATTEMPTS") == 3
    assert literal_setting("GEMINI_TIMEOUT_SECONDS") == 30
    assert literal_setting("GEMINI_BUDGET_SECONDS") == 90
    assert literal_setting("LOCAL_MAX_ATTEMPTS") == 2
    setup = source("setup")
    assert "RUNTIME_PYTHON = Path(sys.executable)" in setup
    assert 'pip_stage(["install", "--prefer-binary", *runtime_index_args, *packages]' in setup
    assert "local_runtime_install_spec(VLLM_VERSION)" in setup
    assert 'pip_stage(["check"]' in setup
    for forbidden in ("VENV_DIR", '"venv"', '"-e"', '"--upgrade"'):
        assert forbidden not in setup


def test_local_runtime_uses_vllm_supported_versions_with_one_cuda_variant():
    get_spec = setup_helpers({"local_runtime_install_spec"})["local_runtime_install_spec"]
    packages, index_args = get_spec(literal_setting("VLLM_VERSION"))
    versions = dict(requirement.split("==", 1) for requirement in packages)
    assert versions == {"vllm": "0.27.1", "torch": "2.13.0+cu130",
                        "torchvision": "0.28.0+cu130", "torchaudio": "2.11.0+cu130"}
    assert index_args == ["--extra-index-url", "https://download.pytorch.org/whl/cu130"]
    # TorchAudio 2.11 uses PyTorch's stable ABI: different base versions are
    # intentional. The original failure was the CUDA build, not base inequality.
    assert versions["torch"].split("+")[0] != versions["torchaudio"].split("+")[0]
    assert {versions[name].split("+")[1] for name in ("torch", "torchvision", "torchaudio")} == {"cu130"}


@pytest.mark.parametrize("version", ["0.27.0", "unknown", ""])
def test_unknown_vllm_version_does_not_guess_torch_wheels(version):
    get_spec = setup_helpers({"local_runtime_install_spec"})["local_runtime_install_spec"]
    with pytest.raises(ValueError):
        get_spec(version)


@pytest.mark.parametrize("install", [False, True])
@pytest.mark.parametrize("needs_local", [False, True])
def test_local_preflight_is_mandatory_even_when_installation_is_skipped(install, needs_local):
    # Execute only the top-level preflight gate with a fake function. Its
    # top-level position is deliberate: nesting under INSTALL_DEPENDENCIES
    # would reproduce the stale/broken-runtime failure on resumed notebooks.
    statements = ast.parse(source("setup")).body
    gates = [node for node in statements if isinstance(node, ast.If)
             and isinstance(node.test, ast.Name) and node.test.id == "NEED_LOCAL_MODEL"
             and any(isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
                     and item.func.id == "verify_local_runtime" for item in ast.walk(node))]
    assert len(gates) == 1
    calls = []
    namespace = {"INSTALL_DEPENDENCIES": install, "NEED_LOCAL_MODEL": needs_local,
                 "verify_local_runtime": lambda: calls.append("checked")}
    exec(compile(ast.Module(body=gates, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    assert calls == (["checked"] if needs_local else [])


def local_preflight_code():
    assignment = next(node for node in ast.parse(source("setup")).body
                      if isinstance(node, ast.Assign) and any(
                          isinstance(target, ast.Name) and target.id == "local_verify_code"
                          for target in node.targets))
    return ast.literal_eval(assignment.value)


def test_local_preflight_imports_the_reported_failure_path_without_loading_model():
    code = local_preflight_code()
    tree = ast.parse(code)
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
               for alias in node.names}
    assert {"torch", "torchaudio", "torchvision"} <= imports
    module_imports = {ast.literal_eval(node.args[0]) for node in ast.walk(tree)
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                      and isinstance(node.func.value, ast.Name)
                      and node.func.value.id == "importlib" and node.func.attr == "import_module"}
    assert {"transformers.audio_utils", "vllm.entrypoints.openai.api_server"} <= module_imports
    for forbidden in ("from_pretrained", "snapshot_download", "LLM(", "run_server("):
        assert forbidden not in code


@pytest.mark.parametrize("overrides", [{}, {"torchaudio": "2.11.0+cu128"}, {"vllm": "0.27.0"}])
def test_local_preflight_rejects_wrong_cuda_wheel_and_stale_runtime_before_import(overrides):
    packages, _ = setup_helpers({"local_runtime_install_spec"})["local_runtime_install_spec"]("0.27.1")
    expected = dict(requirement.split("==", 1) for requirement in packages)
    installed = {**expected, **overrides}
    # Only execute the metadata check with a fake lookup, never GPU imports.
    statements = ast.parse(local_preflight_code()).body
    start = next(index for index, node in enumerate(statements) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "actual" for target in node.targets))
    end = next(index for index, node in enumerate(statements) if isinstance(node, ast.Import)
               and any(alias.name == "torch" for alias in node.names))
    check = compile(ast.Module(body=statements[start:end], type_ignores=[]), str(NOTEBOOK), "exec")
    namespace = {"expected": expected, "version": installed.__getitem__, "json": json}
    if overrides:
        with pytest.raises(RuntimeError, match="Runtime belum sesuai preset"):
            exec(check, namespace)
    else:
        exec(check, namespace)


@pytest.mark.parametrize("returncode", [0, 1])
def test_local_preflight_uses_fresh_interpreter_and_reports_sanitized_failure(tmp_path, capsys, returncode):
    calls = []
    code = local_preflight_code()
    python = tmp_path / "kernel-python"
    log = tmp_path / "install.log"
    packages, _ = setup_helpers({"local_runtime_install_spec"})["local_runtime_install_spec"]("0.27.1")
    expected = dict(requirement.split("==", 1) for requirement in packages)

    def run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return SimpleNamespace(returncode=returncode, stdout="runtime versions\n",
                               stderr="probe: https://example.invalid/report?test=hidden\n")

    namespace = setup_helpers(
        {"verify_local_runtime", "setup_log_line"}, re=re, json=json, runtime_packages=packages,
        RUNTIME_PYTHON=python, REPO_DIR=tmp_path, INSTALL_LOG=log, local_verify_code=code,
        subprocess=SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired),
    )
    if returncode:
        with pytest.raises(RuntimeError, match="Preflight vLLM gagal sebelum model dimuat"):
            namespace["verify_local_runtime"]()
    else:
        namespace["verify_local_runtime"]()
    assert calls == [([str(python), "-c", code, json.dumps(expected)], {
        "cwd": tmp_path, "capture_output": True, "text": True,
        "encoding": "utf-8", "errors": "replace", "timeout": 180,
    })]
    recorded = log.read_text(encoding="utf-8")
    printed = capsys.readouterr().out
    for output in (recorded, printed):
        assert "runtime versions" in output
        assert "https://example.invalid/report?[redacted]" in output
        assert "test=hidden" not in output
    assert ("Import runtime lokal berhasil" in printed) is (returncode == 0)


def test_local_preflight_timeout_stops_setup_without_loading_model(tmp_path):
    def run(arguments, **kwargs):
        raise subprocess.TimeoutExpired(arguments, kwargs["timeout"])

    namespace = setup_helpers(
        {"verify_local_runtime"}, RUNTIME_PYTHON=tmp_path / "kernel-python",
        REPO_DIR=tmp_path, INSTALL_LOG=tmp_path / "install.log", local_verify_code="import-only",
        runtime_packages=["vllm==0.27.1"], json=json,
        subprocess=SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired),
    )
    with pytest.raises(RuntimeError, match="melebihi 180 detik"):
        namespace["verify_local_runtime"]()


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


@pytest.mark.parametrize("start_from", ["segment", "analyze"])
@pytest.mark.parametrize("fault", ["empty", "invalid_json", "failed", "duplicate", "invalid_window",
                                    "missing_keyframes", "missing_frame_file"])
def test_invalid_segmentation_stops_before_model_startup(tmp_path, start_from, fault):
    run, events, _, _ = pipeline_harness(tmp_path, start_from=start_from)
    path = tmp_path / "segment_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if fault == "empty":
        manifest.update(scenes=[], total_scenes=0, keyframes={})
    elif fault == "failed":
        manifest["status"] = "failed"
    elif fault == "duplicate":
        manifest["scenes"] *= 2
        manifest["total_scenes"] = 2
    elif fault == "invalid_window":
        manifest["scenes"][0]["end_time"] = float("nan")
    elif fault == "missing_keyframes":
        manifest["keyframes"] = {}
    elif fault == "missing_frame_file":
        Path(manifest["keyframes"]["scene_001"][0]).unlink()
    path.write_text("{broken" if fault == "invalid_json" else json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match='START_FROM="segment"'):
        run()
    assert events == ([("segment", "run-one")] if start_from == "segment" else [])


def qwen_startup_harness(tmp_path, *, ready_after=65, timeout=900, exited=False):
    """Only fake time/process/HTTP; no server, network, GPU or model is started."""
    clock = SimpleNamespace(now=0.0)

    def sleep(seconds):
        clock.now += seconds

    class Probe:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def connect_ex(self, address):
            return 1

    def urlopen(url, timeout):
        if clock.now < ready_after:
            raise OSError("not listening yet")
        return io.StringIO(json.dumps({"data": [{"id": "fixture-model"}]}))

    process = SimpleNamespace(poll=lambda: 1 if exited else None)
    launches, stopped = [], []

    def popen(command, **kwargs):
        launches.append((command, kwargs))
        return process

    namespace = {
        "socket": SimpleNamespace(socket=Probe), "PORT": 8000,
        "qwen_command": lambda: ["mock-server"],
        "CHILD_ENV": {"GOOGLE_API_KEY": "fake-google-secret", "ELEVENLABS_API_KEY": "fake-eleven-secret"},
        "PROJECT_DIR": tmp_path, "REPO_DIR": ROOT, "json": json,
        "subprocess": SimpleNamespace(Popen=popen, STDOUT=-2),
        "time": SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep),
        "SERVER_TIMEOUT_SECONDS": timeout, "MODEL_NAME": "fixture-model",
        "BASE_URL": "http://127.0.0.1:8000/v1",
        "urllib": SimpleNamespace(request=SimpleNamespace(urlopen=urlopen)),
        "stop_server": lambda item: stopped.append(item),
    }
    definitions = [node for node in ast.parse(source("pipeline")).body if isinstance(node, ast.FunctionDef)
                   and node.name == "start_qwen_server"]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace["start_qwen_server"], launches, stopped, process


def test_qwen_startup_reports_wait_and_readiness_without_exposing_credentials(tmp_path, capsys):
    start, launches, stopped, process = qwen_startup_harness(tmp_path)
    actual, log = start()
    try:
        assert actual is process
        assert stopped == []
        output = capsys.readouterr().out
        assert "Menyiapkan Qwen/vLLM" in output
        assert "30/900 detik" in output and "60/900 detik" in output
        assert output.count("Menunggu Qwen siap:") == 2
        assert "Qwen siap setelah 65 detik" in output
        assert str(tmp_path / "vllm_server.log") in output
        assert "fake-google-secret" not in output and "fake-eleven-secret" not in output
        assert launches[0][1]["env"] == {}
    finally:
        log.close()


@pytest.mark.parametrize("exited", [False, True])
def test_qwen_failed_startup_still_stops_process_and_closes_log(tmp_path, capsys, exited):
    start, launches, stopped, process = qwen_startup_harness(tmp_path, ready_after=100, timeout=35, exited=exited)
    with pytest.raises(RuntimeError, match="berhenti sebelum siap" if exited else "timeout"):
        start()
    assert stopped == [process]
    assert launches[0][1]["stdout"].closed
    assert "Qwen siap setelah" not in capsys.readouterr().out


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


def outputs_harness(tmp_path, monkeypatch, *, preview_failure=None, manifest_failure=None):
    """Run packaging with dummy MP4 bytes and fake display objects, never video decoding."""
    project = tmp_path / "outputs" / "sample"
    project.mkdir(parents=True)
    names = helpers()["expected_output_names"](True)
    for name in names:
        (project / name).write_bytes(b"rendered video fixture")
    config_path = project / "notebook.yaml"
    config_path.write_text("render: {}\n", encoding="utf-8")
    manifest = {"status": "complete", "rendered_videos": [str(project / name) for name in names]}
    if manifest_failure == "incomplete":
        manifest["status"] = "failed"
    elif manifest_failure == "missing":
        (project / names[0]).unlink()
    elif manifest_failure == "unreported":
        manifest["rendered_videos"] = []
    (project / "render_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    bundle = tmp_path / "sample_dubbing_results.zip"
    videos, displayed = [], []

    def video(data=None, **kwargs):
        # Reproduce the Kaggle IPython path check: filename= alone leaves data
        # as None. This stub never reads or decodes actual video content.
        if data is None:
            raise AssertionError("Video must receive a positional path")
        assert bundle.is_file(), "Packaging must finish before optional previews"
        assert kwargs == {"embed": True, "width": 360}
        videos.append(data)
        if preview_failure == "constructor" and len(videos) == 1:
            raise TypeError("simulated IPython preview error")
        return ("video", data)

    def display(value):
        if preview_failure == "display" and value == ("video", str(project / names[0])):
            raise RuntimeError("simulated display rendering error")
        displayed.append(value)

    fake_display = ModuleType("IPython.display")
    fake_display.Video, fake_display.display = video, display
    fake_display.FileLink = lambda path: ("link", path)
    fake_ipython = ModuleType("IPython")
    fake_ipython.display = fake_display
    monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", fake_display)
    monkeypatch.chdir(tmp_path)  # Restore cwd after the notebook's os.chdir.
    namespace = {**helpers(), "Path": Path, "json": json, "os": os,
                 "PROJECT_DIR": project, "WORK_ROOT": tmp_path, "ACTIVE_PROJECT": "sample",
                 "CONFIG_PATH": config_path, "ENABLE_CANVAS": True}

    def run():
        exec(compile(source("outputs"), str(NOTEBOOK), "exec"), namespace)

    return run, project, names, bundle, videos, displayed


@pytest.mark.parametrize("failure", [None, "constructor", "display"])
def test_output_bundle_and_downloads_survive_optional_preview_failure(tmp_path, monkeypatch, failure):
    run, project, names, bundle, videos, displayed = outputs_harness(
        tmp_path, monkeypatch, preview_failure=failure)
    run()
    assert videos == [str(project / name) for name in names]
    assert ("link", bundle.name) in displayed
    for name in names:
        assert ("link", str((project / name).relative_to(tmp_path))) in displayed
        assert (project / name).read_bytes() == b"rendered video fixture"
    with zipfile.ZipFile(bundle) as archive:
        assert set(names) <= set(archive.namelist())
        assert "render_manifest.json" in archive.namelist()
        assert all(archive.read(name) == b"rendered video fixture" for name in names)


def test_output_bundle_preserves_visual_evidence_and_action_misses(tmp_path, monkeypatch):
    run, project, _, bundle, _, _ = outputs_harness(tmp_path, monkeypatch)
    evidence = {
        "visual_timeline.json": {"status": "complete", "per_second": [{"second": 0}]},
        "storyboard.json": {"scenes": []},
        "narration_plan.json": {"action_sync_enabled": True},
        "story_plan.json": {"hook": "Visible work begins"},
        "story_memory.json": {"process": "assembly"},
        "audio/audio_id.actions.json": {"status": "partial", "validated": False,
                                      "segments": [{"anchors": [{"matched": False}]}]},
    }
    for relative, payload in evidence.items():
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    run()
    with zipfile.ZipFile(bundle) as archive:
        for relative, payload in evidence.items():
            assert json.loads(archive.read(relative)) == payload


@pytest.mark.parametrize("failure", ["incomplete", "missing", "unreported"])
def test_outputs_reject_incomplete_or_unverified_render_before_packaging(tmp_path, monkeypatch, failure):
    run, _, _, bundle, videos, displayed = outputs_harness(
        tmp_path, monkeypatch, manifest_failure=failure)
    with pytest.raises(RuntimeError):
        run()
    assert not bundle.exists()
    assert videos == displayed == []


@pytest.mark.parametrize("burn", [False, True])
def test_rerender_applies_subtitle_burn_toggle_without_repeating_pipeline(burn):
    # Execute only the action gate, as if the user edited the cell's toggles.
    gate = next(node for node in ast.parse(source("rerender")).body
                if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                and node.test.id == "RUN_RENDER_AGAIN")
    config = {"render": {"burn_subtitles": not burn}, "subtitle": {"style": "classic"}}
    events = []
    namespace = {**helpers(), "RUN_RENDER_AGAIN": True, "RERENDER_CANVAS": False,
                 "RERENDER_SUBTITLE_STYLE": "karaoke", "RERENDER_BURN_SUBTITLES": burn,
                 "BURN_SUBTITLES": not burn, "CONFIG": config, "ACTIVE_PROJECT": "sample",
                 "save_config": lambda: events.append(("saved", json.loads(json.dumps(config)))),
                 "run_cli": lambda arguments, stage: events.append((stage, arguments))}
    exec(compile(ast.Module(body=[gate], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    assert events == [("saved", {"render": {"burn_subtitles": burn, "ratios": ["source"],
                                           "vertical_layout": "pad"}, "subtitle": {"style": "karaoke"}}),
                      ("render", ["render", "--project", "sample", "--mode", "full"])]


@pytest.mark.parametrize("url", [
    "https://www.xiaohongshu.com/discovery/item/abc123?share=example",
    "https://www.rednote.com/discovery/item/6ad206ee000000001400240a?share=example",
])
def test_remote_ingest_keeps_original_host_share_query_and_rights_flag(url):
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


def test_preflight_requires_rednote_domain_adapter_before_installation():
    setup = source("setup")
    assignments = [node for node in ast.parse(setup).body if isinstance(node, ast.Assign)]
    assignment = next(node for node in assignments if any(
        isinstance(target, ast.Name) and target.id == "required_markers" for target in node.targets))
    markers = ast.literal_eval(assignment.value)
    assert "class RedNoteIE" in markers["dubbingstory/ingest/rednote.py"]
    assert 'return "RedNote"' in markers["dubbingstory/ingest/sources.py"]
    assert "add_info_extractor(RedNoteIE())" in markers["dubbingstory/ingest/youtube.py"]
    assert setup.index("required_markers") < setup.index("if INSTALL_DEPENDENCIES:")


def test_no_original_cleanup_or_youtube_downloader_is_copied():
    code = "\n".join("".join(cell["source"]) for cell in read_notebook()["cells"] if cell["cell_type"] == "code")
    for forbidden in ("rm -rf", "shutil.rmtree", "download_subtitles", "ingest.youtube", "cookiesfrombrowser", "load_dotenv"):
        assert forbidden not in code
    assert "normalize_source_url" in code
    assert 'source.provider not in {"tiktok", "xiaohongshu"}' in code
