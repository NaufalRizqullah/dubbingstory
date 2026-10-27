"""Notebook contract checks for Kaggle; never run installation or media cells."""
import ast
import json
from pathlib import Path

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
