"""
dubbingstory.config — Configuration loader

Merges YAML config files, .env variables, and CLI arguments into
a single SimpleNamespace config object.
"""

import os
import yaml
from types import SimpleNamespace
from pathlib import Path
from dotenv import load_dotenv


# ==============================================================================
# DEFAULTS
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG = CONFIGS_DIR / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    """Flatten nested dict for SimpleNamespace access.

    Example: {"vision": {"provider": "gemini"}} → {"vision_provider": "gemini"}
    """
    items: list[tuple[str, object]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_yaml(path: str | Path) -> dict:
    """Load a YAML config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_narration_styles() -> dict:
    """Load narration styles config."""
    styles_path = CONFIGS_DIR / "narration_styles.yaml"
    if styles_path.exists():
        data = load_yaml(styles_path)
        return data.get("styles", {})
    return {}


def load_tts_voices() -> dict:
    """Load TTS voice config."""
    voices_path = CONFIGS_DIR / "tts_voices.yaml"
    if voices_path.exists():
        return load_yaml(voices_path)
    return {}


def build_config(
    cli_args: list[str] | None = None,
    config_path: str | Path | None = None,
    overrides: dict | None = None,
) -> SimpleNamespace:
    """
    Build the unified config object.

    Priority (highest to lowest):
    1. CLI arguments / overrides dict
    2. Custom config YAML (if provided)
    3. .env file
    4. default.yaml

    Parameters
    ----------
    cli_args : list[str] | None
        Raw CLI arguments (for future argparse integration).
    config_path : str | Path | None
        Path to a custom YAML config to merge on top of defaults.
    overrides : dict | None
        Direct dict overrides (highest priority).

    Returns
    -------
    SimpleNamespace
        Flat config namespace with all settings.
    """
    # Load .env
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    # Load default config
    config = load_yaml(DEFAULT_CONFIG) if DEFAULT_CONFIG.exists() else {}

    # Merge custom config
    if config_path:
        custom = load_yaml(config_path)
        config = _deep_merge(config, custom)

    # Merge overrides
    if overrides:
        config = _deep_merge(config, overrides)

    # Flatten for easy attribute access
    flat = _flatten_dict(config)

    # Inject environment variables
    flat["api_key_gemini"] = os.getenv("GOOGLE_API_KEY", "")
    flat["api_key_nvidia"] = os.getenv("NVIDIA_API_KEY", "")

    # OpenAI-compatible vision (Qwen3-VL via vLLM, HuggingFace, etc.)
    flat["api_key_openai_vision"] = os.getenv("OPENAI_VISION_API_KEY", "")
    flat["api_key_hf"] = os.getenv("HF_API_KEY", "")

    # Vision-related env overrides
    env_vision_provider = os.getenv("VISION_PROVIDER", "")
    if env_vision_provider:
        flat["vision_provider"] = env_vision_provider
    env_vision_base_url = os.getenv("OPENAI_VISION_BASE_URL", "")
    if env_vision_base_url:
        flat["vision_openai_base_url"] = env_vision_base_url

    # image_mode: "data" (base64, default) or "file" (file:// path, efficient
    # for local vLLM but requires --allowed-local-media-path flag)
    env_image_mode = os.getenv("OPENAI_VISION_IMAGE_MODE", "")
    if env_image_mode:
        flat["vision_openai_image_mode"] = env_image_mode

    env_vision_max_tokens = os.getenv("OPENAI_VISION_MAX_TOKENS", "")
    if env_vision_max_tokens:
        try:
            flat["vision_openai_max_tokens"] = int(env_vision_max_tokens)
        except Exception:
            flat["vision_openai_max_tokens"] = env_vision_max_tokens

    env_model_max_context = os.getenv("OPENAI_VISION_MODEL_MAX_CONTEXT", "")
    if env_model_max_context:
        try:
            flat["vision_openai_model_max_context"] = int(env_model_max_context)
        except Exception:
            flat["vision_openai_model_max_context"] = env_model_max_context

    # Inject project root
    flat["project_root"] = str(PROJECT_ROOT)
    flat["configs_dir"] = str(CONFIGS_DIR)

    # Load sub-configs as nested objects
    flat["narration_styles"] = load_narration_styles()
    flat["tts_voice_config"] = load_tts_voices()

    # Summary mode defaults (can be overridden by YAML or CLI)
    flat.setdefault("summary_target_duration", None)  # None = auto (~10-15% of original)
    flat.setdefault("summary_max_scenes", None)        # None = auto
    flat.setdefault("summary_min_scene_score", 0.3)
    flat.setdefault("temporal_chunk_size", 30)

    return SimpleNamespace(**flat)

