"""
Tests for OpenAI-compatible vision analyzer.

Tests the analyzer module without requiring a real API server,
using mocked responses.
"""

import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ── Test JSON extraction ────────────────────────────────────────────────────

class TestExtractJsonFromText:
    """Test the _extract_json_from_text helper."""

    def test_direct_json_object(self):
        from dubbingstory.vision.openai_vision import _extract_json_from_text
        text = '{"scene_id": "scene_001", "action": "mechanic opens engine"}'
        result = _extract_json_from_text(text)
        assert result["scene_id"] == "scene_001"

    def test_direct_json_array(self):
        from dubbingstory.vision.openai_vision import _extract_json_from_text
        text = '[{"scene_id": "scene_001"}, {"scene_id": "scene_002"}]'
        result = _extract_json_from_text(text)
        assert len(result) == 2

    def test_json_in_code_fence(self):
        from dubbingstory.vision.openai_vision import _extract_json_from_text
        text = """Here is the analysis:
```json
{"scene_id": "scene_001", "confidence": 0.8}
```
That's my analysis."""
        result = _extract_json_from_text(text)
        assert result["confidence"] == 0.8

    def test_json_in_plain_code_fence(self):
        from dubbingstory.vision.openai_vision import _extract_json_from_text
        text = """```
{"scene_id": "scene_001"}
```"""
        result = _extract_json_from_text(text)
        assert result["scene_id"] == "scene_001"

    def test_json_with_surrounding_text(self):
        from dubbingstory.vision.openai_vision import _extract_json_from_text
        text = 'The result is: {"scene_id": "scene_001"} and that is all.'
        result = _extract_json_from_text(text)
        assert result["scene_id"] == "scene_001"

    def test_invalid_json_raises(self):
        from dubbingstory.vision.openai_vision import _extract_json_from_text
        with pytest.raises(json.JSONDecodeError):
            _extract_json_from_text("This is not JSON at all")


# ── Test image encoding ─────────────────────────────────────────────────────

class TestImageToBase64:
    """Test image file to base64 URL conversion."""

    def test_jpeg_encoding(self):
        from dubbingstory.vision.openai_vision import _image_to_base64_url
        # Create a tiny temporary JPEG-like file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 10)  # Minimal JPEG header
            tmp_path = f.name

        try:
            result = _image_to_base64_url(tmp_path)
            assert result.startswith("data:image/jpeg;base64,")
            assert len(result) > 30
        finally:
            os.unlink(tmp_path)

    def test_png_encoding(self):
        from dubbingstory.vision.openai_vision import _image_to_base64_url
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG" + b"\x00" * 10)
            tmp_path = f.name

        try:
            result = _image_to_base64_url(tmp_path)
            assert result.startswith("data:image/png;base64,")
        finally:
            os.unlink(tmp_path)


# ── Test retryable detection ────────────────────────────────────────────────

class TestIsRetryable:
    """Test the _is_retryable helper."""

    def test_json_decode_error_is_retryable(self):
        from dubbingstory.vision.openai_vision import _is_retryable
        exc = json.JSONDecodeError("test", "", 0)
        assert _is_retryable(exc) is True

    def test_timeout_error_is_retryable(self):
        from dubbingstory.vision.openai_vision import _is_retryable
        exc = Exception("Connection timeout occurred")
        assert _is_retryable(exc) is True

    def test_connection_refused_is_retryable(self):
        from dubbingstory.vision.openai_vision import _is_retryable
        exc = Exception("Connection refused")
        assert _is_retryable(exc) is True

    def test_auth_error_not_retryable(self):
        from dubbingstory.vision.openai_vision import _is_retryable
        exc = Exception("Authentication failed: invalid API key")
        assert _is_retryable(exc) is False


# ── Test analyzer constructor ───────────────────────────────────────────────

class TestOpenAIVisionAnalyzerInit:
    """Test analyzer initialization."""

    def test_missing_openai_package(self):
        """If openai is not installed, should raise ImportError."""
        from dubbingstory.vision import openai_vision
        original = openai_vision.OpenAI

        try:
            openai_vision.OpenAI = None
            with pytest.raises(ImportError, match="openai package"):
                openai_vision.OpenAIVisionAnalyzer()
        finally:
            openai_vision.OpenAI = original


# ── Test provider selection in scene_understanding ──────────────────────────

class TestCreateAnalyzer:
    """Test _create_analyzer provider factory."""

    def test_gemini_provider(self):
        from dubbingstory.vision.scene_understanding import _create_analyzer
        from dubbingstory.vision.gemini_analyzer import GeminiVideoAnalyzer

        cfg = SimpleNamespace(
            vision_provider="gemini",
            api_key_gemini="test-key",
            vision_gemini_model="gemini-2.5-flash",
            vision_gemini_fallback_model="gemini-2.0-flash",
        )
        analyzer = _create_analyzer(cfg)
        assert isinstance(analyzer, GeminiVideoAnalyzer)

    @patch("dubbingstory.vision.openai_vision.OpenAI")
    def test_openai_provider(self, mock_openai_cls):
        from dubbingstory.vision.scene_understanding import _create_analyzer
        from dubbingstory.vision.openai_vision import OpenAIVisionAnalyzer

        cfg = SimpleNamespace(
            vision_provider="openai",
            api_key_openai_vision="",
            api_key_hf="",
            vision_openai_base_url="http://127.0.0.1:8000/v1",
            vision_openai_model="Qwen/Qwen3-VL-2B-Instruct",
            vision_openai_temperature=0.2,
            vision_openai_max_tokens=2048,
        )
        analyzer = _create_analyzer(cfg)
        assert isinstance(analyzer, OpenAIVisionAnalyzer)
        # Should use "local" as default API key when none provided
        mock_openai_cls.assert_called_once_with(
            api_key="local",
            base_url="http://127.0.0.1:8000/v1",
        )

    def test_unknown_provider_raises(self):
        from dubbingstory.vision.scene_understanding import _create_analyzer

        cfg = SimpleNamespace(vision_provider="unknown_provider")
        with pytest.raises(ValueError, match="Unknown vision provider"):
            _create_analyzer(cfg)

    def test_gemini_without_key_raises(self):
        from dubbingstory.vision.scene_understanding import _create_analyzer

        cfg = SimpleNamespace(
            vision_provider="gemini",
            api_key_gemini="",
        )
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            _create_analyzer(cfg)


# ── Test config loading ─────────────────────────────────────────────────────

class TestConfigVisionProvider:
    """Test that config properly loads vision provider settings."""

    def test_default_provider_is_gemini(self):
        """Default config should use gemini as vision provider."""
        from dubbingstory.config import build_config

        cfg = build_config()
        provider = getattr(cfg, "vision_provider", "gemini")
        assert provider == "gemini"

    def test_openai_model_default(self):
        """OpenAI model default should be Qwen3-VL-2B."""
        from dubbingstory.config import build_config

        cfg = build_config()
        model = getattr(cfg, "vision_openai_model", "")
        assert "Qwen" in model or model == ""  # Default from YAML

    @patch.dict(os.environ, {"VISION_PROVIDER": "openai"})
    def test_env_override_provider(self):
        """VISION_PROVIDER env var should override config."""
        from dubbingstory.config import build_config

        cfg = build_config()
        assert cfg.vision_provider == "openai"

    @patch.dict(os.environ, {"OPENAI_VISION_BASE_URL": "http://custom:9000/v1"})
    def test_env_override_base_url(self):
        """OPENAI_VISION_BASE_URL env var should override config."""
        from dubbingstory.config import build_config

        cfg = build_config()
        assert cfg.vision_openai_base_url == "http://custom:9000/v1"
