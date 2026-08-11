"""
dubbingstory.vision.openai_vision — OpenAI-compatible vision analyzer

Connects to any OpenAI-compatible vision endpoint:
- vLLM serving Qwen3-VL locally (default)
- HuggingFace Inference API
- Ollama, LM Studio, etc.

Designed for use with locally-downloaded HuggingFace models
served via vLLM on Colab/Kaggle.

Model discovery:
    Browse vision models at https://huggingface.co/models?pipeline_tag=image-text-to-text
    Recommended: Qwen/Qwen3-VL-2B-Instruct (fits T4 16GB)
    Alternatives: Qwen/Qwen3-VL-4B-Instruct, Qwen/Qwen3-VL-8B-Instruct
"""

import base64
import json
import os
import re
import time
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # Will raise clear error when used


# ── Retry Config (same as gemini_analyzer.py) ────────────────────────────────
MAX_ATTEMPTS = 6
INITIAL_WAIT_SECONDS = 5
WAIT_INCREMENT_SECONDS = 5
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is retryable."""
    if isinstance(exc, json.JSONDecodeError):
        return True
    # Check for HTTP status code in exception
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and value in RETRYABLE_STATUS_CODES:
            return True
    msg = str(exc).lower()
    keywords = (
        "timeout", "temporarily unavailable", "deadline",
        "connection reset", "connection aborted", "service unavailable",
        "connection refused", "connect timeout",
    )
    return any(k in msg for k in keywords)


def _extract_json_from_text(text: str) -> dict | list:
    """Extract JSON from text that might contain markdown code fences or extra text.

    Tries in order:
    1. Direct JSON parse
    2. Extract from ```json ... ``` code fence
    3. Find first { ... } or [ ... ] block
    """
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try code fence extraction
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding JSON object or array
    for pattern in [r"\{.*\}", r"\[.*\]"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    raise json.JSONDecodeError(
        f"Could not extract JSON from response: {text[:200]}...",
        text, 0,
    )


def _image_to_base64_url(image_path: str) -> str:
    """Convert an image file to a base64 data URL for OpenAI Vision API."""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    mime = mime_map.get(ext, "image/jpeg")

    return f"data:{mime};base64,{data}"


class OpenAIVisionAnalyzer:
    """
    Vision analyzer using OpenAI-compatible API.

    Works with any server that implements the OpenAI Chat Completions API
    with vision support, including:

    - vLLM serving Qwen3-VL (recommended for Colab/Kaggle)
    - HuggingFace Inference Endpoints
    - Ollama, LM Studio, etc.

    Parameters
    ----------
    api_key : str
        API key. For local vLLM, use any non-empty string (e.g. "local").
    base_url : str
        Base URL of the API server (e.g. "http://127.0.0.1:8000/v1").
    model : str
        Model name (e.g. "Qwen/Qwen3-VL-2B-Instruct").
    temperature : float
        Sampling temperature. Lower = more deterministic.
    max_tokens : int
        Maximum tokens in response.

    Model Discovery
    ---------------
    Browse HuggingFace for vision-language models:
        https://huggingface.co/models?pipeline_tag=image-text-to-text&sort=trending

    Popular choices:
        - Qwen/Qwen3-VL-2B-Instruct  (16GB VRAM, T4 friendly)
        - Qwen/Qwen3-VL-4B-Instruct  (needs quantization on T4)
        - Qwen/Qwen3-VL-8B-Instruct  (24GB+ VRAM)
        - meta-llama/Llama-4-Scout-17B-16E-Instruct
    """

    def __init__(
        self,
        api_key: str = "local",
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "Qwen/Qwen3-VL-2B-Instruct",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ):
        if OpenAI is None:
            raise ImportError(
                "openai package is required for OpenAI-compatible vision.\n"
                "Install with: pip install openai>=1.0.0\n"
                "Or: pip install dubbingstory[openai-vision]"
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _build_image_content(self, image_paths: list[str]) -> list[dict]:
        """Build OpenAI Vision API content parts from image file paths."""
        parts = []
        for path in image_paths:
            if not os.path.exists(path):
                continue
            url = _image_to_base64_url(path)
            parts.append({
                "type": "image_url",
                "image_url": {"url": url},
            })
        return parts

    def _call_with_retry(
        self,
        messages: list[dict],
    ) -> dict | list:
        """
        Call the OpenAI-compatible API with retry logic.

        Returns parsed JSON from the response.
        """
        last_exc = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                print(f"      [Vision] Attempt {attempt}/{MAX_ATTEMPTS} ({self.model})...")

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                text = response.choices[0].message.content
                if not text or not text.strip():
                    raise ValueError("Vision model returned empty response.")

                return _extract_json_from_text(text)

            except Exception as exc:
                last_exc = exc
                print(
                    f"      [Vision] Attempt {attempt} failed | "
                    f"error={str(exc)[:120]}"
                )

                if (not _is_retryable(exc)) or attempt == MAX_ATTEMPTS:
                    break

                wait = INITIAL_WAIT_SECONDS + ((attempt - 1) * WAIT_INCREMENT_SECONDS)
                print(f"      [Vision] Retrying in {wait}s...")
                time.sleep(wait)

        raise RuntimeError(
            f"Vision analysis failed after {MAX_ATTEMPTS} attempts. "
            f"Last error: {last_exc}"
        ) from last_exc

    def analyze_scene_from_frames(
        self,
        keyframe_paths: list[str],
        prompt: str,
    ) -> dict:
        """
        Analyze a scene using its keyframe images.

        Parameters
        ----------
        keyframe_paths : list[str]
            Paths to keyframe images.
        prompt : str
            Analysis prompt.

        Returns
        -------
        dict
            Structured scene analysis (JSON).
        """
        # Build multimodal content: images + text
        content = self._build_image_content(keyframe_paths)
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        return self._call_with_retry(messages)

    def analyze_temporal_flow(
        self,
        scene_analyses: list[dict],
        prompt: str,
    ) -> dict:
        """
        Analyze the temporal flow across all scenes.

        This is a text-only call (no images) — uses the same model
        to understand relationships between scenes.

        Parameters
        ----------
        scene_analyses : list[dict]
            Per-scene analysis results.
        prompt : str
            Temporal flow prompt.

        Returns
        -------
        dict
            Enriched scene data with narrative context.
        """
        messages = [{"role": "user", "content": prompt}]

        return self._call_with_retry(messages)

    def health_check(self) -> bool:
        """Check if the vision server is reachable and serving the expected model."""
        try:
            models = self.client.models.list()
            model_ids = [m.id for m in models.data]
            if self.model in model_ids:
                print(f"      [Vision] ✅ Server ready — model '{self.model}' available")
                return True
            else:
                print(
                    f"      [Vision] ⚠️ Server reachable but model '{self.model}' "
                    f"not found. Available: {model_ids}"
                )
                return True  # Server is up, model name might just differ
        except Exception as exc:
            print(f"      [Vision] ❌ Server not reachable: {exc}")
            return False
