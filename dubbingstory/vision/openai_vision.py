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
    """Extract JSON from text using robust fallback parser."""
    text = (text or "").strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try code fence extraction
    text_clean = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.I)
    text_clean = re.sub(r"\s*```\s*$", "", text_clean)

    starts = [i for i in (text_clean.find("{"), text_clean.find("[")) if i >= 0]
    if not starts:
        raise json.JSONDecodeError("No JSON start found", text, 0)

    start = min(starts)
    decoder = json.JSONDecoder()
    try:
        obj, _end = decoder.raw_decode(text_clean[start:])
        return obj
    except json.JSONDecodeError:
        raise json.JSONDecodeError(
            f"Could not extract JSON from response.",
            text_clean, start,
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


# Token estimation helper: use tiktoken if available, otherwise fall back
# to a conservative char-based heuristic (chars / 4).
try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None


def _estimate_tokens(text: str, model: str | None = None) -> int:
    """Estimate token count for a text. Prefer tiktoken when available.

    This is a best-effort estimate and used for budgeting only.
    """
    if not text:
        return 0
    if tiktoken is not None:
        try:
            # Choose an encoding based on model if possible; fall back to 'cl100k_base'
            enc_name = None
            if model:
                try:
                    enc = tiktoken.encoding_for_model(model)
                except Exception:
                    enc = tiktoken.get_encoding("cl100k_base")
            else:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            # Fallback to heuristic below
            pass
    # Heuristic: assume ~4 characters per token (conservative)
    return max(1, len(text) // 4)


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
        Maximum tokens in response (request default). This will be
        dynamically adjusted based on prompt size to avoid context errors.
    model_max_context : int | None
        (Optional) Known maximum context length of the model. If None, a
        conservative default of 8192 is used.
    """

    def __init__(
        self,
        api_key: str = "local",
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "Qwen/Qwen3-VL-2B-Instruct",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        model_max_context: int | None = None,
        image_mode: str = "data",
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
        # If not provided, default to 8192 (can be overridden via cfg)
        self.model_max_context = int(model_max_context or 8192)
        # image_mode: how images are sent to the vision API
        #   "data"  → base64 data URI (default, always works)
        #   "file"  → file:// path (efficient, but requires vLLM flag
        #             --allowed-local-media-path)
        self.image_mode = image_mode or "data"
        # Estimated token cost per image for Qwen3-VL vision models.
        # Qwen3-VL-2B typically uses ~1300-2600 tokens per image depending
        # on resolution. This is used for token budgeting to avoid context
        # overflow errors that the text-only estimation misses entirely.
        self.image_token_cost = 1300

    def _build_image_content(self, image_paths: list[str]) -> list[dict]:
        """Build OpenAI Vision API content parts from image file paths."""
        parts = []
        for path in image_paths:
            if not os.path.exists(path):
                continue
            if self.image_mode == "file":
                url = f"file://{os.path.abspath(path)}"
            else:
                url = _image_to_base64_url(path)
            parts.append({
                "type": "image_url",
                "image_url": {"url": url},
            })
        return parts

    def _call_with_retry(
        self,
        messages: list[dict],
        max_tokens_override: int | None = None,
    ) -> dict | list:
        """
        Call the OpenAI-compatible API with smart retry logic.
        """
        last_exc = None
        format_retries_left = 1
        network_retries_left = 3

        while format_retries_left >= 0 and network_retries_left >= 0:
            try:
                # Estimate prompt/input token usage to avoid exceeding model
                # context when combined with requested output.
                text_parts = []
                n_images = 0
                for m in messages:
                    content = m.get("content")
                    if isinstance(content, str):
                        text_parts.append(content)
                    elif isinstance(content, list):
                        for p in content:
                            if isinstance(p, dict):
                                if p.get("type") == "text":
                                    text_parts.append(p.get("text", ""))
                                elif p.get("type") == "image_url":
                                    n_images += 1
                prompt_text = "\n".join(text_parts)

                text_tokens = _estimate_tokens(prompt_text, model=self.model)
                image_tokens = n_images * self.image_token_cost
                input_tokens = text_tokens + image_tokens
                safety_margin = 64
                allowed_output = self.model_max_context - input_tokens - safety_margin

                if allowed_output <= 0:
                    # Prompt is already too large for the model — fail fast with guidance
                    raise RuntimeError(
                        f"Prompt too large for model context ({self.model_max_context}): "
                        f"estimated input tokens={input_tokens} "
                        f"(text={text_tokens} + images={n_images}×{self.image_token_cost}={image_tokens}). "
                        f"Summarize or trim scene_analyses before calling temporal analysis."
                    )

                # Allocate output tokens: use override if provided (e.g. for
                # temporal analysis which is text-only and needs large output),
                # otherwise cap at self.max_tokens (safe for vision calls).
                effective_max = max_tokens_override if max_tokens_override else self.max_tokens
                call_max_tokens = min(int(effective_max), int(allowed_output))

                print(
                    f"      [Vision] token_budget text={text_tokens} images={n_images}×{self.image_token_cost}={image_tokens} "
                    f"total_input={input_tokens} model_max={self.model_max_context} "
                    f"allowed_output={allowed_output} using_max_tokens={call_max_tokens}"
                )

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=call_max_tokens,
                    response_format={"type": "json_object"},
                )

                choice = response.choices[0]
                text = choice.message.content or ""
                finish_reason = choice.finish_reason

                print(f"      [Vision] finish_reason={finish_reason} chars={len(text)}")

                if finish_reason == "length":
                    raise RuntimeError("finish_reason=length")

                try:
                    return _extract_json_from_text(text)
                except json.JSONDecodeError as e:
                    print("      [Vision] RAW HEAD:", repr(text[:400]))
                    print("      [Vision] RAW TAIL:", repr(text[-800:]))
                    raise e

            except json.JSONDecodeError as exc:
                last_exc = exc
                format_retries_left -= 1
                if format_retries_left < 0:
                    break
                print(f"      [Vision] Format error, retrying... ({format_retries_left} left)")
            except Exception as exc:
                last_exc = exc
                if str(exc) == "finish_reason=length":
                    print("      [Vision] Output truncated (finish_reason=length), not retrying.")
                    break
                if getattr(exc, "status_code", None) == 400:
                    print(f"      [Vision] HTTP 400 Context Length error, not retrying: {exc}")
                    break
                
                if not _is_retryable(exc):
                    break
                
                network_retries_left -= 1
                if network_retries_left < 0:
                    break
                
                wait = INITIAL_WAIT_SECONDS
                print(f"      [Vision] Network error {str(exc)[:120]} | Retrying in {wait}s...")
                time.sleep(wait)

        raise RuntimeError(f"Vision analysis failed. Last error: {last_exc}") from last_exc

    def analyze_scene_from_frames(
        self,
        keyframe_paths: list[str],
        prompt: str,
    ) -> dict:
        """
        Analyze a scene using its keyframe images.

        If the first attempt fails due to token limits (finish_reason=length
        or context overflow), automatically retries with fewer images.

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

        try:
            return self._call_with_retry(messages)
        except RuntimeError as exc:
            err_msg = str(exc).lower()
            is_token_issue = (
                "finish_reason=length" in err_msg
                or "prompt too large" in err_msg
                or "context length" in err_msg
            )
            # Only retry with fewer images if we had more than 1
            if is_token_issue and len(keyframe_paths) > 1:
                reduced = keyframe_paths[:1]
                print(
                    f"      [Vision] Retrying with {len(reduced)} image(s) "
                    f"(reduced from {len(keyframe_paths)}) to fit context..."
                )
                content = self._build_image_content(reduced)
                content.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content}]
                return self._call_with_retry(messages)
            raise

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

        # Temporal analysis is text-only (no images) and needs a large output
        # to fit narrative data for all scenes in the chunk. Use the full
        # allowed output budget instead of the default max_tokens (which is
        # tuned for per-scene vision calls and usually only 2048).
        return self._call_with_retry(
            messages,
            max_tokens_override=self.model_max_context,
        )

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
