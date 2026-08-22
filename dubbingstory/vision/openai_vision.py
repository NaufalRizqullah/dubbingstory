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

from dubbingstory.vision.schemas import (
    CHEAP_SCENE_SCHEMA,
    DEEP_SCENE_SCHEMA,
    VISION_SCHEMA_VERSION,
    build_temporal_flow_schema,
    normalize_scene_analysis,
)

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
        structured_outputs: bool = True,
        allow_schema_fallback: bool = False,
        repetition_penalty: float = 1.05,
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
        # vLLM can enforce JSON Schema during decoding. This is the primary
        # protection against array repetition/runaway output.
        self.structured_outputs = bool(structured_outputs)
        self.allow_schema_fallback = bool(allow_schema_fallback)
        self.repetition_penalty = float(repetition_penalty or 1.0)
        # Estimated token cost per image for Qwen3-VL vision models.
        # Qwen3-VL-2B typically uses ~1300-2600 tokens per image depending
        # on resolution. This is used for token budgeting to avoid context
        # overflow errors that the text-only estimation misses entirely.
        self.image_token_cost = 1300

    def cache_fingerprint(self) -> dict:
        """Settings that materially affect deterministic vision-cache output."""
        return {
            "schema_version": VISION_SCHEMA_VERSION,
            "structured_outputs": self.structured_outputs,
            "allow_schema_fallback": self.allow_schema_fallback,
            "repetition_penalty": self.repetition_penalty,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "model_max_context": self.model_max_context,
            "image_mode": self.image_mode,
        }

    @staticmethod
    def _response_format_for_schema(schema: dict, schema_name: str) -> dict:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        }

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
        response_schema: dict | None = None,
        schema_name: str = "vision_response",
        trace_id: str = "request",
    ) -> dict | list:
        """Call the OpenAI-compatible API with bounded structured output."""
        last_exc = None
        format_retries_left = 1
        network_retries_left = 3
        schema_enabled = bool(response_schema is not None and self.structured_outputs)
        prefix = f"[Vision {trace_id}]"

        while format_retries_left >= 0 and network_retries_left >= 0:
            try:
                text_parts = []
                n_images = 0
                for m in messages:
                    content = m.get("content")
                    if isinstance(content, str):
                        text_parts.append(content)
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict):
                                if part.get("type") == "text":
                                    text_parts.append(part.get("text", ""))
                                elif part.get("type") == "image_url":
                                    n_images += 1
                prompt_text = "\n".join(text_parts)

                text_tokens = _estimate_tokens(prompt_text, model=self.model)
                image_tokens = n_images * self.image_token_cost
                input_tokens = text_tokens + image_tokens
                safety_margin = 64
                allowed_output = self.model_max_context - input_tokens - safety_margin

                if allowed_output <= 0:
                    raise RuntimeError(
                        f"Prompt too large for model context ({self.model_max_context}): "
                        f"estimated input tokens={input_tokens} "
                        f"(text={text_tokens} + images={n_images}×{self.image_token_cost}={image_tokens}). "
                        f"Summarize or trim scene_analyses before calling temporal analysis."
                    )

                effective_max = max_tokens_override if max_tokens_override else self.max_tokens
                call_max_tokens = min(int(effective_max), int(allowed_output))
                print(
                    f"      {prefix} token_budget text={text_tokens} images={n_images}×{self.image_token_cost}={image_tokens} "
                    f"total_input={input_tokens} model_max={self.model_max_context} "
                    f"allowed_output={allowed_output} using_max_tokens={call_max_tokens} "
                    f"schema={'on' if schema_enabled else 'json_object'}"
                )

                response_format = (
                    self._response_format_for_schema(response_schema, schema_name)
                    if schema_enabled and response_schema is not None
                    else {"type": "json_object"}
                )
                extra_body = {"request_id": trace_id}
                if self.repetition_penalty != 1.0:
                    extra_body["repetition_penalty"] = self.repetition_penalty

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=call_max_tokens,
                    response_format=response_format,
                    extra_body=extra_body,
                )

                choice = response.choices[0]
                text = choice.message.content or ""
                finish_reason = choice.finish_reason
                usage = getattr(response, "usage", None)
                if usage is not None:
                    print(
                        f"      {prefix} usage "
                        f"prompt={getattr(usage, 'prompt_tokens', '?')} "
                        f"completion={getattr(usage, 'completion_tokens', '?')} "
                        f"total={getattr(usage, 'total_tokens', '?')}"
                    )
                print(f"      {prefix} finish_reason={finish_reason} chars={len(text)}")
                if finish_reason == "length":
                    print(f"      {prefix} LENGTH RAW HEAD:", repr(text[:600]))
                    print(f"      {prefix} LENGTH RAW TAIL:", repr(text[-1200:]))

                try:
                    obj = _extract_json_from_text(text)
                    if finish_reason == "length":
                        print(f"      {prefix} parsed valid JSON despite length stop (chars={len(text)})")
                    return obj
                except json.JSONDecodeError as exc:
                    if finish_reason == "length":
                        print(f"      {prefix} truncated JSON; triggering context reduction/rescue")
                        raise RuntimeError("finish_reason=length") from exc
                    print(f"      {prefix} RAW HEAD:", repr(text[:400]))
                    print(f"      {prefix} RAW TAIL:", repr(text[-800:]))
                    raise

            except json.JSONDecodeError as exc:
                last_exc = exc
                format_retries_left -= 1
                if format_retries_left < 0:
                    break
                print(f"      {prefix} format error, retrying ({format_retries_left} left)")
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()

                # Some third-party OpenAI-compatible servers do not implement
                # json_schema. vLLM does. Fallback is opt-in so the local target
                # keeps the hard decoder constraint instead of silently weakening it.
                if (
                    schema_enabled
                    and self.allow_schema_fallback
                    and getattr(exc, "status_code", None) == 400
                    and any(term in msg for term in ("json_schema", "response_format", "structured"))
                ):
                    schema_enabled = False
                    print(f"      {prefix} server rejected json_schema; falling back to json_object")
                    continue

                if str(exc) == "finish_reason=length":
                    print(f"      {prefix} output truncated (finish_reason=length), not retrying same payload")
                    break
                if getattr(exc, "status_code", None) == 400:
                    print(f"      {prefix} HTTP 400, not retrying: {exc}")
                    break
                if not _is_retryable(exc):
                    break

                network_retries_left -= 1
                if network_retries_left < 0:
                    break
                wait = INITIAL_WAIT_SECONDS
                print(f"      {prefix} network error {str(exc)[:120]} | retrying in {wait}s")
                time.sleep(wait)

        raise RuntimeError(f"Vision analysis failed. Last error: {last_exc}") from last_exc

    def analyze_scene_from_frames(
        self,
        keyframe_paths: list[str],
        prompt: str,
        analysis_kind: str = "deep",
        trace_id: str = "scene",
    ) -> dict:
        """Analyze one scene with strict bounded JSON when supported."""
        scene_max_tokens = min(int(self.max_tokens), 640)
        schema = CHEAP_SCENE_SCHEMA if analysis_kind == "cheap" else DEEP_SCENE_SCHEMA
        schema_name = "cheap_scene_analysis" if analysis_kind == "cheap" else "deep_scene_analysis"

        content = self._build_image_content(keyframe_paths)
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        try:
            result = self._call_with_retry(
                messages,
                max_tokens_override=scene_max_tokens,
                response_schema=schema,
                schema_name=schema_name,
                trace_id=trace_id,
            )
            return normalize_scene_analysis(result, kind=analysis_kind)
        except RuntimeError as exc:
            err_msg = str(exc).lower()
            is_token_issue = (
                "finish_reason=length" in err_msg
                or "prompt too large" in err_msg
                or "context length" in err_msg
            )
            if is_token_issue and len(keyframe_paths) > 1:
                middle = keyframe_paths[len(keyframe_paths) // 2]
                rescue_prompt = (
                    prompt
                    + "\n\nRESCUE MODE: Return the requested JSON object immediately. "
                    "visible_objects must contain at most 8 unique items; text_visible at most 4 unique literal strings. "
                    "If text is not clearly legible use an empty text_visible array. "
                    "Do not infer object names as visible text. Do not repeat keys, array items, sentences, or explanations."
                )
                print(
                    f"      [Vision {trace_id}] retrying one representative middle frame "
                    f"after token/runaway failure ({len(keyframe_paths)} -> 1 image)"
                )
                content = self._build_image_content([middle])
                content.append({"type": "text", "text": rescue_prompt})
                messages = [{"role": "user", "content": content}]
                result = self._call_with_retry(
                    messages,
                    max_tokens_override=384,
                    response_schema=schema,
                    schema_name=schema_name,
                    trace_id=f"{trace_id}-rescue",
                )
                return normalize_scene_analysis(result, kind=analysis_kind)
            raise

    def analyze_temporal_flow(
        self,
        scene_analyses: list[dict],
        prompt: str,
        trace_id: str = "temporal",
    ) -> dict:
        """Analyze temporal flow with a schema bounded to the current chunk."""
        messages = [{"role": "user", "content": prompt}]
        scene_ids = [str(item.get("scene_id", "")) for item in scene_analyses if item.get("scene_id")]
        schema = build_temporal_flow_schema(scene_ids)
        result = self._call_with_retry(
            messages,
            max_tokens_override=min(self.model_max_context, 4096),
            response_schema=schema,
            schema_name="temporal_flow_analysis",
            trace_id=trace_id,
        )
        if not isinstance(result, dict):
            raise ValueError("Temporal vision response must be a JSON object")
        return result

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
