"""OpenAI-compatible vision analyzer for local Qwen3-VL/vLLM and peers.

The important distinction in this version is that ``finish_reason=length`` is
not treated as one generic failure anymore.  It is classified as either:

* clean output truncation -> retry same visual evidence with a larger budget;
* repetition/runaway -> one representative frame + compact rescue schema;
* context-budget pressure -> reduce visual input, not the output budget.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import time
from collections import Counter
from io import BytesIO
from typing import Any

from dubbingstory.vision.schemas import (
    CHEAP_SCENE_SCHEMA,
    DEEP_SCENE_RESCUE_SCHEMA,
    DEEP_SCENE_SCHEMA,
    VISION_SCHEMA_VERSION,
    build_temporal_flow_schema,
    normalize_scene_analysis,
)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from jsonschema import Draft202012Validator
except ImportError:
    Draft202012Validator = None


MAX_ATTEMPTS = 6
INITIAL_WAIT_SECONDS = 5
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class VisionGenerationError(RuntimeError):
    """Base error for a generation policy failure."""


class VisionOutputTruncatedError(VisionGenerationError):
    """The model hit the output cap without strong repetition evidence."""


class VisionRunawayError(VisionGenerationError):
    """The model hit the output cap and the response is strongly repetitive."""


class VisionContextBudgetError(VisionGenerationError):
    """The estimated/actual prompt leaves too little room for useful output."""


class VisionSchemaValidationError(VisionGenerationError):
    """Parsed JSON did not satisfy the client-side schema."""


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (json.JSONDecodeError, VisionSchemaValidationError)):
        return True
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and value in RETRYABLE_STATUS_CODES:
            return True
    msg = str(exc).lower()
    keywords = (
        "timeout",
        "temporarily unavailable",
        "deadline",
        "connection reset",
        "connection aborted",
        "service unavailable",
        "connection refused",
        "connect timeout",
    )
    return any(k in msg for k in keywords)


def _looks_like_context_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        term in msg
        for term in (
            "context length",
            "maximum context",
            "max context",
            "prompt too long",
            "prompt too large",
            "too many tokens",
            "context window",
        )
    )


def _extract_json_from_text(text: str) -> dict | list:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

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
        raise json.JSONDecodeError("Could not extract JSON from response.", text_clean, start)


def _validate_response_schema(obj: Any, schema: dict | None) -> None:
    """Client-side validation catches weak/buggy structured-output servers."""
    if schema is None or Draft202012Validator is None:
        return
    errors = sorted(Draft202012Validator(schema).iter_errors(obj), key=lambda err: list(err.path))
    if not errors:
        return
    first = errors[0]
    path = ".".join(str(p) for p in first.path) or "<root>"
    raise VisionSchemaValidationError(f"schema validation failed at {path}: {first.message}")


def _image_to_base64_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")
    return f"data:{mime};base64,{data}"


try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None


def _estimate_tokens(text: str, model: str | None = None) -> int:
    if not text:
        return 0
    if tiktoken is not None:
        try:
            if model:
                try:
                    enc = tiktoken.encoding_for_model(model)
                except Exception:
                    enc = tiktoken.get_encoding("cl100k_base")
            else:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def _qwen3vl_visual_tokens(width: int, height: int) -> int:
    """Approximate Qwen3-VL image tokens from its public processor geometry.

    Qwen3-VL uses patch_size=16 and merge_size=2, therefore one visual token
    corresponds roughly to a 32x32 pixel cell after smart resize.  The public
    processor also uses an image-area floor of 65,536 pixels.  A small overhead
    is added for vision boundary/special tokens.
    """
    if width <= 0 or height <= 0:
        return 0
    factor = 16 * 2
    min_pixels = 65536
    max_pixels = 16777216
    area = float(width * height)
    scale = 1.0
    if area < min_pixels:
        scale = math.sqrt(min_pixels / area)
    elif area > max_pixels:
        scale = math.sqrt(max_pixels / area)

    scaled_w = max(factor, int(round((width * scale) / factor)) * factor)
    scaled_h = max(factor, int(round((height * scale) / factor)) * factor)
    return max(1, (scaled_w // factor) * (scaled_h // factor)) + 8


def _image_dimensions_from_url(url: str) -> tuple[int, int] | None:
    if Image is None or not url:
        return None
    try:
        if url.startswith("file://"):
            with Image.open(url[7:]) as image:
                return image.size
        if url.startswith("data:") and "," in url:
            payload = url.split(",", 1)[1]
            with Image.open(BytesIO(base64.b64decode(payload))) as image:
                return image.size
    except Exception:
        return None
    return None


def _strip_json_keys_for_repetition(text: str) -> str:
    # Repeated schema keys in an array are legitimate.  Remove them before
    # measuring n-gram repetition so temporal JSON is not falsely classified.
    return re.sub(r'"(?:[^"\\]|\\.)+"\s*:', " ", text or "")


def _repetition_score(text: str) -> float:
    """Return 0..1 evidence of generation-loop repetition.

    The score is based on duplicate 5-grams after JSON keys are removed, plus
    repeated long quoted values.  It intentionally ignores short outputs.
    """
    cleaned = _strip_json_keys_for_repetition(text).lower()
    words = re.findall(r"[\w'-]+", cleaned, flags=re.UNICODE)
    if len(words) < 25:
        return 0.0

    n = 5
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    duplicate_ratio = 1.0 - (len(counts) / max(1, len(grams)))
    max_occurrence = max(counts.values(), default=1)
    burst_score = min(1.0, max(0, max_occurrence - 1) / 5.0)

    quoted_values = [
        value.strip().casefold()
        for value in re.findall(r'"((?:[^"\\]|\\.)*)"', cleaned)
        if len(value.strip()) >= 16
    ]
    quote_counts = Counter(quoted_values)
    quote_repeat = max(quote_counts.values(), default=1)
    quote_score = min(1.0, max(0, quote_repeat - 1) / 4.0)
    return max(duplicate_ratio, burst_score, quote_score)


class OpenAIVisionAnalyzer:
    """Vision analyzer using an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        api_key: str = "local",
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "Qwen/Qwen3-VL-2B-Instruct",
        temperature: float = 0.7,
        max_tokens: int = 640,
        model_max_context: int | None = None,
        image_mode: str = "data",
        structured_outputs: bool = True,
        allow_schema_fallback: bool = False,
        repetition_penalty: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 20,
        presence_penalty: float = 1.5,
        image_token_cost: int = 512,
        truncation_retry_tokens: int = 896,
        runaway_rescue_tokens: int = 448,
        temporal_truncation_retry_tokens: int = 6144,
        runaway_threshold: float = 0.35,
    ):
        if OpenAI is None:
            raise ImportError(
                "openai package is required for OpenAI-compatible vision.\n"
                "Install with: pip install openai>=1.0.0"
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.model_max_context = int(model_max_context or 8192)
        self.image_mode = image_mode or "data"
        self.structured_outputs = bool(structured_outputs)
        self.allow_schema_fallback = bool(allow_schema_fallback)
        self.repetition_penalty = float(repetition_penalty or 1.0)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.presence_penalty = float(presence_penalty)
        self.image_token_cost = max(1, int(image_token_cost))
        self.truncation_retry_tokens = max(1, int(truncation_retry_tokens))
        self.runaway_rescue_tokens = max(1, int(runaway_rescue_tokens))
        self.temporal_truncation_retry_tokens = max(1, int(temporal_truncation_retry_tokens))
        self.runaway_threshold = max(0.0, min(1.0, float(runaway_threshold)))

    def cache_fingerprint(self) -> dict:
        return {
            "schema_version": VISION_SCHEMA_VERSION,
            "structured_outputs": self.structured_outputs,
            "allow_schema_fallback": self.allow_schema_fallback,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
            "max_tokens": self.max_tokens,
            "model_max_context": self.model_max_context,
            "image_mode": self.image_mode,
            "image_budget_estimator": "qwen3vl-32px-grid-v1",
            "image_token_cost_fallback": self.image_token_cost,
            "truncation_retry_tokens": self.truncation_retry_tokens,
            "runaway_rescue_tokens": self.runaway_rescue_tokens,
            "runaway_threshold": self.runaway_threshold,
        }

    @staticmethod
    def _response_format_for_schema(schema: dict, schema_name: str) -> dict:
        return {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        }

    def _build_image_content(self, image_paths: list[str]) -> list[dict]:
        parts: list[dict] = []
        for path in image_paths:
            if not os.path.exists(path):
                continue
            if self.image_mode == "file":
                url = f"file://{os.path.abspath(path)}"
            else:
                url = _image_to_base64_url(path)
            parts.append({"type": "image_url", "image_url": {"url": url}})
        return parts

    def _estimate_image_tokens(self, url: str) -> int:
        dimensions = _image_dimensions_from_url(url)
        if dimensions and "qwen3-vl" in self.model.casefold():
            return _qwen3vl_visual_tokens(*dimensions)
        return self.image_token_cost

    def _budget_messages(self, messages: list[dict]) -> tuple[int, int, int]:
        text_parts: list[str] = []
        image_tokens = 0
        n_images = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                text_parts.append(content)
                continue
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    text_parts.append(str(part.get("text", "") or ""))
                elif part.get("type") == "image_url":
                    image = part.get("image_url") or {}
                    url = image.get("url", "") if isinstance(image, dict) else ""
                    n_images += 1
                    image_tokens += self._estimate_image_tokens(url)
        text_tokens = _estimate_tokens("\n".join(text_parts), model=self.model)
        return text_tokens, image_tokens, n_images

    def _call_with_retry(
        self,
        messages: list[dict],
        max_tokens_override: int | None = None,
        response_schema: dict | None = None,
        schema_name: str = "vision_response",
        trace_id: str = "request",
        min_output_tokens: int = 0,
        temperature_override: float | None = None,
        presence_penalty_override: float | None = None,
        repetition_penalty_override: float | None = None,
        runaway_threshold_override: float | None = None,
    ) -> dict | list:
        """Call API; network/format retry is separate from length recovery."""
        last_exc: Exception | None = None
        format_retries_left = 1
        network_retries_left = 3
        schema_enabled = bool(response_schema is not None and self.structured_outputs)
        prefix = f"[Vision {trace_id}]"

        while format_retries_left >= 0 and network_retries_left >= 0:
            try:
                text_tokens, image_tokens, n_images = self._budget_messages(messages)
                input_tokens = text_tokens + image_tokens
                safety_margin = 64
                allowed_output = self.model_max_context - input_tokens - safety_margin
                effective_max = int(max_tokens_override if max_tokens_override is not None else self.max_tokens)

                if allowed_output <= 0 or (min_output_tokens and allowed_output < int(min_output_tokens)):
                    raise VisionContextBudgetError(
                        f"prompt leaves only {allowed_output} output tokens in context={self.model_max_context}; "
                        f"estimated input={input_tokens} (text={text_tokens}, images={image_tokens})"
                    )

                call_max_tokens = min(effective_max, int(allowed_output))
                print(
                    f"      {prefix} token_budget text={text_tokens} images={n_images}/{image_tokens} "
                    f"total_input={input_tokens} model_max={self.model_max_context} "
                    f"allowed_output={allowed_output} using_max_tokens={call_max_tokens} "
                    f"schema={'on' if schema_enabled else 'json_object'}"
                )

                response_format = (
                    self._response_format_for_schema(response_schema, schema_name)
                    if schema_enabled and response_schema is not None
                    else {"type": "json_object"}
                )
                rep_penalty = (
                    self.repetition_penalty
                    if repetition_penalty_override is None
                    else float(repetition_penalty_override)
                )
                extra_body: dict[str, Any] = {"request_id": trace_id}
                if self.top_k > 0:
                    extra_body["top_k"] = self.top_k
                if rep_penalty != 1.0:
                    extra_body["repetition_penalty"] = rep_penalty

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=(
                        self.temperature if temperature_override is None else float(temperature_override)
                    ),
                    top_p=self.top_p,
                    presence_penalty=(
                        self.presence_penalty
                        if presence_penalty_override is None
                        else float(presence_penalty_override)
                    ),
                    max_tokens=call_max_tokens,
                    response_format=response_format,
                    extra_body=extra_body,
                )

                choice = response.choices[0]
                text = choice.message.content or ""
                finish_reason = choice.finish_reason
                usage = getattr(response, "usage", None)
                completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
                if usage is not None:
                    print(
                        f"      {prefix} usage prompt={getattr(usage, 'prompt_tokens', '?')} "
                        f"completion={completion_tokens if completion_tokens is not None else '?'} "
                        f"total={getattr(usage, 'total_tokens', '?')}"
                    )

                completion_ratio = (
                    float(completion_tokens) / float(call_max_tokens)
                    if isinstance(completion_tokens, int) and call_max_tokens > 0
                    else None
                )
                ratio_text = f" ratio={completion_ratio:.2f}" if completion_ratio is not None else ""
                print(f"      {prefix} finish_reason={finish_reason} chars={len(text)}{ratio_text}")

                parsed: dict | list | None = None
                parse_error: json.JSONDecodeError | None = None
                try:
                    parsed = _extract_json_from_text(text)
                except json.JSONDecodeError as exc:
                    parse_error = exc

                if finish_reason == "length":
                    repetition = _repetition_score(text)
                    schema_valid = False
                    if parsed is not None:
                        try:
                            _validate_response_schema(parsed, response_schema if schema_enabled else None)
                            schema_valid = True
                        except VisionSchemaValidationError:
                            schema_valid = False
                    print(
                        f"      {prefix} length_diagnosis repetition_score={repetition:.3f} "
                        f"schema_valid={schema_valid} max_tokens={call_max_tokens}"
                    )
                    print(f"      {prefix} LENGTH RAW HEAD:", repr(text[:600]))
                    print(f"      {prefix} LENGTH RAW TAIL:", repr(text[-1200:]))
                    length_runaway_threshold = (
                        self.runaway_threshold
                        if runaway_threshold_override is None
                        else float(runaway_threshold_override)
                    )
                    if repetition >= length_runaway_threshold:
                        raise VisionRunawayError(
                            f"finish_reason=length runaway repetition_score={repetition:.3f}"
                        )
                    raise VisionOutputTruncatedError(
                        f"finish_reason=length clean_truncation repetition_score={repetition:.3f}"
                    )

                if parse_error is not None:
                    print(f"      {prefix} RAW HEAD:", repr(text[:400]))
                    print(f"      {prefix} RAW TAIL:", repr(text[-800:]))
                    raise parse_error

                assert parsed is not None
                _validate_response_schema(parsed, response_schema if schema_enabled else None)
                if completion_ratio is not None and completion_ratio >= 0.90:
                    print(
                        f"      {prefix} ⚠️ near_output_cap=true completion_ratio={completion_ratio:.2f}"
                    )
                return parsed

            except (VisionOutputTruncatedError, VisionRunawayError, VisionContextBudgetError):
                raise
            except VisionSchemaValidationError as exc:
                last_exc = exc
                format_retries_left -= 1
                if format_retries_left < 0:
                    break
                print(f"      {prefix} schema error; retrying same request ({format_retries_left} left): {exc}")
            except json.JSONDecodeError as exc:
                last_exc = exc
                format_retries_left -= 1
                if format_retries_left < 0:
                    break
                print(f"      {prefix} format error, retrying ({format_retries_left} left)")
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()

                if _looks_like_context_error(exc):
                    raise VisionContextBudgetError(str(exc)) from exc

                if (
                    schema_enabled
                    and self.allow_schema_fallback
                    and getattr(exc, "status_code", None) == 400
                    and any(term in msg for term in ("json_schema", "response_format", "structured"))
                ):
                    schema_enabled = False
                    print(f"      {prefix} server rejected json_schema; falling back to json_object")
                    continue

                if getattr(exc, "status_code", None) == 400:
                    print(f"      {prefix} HTTP 400, not retrying: {exc}")
                    break
                if not _is_retryable(exc):
                    break

                network_retries_left -= 1
                if network_retries_left < 0:
                    break
                print(
                    f"      {prefix} network error {str(exc)[:120]} | "
                    f"retrying in {INITIAL_WAIT_SECONDS}s"
                )
                time.sleep(INITIAL_WAIT_SECONDS)

        raise RuntimeError(f"Vision analysis failed. Last error: {last_exc}") from last_exc

    def _messages_for_scene(self, keyframe_paths: list[str], prompt: str) -> list[dict]:
        content = self._build_image_content(keyframe_paths)
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    @staticmethod
    def _representative_frame(keyframe_paths: list[str]) -> list[str]:
        if not keyframe_paths:
            return []
        return [keyframe_paths[len(keyframe_paths) // 2]]

    def _runaway_rescue(
        self,
        keyframe_paths: list[str],
        prompt: str,
        *,
        analysis_kind: str,
        trace_id: str,
    ) -> dict:
        representative = self._representative_frame(keyframe_paths)
        rescue_schema = CHEAP_SCENE_SCHEMA if analysis_kind == "cheap" else DEEP_SCENE_RESCUE_SCHEMA
        rescue_prompt = (
            prompt
            + "\n\nRUNAWAY RESCUE MODE: Return the JSON object immediately and be terse. "
            "Do not repeat words, phrases, keys, objects, OCR strings, or explanations. "
            "Use only the strongest visible evidence from this representative frame."
        )
        print(
            f"      [Vision {trace_id}] runaway rescue: {len(keyframe_paths)} -> "
            f"{len(representative)} image, compact schema, max_tokens={self.runaway_rescue_tokens}"
        )
        result = self._call_with_retry(
            self._messages_for_scene(representative, rescue_prompt),
            max_tokens_override=self.runaway_rescue_tokens,
            min_output_tokens=min(256, self.runaway_rescue_tokens),
            response_schema=rescue_schema,
            schema_name=(
                "cheap_scene_analysis_rescue"
                if analysis_kind == "cheap"
                else "deep_scene_analysis_rescue"
            ),
            trace_id=f"{trace_id}-runaway-rescue",
            presence_penalty_override=max(self.presence_penalty, 1.5),
            repetition_penalty_override=max(self.repetition_penalty, 1.05),
        )
        return normalize_scene_analysis(result, kind=analysis_kind)

    def analyze_scene_from_frames(
        self,
        keyframe_paths: list[str],
        prompt: str,
        analysis_kind: str = "deep",
        trace_id: str = "scene",
    ) -> dict:
        """Analyze one scene with error-specific recovery policies."""
        scene_max_tokens = min(int(self.max_tokens), 640)
        schema = CHEAP_SCENE_SCHEMA if analysis_kind == "cheap" else DEEP_SCENE_SCHEMA
        schema_name = "cheap_scene_analysis" if analysis_kind == "cheap" else "deep_scene_analysis"
        messages = self._messages_for_scene(keyframe_paths, prompt)

        try:
            result = self._call_with_retry(
                messages,
                max_tokens_override=scene_max_tokens,
                min_output_tokens=min(384, scene_max_tokens),
                response_schema=schema,
                schema_name=schema_name,
                trace_id=trace_id,
            )
            return normalize_scene_analysis(result, kind=analysis_kind)

        except VisionRunawayError:
            return self._runaway_rescue(
                keyframe_paths, prompt, analysis_kind=analysis_kind, trace_id=trace_id
            )

        except VisionContextBudgetError:
            representative = self._representative_frame(keyframe_paths)
            if len(representative) == len(keyframe_paths):
                raise
            print(
                f"      [Vision {trace_id}] context rescue: {len(keyframe_paths)} -> "
                f"{len(representative)} image; preserving {scene_max_tokens}-token output budget"
            )
            result = self._call_with_retry(
                self._messages_for_scene(representative, prompt),
                max_tokens_override=scene_max_tokens,
                min_output_tokens=min(384, scene_max_tokens),
                response_schema=schema,
                schema_name=schema_name,
                trace_id=f"{trace_id}-context-rescue",
            )
            return normalize_scene_analysis(result, kind=analysis_kind)

        except VisionOutputTruncatedError:
            retry_tokens = max(scene_max_tokens, self.truncation_retry_tokens)
            print(
                f"      [Vision {trace_id}] clean truncation: retrying SAME frames "
                f"with max_tokens={retry_tokens} (was {scene_max_tokens})"
            )
            try:
                result = self._call_with_retry(
                    messages,
                    max_tokens_override=retry_tokens,
                    min_output_tokens=min(512, retry_tokens),
                    response_schema=schema,
                    schema_name=schema_name,
                    trace_id=f"{trace_id}-truncation-retry",
                )
                return normalize_scene_analysis(result, kind=analysis_kind)
            except VisionRunawayError:
                return self._runaway_rescue(
                    keyframe_paths, prompt, analysis_kind=analysis_kind, trace_id=trace_id
                )
            except VisionOutputTruncatedError:
                # A second clean truncation means the requested response is still
                # too verbose.  Shrink the *schema contract*, not just max_tokens.
                representative = self._representative_frame(keyframe_paths)
                compact_schema = (
                    CHEAP_SCENE_SCHEMA if analysis_kind == "cheap" else DEEP_SCENE_RESCUE_SCHEMA
                )
                compact_prompt = (
                    prompt
                    + "\n\nCOMPACT TRUNCATION RECOVERY: Preserve only the strongest evidence. "
                    "Use short phrases and finish every required JSON field."
                )
                print(
                    f"      [Vision {trace_id}] second clean truncation: compact schema + "
                    f"representative frame, max_tokens={scene_max_tokens}"
                )
                result = self._call_with_retry(
                    self._messages_for_scene(representative, compact_prompt),
                    max_tokens_override=scene_max_tokens,
                    min_output_tokens=min(384, scene_max_tokens),
                    response_schema=compact_schema,
                    schema_name=f"{schema_name}_compact",
                    trace_id=f"{trace_id}-truncation-compact",
                )
                return normalize_scene_analysis(result, kind=analysis_kind)

    def analyze_temporal_flow(
        self,
        scene_analyses: list[dict],
        prompt: str,
        trace_id: str = "temporal",
    ) -> dict:
        """Analyze temporal flow; clean truncation gets budget, runaway gets split upstream."""
        messages = [{"role": "user", "content": prompt}]
        scene_ids = [str(item.get("scene_id", "")) for item in scene_analyses if item.get("scene_id")]
        schema = build_temporal_flow_schema(scene_ids)
        normal_tokens = min(self.model_max_context, 4096)
        try:
            result = self._call_with_retry(
                messages,
                max_tokens_override=normal_tokens,
                min_output_tokens=min(768, normal_tokens),
                response_schema=schema,
                schema_name="temporal_flow_analysis",
                trace_id=trace_id,
                # Temporal JSON repeats the same object keys by design; use a
                # more conservative runaway threshold to avoid false positives.
                runaway_threshold_override=max(self.runaway_threshold, 0.60),
            )
        except VisionOutputTruncatedError:
            retry_tokens = min(
                self.model_max_context - 64,
                max(normal_tokens, self.temporal_truncation_retry_tokens),
            )
            print(
                f"      [Vision {trace_id}] temporal clean truncation: "
                f"retrying with max_tokens={retry_tokens}"
            )
            result = self._call_with_retry(
                messages,
                max_tokens_override=retry_tokens,
                min_output_tokens=min(1024, retry_tokens),
                response_schema=schema,
                schema_name="temporal_flow_analysis",
                trace_id=f"{trace_id}-truncation-retry",
                runaway_threshold_override=max(self.runaway_threshold, 0.60),
            )
        if not isinstance(result, dict):
            raise ValueError("Temporal vision response must be a JSON object")
        return result

    def health_check(self) -> bool:
        try:
            models = self.client.models.list()
            model_ids = [m.id for m in models.data]
            if self.model in model_ids:
                print(f"      [Vision] ✅ Server ready — model '{self.model}' available")
                return True
            print(
                f"      [Vision] ⚠️ Server reachable but model '{self.model}' not found. "
                f"Available: {model_ids}"
            )
            return True
        except Exception as exc:
            print(f"      [Vision] ❌ Server not reachable: {exc}")
            return False
