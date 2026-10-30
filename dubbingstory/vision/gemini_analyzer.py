"""
dubbingstory.vision.gemini_analyzer — Gemini API visual analysis

Uses Gemini's multimodal capabilities to analyze video scenes
from keyframe images or uploaded video segments.
Includes retry logic adapted from opensource-clipping.
"""

import json
import math
import os
import re
import time
import base64

from google import genai
from google.genai import types


# Total attempts include the original HTTP request. The SDK's own retries are
# disabled so these limits describe real requests, rather than nested loops.
MAX_ATTEMPTS = 3
INITIAL_WAIT_SECONDS = 5
WAIT_INCREMENT_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 30
CALL_BUDGET_SECONDS = 90
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class GeminiGenerationError(RuntimeError):
    """Safe failure metadata for provider routing, without provider response text."""

    def __init__(self, *, attempts, status_code=None, retryable=False,
                 service_failure=False, service_failures=0, service_streak_reset=False,
                 reason="request_failed"):
        self.attempts = attempts
        self.status_code = status_code
        self.retryable = retryable
        self.service_failure = service_failure
        # Consecutive service failures at the end of this call. A valid provider
        # response rejected by the JSON/language validator resets this streak.
        self.service_failures = service_failures
        self.consecutive_service_failures = service_failures
        self.service_streak_reset = service_streak_reset
        self.reason = reason
        status = f", HTTP {status_code}" if status_code is not None else ""
        super().__init__(f"Gemini generation failed after {attempts} attempt(s): {reason}{status}")


class _ResponseValidationError(ValueError):
    """A semantic response error that should trigger a fresh generation attempt."""


def _extract_status_code(exc: Exception):
    """Extract HTTP status code from various exception types."""
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    match = re.search(r"\b([45]\d{2})\b", str(exc))
    return int(match.group(1)) if match else None


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is retryable."""
    if isinstance(exc, (json.JSONDecodeError, _ResponseValidationError)):
        return True
    code = _extract_status_code(exc)
    if code is not None:
        return code in RETRYABLE_STATUS_CODES
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    msg = str(exc).lower()
    keywords = (
        "timeout", "temporarily unavailable", "deadline",
        "connection reset", "connection aborted", "service unavailable",
        "resource_exhausted",
    )
    return (any(k in msg for k in keywords)
            or type(exc).__name__.lower() in {
                "connecterror", "readerror", "writeerror", "remoteprotocolerror",
                "connecttimeout", "readtimeout", "writetimeout", "pooltimeout",
            })


def _safe_failure_reason(exc: Exception) -> str:
    if isinstance(exc, (json.JSONDecodeError, _ResponseValidationError)):
        return "response_validation"
    code = _extract_status_code(exc)
    if code in {401, 403}:
        return "authentication_or_access"
    if code == 429:
        return "quota_or_rate_limit"
    if code in RETRYABLE_STATUS_CODES:
        return "service_unavailable"
    if code is not None:
        return "request_rejected"
    if _is_retryable(exc):
        return "connection_or_timeout"
    return "request_failed"


def _seconds(value, name: str, *, allow_zero: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0 or (not allow_zero and number == 0):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'} and finite")
    return number


def _http_options(timeout_seconds: float):
    # Google Gen AI HttpOptions uses milliseconds. Request-level options also
    # cover callers/tests that supply a pre-existing client instead of __init__.
    return types.HttpOptions(
        timeout=max(1, int(timeout_seconds * 1000)),
        retry_options=types.HttpRetryOptions(attempts=1),
    )


def _retry_after_seconds(exc: Exception) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or getattr(exc, "headers", None)
    if headers is None:
        return 0.0
    try:
        value = float(headers.get("Retry-After", headers.get("retry-after", 0)))
        return value if math.isfinite(value) and value > 0 else 0.0
    except (TypeError, ValueError, AttributeError):
        return 0.0


class GeminiVideoAnalyzer:
    """
    Gemini API wrapper for visual scene analysis.

    Supports two modes:
    - Keyframes mode: Send extracted keyframe images
    - Video upload mode: Upload scene video via Files API
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        fallback_model: str = "gemini-2.0-flash",
        *,
        max_attempts: int | None = None,
        timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
        budget_seconds: float = CALL_BUDGET_SECONDS,
        initial_wait_seconds: float = INITIAL_WAIT_SECONDS,
        wait_increment_seconds: float = WAIT_INCREMENT_SECONDS,
    ):
        attempt_limit = MAX_ATTEMPTS if max_attempts is None else max_attempts
        if isinstance(attempt_limit, bool) or int(attempt_limit) != attempt_limit or attempt_limit < 1:
            raise ValueError("max_attempts must be a positive integer")
        self.max_attempts = int(attempt_limit)
        self.timeout_seconds = _seconds(timeout_seconds, "timeout_seconds")
        self.budget_seconds = _seconds(budget_seconds, "budget_seconds")
        self.initial_wait_seconds = _seconds(initial_wait_seconds, "initial_wait_seconds", allow_zero=True)
        self.wait_increment_seconds = _seconds(wait_increment_seconds, "wait_increment_seconds", allow_zero=True)
        self.client = genai.Client(api_key=api_key, http_options=_http_options(self.timeout_seconds))
        self.model = model
        self.fallback_model = fallback_model

    def _load_image_as_part(self, image_path: str) -> types.Part:
        """Load an image file as a Gemini API Part."""
        with open(image_path, "rb") as f:
            data = f.read()

        # Determine mime type
        ext = os.path.splitext(image_path)[1].lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        mime = mime_map.get(ext, "image/jpeg")

        return types.Part.from_bytes(data=data, mime_type=mime)

    def _generate_with_retry(
        self,
        contents: list,
        response_schema: str = "json",
        *,
        temperature: float = 0.2,
        response_validator=None,
        task_name: str = "analysis",
        model_override: str | None = None,
        max_output_tokens: int | None = None,
    ) -> dict | list:
        """Call Gemini with retry logic and optional semantic validation.

        ``response_validator`` receives the decoded JSON payload and returns either
        ``None`` (accepted) or a human-readable error string (retry). This is used
        by narration generation to reject a response written in the wrong language
        without losing the original prompt/language contract on the next attempt.
        """
        max_attempts = getattr(self, "max_attempts", MAX_ATTEMPTS)
        timeout_seconds = getattr(self, "timeout_seconds", REQUEST_TIMEOUT_SECONDS)
        budget_seconds = getattr(self, "budget_seconds", CALL_BUDGET_SECONDS)
        initial_wait = getattr(self, "initial_wait_seconds", INITIAL_WAIT_SECONDS)
        wait_increment = getattr(self, "wait_increment_seconds", WAIT_INCREMENT_SECONDS)
        deadline = time.monotonic() + budget_seconds
        attempts = 0
        status_code = None
        retryable = False
        service_failure = False
        service_failures = 0
        service_streak_reset = False
        reason = "request_failed"
        base_contents = list(contents)
        attempt_contents = list(base_contents)
        primary_model = model_override or self.model

        for attempt in range(1, max_attempts + 1):
            remaining = deadline - time.monotonic()
            if remaining < 0.001:
                reason = "budget_exhausted"
                break
            # A single-attempt policy must still call the requested primary.
            model = (self.fallback_model if attempt == max_attempts and attempt > 1
                     and self.fallback_model else primary_model)
            config = types.GenerateContentConfig(
                response_mime_type="application/json", temperature=temperature,
                http_options=_http_options(min(timeout_seconds, remaining)),
                **({"max_output_tokens": int(max_output_tokens)} if max_output_tokens is not None else {}),
            )

            try:
                attempts += 1
                print(
                    f"      [Gemini:{task_name}] Attempt {attempt}/{max_attempts} "
                    f"({model})..."
                )

                response = self.client.models.generate_content(
                    model=model,
                    contents=attempt_contents,
                    config=config,
                )

                text = getattr(response, "text", None)
                if not text or not text.strip():
                    raise _ResponseValidationError("Gemini returned empty response.")

                payload = json.loads(text)
                if not isinstance(payload, (dict, list)):
                    raise _ResponseValidationError("Response must be a JSON object or array.")
                if response_validator is not None:
                    validation_error = response_validator(payload)
                    if validation_error:
                        raise _ResponseValidationError(str(validation_error))
                if time.monotonic() >= deadline:
                    raise TimeoutError("Generation exceeded its call budget")
                return payload

            except Exception as exc:
                status_code = _extract_status_code(exc)
                retryable = _is_retryable(exc)
                service_failure = retryable and not isinstance(exc, (json.JSONDecodeError, _ResponseValidationError))
                service_failures = service_failures + 1 if service_failure else 0
                service_streak_reset = service_streak_reset or not service_failure
                reason = _safe_failure_reason(exc)

                print(
                    f"      [Gemini:{task_name}] Attempt {attempt} failed | "
                    f"status={status_code} | reason={reason}"
                )

                if (not retryable) or attempt == max_attempts:
                    break

                # For semantic validation failures, keep the ORIGINAL request and
                # add only a correction note. The original prompt contains the
                # selected language code/name, so retry can never silently lose it.
                if isinstance(exc, _ResponseValidationError):
                    retry_note = (
                        "RETRY VALIDATION FAILURE: The previous JSON response was rejected. "
                        f"Reason: {exc}. Re-read and obey every original requirement, "
                        "especially the required output language. Return a fresh complete JSON response."
                    )
                    if all(isinstance(item, str) for item in base_contents):
                        attempt_contents = list(base_contents) + [retry_note]
                    else:
                        attempt_contents = list(base_contents)

                wait = max(initial_wait + ((attempt - 1) * wait_increment), _retry_after_seconds(exc))
                if wait >= deadline - time.monotonic():
                    reason = "budget_exhausted"
                    break
                print(f"      [Gemini:{task_name}] Retrying in {wait}s...")
                time.sleep(wait)

        raise GeminiGenerationError(
            attempts=attempts, status_code=status_code, retryable=retryable,
            service_failure=service_failure, service_failures=service_failures,
            service_streak_reset=service_streak_reset,
            reason=reason,
        ) from None

    def generate_json_with_retry(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        response_validator=None,
        task_name: str = "json",
        model: str | None = None,
    ) -> dict | list:
        """Generate text-only JSON while preserving the exact prompt across retries."""
        return self._generate_with_retry(
            [prompt],
            temperature=temperature,
            response_validator=response_validator,
            task_name=task_name,
            model_override=model,
        )

    def analyze_timed_frames(self, frames: list[dict], prompt: str,
                             max_output_tokens: int = 1536) -> list[dict]:
        """Preserve every sampled frame and validate the same contract as local vision."""
        from dubbingstory.vision.schemas import normalize_timed_frames

        if not frames or any(not os.path.isfile(frame["path"]) for frame in frames):
            raise ValueError("Every temporal frame image must exist")
        ids = [frame["frame_id"] for frame in frames]
        parts = [self._load_image_as_part(frame["path"]) for frame in frames]
        parts.append(types.Part.from_text(text=prompt))

        def validate(payload):
            try:
                normalize_timed_frames(payload, ids)
            except ValueError as exc:
                return str(exc)
            return None

        try:
            payload = self._generate_with_retry(
                [types.Content(role="user", parts=parts)], response_validator=validate,
                task_name="timed-frames", max_output_tokens=max_output_tokens,
            )
        except GeminiGenerationError as exc:
            if exc.reason == "response_validation" and not exc.service_failure:
                # The dense caller can divide a malformed/truncated batch and
                # retain all frames. Do not multiply retries during an outage.
                raise ValueError("Gemini timed frame response remained incomplete") from None
            raise
        return normalize_timed_frames(payload, ids)

    def analyze_scene_from_frames(
        self,
        keyframe_paths: list[str],
        prompt: str,
    ) -> dict:
        """
        Analyze a scene using its keyframes.

        Parameters
        ----------
        keyframe_paths : list[str]
            Paths to keyframe images.
        prompt : str
            Analysis prompt (from prompts.py).

        Returns
        -------
        dict
            Structured scene analysis.
        """
        # Build content parts: images + text prompt
        parts = []
        for kf_path in keyframe_paths:
            if os.path.exists(kf_path):
                parts.append(self._load_image_as_part(kf_path))

        parts.append(types.Part.from_text(text=prompt))

        contents = [types.Content(role="user", parts=parts)]

        return self._generate_with_retry(contents)

    def analyze_scene_from_video(
        self,
        video_path: str,
        prompt: str,
    ) -> dict:
        """
        Analyze a scene by uploading the video segment.

        Uses Gemini Files API for better temporal understanding.

        Parameters
        ----------
        video_path : str
            Path to the scene video file.
        prompt : str
            Analysis prompt.

        Returns
        -------
        dict
            Structured scene analysis.
        """
        print(f"      📤 Uploading video to Gemini Files API...")

        # Upload video
        video_file = self.client.files.upload(file=video_path)

        # Wait for processing
        while video_file.state == "PROCESSING":
            print("      ⏳ Waiting for video processing...")
            time.sleep(5)
            video_file = self.client.files.get(name=video_file.name)

        if video_file.state == "FAILED":
            raise RuntimeError(f"Video processing failed: {video_file.state}")

        # Analyze
        contents = [video_file, prompt]

        result = self._generate_with_retry(contents)

        # Clean up uploaded file
        try:
            self.client.files.delete(name=video_file.name)
        except Exception:
            pass  # Non-critical cleanup

        return result

    def analyze_temporal_flow(
        self,
        scene_analyses: list[dict],
        prompt: str,
    ) -> dict:
        """
        Analyze the temporal flow across all scenes.

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
        contents = [types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )]

        return self._generate_with_retry(contents)
