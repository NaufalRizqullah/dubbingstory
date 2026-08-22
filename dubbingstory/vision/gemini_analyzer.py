"""
dubbingstory.vision.gemini_analyzer — Gemini API visual analysis

Uses Gemini's multimodal capabilities to analyze video scenes
from keyframe images or uploaded video segments.
Includes retry logic adapted from opensource-clipping.
"""

import json
import os
import re
import time
import base64

from google import genai
from google.genai import types


# ── Retry Config (adapted from clipping/engine.py) ───────────────────────────
MAX_ATTEMPTS = 8
INITIAL_WAIT_SECONDS = 30
WAIT_INCREMENT_SECONDS = 15
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


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
    match = re.search(r"\b(408|429|500|502|503|504)\b", str(exc))
    return int(match.group(1)) if match else None


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is retryable."""
    if isinstance(exc, (json.JSONDecodeError, _ResponseValidationError)):
        return True
    code = _extract_status_code(exc)
    if code in RETRYABLE_STATUS_CODES:
        return True
    msg = str(exc).lower()
    keywords = (
        "timeout", "temporarily unavailable", "deadline",
        "connection reset", "connection aborted", "service unavailable",
        "resource_exhausted",
    )
    return any(k in msg for k in keywords)


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
    ):
        self.client = genai.Client(api_key=api_key)
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
    ) -> dict | list:
        """Call Gemini with retry logic and optional semantic validation.

        ``response_validator`` receives the decoded JSON payload and returns either
        ``None`` (accepted) or a human-readable error string (retry). This is used
        by narration generation to reject a response written in the wrong language
        without losing the original prompt/language contract on the next attempt.
        """
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=temperature,
        )

        last_exc = None
        status_code = None
        base_contents = list(contents)
        attempt_contents = list(base_contents)
        primary_model = model_override or self.model

        for attempt in range(1, MAX_ATTEMPTS + 1):
            model = primary_model if attempt <= MAX_ATTEMPTS - 1 else self.fallback_model

            try:
                print(
                    f"      [Gemini:{task_name}] Attempt {attempt}/{MAX_ATTEMPTS} "
                    f"({model})..."
                )

                response = self.client.models.generate_content(
                    model=model,
                    contents=attempt_contents,
                    config=config,
                )

                text = getattr(response, "text", None)
                if not text or not text.strip():
                    raise ValueError("Gemini returned empty response.")

                payload = json.loads(text)
                if response_validator is not None:
                    validation_error = response_validator(payload)
                    if validation_error:
                        raise _ResponseValidationError(str(validation_error))
                return payload

            except Exception as exc:
                last_exc = exc
                status_code = _extract_status_code(exc)
                retryable = _is_retryable(exc)

                print(
                    f"      [Gemini:{task_name}] Attempt {attempt} failed | "
                    f"status={status_code} | error={str(exc)[:180]}"
                )

                if (not retryable) or attempt == MAX_ATTEMPTS:
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

                wait = INITIAL_WAIT_SECONDS + ((attempt - 1) * WAIT_INCREMENT_SECONDS)
                print(f"      [Gemini:{task_name}] Retrying in {wait}s...")
                time.sleep(wait)

        raise RuntimeError(
            f"Gemini {task_name} failed after {MAX_ATTEMPTS} attempts. "
            f"Last error: {last_exc}"
        ) from last_exc

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
