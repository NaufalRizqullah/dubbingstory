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
    if isinstance(exc, json.JSONDecodeError):
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
    ) -> dict:
        """
        Call Gemini with retry logic.

        Adapted from clipping/engine.py _generate_json_with_retry.
        """
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,  # Low temperature for more deterministic analysis
        )

        last_exc = None
        status_code = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            model = self.model if attempt <= MAX_ATTEMPTS - 1 else self.fallback_model

            try:
                print(f"      [Gemini] Attempt {attempt}/{MAX_ATTEMPTS} ({model})...")

                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )

                text = getattr(response, "text", None)
                if not text or not text.strip():
                    raise ValueError("Gemini returned empty response.")

                return json.loads(text)

            except Exception as exc:
                last_exc = exc
                status_code = _extract_status_code(exc)
                retryable = _is_retryable(exc)

                print(
                    f"      [Gemini] Attempt {attempt} failed | "
                    f"status={status_code} | error={str(exc)[:100]}"
                )

                if (not retryable) or attempt == MAX_ATTEMPTS:
                    break

                wait = INITIAL_WAIT_SECONDS + ((attempt - 1) * WAIT_INCREMENT_SECONDS)
                print(f"      [Gemini] Retrying in {wait}s...")
                time.sleep(wait)

        raise RuntimeError(
            f"Gemini analysis failed after {MAX_ATTEMPTS} attempts. "
            f"Last error: {last_exc}"
        ) from last_exc

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
