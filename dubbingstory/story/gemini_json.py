"""Shared retry-safe Gemini JSON helper for story-layer modules.

All story/narration planning modules should route text-only Gemini JSON calls
through this helper (or the narration-specific wrapper in script_writer) rather
than calling the SDK directly.  This keeps retry behaviour, fallback models and
response validation consistent.
"""

from __future__ import annotations

from typing import Any, Callable


Validator = Callable[[Any], str | None]


def build_story_client(cfg, *, model_attr: str | None = None):
    """Return ``(GeminiVideoAnalyzer, model_name)`` for a story-layer task."""
    from dubbingstory.vision.gemini_analyzer import GeminiVideoAnalyzer

    api_key = getattr(cfg, "api_key_gemini", "")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not available for story-layer Gemini call")

    model = ""
    if model_attr:
        model = str(getattr(cfg, model_attr, "") or "")
    model = model or str(getattr(cfg, "story_model", "") or "")
    model = model or str(getattr(cfg, "vision_gemini_model", "gemini-2.5-flash") or "gemini-2.5-flash")
    fallback = str(getattr(cfg, "vision_gemini_fallback_model", "gemini-2.0-flash") or "gemini-2.0-flash")
    return GeminiVideoAnalyzer(api_key=api_key, model=model, fallback_model=fallback), model


def generate_story_json(
    cfg,
    prompt: str,
    *,
    task_name: str,
    temperature: float = 0.2,
    response_validator: Validator | None = None,
    model_attr: str | None = None,
) -> Any:
    """Generate JSON with the centralized Gemini retry wrapper.

    The original prompt object is reused for every retry by
    :class:`GeminiVideoAnalyzer`; validation failures add a correction note but
    never rebuild task/language contracts from a previous model response.
    """
    analyzer, model = build_story_client(cfg, model_attr=model_attr)
    return analyzer.generate_json_with_retry(
        prompt,
        temperature=temperature,
        response_validator=response_validator,
        task_name=task_name,
        model=model,
    )
