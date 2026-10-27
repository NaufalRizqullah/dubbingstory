"""Shared retry-safe text JSON helper for story-layer modules.

All story/narration planning modules should route text-only JSON calls
through this helper (or the narration-specific wrapper in script_writer) rather
than calling the SDK directly.  This keeps retry behaviour, fallback models and
response validation consistent.
"""

from __future__ import annotations

from typing import Any, Callable

from dubbingstory.llm.text_client import (
    TextGenerationError,
    TextLLMClient,
    strict_text_routing,
    text_llm_available,
)


Validator = Callable[[Any], str | None]


def build_story_client(cfg, *, model_attr: str | None = None):
    """Return a routed text client without changing the caller's model choice."""
    model = ""
    if model_attr:
        model = str(getattr(cfg, model_attr, "") or "")
    model = model or str(getattr(cfg, "story_model", "") or "")
    model = model or str(getattr(cfg, "vision_gemini_model", "gemini-2.5-flash") or "gemini-2.5-flash")
    return TextLLMClient(cfg, model), model


def raise_if_strict_text_error(cfg, exc: Exception, task_name: str) -> None:
    """Do not hide failed routed generation behind an invented/template story."""
    if isinstance(exc, TextGenerationError):
        raise exc
    if strict_text_routing(cfg):
        raise TextGenerationError(f"{task_name}: generated text failed validation") from None


def validate_scene_items(payload: Any, expected_ids: set[str]) -> str | None:
    """Check complete scene coverage before normalizers can fill missing rows."""
    if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
        return "response must be an array of objects"
    ids = [item.get("scene_id") for item in payload]
    if any(not isinstance(sid, str) or not sid for sid in ids):
        return "every item must have a nonempty scene_id"
    if len(ids) != len(set(ids)) or set(ids) != expected_ids:
        return "include every supplied scene_id exactly once, without extra IDs"
    return None


def validate_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value)


def generate_story_json(
    cfg,
    prompt: str,
    *,
    task_name: str,
    temperature: float = 0.2,
    response_validator: Validator | None = None,
    model_attr: str | None = None,
) -> Any:
    """Generate JSON with the configured bounded text route.

    The original prompt object is reused for every retry by
    text client; validation failures add a correction note but
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
