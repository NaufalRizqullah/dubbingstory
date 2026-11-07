"""Content-bound, validated text batch caches; never store prompts or credentials."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


def payload_hash(payload) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path, payload) -> None:
    target = Path(path)
    temporary = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=target.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def cached_json(project_dir, cfg, *, task, prompt, response_validator, generate,
                version, namespace="story", evidence_identity=""):
    """Reuse only complete responses validated against the current task contract.

    Model/configuration names are allowlisted into the key. Evidence and prompt
    participate only as hashes; run IDs do not invalidate reusable good batches.
    """
    if namespace not in {"story", "narration"}:
        raise ValueError("Unsupported text batch cache namespace")
    identity = {
        "version": version, "task": task, "prompt_sha256": payload_hash(prompt),
        "evidence_sha256": payload_hash(evidence_identity),
        "schema": getattr(response_validator, "response_schema", None),
        "configuration": {key: getattr(cfg, key, None) for key in (
            "llm_provider", "llm_fallback_enabled", "llm_local_model", "story_model",
            "narration_model", "vision_gemini_model", "llm_local_context_window",
            "llm_batching_output_tokens", "llm_local_max_tokens",
            "llm_gemini_max_output_tokens", "llm_gemini_thinking_budget",
            "llm_local_base_url", "llm_local_structured_outputs", "llm_local_allow_schema_fallback",
        )},
    }
    key = payload_hash(identity)
    path = Path(project_dir) / f"{namespace}_cache" / (key + ".json") if project_dir else None
    if path is not None:
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            payload = cached["payload"]
            if (cached.get("cache_key") == key and cached.get("status") == "complete"
                    and cached.get("payload_sha256") == payload_hash(payload)
                    and not response_validator(payload)):
                print(f"   Text batch {task}: cached, skip generation")
                return payload
        except (OSError, ValueError, TypeError, KeyError, RecursionError):
            pass
    payload = generate()
    error = response_validator(payload)
    if error:
        from dubbingstory.llm.text_client import TextGenerationError
        raise TextGenerationError(f"{task}: batch response failed validation: {error}")
    if path is not None:
        atomic_json(path, {"schema_version": 1, "cache_key": key, "status": "complete",
                           "payload_sha256": payload_hash(payload), "payload": payload})
    return payload
