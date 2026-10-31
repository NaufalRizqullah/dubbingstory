"""Bounded text-only Gemini calls with opt-in, run-scoped local fallback.

The local endpoint is the existing vLLM server. No model is loaded here. Its
tokenizer checks the complete prompt before generation; evidence is never
silently truncated to fit a smaller model. State contains routing metadata only.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import time
from urllib.parse import urlsplit
from uuid import uuid4

import requests

from dubbingstory.llm.diagnostics import LLMDiagnostics, response_usage


class TextGenerationError(RuntimeError):
    """A configured text route failed; do not substitute invented narration."""


def strict_text_routing(cfg) -> bool:
    return (getattr(cfg, "llm_provider", "gemini") == "local"
            or bool(getattr(cfg, "llm_fallback_enabled", False)))


def text_llm_available(cfg) -> bool:
    return (getattr(cfg, "llm_provider", "gemini") == "local"
            or bool(getattr(cfg, "api_key_gemini", "")))


def gemini_retry_options(cfg) -> dict:
    """One retry policy for Gemini vision and text, without changing providers."""
    defaults = {"max_attempts": 3, "timeout_seconds": 30, "budget_seconds": 90,
                "initial_wait_seconds": 5, "wait_increment_seconds": 5}
    return {key: getattr(cfg, "llm_gemini_" + key, value)
            for key, value in defaults.items()}


def _positive(cfg, key: str, default: float) -> float:
    raw = getattr(cfg, key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise TextGenerationError(f"{key} must be a finite positive number") from None
    if isinstance(raw, bool) or not math.isfinite(value) or value <= 0:
        raise TextGenerationError(f"{key} must be a finite positive number")
    return value


def _positive_int(cfg, key: str, default: int) -> int:
    value = _positive(cfg, key, default)
    if not value.is_integer():
        raise TextGenerationError(f"{key} must be a positive integer")
    return int(value)


class _LocalFailure(Exception):
    def __init__(self, category: str, *, retryable: bool = True, detail="", status_code=None):
        self.category = category
        self.retryable = retryable
        self.detail = detail
        self.status_code = status_code


class TextLLMClient:
    def __init__(self, cfg, model: str):
        self.cfg = cfg
        self.model = model
        self.provider = str(getattr(cfg, "llm_provider", "gemini"))
        if self.provider not in {"gemini", "local"}:
            raise TextGenerationError("llm.provider must be gemini or local")
        self.fallback = bool(getattr(cfg, "llm_fallback_enabled", False))
        self.diagnostics = LLMDiagnostics.from_config(cfg)

    def _read_state(self) -> dict:
        run_id = str(getattr(self.cfg, "llm_fallback_run_id", "") or "")
        state_path = getattr(self.cfg, "llm_fallback_state_path", "")
        if state_path:
            if not run_id:
                raise TextGenerationError("llm.fallback.run_id is required with state_path")
            try:
                state = json.loads(Path(state_path).read_text(encoding="utf-8"))
            except FileNotFoundError:
                return {}
            except (OSError, ValueError):
                raise TextGenerationError("Cannot read LLM routing state; use a fresh state_path") from None
        else:
            state = getattr(self.cfg, "_text_llm_state", {})
        if not isinstance(state, dict):
            raise TextGenerationError("Invalid LLM routing state")
        if state.get("run_id") != run_id:
            return {}
        if state and (state.get("schema_version") != 1
                      or state.get("provider") != "local"):
            raise TextGenerationError("Unsupported LLM routing state")
        return state

    def _activate_local(self, *, reason: str, attempts: int) -> None:
        state = {"schema_version": 1,
                 "run_id": str(getattr(self.cfg, "llm_fallback_run_id", "") or ""),
                 "provider": "local", "reason": reason, "gemini_attempts": attempts}
        state_path = getattr(self.cfg, "llm_fallback_state_path", "")
        if state_path:
            target = Path(state_path)
            temporary = None
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                                 dir=target.parent, prefix=target.name + ".",
                                                 suffix=".tmp", delete=False) as handle:
                    temporary = Path(handle.name)
                    json.dump(state, handle)
                os.replace(temporary, target)
            except OSError:
                raise TextGenerationError("Cannot persist LLM routing state") from None
            finally:
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
        self.cfg._text_llm_state = state

    def _generate_gemini(self, prompt, **kwargs):
        from dubbingstory.vision.gemini_analyzer import GeminiVideoAnalyzer

        call_id = kwargs.pop("_call_id", None) or uuid4().hex
        key = getattr(self.cfg, "api_key_gemini", "")
        if not key:
            raise TextGenerationError("Gemini key unavailable; configure llm.provider: local to use Qwen only")
        # Opt-in local fallback replaces the old last-attempt cloud-model switch.
        fallback_model = (self.model if self.fallback else
                          getattr(self.cfg, "vision_gemini_fallback_model", "gemini-2.0-flash"))
        analyzer = GeminiVideoAnalyzer(api_key=key, model=self.model,
                                       fallback_model=fallback_model,
                                       **gemini_retry_options(self.cfg))
        analyzer.diagnostics = self.diagnostics
        analyzer.diagnostics_call_id = call_id
        try:
            return analyzer.generate_json_with_retry(prompt, **kwargs)
        finally:
            # A text task owns this SDK client, including its connection pools.
            try:
                analyzer.client.close()
            except Exception:
                pass

    def generate_json_with_retry(self, prompt, *, temperature=0.2,
                                 response_validator=None, task_name="text", model=None):
        kwargs = dict(temperature=temperature, response_validator=response_validator,
                      task_name=task_name, model=model or self.model, _call_id=uuid4().hex)
        if self.provider == "local" or (self.fallback and self._read_state()):
            return self._generate_local(prompt, **kwargs)

        from dubbingstory.vision.gemini_analyzer import GeminiGenerationError

        try:
            return self._generate_gemini(prompt, **kwargs)
        except GeminiGenerationError as exc:
            if not self.fallback:
                raise
            if not exc.retryable:
                raise TextGenerationError("Gemini rejected the request; check model/access/configuration") from None
            # A bounded call has already exhausted its retries or deadline.
            # Only service outages latch the run; bad output retries locally for
            # this task without declaring the cloud service unavailable.
            if exc.service_failure:
                self._activate_local(reason="gemini_service_exhausted", attempts=exc.attempts)
                print(f"   LLM {task_name}: Gemini service exhausted; local route active for this run")
            else:
                print(f"   LLM {task_name}: Gemini output rejected; trying local for this task")
            return self._generate_local(prompt, **kwargs)

    def _generate_local(self, prompt, *, temperature, response_validator, task_name, model,
                        _call_id=None):
        del model  # Cloud model names must never leak into the local request.
        base_url = str(getattr(self.cfg, "llm_local_base_url", "http://127.0.0.1:8000/v1")).rstrip("/")
        url = urlsplit(base_url)
        if (url.scheme not in {"http", "https"} or url.hostname not in {"localhost", "127.0.0.1", "::1"}
                or url.username or url.password or url.query or url.fragment or url.path != "/v1"):
            raise TextGenerationError("llm.local.base_url must be a loopback vLLM URL ending in /v1")
        local_model = str(getattr(self.cfg, "llm_local_model", "Qwen/Qwen3-VL-4B-Instruct"))
        timeout = _positive(self.cfg, "llm_local_timeout_seconds", 60)
        budget = _positive(self.cfg, "llm_local_budget_seconds", 120)
        attempts = _positive_int(self.cfg, "llm_local_max_attempts", 2)
        max_tokens = _positive_int(self.cfg, "llm_local_max_tokens", 2048)
        context_window = _positive_int(self.cfg, "llm_local_context_window", 10240)
        deadline = time.monotonic() + budget
        started = time.monotonic()
        category = "budget_exhausted"
        used = 0
        correction = ""
        last_validation_error = ""
        diagnostics = self.diagnostics
        call_id = _call_id or uuid4().hex
        payload_schema = getattr(response_validator, "response_schema", None)
        response_format = {"type": "json_object"}
        if isinstance(payload_schema, dict):
            response_format = {"type": "json_schema", "json_schema": {
                "name": "dubbingstory_result", "strict": True,
                "schema": {"type": "object", "properties": {"result": payload_schema},
                           "required": ["result"], "additionalProperties": False},
            }}
        system = (
            'Follow the supplied video evidence, language and JSON contract. '
            'Return a JSON object with exactly one key "result" whose value is '
            'the requested JSON object or array. The user output contract describes '
            'the VALUE of result; always wrap that value in {"result": ...}. '
            'Do not add facts, markdown or commentary.'
        )
        # No environment proxy/auth forwarding and no redirects from local to
        # external services. requests itself performs no automatic retries.
        with requests.Session() as session:
            session.trust_env = False
            for attempt in range(1, attempts + 1):
                if time.monotonic() >= deadline:
                    break
                used = attempt
                messages = [{"role": "system", "content": system + correction},
                            {"role": "user", "content": prompt}]
                attempt_started = time.monotonic()
                content, payload, raw = None, None, {}
                finish_reason, validation_error, status_code = None, "", None
                error_type = ""
                try:
                    tokenized = self._post(session, base_url[:-3] + "/tokenize",
                                           {"model": local_model, "messages": messages,
                                            "add_generation_prompt": True}, timeout, deadline)
                    count = tokenized.get("count")
                    server_window = tokenized.get("max_model_len")
                    if type(count) is not int or count < 0 or type(server_window) is not int or server_window <= 0:
                        raise _LocalFailure("invalid_tokenizer_response", retryable=False)
                    available = min(context_window, server_window) - count - 64
                    if available < max_tokens:
                        raise _LocalFailure("context_budget_exceeded: increase MAX_MODEL_LEN or reduce analysis evidence", retryable=False)
                    raw = self._post(session, base_url + "/chat/completions",
                                     {"model": local_model, "messages": messages,
                                      "temperature": temperature, "max_tokens": max_tokens,
                                      "response_format": response_format, "stream": False},
                                     timeout, deadline)
                    status_code = 200
                    choices = raw.get("choices", [])
                    choice = choices[0] if choices else {}
                    finish_reason = choice.get("finish_reason")
                    content = choice.get("message", {}).get("content")
                    if finish_reason != "stop":
                        raise _LocalFailure("incomplete_output")
                    if not isinstance(content, str) or not content.strip():
                        raise _LocalFailure("empty_output")
                    try:
                        envelope = json.loads(content)
                    except RecursionError:
                        raise _LocalFailure("invalid_json_response", detail="JSON nesting limit exceeded") from None
                    if not isinstance(envelope, dict) or set(envelope) != {"result"}:
                        raise _LocalFailure("invalid_result_envelope")
                    payload = envelope["result"]
                    if not isinstance(payload, (dict, list)):
                        raise _LocalFailure("invalid_json_shape")
                    validation_error = response_validator(payload) if response_validator else None
                    if validation_error:
                        raise _LocalFailure("schema_or_language_rejected", detail=str(validation_error))
                    if time.monotonic() >= deadline:
                        raise _LocalFailure("budget_exhausted", retryable=False)
                    diagnostics.record(
                        call_id=call_id, task=task_name, provider="local", model=local_model,
                        attempt=attempt, outcome="accepted", raw_response=content, payload=payload,
                        finish_reason=finish_reason, usage=response_usage(raw.get("usage")),
                        provider_request_id=raw.get("id"), http_status=200,
                        latency_seconds=time.monotonic() - attempt_started,
                    )
                    print(f"   LLM {task_name}: local attempt {attempt}/{attempts} succeeded ({time.monotonic() - started:.1f}s)")
                    return payload
                except _LocalFailure as exc:
                    error_type = type(exc).__name__
                    category = exc.category
                    retryable = exc.retryable
                    validation_error = exc.detail
                    status_code = exc.status_code if exc.status_code is not None else status_code
                except json.JSONDecodeError as exc:
                    error_type = type(exc).__name__
                    category, retryable = "invalid_json_response", True
                    validation_error = f"JSON parse error at line {exc.lineno}, column {exc.colno}: {exc.msg}"
                except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
                    error_type = type(exc).__name__
                    category, retryable = "invalid_json_response", True
                except requests.RequestException as exc:
                    error_type = type(exc).__name__
                    category, retryable = "local_transport_error", True
                safe_error = diagnostics.safe_text(validation_error, 1200)
                last_validation_error = safe_error
                diagnostics.record(
                    call_id=call_id, task=task_name, provider="local", model=local_model,
                    attempt=attempt, outcome="rejected", category=category,
                    error_type=error_type,
                    validation_error=safe_error, raw_response=content, payload=payload,
                    finish_reason=finish_reason, usage=response_usage(raw.get("usage")),
                    provider_request_id=raw.get("id"), http_status=status_code,
                    latency_seconds=time.monotonic() - attempt_started,
                )
                print(f"   LLM {task_name}: local attempt {attempt}/{attempts} failed ({category})")
                if safe_error:
                    print(f"      Validation: {safe_error[:500]}")
                if category in {"schema_or_language_rejected", "invalid_result_envelope", "invalid_json_shape",
                                "invalid_json_response", "incomplete_output", "empty_output"}:
                    correction = "\n" + diagnostics.correction(safe_error or category, content)
                if not retryable or attempt == attempts:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 1:
                    break
                time.sleep(1)
        detail = f" ({last_validation_error[:500]})" if last_validation_error else ""
        raise TextGenerationError(f"Local LLM failed after {used} attempt(s): {category}{detail}. Saved artifacts can be resumed.") from None

    @staticmethod
    def _post(session, url: str, body: dict, timeout: float, deadline: float) -> dict:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _LocalFailure("budget_exhausted", retryable=False)
        with session.post(url, json=body, timeout=min(timeout, remaining), allow_redirects=False) as response:
            if not 200 <= response.status_code < 300:
                retryable = response.status_code in {408, 429, 500, 502, 503, 504}
                raise _LocalFailure(f"local_http_{response.status_code}", retryable=retryable,
                                    status_code=response.status_code)
            result = response.json()
        if not isinstance(result, dict):
            raise _LocalFailure("invalid_response_object")
        if time.monotonic() >= deadline:
            raise _LocalFailure("budget_exhausted", retryable=False)
        return result
