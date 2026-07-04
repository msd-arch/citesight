"""Provider-agnostic chat LLM for routing, verification, composing, and judging.

NO paid APIs. Every provider speaks the OpenAI chat-completions protocol:

  groq               free tier, fast — app default for the agent
  gemini             OpenAI-compat endpoint — judge default (generous free volume)
  ollama             local, keyless, no rate limits
  openai_compatible  any endpoint, fully env-configured
  self_vlm           text-only reasoning through the already-loaded Qwen2.5-VL
  fake               deterministic canned responses for CI / offline tests

Free-tier protection: request pacing, exponential backoff honoring Retry-After
on 429, and optional on-disk response caching (eval harness turns it on).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable

from citesight.config.settings import Settings
from citesight.utils.json_parsing import extract_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderSpec:
    base_url: str | None
    api_key_env: str | None  # None -> no key required
    default_model: str


PROVIDERS: dict[str, ProviderSpec] = {
    "groq": ProviderSpec(
        "https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"
    ),
    "gemini": ProviderSpec(
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        "gemini-2.0-flash",
    ),
    "ollama": ProviderSpec("http://localhost:11434/v1", None, "qwen2.5:7b-instruct"),
}


class LlmError(RuntimeError):
    pass


class ChatLLM:
    """Interface: complete() -> text, complete_json() -> parsed dict."""

    model_id: str = "?"

    def complete(
        self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 1024
    ) -> str:
        raise NotImplementedError

    def complete_json(
        self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 1024
    ) -> dict:
        """complete() + robust JSON parse; one repair round-trip on failure."""
        raw = self.complete(system, user, temperature, max_tokens)
        try:
            return extract_json(raw)
        except (ValueError, json.JSONDecodeError):
            logger.warning("non-JSON LLM output, attempting repair: %r", raw[:200])
            repaired = self.complete(
                system,
                "Your previous reply was not valid JSON. Reply with ONLY the "
                f"corrected JSON object, nothing else:\n\n{raw}",
                temperature,
                max_tokens,
            )
            return extract_json(repaired)


class _RateLimiter:
    def __init__(self, per_min: float) -> None:
        self.min_interval = 60.0 / per_min if per_min > 0 else 0.0
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        delta = self.min_interval - (now - self._last)
        if delta > 0:
            time.sleep(delta)
        self._last = time.monotonic()


class OpenAICompatLLM(ChatLLM):
    """OpenAI-protocol client with pacing, Retry-After-aware backoff, disk cache."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        requests_per_min: float = 25.0,
        max_retries: int = 5,
        cache_dir: str | None = None,
    ) -> None:
        from openai import OpenAI

        self.model_id = model
        self._client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        self._limiter = _RateLimiter(requests_per_min)
        self._max_retries = max_retries
        self._cache_dir = cache_dir

    def _cache_path(self, payload: dict) -> str | None:
        if not self._cache_dir:
            return None
        key = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        return os.path.join(self._cache_dir, f"{key}.json")

    def complete(
        self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 1024
    ) -> str:
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        cache_path = self._cache_path(payload)
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)["content"]

        from openai import APIConnectionError, APIStatusError, RateLimitError

        delay = 2.0
        for attempt in range(self._max_retries + 1):
            self._limiter.wait()
            try:
                resp = self._client.chat.completions.create(**payload)
                content = resp.choices[0].message.content or ""
                if cache_path:
                    os.makedirs(self._cache_dir, exist_ok=True)
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump({"content": content}, f)
                return content
            except RateLimitError as exc:
                wait = _retry_after_seconds(exc) or delay
                logger.warning(
                    "429 from %s (attempt %d/%d), sleeping %.1fs",
                    self.model_id, attempt + 1, self._max_retries, wait,
                )
                time.sleep(wait)
                delay = min(delay * 2, 60)
            except (APIConnectionError, APIStatusError) as exc:
                status = getattr(exc, "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    raise LlmError(f"{self.model_id}: {exc}") from exc
                if attempt >= self._max_retries:
                    raise LlmError(f"{self.model_id}: retries exhausted: {exc}") from exc
                logger.warning("transient LLM error (%s), retrying in %.1fs", exc, delay)
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise LlmError(f"{self.model_id}: retries exhausted (rate limited)")


def _retry_after_seconds(exc: Exception) -> float | None:
    resp = getattr(exc, "response", None)
    if resp is None:
        return None
    header = resp.headers.get("retry-after") if hasattr(resp, "headers") else None
    if header is None:
        return None
    try:
        return float(header)
    except ValueError:
        return None


class FakeLLM(ChatLLM):
    """Deterministic backend for CI/offline tests.

    `responses` maps a substring of the user prompt -> canned reply (checked in
    insertion order); `script` (if given) pops replies in sequence instead.
    """

    model_id = "fake-llm"

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        script: list[str] | None = None,
    ) -> None:
        self.responses = responses or {}
        self.script = list(script) if script else None
        self.calls: list[dict] = []

    def complete(
        self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 1024
    ) -> str:
        self.calls.append({"system": system, "user": user})
        if self.script is not None:
            if not self.script:
                raise LlmError("FakeLLM script exhausted")
            return self.script.pop(0)
        for needle, reply in self.responses.items():
            if needle in user or needle in system:
                return reply
        return '{"answer": "fake", "claims": []}'


class SelfVlmLLM(ChatLLM):
    """Route text-only reasoning through the already-loaded Qwen2.5-VL.

    Zero extra dependencies/keys: reuses the VLM answerer's model + processor
    with an image-free chat. Slow on CPU, but fully local.
    """

    def __init__(self, settings: Settings) -> None:
        from citesight.models.vlm import QwenVlAnswerer

        self._answerer = QwenVlAnswerer(settings)
        self.model_id = f"self_vlm:{self._answerer.model_id}"

    def complete(
        self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 1024
    ) -> str:
        import torch

        self._answerer._ensure_loaded()
        model, processor = self._answerer._model, self._answerer._processor
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ]
        chat_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[chat_text], padding=True, return_tensors="pt").to(
            model.device
        )
        with torch.no_grad():
            generated = model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False
            )
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def make_llm(
    role: str,
    settings: Settings,
    cache: bool | None = None,
    fake_factory: Callable[[], ChatLLM] | None = None,
) -> ChatLLM:
    """Build the LLM for a role ('agent' | 'judge') from settings."""
    role_field = "agent_llm_provider" if role == "agent" else "judge_llm_provider"
    provider = getattr(settings, role_field)
    # LLM_PROVIDER fallback: applies when the role-specific var wasn't set explicitly
    if role_field not in settings.model_fields_set and settings.llm_provider:
        provider = settings.llm_provider
    model_override = (
        settings.agent_llm_model if role == "agent" else settings.judge_llm_model
    )
    if provider == "fake":
        return fake_factory() if fake_factory else FakeLLM()
    if provider == "self_vlm":
        return SelfVlmLLM(settings)

    if provider == "openai_compatible":
        if not settings.openai_compatible_base_url or not settings.openai_compatible_model:
            raise LlmError(
                "provider=openai_compatible requires OPENAI_COMPATIBLE_BASE_URL "
                "and OPENAI_COMPATIBLE_MODEL in the environment"
            )
        base_url = settings.openai_compatible_base_url
        model = model_override or settings.openai_compatible_model
        key_env = settings.openai_compatible_api_key_env
    else:
        spec = PROVIDERS[provider]
        base_url = spec.base_url
        model = model_override or spec.default_model
        key_env = spec.api_key_env

    api_key = os.environ.get(key_env, "") if key_env else "not-needed"
    if key_env and not api_key:
        raise LlmError(
            f"provider={provider} needs {key_env} in the environment (see .env.example)"
        )
    use_cache = settings.llm_cache_enabled if cache is None else cache
    llm = OpenAICompatLLM(
        base_url=base_url,
        api_key=api_key or "not-needed",
        model=model,
        requests_per_min=settings.llm_requests_per_min,
        max_retries=settings.llm_max_retries,
        cache_dir=str(settings.llm_cache_dir) if use_cache else None,
    )
    logger.info("llm[%s]: provider=%s model=%s cache=%s", role, provider, model, use_cache)
    return llm
