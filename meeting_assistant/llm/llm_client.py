"""
llm/llm_client.py — Pluggable LLM backend abstraction.

All agents call this module. To swap from OpenAI to another provider,
only this file needs to change — no agent code is touched.

Design contract:
  client = LLMClient()
  response: str = client.chat(messages, model, temperature, max_tokens)

`messages` follows the standard OpenAI format:
  [
    {"role": "system", "content": "You are ..."},
    {"role": "user",   "content": "Summarise this: ..."},
  ]
"""

import json
from typing import Any

from loguru import logger
from openai import OpenAI, APITimeoutError, RateLimitError, APIError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

from config import settings


# Map tenacity's logging to loguru
_tenacity_logger = logging.getLogger("tenacity")


class LLMError(Exception):
    """Raised when the LLM call fails after all retries are exhausted."""
    pass


class LLMClient:
    """
    Thin wrapper around the Gemini API (via its OpenAI-compatible endpoint).

    Why a wrapper instead of calling the API directly?
    - Centralises retry logic (network blips, rate limits)
    - Enforces JSON mode when requested
    - Makes the backend swappable (just change base_url + api_key in config)
    - Provides a single place to add logging, cost tracking, or caching

    Usage:
        client = LLMClient()

        # Plain text response
        text = client.chat(messages, model="gemini-2.0-flash")

        # Guaranteed JSON object response (no markdown fences, no preamble)
        data = client.chat_json(messages, model="gemini-2.0-flash")
        # data is already a dict — no json.loads() needed
    """

    def __init__(self) -> None:
        if not settings.gemini_api_key:
            raise LLMError(
                "Gemini API key not found. "
                "Set the GEMINI_API_KEY environment variable.\n"
                "Get a free key at https://aistudio.google.com"
            )
        # Gemini exposes an OpenAI-compatible endpoint, so the openai SDK
        # works as-is — only the api_key and base_url differ.
        self._client = OpenAI(
            api_key=settings.gemini_api_key,
            base_url=settings.llm.base_url,
            timeout=settings.llm.timeout,
        )
        logger.info("LLMClient initialised (Gemini backend via AI Studio)")

    # ── Core method ───────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((APITimeoutError, RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(_tenacity_logger, logging.WARNING),
        reraise=True,
    )
    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Send a chat completion request and return the response as a string.

        Args:
            messages:    List of {"role": ..., "content": ...} dicts.
            model:       Override the model from config (e.g. "gemini-2.0-flash").
            temperature: Override the temperature from config.
            max_tokens:  Override max tokens from config.

        Returns:
            The assistant's reply as a plain string.

        Raises:
            LLMError: If the API call fails after retries.
        """
        _model = model or settings.llm.summary_model
        _temperature = temperature if temperature is not None else settings.llm.summary_temperature
        _max_tokens = max_tokens or settings.llm.max_tokens

        try:
            response = self._client.chat.completions.create(
                model=_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=_temperature,
                max_tokens=_max_tokens,
            )
            content = response.choices[0].message.content or ""
            logger.debug(
                f"LLM call OK | model={_model} | "
                f"prompt_tokens={response.usage.prompt_tokens} | "
                f"completion_tokens={response.usage.completion_tokens}"
            )
            return content.strip()

        except (APITimeoutError, RateLimitError):
            # tenacity will retry on these
            raise
        except APIError as e:
            raise LLMError(f"Gemini API error: {e}") from e

    # ── JSON mode ─────────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((APITimeoutError, RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(_tenacity_logger, logging.WARNING),
        reraise=True,
    )
    def chat_json(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Like `chat()`, but forces the model to return a valid JSON object.

        Uses OpenAI-compatible `response_format={"type": "json_object"}` feature,
        which Gemini's endpoint fully supports.
        The response is parsed and returned as a dict — no json.loads() needed.

        IMPORTANT: Your system prompt MUST mention the word "JSON" (e.g.
        "Respond only with a JSON object."). Gemini requires this when
        json_object mode is enabled, otherwise the API returns a 400 error.

        Args:
            messages:    List of {"role": ..., "content": ...} dicts.
            model:       Defaults to settings.llm.extraction_model.
            temperature: Defaults to settings.llm.extraction_temperature.
            max_tokens:  Defaults to settings.llm.max_tokens.

        Returns:
            Parsed dict from the model's JSON response.

        Raises:
            LLMError: If the API call fails or the response isn't valid JSON.
        """
        _model = model or settings.llm.extraction_model
        _temperature = temperature if temperature is not None else settings.llm.extraction_temperature
        _max_tokens = max_tokens or settings.llm.max_tokens

        raw = "{}"  # initialised here so it's always bound in the except block
        try:
            response = self._client.chat.completions.create(
                model=_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=_temperature,
                max_tokens=_max_tokens,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            logger.debug(f"LLM JSON call OK | model={_model} | raw={raw[:120]}")
            return json.loads(raw)

        except json.JSONDecodeError as e:
            raise LLMError(f"Model returned invalid JSON: {e}\nRaw: {raw}") from e
        except (APITimeoutError, RateLimitError, APIError) as e:
            raise LLMError(f"Gemini API error in chat_json: {e}") from e


# ── Convenience singleton ──────────────────────────────────────────────────────
# Agents can import this directly instead of instantiating LLMClient themselves.
# It is lazily created on first access to avoid import-time side effects.

_client_instance: LLMClient | None = None


def get_client() -> LLMClient:
    """Return the shared LLMClient singleton."""
    global _client_instance
    if _client_instance is None:
        _client_instance = LLMClient()
    return _client_instance
