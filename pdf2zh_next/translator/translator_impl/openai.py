import json
import logging
from typing import Any

import httpx
import openai
from babeldoc.utils.atomic_integer import AtomicInteger
from pdf2zh_next.config.model import SettingsModel
from pdf2zh_next.translator.base_rate_limiter import BaseRateLimiter
from pdf2zh_next.translator.base_translator import BaseTranslator
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

logger = logging.getLogger(__name__)


def _coerce_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8", errors="replace")
        except Exception:
            return str(value)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                if "text" in item and item["text"] is not None:
                    parts.append(str(item["text"]))
                    continue
                if "content" in item and item["content"] is not None:
                    parts.append(str(item["content"]))
                    continue
        if parts:
            return "".join(parts)
    return str(value)


def _extract_message_content_from_response(response: Any) -> str:
    """Extract assistant text from OpenAI/OpenAI-compatible responses.

    Some OpenAI-compatible proxies return payloads like `{\"content\": \"...\"}`
    or even plain strings, which would otherwise crash on `.choices[0]...`.
    """

    if response is None:
        raise RuntimeError("OpenAI response is None")

    if isinstance(response, str):
        text = response.strip()
        if text and text[0] in "{[":
            try:
                parsed = json.loads(text)
            except Exception:
                return text
            try:
                return _extract_message_content_from_response(parsed)
            except Exception:
                return text
        return text

    if isinstance(response, (bytes, bytearray)):
        return (bytes(response).decode("utf-8", errors="replace")).strip()

    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message") or first.get("delta") or {}
                if isinstance(msg, dict):
                    content = _coerce_text(msg.get("content"))
                    if content is not None:
                        return content.strip()
                    text = _coerce_text(msg.get("text"))
                    if text is not None:
                        return text.strip()
                text = _coerce_text(first.get("text"))
                if text is not None:
                    return text.strip()
            else:
                msg = getattr(first, "message", None) or getattr(first, "delta", None)
                if msg is not None:
                    content = _coerce_text(getattr(msg, "content", None))
                    if content is not None:
                        return content.strip()
                    text = _coerce_text(getattr(msg, "text", None))
                    if text is not None:
                        return text.strip()
                text = _coerce_text(getattr(first, "text", None))
                if text is not None:
                    return text.strip()

        for key in ("content", "translated_text", "translation", "result", "output_text"):
            content = _coerce_text(response.get(key))
            if content is not None:
                return content.strip()

        for key in ("error", "detail", "message"):
            if key in response and response.get(key):
                raise RuntimeError(
                    f"OpenAI API returned error payload: {key}={response.get(key)}"
                )

        raise RuntimeError(
            f"Unexpected OpenAI response dict shape (no choices/content). Keys={list(response.keys())[:20]}"
        )

    # OpenAI "responses" API convenience
    output_text = getattr(response, "output_text", None)
    if output_text:
        coerced = _coerce_text(output_text)
        if coerced is not None:
            return coerced.strip()

    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        first = choices[0]
        msg = getattr(first, "message", None) or getattr(first, "delta", None)
        if msg is not None:
            content = _coerce_text(getattr(msg, "content", None))
            if content is not None:
                return content.strip()
            text = _coerce_text(getattr(msg, "text", None))
            if text is not None:
                return text.strip()
        text = _coerce_text(getattr(first, "text", None))
        if text is not None:
            return text.strip()

    if choices is None:
        content = _coerce_text(getattr(response, "content", None))
        if content is not None:
            return content.strip()

    # Pydantic model (OpenAI SDK) -> dict (fallback for unusual types)
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            return _extract_message_content_from_response(model_dump())
        except Exception:
            pass

    to_dict = getattr(response, "dict", None)
    if callable(to_dict):
        try:
            return _extract_message_content_from_response(to_dict())
        except Exception:
            pass

    raise RuntimeError(f"Unexpected OpenAI response type: {type(response).__name__}")


class OpenAITranslator(BaseTranslator):
    # https://github.com/openai/openai-python
    name = "openai"

    def __init__(
        self,
        settings: SettingsModel,
        rate_limiter: BaseRateLimiter,
    ):
        super().__init__(settings, rate_limiter)
        self.timeout = settings.translate_engine_settings.openai_timeout
        self.client = openai.OpenAI(
            base_url=settings.translate_engine_settings.openai_base_url,
            api_key=settings.translate_engine_settings.openai_api_key,
            timeout=float(self.timeout) if self.timeout else openai.NOT_GIVEN,
            http_client=httpx.Client(
                limits=httpx.Limits(
                    max_connections=None, max_keepalive_connections=None
                )
            ),
        )
        self.options = {}
        self.temperature = settings.translate_engine_settings.openai_temperature
        self.reasoning_effort = (
            settings.translate_engine_settings.openai_reasoning_effort
        )
        self.send_temperature = (
            settings.translate_engine_settings.openai_send_temprature
        )
        self.send_reasoning_effort = (
            settings.translate_engine_settings.openai_send_reasoning_effort
        )

        if self.send_temperature and self.temperature:
            self.add_cache_impact_parameters("temperature", self.temperature)
            self.options["temperature"] = float(self.temperature)
        if self.send_reasoning_effort and self.reasoning_effort:
            self.add_cache_impact_parameters("reasoning_effort", self.reasoning_effort)
            self.options["reasoning_effort"] = self.reasoning_effort

        self.model = settings.translate_engine_settings.openai_model
        if settings.translate_engine_settings.openai_base_url:
            self.add_cache_impact_parameters(
                "base_url", settings.translate_engine_settings.openai_base_url
            )
        self.add_cache_impact_parameters("model", self.model)
        self.add_cache_impact_parameters("prompt", self.prompt(""))
        self.token_count = AtomicInteger()
        self.prompt_token_count = AtomicInteger()
        self.completion_token_count = AtomicInteger()
        self.cache_hit_prompt_token_count = AtomicInteger()

        self.enable_json_mode = (
            settings.translate_engine_settings.openai_enable_json_mode
        )
        if self.enable_json_mode:
            self.add_cache_impact_parameters("enable_json_mode", self.enable_json_mode)

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_translate(self, text, rate_limit_params: dict = None) -> str:
        options = self.options.copy()
        if (
            self.enable_json_mode
            and rate_limit_params
            and rate_limit_params.get("request_json_mode", False)
        ):
            options["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(
            model=self.model,
            **options,
            messages=self.prompt(text),
        )
        try:
            if hasattr(response, "usage") and response.usage:
                if hasattr(response.usage, "total_tokens"):
                    self.token_count.inc(response.usage.total_tokens)
                if hasattr(response.usage, "prompt_tokens"):
                    self.prompt_token_count.inc(response.usage.prompt_tokens)
                if hasattr(response.usage, "completion_tokens"):
                    self.completion_token_count.inc(response.usage.completion_tokens)
                if hasattr(response.usage, "prompt_cache_hit_tokens"):
                    self.cache_hit_prompt_token_count.inc(
                        response.usage.prompt_cache_hit_tokens
                    )
                elif hasattr(response.usage, "prompt_tokens_details") and hasattr(
                    response.usage.prompt_tokens_details, "cached_tokens"
                ):
                    self.cache_hit_prompt_token_count.inc(
                        response.usage.prompt_tokens_details.cached_tokens
                    )
        except Exception as e:
            logger.error(f"Error getting token usage: {e}")
            pass
        message = _extract_message_content_from_response(response)
        message = self._remove_cot_content(message)
        return message

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_llm_translate(self, text, rate_limit_params: dict = None):
        if text is None:
            return None
        options = self.options.copy()
        if (
            self.enable_json_mode
            and rate_limit_params
            and rate_limit_params.get("request_json_mode", False)
        ):
            options["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(
            model=self.model,
            **options,
            messages=[
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )
        try:
            if hasattr(response, "usage") and response.usage:
                if hasattr(response.usage, "total_tokens"):
                    self.token_count.inc(response.usage.total_tokens)
                if hasattr(response.usage, "prompt_tokens"):
                    self.prompt_token_count.inc(response.usage.prompt_tokens)
                if hasattr(response.usage, "completion_tokens"):
                    self.completion_token_count.inc(response.usage.completion_tokens)
                if hasattr(response.usage, "prompt_cache_hit_tokens"):
                    self.cache_hit_prompt_token_count.inc(
                        response.usage.prompt_cache_hit_tokens
                    )
                elif hasattr(response.usage, "prompt_tokens_details") and hasattr(
                    response.usage.prompt_tokens_details, "cached_tokens"
                ):
                    self.cache_hit_prompt_token_count.inc(
                        response.usage.prompt_tokens_details.cached_tokens
                    )
        except Exception as e:
            logger.error(f"Error getting token usage: {e}")
            pass
        message = _extract_message_content_from_response(response)
        message = self._remove_cot_content(message)
        return message
