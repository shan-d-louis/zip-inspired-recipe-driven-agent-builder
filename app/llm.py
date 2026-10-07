"""LLM provider wrapper: Gemini first, Groq as fallback, or a scripted fake.

LLM_MODE=fake (default) needs no keys and makes no network calls.
LLM_MODE=live builds the real providers from env vars. Model IDs always come
from env, because free-tier model names change.

We use a small invoke_with_fallback helper instead of LangChain's
with_fallbacks() because we must report which provider answered.
"""

import asyncio
import logging
import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

logger = logging.getLogger(__name__)

Provider = tuple[str, BaseChatModel]  # (name shown in the UI, chat model)


class LLMUnavailable(Exception):
    """No provider could answer. The message is safe to show to users."""


class ToolCallRejected(Exception):
    """The provider refused the model's own tool call (e.g. a tool that doesn't exist).

    That's a model mistake, not an outage: the engine tells the model and lets it try again.
    The message is the provider's short explanation, meant for the model, not for logs.
    """


def _tool_call_rejection(error: BaseException) -> str | None:
    """Groq answers an invalid tool call with HTTP 400, code 'tool_use_failed'."""
    for err in _error_chain(error):
        body = getattr(err, "body", None)
        if isinstance(body, dict):
            detail = body.get("error", body)
            if isinstance(detail, dict) and detail.get("code") == "tool_use_failed":
                return str(detail.get("message", "invalid tool call"))[:300]
    return None


def llm_mode() -> str:
    mode = os.getenv("LLM_MODE", "fake").strip().lower()
    if mode not in {"fake", "live"}:
        raise ValueError("LLM_MODE must be 'fake' or 'live'")
    return mode


def timeout_seconds() -> float:
    return float(os.getenv("LLM_TIMEOUT_SECONDS", "12"))


def get_providers() -> list[Provider]:
    """Providers in the order to try them."""
    if llm_mode() == "fake":
        from app.fake_llm import ScriptedChatModel

        return [("fake", ScriptedChatModel())]

    timeout = timeout_seconds()
    providers: list[Provider] = []
    # Provider packages are imported only in live mode.
    if os.getenv("GEMINI_API_KEY") and os.getenv("PRIMARY_MODEL"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        # max_retries=1 means one attempt: we do our own fallback instead of waiting.
        providers.append(("gemini", ChatGoogleGenerativeAI(
            model=os.environ["PRIMARY_MODEL"], api_key=os.environ["GEMINI_API_KEY"],
            timeout=timeout, max_retries=1,
        )))
    if os.getenv("GROQ_API_KEY") and os.getenv("FALLBACK_MODEL"):
        from langchain_groq import ChatGroq

        # Reasoning models (e.g. gpt-oss) spend most output tokens thinking; "low" keeps
        # runs cheap on the free tier. Set it empty for models without a reasoning setting.
        effort = os.getenv("GROQ_REASONING_EFFORT", "low").strip() or None
        providers.append(("groq", ChatGroq(
            model=os.environ["FALLBACK_MODEL"], api_key=os.environ["GROQ_API_KEY"],
            timeout=timeout, max_retries=0, reasoning_effort=effort,
        )))
    if not providers:
        raise LLMUnavailable("No AI provider is configured on the server.")
    return providers


def _error_chain(error: BaseException):
    """The error plus the errors it was raised from (wrappers hide status codes)."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        yield error
        error = error.__cause__ or error.__context__


# Gemini answers an invalid API key with 400 (not 401), recognisable by its message.
BAD_KEY_HINTS = ("API key not valid", "API_KEY_INVALID")


def should_fall_back(error: BaseException) -> bool:
    """True when another provider could succeed where this one failed:
    rate limits (429), server errors (5xx), timeouts, dropped connections,
    and auth/permission problems with this provider's key (401, 403, Gemini's
    bad-key 400). Any other 4xx is a bad request that would fail everywhere.
    """
    chain = list(_error_chain(error))
    for err in chain:
        if isinstance(err, TimeoutError):
            return True
        name = type(err).__name__
        if "Timeout" in name or name in {"APIConnectionError", "ConnectError"}:
            return True
        for attr in ("status_code", "code"):
            status = getattr(err, attr, None)
            if isinstance(status, int) and not isinstance(status, bool):
                if status == 400:
                    return any(hint in str(e) for e in chain for hint in BAD_KEY_HINTS)
                return status in (401, 403, 429) or 500 <= status < 600
    return False


def log_provider_config() -> None:
    """Log the mode and which providers are configured (names and model IDs, never keys)."""
    try:
        providers = get_providers()
    except LLMUnavailable:
        logger.warning("LLM mode: %s; no providers configured", llm_mode())
        return
    described = ", ".join(f"{name} ({getattr(model, 'model', None) or getattr(model, 'model_name', 'scripted')})"
                          for name, model in providers)
    logger.info("LLM mode: %s; providers in order: %s", llm_mode(), described)


async def invoke_with_fallback(
    messages: list[BaseMessage],
    tools: list[Any] | None = None,
    providers: list[Provider] | None = None,
    usage: "UsageMeter | None" = None,
) -> tuple[AIMessage, str]:
    """Ask each provider in turn. Returns (response, provider_name).

    With a single provider there is nothing to fall back to, so a rate limit
    (429) gets one retry after the server's Retry-After delay (capped at 5 s).
    The run's overall timeout still applies around all of this.
    """
    providers = providers if providers is not None else get_providers()
    for name, model in providers:
        runnable = model.bind_tools(tools) if tools else model
        retried = False
        while True:
            try:
                response = await asyncio.wait_for(runnable.ainvoke(messages), timeout=timeout_seconds())
                if usage is not None:
                    usage.add(response)
                return response, name
            except Exception as error:
                if (rejection := _tool_call_rejection(error)) is not None:
                    raise ToolCallRejected(rejection) from None
                if len(providers) == 1 and not retried and status_code(error) == 429:
                    delay = retry_after_seconds(error)
                    logger.warning("LLM provider %s rate limited; retrying once in %.1f s", name, delay)
                    await asyncio.sleep(delay)
                    retried = True
                    continue
                if not should_fall_back(error):
                    raise
                # Log the error type only: messages can echo request details.
                logger.warning("LLM provider %s failed with %s; trying next provider", name, type(error).__name__)
                break
    raise LLMUnavailable("The AI providers are busy right now. Please try again in a minute.")


MAX_RETRY_AFTER_SECONDS = 5.0
DEFAULT_RETRY_AFTER_SECONDS = 2.0


def status_code(error: BaseException) -> int | None:
    """The HTTP status anywhere in the error chain, if any."""
    for err in _error_chain(error):
        for attr in ("status_code", "code"):
            status = getattr(err, attr, None)
            if isinstance(status, int) and not isinstance(status, bool):
                return status
    return None


def retry_after_seconds(error: BaseException) -> float:
    """The server's Retry-After header (seconds), capped at 5 s; 2 s if absent or unreadable."""
    for err in _error_chain(error):
        headers = getattr(getattr(err, "response", None), "headers", None)
        if headers and headers.get("retry-after"):
            try:
                return min(max(float(headers["retry-after"]), 0.0), MAX_RETRY_AFTER_SECONDS)
            except ValueError:
                break  # an HTTP date instead of seconds: use the default
    return DEFAULT_RETRY_AFTER_SECONDS


class UsageMeter:
    """Adds up LLM calls and token counts for one run (counts only, never content)."""

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.reported = False  # did any provider report token usage?

    def add(self, response: AIMessage) -> None:
        self.calls += 1
        usage = getattr(response, "usage_metadata", None)
        if usage:
            self.reported = True
            self.input_tokens += usage.get("input_tokens", 0)
            self.output_tokens += usage.get("output_tokens", 0)

    def describe(self) -> str:
        if not self.reported:
            return f"llm_calls={self.calls} tokens=not reported"
        return (f"llm_calls={self.calls} input_tokens={self.input_tokens} "
                f"output_tokens={self.output_tokens} total_tokens={self.input_tokens + self.output_tokens}")
