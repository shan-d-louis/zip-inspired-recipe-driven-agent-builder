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

        providers.append(("groq", ChatGroq(
            model=os.environ["FALLBACK_MODEL"], api_key=os.environ["GROQ_API_KEY"],
            timeout=timeout, max_retries=0,
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
) -> tuple[AIMessage, str]:
    """Ask each provider in turn. Returns (response, provider_name)."""
    providers = providers if providers is not None else get_providers()
    for name, model in providers:
        runnable = model.bind_tools(tools) if tools else model
        try:
            response = await asyncio.wait_for(runnable.ainvoke(messages), timeout=timeout_seconds())
            return response, name
        except Exception as error:
            if not should_fall_back(error):
                raise
            # Log the error type only: messages can echo request details.
            logger.warning("LLM provider %s failed with %s; trying next provider", name, type(error).__name__)
    raise LLMUnavailable("The AI providers are busy right now. Please try again in a minute.")
