import asyncio
import logging
import os
import subprocess
import sys

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app import llm
from app.fake_llm import ScriptedChatModel

FAKE_KEY = "sk-test-SECRET-1234567890"


class StatusError(Exception):
    """Looks like an SDK error with an HTTP status (e.g. groq.RateLimitError)."""

    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code} for key {FAKE_KEY}")
        self.status_code = status_code


class WrapperError(Exception):
    """Looks like a LangChain wrapper that hides the status code."""


class BrokenModel(BaseChatModel):
    """Raises whatever error factory it is given; counts calls."""

    error_factory: object
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "broken"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls += 1
        raise self.error_factory()


class SlowModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "slow"

    def _generate(self, *args, **kwargs):
        raise NotImplementedError

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        await asyncio.sleep(5)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="late"))])


def wrapped_429():
    try:
        raise StatusError(429)
    except StatusError as cause:
        try:
            raise WrapperError("Error calling model") from cause
        except WrapperError as error:
            return error


MESSAGES = [HumanMessage(content="hello")]


# --- Mode selection and lazy imports ---


def test_default_mode_is_fake(monkeypatch):
    monkeypatch.delenv("LLM_MODE", raising=False)
    providers = llm.get_providers()
    assert [name for name, _ in providers] == ["fake"]
    assert isinstance(providers[0][1], ScriptedChatModel)


def test_invalid_mode_rejected(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "turbo")
    with pytest.raises(ValueError):
        llm.get_providers()


def test_fake_mode_never_imports_provider_packages():
    code = (
        "import sys, app.llm; app.llm.get_providers();"
        "print('langchain_google_genai' in sys.modules, 'langchain_groq' in sys.modules)"
    )
    env = {**os.environ, "LLM_MODE": "fake"}  # full env: a bare env makes Windows write stray cache dirs
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip() == "False False"


def test_live_mode_builds_both_providers_from_env(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "live")
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    monkeypatch.setenv("PRIMARY_MODEL", "primary-model-from-env")
    monkeypatch.setenv("FALLBACK_MODEL", "fallback-model-from-env")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "7")
    (g_name, gemini), (q_name, groq) = llm.get_providers()
    assert (g_name, q_name) == ("gemini", "groq")
    assert "primary-model-from-env" in gemini.model
    assert groq.model_name == "fallback-model-from-env"
    assert gemini.timeout == 7 and groq.request_timeout == 7
    assert FAKE_KEY not in repr(gemini) and FAKE_KEY not in repr(groq)  # keys are masked


def test_live_mode_skips_unconfigured_provider(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "live")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    monkeypatch.setenv("FALLBACK_MODEL", "fallback-model-from-env")
    assert [name for name, _ in llm.get_providers()] == ["groq"]


def test_live_mode_without_any_keys_is_unavailable(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "live")
    for var in ("GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(llm.LLMUnavailable):
        llm.get_providers()


# --- Which errors trigger the fallback ---


@pytest.mark.parametrize(
    "error,expected",
    [
        (StatusError(429), True),
        (StatusError(500), True),
        (StatusError(503), True),
        (TimeoutError(), True),
        (type("APITimeoutError", (Exception,), {})(), True),
        (type("APIConnectionError", (Exception,), {})(), True),
        (wrapped_429(), True),
        (StatusError(400), False),
        (StatusError(401), False),
        (StatusError(404), False),
        (ValueError("bad tool schema"), False),
    ],
)
def test_is_retryable(error, expected):
    assert llm.is_retryable(error) is expected


# --- invoke_with_fallback ---


@pytest.mark.parametrize("error_factory", [lambda: StatusError(429), wrapped_429, lambda: StatusError(503)])
async def test_primary_rate_limited_fallback_answers(error_factory):
    primary = BrokenModel(error_factory=error_factory)
    response, provider = await llm.invoke_with_fallback(MESSAGES, providers=[("gemini", primary), ("groq", ScriptedChatModel())])
    assert provider == "groq"
    assert primary.calls == 1
    assert "Scripted result" in response.content


async def test_primary_bad_request_does_not_fall_back():
    primary = BrokenModel(error_factory=lambda: StatusError(400))
    fallback = BrokenModel(error_factory=lambda: StatusError(500))
    with pytest.raises(StatusError):
        await llm.invoke_with_fallback(MESSAGES, providers=[("gemini", primary), ("groq", fallback)])
    assert fallback.calls == 0


async def test_primary_timeout_falls_back(monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "0.05")
    _, provider = await llm.invoke_with_fallback(MESSAGES, providers=[("gemini", SlowModel()), ("groq", ScriptedChatModel())])
    assert provider == "groq"


async def test_all_providers_failing_gives_friendly_error():
    providers = [("gemini", BrokenModel(error_factory=lambda: StatusError(429))),
                 ("groq", BrokenModel(error_factory=lambda: StatusError(503)))]
    with pytest.raises(llm.LLMUnavailable) as info:
        await llm.invoke_with_fallback(MESSAGES, providers=providers)
    assert "try again" in str(info.value).lower()
    assert FAKE_KEY not in str(info.value)


async def test_fallback_logs_never_contain_secrets(caplog):
    caplog.set_level(logging.DEBUG)
    primary = BrokenModel(error_factory=lambda: StatusError(429))
    await llm.invoke_with_fallback(MESSAGES, providers=[("gemini", primary), ("groq", ScriptedChatModel())])
    assert "gemini failed with StatusError" in caplog.text
    assert FAKE_KEY not in caplog.text


async def test_tools_are_bound_for_each_provider():
    from app.tools import build_tools

    tools = [t.as_langchain_tool() for t in build_tools(["company_context"], "3")]
    primary = BrokenModel(error_factory=lambda: StatusError(429))
    response, provider = await llm.invoke_with_fallback(
        MESSAGES, tools=tools, providers=[("gemini", primary), ("groq", ScriptedChatModel())])
    assert provider == "groq"
    assert response.tool_calls[0]["name"] == "company_context"
