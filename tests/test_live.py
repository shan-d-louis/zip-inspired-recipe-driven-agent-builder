"""Real API calls. Skipped by default; run with:  pytest -m live"""

import os

import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from app import llm
from app.tools import build_tools

pytestmark = pytest.mark.live

PROVIDERS = {"gemini": ("GEMINI_API_KEY", "PRIMARY_MODEL"), "groq": ("GROQ_API_KEY", "FALLBACK_MODEL")}


@pytest.mark.parametrize("provider_name", list(PROVIDERS))
async def test_provider_makes_one_tool_call(provider_name, monkeypatch):
    load_dotenv()
    if not all(os.getenv(var) for var in PROVIDERS[provider_name]):
        pytest.skip(f"{provider_name} is not configured in .env")
    monkeypatch.setenv("LLM_MODE", "live")
    providers = dict(llm.get_providers())

    tools = [t.as_langchain_tool() for t in build_tools(["company_context"], "3")]
    prompt = "Use the company_context tool to look up our data residency policy."
    response, used = await llm.invoke_with_fallback(
        [HumanMessage(content=prompt)], tools=tools, providers=[(provider_name, providers[provider_name])])

    assert used == provider_name
    assert response.tool_calls and response.tool_calls[0]["name"] == "company_context"
