"""Live smoke test: one tool call on each configured provider.

Usage:  python scripts/live_smoke.py
Reads keys and model IDs from .env. Prints provider, latency and the tool call,
never the keys. Exits 1 if any provider fails.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

from app import llm  # noqa: E402
from app.tools import build_tools  # noqa: E402

PROMPT = "Use the company_context tool to look up our data residency policy."


async def check(name: str, model) -> bool:
    tools = [t.as_langchain_tool() for t in build_tools(["company_context"], "3")]
    start = time.perf_counter()
    try:
        response, _ = await llm.invoke_with_fallback([HumanMessage(content=PROMPT)], tools=tools,
                                                     providers=[(name, model)])
    except Exception as error:  # report the type only, never the message
        print(f"[FAIL] {name}: {type(error).__name__}")
        return False
    ms = int((time.perf_counter() - start) * 1000)
    if not response.tool_calls:
        print(f"[FAIL] {name}: answered without calling a tool ({ms} ms)")
        return False
    call = response.tool_calls[0]
    print(f"[ OK ] {name}: {call['name']}({call['args']}) in {ms} ms")
    return True


async def main() -> int:
    load_dotenv()
    os.environ["LLM_MODE"] = "live"
    providers = llm.get_providers()
    print(f"Configured providers: {', '.join(name for name, _ in providers)}")
    results = [await check(name, model) for name, model in providers]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
