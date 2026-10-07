"""The execution engine: one linear LangGraph that every recipe runs through."""

import asyncio
import logging
import os
import time

from langgraph.graph import END, START, StateGraph

from app.engine.nodes import EngineError, final_llm_call, orchestration, post_processing, preprocessing
from app.engine.state import AgentState
from app.llm import LLMUnavailable, Provider, UsageMeter, get_providers
from app.models import OutputFormat, Recipe, RunOutput, RunUsage

logger = logging.getLogger(__name__)


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("preprocessing", preprocessing)
    graph.add_node("orchestration", orchestration)
    graph.add_node("final_llm_call", final_llm_call)
    graph.add_node("post_processing", post_processing)
    graph.add_edge(START, "preprocessing")
    graph.add_edge("preprocessing", "orchestration")
    graph.add_edge("orchestration", "final_llm_call")
    graph.add_edge("final_llm_call", "post_processing")
    graph.add_edge("post_processing", END)
    return graph.compile()


GRAPH = build_graph()


def run_timeout_seconds() -> float:
    """Budget for the whole run: every LLM call, tool call and JSON retry counts against it."""
    return float(os.getenv("RUN_TIMEOUT_S", "25"))


async def run_recipe(recipe: Recipe, request_id: str, providers: list[Provider] | None = None) -> RunOutput:
    """Run one agent. Never raises: failures come back as ok=False with a friendly error.

    Logs one line per run with its cost (LLM calls and tokens), never content.
    """
    usage = UsageMeter()
    start = time.perf_counter()
    output = await _run(recipe, request_id, providers, usage)
    duration_ms = int((time.perf_counter() - start) * 1000)
    logger.info("Run %s on request %s: ok=%s provider=%s %s duration_ms=%d", recipe.id, request_id, output.ok,
                output.provider or "-", usage.describe(), duration_ms)
    return output.model_copy(update={"usage": RunUsage(
        llm_calls=usage.calls, duration_ms=duration_ms,
        input_tokens=usage.input_tokens if usage.reported else None,
        output_tokens=usage.output_tokens if usage.reported else None)})


async def _run(recipe: Recipe, request_id: str, providers: list[Provider] | None, usage: UsageMeter) -> RunOutput:
    base = {"recipe_id": recipe.id, "request_id": request_id}
    try:
        providers = providers if providers is not None else get_providers()
        state = await asyncio.wait_for(
            GRAPH.ainvoke({"recipe": recipe, "request_id": request_id},
                          config={"configurable": {"providers": providers, "usage": usage}}),
            timeout=run_timeout_seconds(),
        )
    except (EngineError, LLMUnavailable) as error:
        return RunOutput(ok=False, error=str(error), **base)
    except TimeoutError:
        return RunOutput(ok=False, error="The agent took too long to answer. Please try again.", **base)
    except Exception as error:  # Exception, not BaseException: cancellation and Ctrl+C still propagate
        logger.error("Agent run failed with %s", type(error).__name__)  # type only: no prompts or data in logs
        return RunOutput(ok=False, error="Something went wrong while running the agent. Please try again.", **base)

    common = {"provider": state.get("provider"), "trace": state["trace"], **base}
    if state.get("error"):
        return RunOutput(ok=False, error=state["error"], **common)
    if recipe.output_format is OutputFormat.STRUCTURED:
        return RunOutput(ok=True, findings=state["final_result"], **common)
    return RunOutput(ok=True, markdown=state["final_result"], citations=state["citations"], **common)
