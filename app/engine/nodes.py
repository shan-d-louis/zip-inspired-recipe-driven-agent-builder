"""The four nodes of the execution engine. Every agent runs through these.

Preprocessing -> Orchestration -> FinalLlmCall -> PostProcessing
"""

import json
import os
import re
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from app import data_store, prompts
from app.engine.citations import extract_citations, verify_findings
from app.engine.state import AgentState
from app.llm import invoke_with_fallback
from app.models import Findings, OutputFormat, ToolExecutionResult, TraceStep
from app.tools import build_tools

MAX_CONTEXT_CHARS = 6000  # total tool output carried into the final call
MAX_JSON_RETRIES = 2
CONTEXT_FULL = "\n[Context budget reached: no more tool results will be used.]"


class EngineError(Exception):
    """A failure whose message is safe to show to users."""


def max_agent_steps() -> int:
    return int(os.getenv("MAX_AGENT_STEPS", "4"))


def _providers(config: RunnableConfig):
    return config["configurable"]["providers"]


# --- 1. Preprocessing: no LLM. Resolve the request, vendor and documents. ---


async def preprocessing(state: AgentState) -> dict:
    request = data_store.get_request(state["request_id"])
    if request is None:
        raise EngineError(f"Purchase request '{state['request_id']}' does not exist.")
    vendor = data_store.get_vendor(request["vendor_id"])
    documents = [{"type": "document", "name": d["name"], "title": d["title"]}
                 for d in data_store.get_documents(request["id"])]
    return {
        "referenced_objects": [{"type": "request", **request}, {"type": "vendor", **vendor}, *documents],
        "search_query": state["recipe"].prompt,
        "accumulated_context": [],
        "trace": [],
    }


# --- 2. Orchestration: a ReAct loop that only gathers information. ---


def _task_message(state: AgentState) -> HumanMessage:
    request, vendor, *documents = state["referenced_objects"]
    return HumanMessage(content=(
        f"{state['search_query']}\n\n"
        f"Purchase request {request['id']}: {request['title']} ({request['amount']} {request['currency']}). "
        f"Vendor: {vendor['name']}. Attached documents: {', '.join(d['name'] for d in documents)}."
    ))


def _summarize(args: dict, result: ToolExecutionResult) -> str:
    asked = ", ".join(f"{k}={v!r}" for k, v in args.items())
    if len(asked) > 60:
        asked = asked[:57] + "..."
    raw = result.raw_output
    if raw is None:
        found = "invalid input"
    elif isinstance(raw, list):
        found = f"{len(raw)} match(es): {', '.join(c['chunk_id'] for c in raw)}" if raw else "no matches"
    else:
        found = f"returned {', '.join(raw)}"
    return f"{asked} -> {found}"


async def orchestration(state: AgentState, config: RunnableConfig) -> dict:
    recipe = state["recipe"]
    tools = {t.name: t for t in build_tools(recipe.tools, state["request_id"])}
    lc_tools = [t.as_langchain_tool() for t in tools.values()]
    messages = [SystemMessage(content=prompts.ORCHESTRATION_SYSTEM), _task_message(state)]
    context: list[ToolExecutionResult] = []
    trace: list[TraceStep] = []
    earlier: dict[str, ToolExecutionResult] = {}  # (tool, args) -> result, to skip repeated calls
    used_chars = 0
    provider = None

    for _ in range(max_agent_steps()):  # hard cap on LLM calls in this loop
        reply, provider = await invoke_with_fallback(messages, tools=lc_tools, providers=_providers(config))
        messages.append(reply)
        if not reply.tool_calls:
            break
        context_full = False
        for call in reply.tool_calls:
            start = time.perf_counter()
            key = f"{call['name']}:{json.dumps(call['args'], sort_keys=True)}"
            if key in earlier:
                result = earlier[key]
                content = f"(You already made this exact call; here is the same result again.)\n{result.ai_readable_string}"
                summary = "repeated call, reused earlier result"
            elif call["name"] not in tools:
                content = f"Tool '{call['name']}' is not enabled for this agent."
                summary = "not enabled for this agent"
            else:
                result = await tools[call["name"]].run(call["args"])
                earlier[key] = result
                if used_chars + len(result.ai_readable_string) > MAX_CONTEXT_CHARS:
                    room = max(MAX_CONTEXT_CHARS - used_chars, 0)
                    result = result.model_copy(
                        update={"ai_readable_string": result.ai_readable_string[:room] + CONTEXT_FULL})
                    context_full = True
                used_chars += len(result.ai_readable_string)
                context.append(result)
                content = result.ai_readable_string
                summary = _summarize(call["args"], result)
            trace.append(TraceStep(step=len(trace) + 1, tool=call["name"], summary=summary,
                                   duration_ms=int((time.perf_counter() - start) * 1000)))
            messages.append(ToolMessage(content=content, tool_call_id=call["id"]))
        if context_full:
            break

    return {"accumulated_context": context, "trace": trace, "provider": provider}


# --- 3. FinalLlmCall: one separate call, no tools. Researcher vs writer. ---


async def final_llm_call(state: AgentState, config: RunnableConfig) -> dict:
    recipe = state["recipe"]
    if recipe.output_format is OutputFormat.STRUCTURED:
        output_format = prompts.structured_format(recipe.include_citations)
    else:
        citations = prompts.MARKDOWN_CITATIONS if recipe.include_citations else prompts.MARKDOWN_NO_CITATIONS
        output_format = f"{prompts.MARKDOWN_FORMAT}\n{citations}"
    evidence = "\n\n".join(f"--- from {r.tool_name} ---\n{r.ai_readable_string}"
                           for r in state["accumulated_context"]) or "(no evidence was gathered)"
    messages = [
        SystemMessage(content=f"{prompts.FINAL_SYSTEM}\n\n{output_format}"),
        HumanMessage(content=f"Task: {recipe.prompt}\n\nEvidence:\n{evidence}"),
    ]
    reply, provider = await invoke_with_fallback(messages, providers=_providers(config))  # no tools bound
    return {"final_messages": [*messages, reply], "final_llm_result": reply.text, "provider": provider}


# --- 4. PostProcessing: validate, retry, verify citations. ---


def strip_fences(text: str) -> str:
    """Remove ```json fences and any chatter around the JSON object."""
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


async def post_processing(state: AgentState, config: RunnableConfig) -> dict:
    recipe = state["recipe"]
    context = state["accumulated_context"]
    text = state["final_llm_result"]

    if recipe.output_format is OutputFormat.MARKDOWN:
        citations = extract_citations(text, context) if recipe.include_citations else []
        return {"final_result": text, "citations": citations}

    messages = state["final_messages"]
    provider = state["provider"]
    for attempt in range(MAX_JSON_RETRIES + 1):
        try:
            findings = Findings.model_validate_json(strip_fences(text))
            break
        except ValidationError as error:
            if attempt == MAX_JSON_RETRIES:
                return {"error": "The agent could not produce valid structured output after 3 tries. "
                                 "Please run it again or switch to Markdown."}
            messages = [*messages, HumanMessage(content=prompts.json_retry_message(str(error)[:500]))]
            reply, provider = await invoke_with_fallback(messages, providers=_providers(config))
            messages.append(reply)
            text = reply.text
    # Always overwrite 'verified' from our own text match; the model's value is ignored.
    return {"final_result": verify_findings(findings, context), "provider": provider}
