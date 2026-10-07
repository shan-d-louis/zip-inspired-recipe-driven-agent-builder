"""The scripted model must behave like a predictable tool-calling LLM."""

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.fake_llm import ScriptedChatModel
from app.models import Findings
from app.prompts import JSON_INSTRUCTION
from app.tools import build_tools

PROMPT = "Flag any conflict between this vendor's data terms and our data residency policy"


async def run_tool_loop(model, tool_names, request_id="3", max_steps=10):
    """A minimal ReAct loop (Step 4 builds the real one)."""
    tools = {t.name: t for t in build_tools(tool_names, request_id)}
    bound = model.bind_tools([t.as_langchain_tool() for t in tools.values()])
    messages = [HumanMessage(content=PROMPT)]
    results = []
    for _ in range(max_steps):
        reply = await bound.ainvoke(messages)
        messages.append(reply)
        if not reply.tool_calls:
            break
        for call in reply.tool_calls:
            result = await tools[call["name"]].run(call["args"])
            results.append((call, result))
            messages.append(ToolMessage(content=result.ai_readable_string, tool_call_id=call["id"]))
    return results, messages


async def test_calls_each_enabled_tool_once_in_order_then_stops():
    names = ["api_data", "document_retrieval", "company_context"]
    results, messages = await run_tool_loop(ScriptedChatModel(), names)
    assert [call["name"] for call, _ in results] == names
    assert results[0][0]["args"] == {"section": "summary"}
    assert results[1][0]["args"]["query"].startswith("Flag any conflict")
    assert not messages[-1].tool_calls


async def test_is_deterministic():
    first, _ = await run_tool_loop(ScriptedChatModel(), ["document_retrieval", "company_context"])
    second, _ = await run_tool_loop(ScriptedChatModel(), ["document_retrieval", "company_context"])
    assert [(c, r.ai_readable_string) for c, r in first] == [(c, r.ai_readable_string) for c, r in second]


async def context_text(tool_names):
    results, _ = await run_tool_loop(ScriptedChatModel(), tool_names)
    return "\n\n".join(r.ai_readable_string for _, r in results)


async def test_structured_reply_is_valid_findings_with_real_quotes():
    context = await context_text(["document_retrieval", "company_context"])
    reply = await ScriptedChatModel().ainvoke(
        [SystemMessage(content=f"{JSON_INSTRUCTION} {{...}}"), HumanMessage(content=context)])
    findings = Findings.model_validate_json(reply.content)
    assert findings.findings
    for finding in findings.findings:
        assert finding.quote and finding.quote in context  # copied verbatim, so citations can verify


async def test_markdown_reply_when_no_json_requested():
    context = await context_text(["document_retrieval"])
    reply = await ScriptedChatModel().ainvoke([HumanMessage(content=context)])
    assert reply.content.startswith("## Scripted result")
    assert '> "' in reply.content


async def test_invalid_json_on_demand_then_valid():
    model = ScriptedChatModel(invalid_json_replies=1)
    messages = [SystemMessage(content=JSON_INSTRUCTION), HumanMessage(content="no context")]
    first = await model.ainvoke(messages)
    second = await model.ainvoke(messages)
    assert not _is_valid(first.content)
    assert _is_valid(second.content)


def _is_valid(text: str) -> bool:
    try:
        Findings.model_validate_json(text)
        return True
    except ValueError:
        return False
