"""ScriptedChatModel: a deterministic stand-in for a real LLM (LLM_MODE=fake).

Used by every offline test and for running the app with no API keys. It
behaves like a well-mannered tool-calling model:
  - with tools bound: calls each enabled tool once, in order, then stops
  - without tools: answers with canned markdown, or valid Findings JSON when
    the prompt contains JSON_INSTRUCTION
Its quotes are copied from the tool results it was shown, so citation
checks pass. It can send broken JSON on demand to exercise the retry path.
"""

import json
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, PrivateAttr

from app.prompts import JSON_INSTRUCTION
from app.tools.base import MAX_QUERY_CHARS

LABEL = re.compile(r"^\[([\w-]+(?:#\d+)?)\]$")  # "[fjord-dpa#3]" or "[POL-03]" as printed by the tools


class ScriptedChatModel(BaseChatModel):
    tool_names: list[str] = Field(default_factory=list)
    invalid_json_replies: int = 0  # how many structured replies to break (retry tests)
    _invalid_sent: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "ScriptedChatModel":
        names = [convert_to_openai_tool(t)["function"]["name"] for t in tools]
        return self.model_copy(update={"tool_names": names})

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._reply(messages))])

    # --- behaviour ---

    def _reply(self, messages: list[BaseMessage]) -> AIMessage:
        if self.tool_names:
            return self._next_tool_call(messages)
        text = "\n".join(str(m.content) for m in messages)
        if JSON_INSTRUCTION in text:
            if self._invalid_sent < self.invalid_json_replies:
                self._invalid_sent += 1
                return AIMessage(content='{"summary": "unfinished", "findings": [')
            return AIMessage(content=self._findings_json(text))
        return AIMessage(content=self._markdown(text))

    def _next_tool_call(self, messages: list[BaseMessage]) -> AIMessage:
        called = [c["name"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
        remaining = [name for name in self.tool_names if name not in called]
        if not remaining:
            return AIMessage(content="I have gathered enough information.")
        name = remaining[0]
        if name == "api_data":
            args = {"section": "summary"}
        else:
            human = next((str(m.content) for m in messages if isinstance(m, HumanMessage)), "")
            args = {"query": " ".join(human.split()[:25])[:MAX_QUERY_CHARS] or "policy"}
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": f"call_{len(called) + 1}", "type": "tool_call"}],
        )

    @staticmethod
    def _evidence(text: str, limit: int = 2) -> list[tuple[str, str]]:
        """(source, quote) pairs copied verbatim from labelled tool results."""
        found: list[tuple[str, str]] = []
        label = None
        for line in text.splitlines():
            line = line.strip()
            if match := LABEL.match(line):
                label = match.group(1)
                continue
            if label and len(line) >= 30 and not line.startswith(("#", "|")):
                quote = line.split(". ")[0][:150]
                source = label.split("#")[0]
                if source not in {s for s, _ in found}:
                    found.append((source, quote))
                label = None
            if len(found) == limit:
                break
        return found

    def _findings_json(self, text: str) -> str:
        evidence = self._evidence(text)
        findings = [
            {"severity": "medium", "title": f"Review {source}", "detail": "Scripted finding from a retrieved passage.",
             "source": source, "quote": quote}
            for source, quote in evidence
        ] or [{"severity": "info", "title": "No passages retrieved", "detail": "Nothing to cite.",
               "source": None, "quote": None}]
        return json.dumps({"summary": f"Scripted result (fake mode) based on {len(evidence)} passage(s).",
                           "findings": findings})

    def _markdown(self, text: str) -> str:
        lines = ["## Scripted result (fake mode)", "", "This is a canned answer from the scripted model."]
        for source, quote in self._evidence(text):
            lines += ["", f'> "{quote}" — {source}']
        return "\n".join(lines)
