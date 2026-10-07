"""Core data models shared by the engine, tools and API.

A Recipe is the whole definition of an agent: creating an agent means
creating one of these, never writing new code.
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# Names of the tools a recipe may enable. The tool registry (app/tools) must
# match this list; a test checks that they stay in sync.
KNOWN_TOOLS = ("document_retrieval", "api_data", "company_context")


class OutputFormat(str, Enum):
    MARKDOWN = "markdown"
    STRUCTURED = "structured"  # JSON validated against the Findings schema


class Recipe(BaseModel):
    id: str
    name: str = Field(max_length=60)
    prompt: str = Field(max_length=600)
    tools: list[str] = Field(min_length=1, max_length=3)
    output_format: OutputFormat
    include_citations: bool = True
    is_preset: bool = False
    default_request_id: str | None = None  # presets only: the request they are cached for

    @field_validator("name", "prompt")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("tools")
    @classmethod
    def known_unique_tools(cls, tools: list[str]) -> list[str]:
        unknown = [t for t in tools if t not in KNOWN_TOOLS]
        if unknown:
            raise ValueError(f"unknown tools: {', '.join(unknown)}")
        if len(set(tools)) != len(tools):
            raise ValueError("tools must not repeat")
        return tools


# --- Structured output: one fixed, generic schema so users never write JSON ---


class Finding(BaseModel):
    severity: Literal["high", "medium", "low", "info"]
    title: str
    detail: str
    source: str | None = None  # document or policy name
    quote: str | None = None  # short supporting passage
    verified: bool | None = None  # set by post-processing, never by the model


class Findings(BaseModel):
    summary: str
    findings: list[Finding]


class Citation(BaseModel):
    """A quote cited in a markdown result, checked against the source text."""

    source: str
    quote: str
    verified: bool


# --- Engine bookkeeping ---


class TraceStep(BaseModel):
    step: int
    tool: str
    summary: str  # short, human-readable: what was asked and what came back
    duration_ms: int


class ToolExecutionResult(BaseModel):
    tool_name: str
    ai_readable_string: str  # what the LLM sees
    raw_output: Any  # what post-processing uses (e.g. chunks for citation checks)
