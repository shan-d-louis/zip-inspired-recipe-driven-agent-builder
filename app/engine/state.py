"""The shared state the four nodes read from and write to."""

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage

from app.models import Citation, Recipe, ToolExecutionResult, TraceStep


class AgentState(TypedDict, total=False):
    # Inputs
    recipe: Recipe
    request_id: str
    # Preprocessing
    referenced_objects: list[dict]  # resolved purchase request, vendor, documents
    search_query: str  # for document retrieval
    # Orchestration
    accumulated_context: list[ToolExecutionResult]
    trace: list[TraceStep]  # for the UI
    provider: str  # which LLM answered last
    # FinalLlmCall
    final_messages: list[BaseMessage]  # kept so post-processing can ask for a corrected reply
    final_llm_result: str
    # PostProcessing
    final_result: Any  # markdown string or Findings
    citations: list[Citation]
    error: str | None
