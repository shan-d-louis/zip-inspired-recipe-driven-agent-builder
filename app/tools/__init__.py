"""The tool registry: every block a recipe can enable, by name."""

from app.tools.api_data import ApiDataTool
from app.tools.base import ToolBase
from app.tools.company_context import CompanyContextTool
from app.tools.document_retrieval import DocumentRetrievalTool

TOOL_REGISTRY: dict[str, type[ToolBase]] = {
    tool.name: tool for tool in (DocumentRetrievalTool, ApiDataTool, CompanyContextTool)
}


def build_tools(names: list[str], request_id: str) -> list[ToolBase]:
    """Create fresh tool instances for one run, each bound to the selected request.

    New instances every call, so two runs never share tool state.
    """
    unknown = [n for n in names if n not in TOOL_REGISTRY]
    if unknown:
        raise ValueError(f"unknown tools: {', '.join(unknown)}")
    if len(set(names)) != len(names):
        raise ValueError("tools must not repeat")
    return [TOOL_REGISTRY[name](request_id) for name in names]
