"""Honest citations: a quote is 'verified' only if it really appears in the
source it names, within the raw text the agent actually retrieved.

The model's own claim is ignored: every finding's `verified` is overwritten
with the result of this text match.
"""

import json
import re

from app.models import Citation, Findings, ToolExecutionResult

QUOTE_CHARS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})

# A markdown citation line:  > "exact words" — source
MARKDOWN_CITATION = re.compile(r'^\s*>\s*["“](.+?)["”]\s*[—–-]+\s*(.+?)\s*$', re.MULTILINE)


def normalize(text: str) -> str:
    """Lowercase, straighten curly quotes, collapse whitespace."""
    return " ".join(text.translate(QUOTE_CHARS).lower().split())


def normalize_source(source: str) -> str:
    """'fjord-dpa.md', '[fjord-dpa#3]' and 'Fjord-DPA' all become 'fjord-dpa'."""
    source = source.strip().strip("[]").lower()
    return source.split("#")[0].removesuffix(".md")


def source_texts(context: list[ToolExecutionResult]) -> dict[str, list[str]]:
    """Every retrieved text, keyed by normalized source name (full raw text, not the capped view)."""
    texts: dict[str, list[str]] = {}
    for result in context:
        if isinstance(result.raw_output, list):  # document_retrieval / company_context chunks
            for chunk in result.raw_output:
                texts.setdefault(normalize_source(chunk["doc_name"]), []).append(chunk["text"])
        elif isinstance(result.raw_output, dict):  # api_data records
            texts.setdefault(result.tool_name, []).append(json.dumps(result.raw_output))
    return texts


def is_verified(source: str | None, quote: str | None, texts: dict[str, list[str]]) -> bool:
    if not source or not quote:
        return False
    needle = normalize(quote).strip(" .\"'")
    if not needle:
        return False
    return any(needle in normalize(text) for text in texts.get(normalize_source(source), []))


def verify_findings(findings: Findings, context: list[ToolExecutionResult]) -> Findings:
    texts = source_texts(context)
    checked = [f.model_copy(update={"verified": is_verified(f.source, f.quote, texts)}) for f in findings.findings]
    return findings.model_copy(update={"findings": checked})


def extract_citations(markdown: str, context: list[ToolExecutionResult]) -> list[Citation]:
    texts = source_texts(context)
    return [
        Citation(source=source, quote=quote, verified=is_verified(source, quote, texts))
        for quote, source in MARKDOWN_CITATION.findall(markdown)
    ]
