"""Honest citations: a quote is 'verified' only if it really appears in the
source it names, within the raw text the agent actually retrieved.

The model's own claim is ignored: every finding's `verified` is overwritten
with the result of this text match.
"""

import re

from app.models import Citation, Findings, ToolExecutionResult
from app.tools.api_data import record_lines

LOOKALIKES = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',  # curly quotes
    "‐": "-", "‑": "-",                                  # Unicode hyphens (auto‑renewal)
    " ": " ", " ": " ", " ": " ",                   # no-break spaces (55 000)
})

# Citation styles models actually use:
#   > "exact words" — source                (the style we ask for)
#   “exact words” 【source】 or `field: value` (CAD) 【source】   (common in OpenAI-style models)
#   “exact words” — `source`  (inside a bullet or table cell)
LINE_CITATION = re.compile(r'^\s*>\s*["“](.+?)["”]\s*[—–-]+\s*(.+?)\s*$', re.MULTILINE)
BRACKET_CITATION = re.compile(r'["“`]([^"“”`\n]{3,}?)["”`][^"“”`【\n]{0,40}?【\s*([^】\n]+?)\s*】')
DASH_CITATION = re.compile(r'["“]([^"“”\n]{3,}?)["”]\s*[—–-]\s*`?([\w.#\-‑]+)`?')


def normalize(text: str) -> str:
    """Lowercase, unify lookalike characters, collapse whitespace."""
    return " ".join(text.translate(LOOKALIKES).lower().split())


def normalize_source(source: str) -> str:
    """'fjord-dpa.md', '[fjord-dpa#3]', 'Fjord‑DPA' all become 'fjord-dpa'."""
    source = source.translate(LOOKALIKES).strip().strip("[]【】").lower()
    return source.split("#")[0].removesuffix(".md")


RECORDS = "api_data"


def source_texts(context: list[ToolExecutionResult]) -> dict[str, list[str]]:
    """Every retrieved text, keyed by normalized source name (full raw text, not the capped view).

    Documents and policies give whole chunks; api_data gives one entry per 'field: value' line.
    """
    texts: dict[str, list[str]] = {}
    for result in context:
        if isinstance(result.raw_output, list):  # document_retrieval / company_context chunks
            for chunk in result.raw_output:
                texts.setdefault(normalize_source(chunk["doc_name"]), []).append(chunk["text"])
        elif isinstance(result.raw_output, dict):  # api_data records
            texts.setdefault(RECORDS, []).extend(record_lines(result.raw_output))
    return texts


MIN_QUOTE_WORDS = 5  # shorter quotes ("the vendor", "7%") match too easily to prove anything


def _trim(text: str) -> str:
    """Normalize, then drop surrounding quote marks and full stops."""
    return normalize(text).strip(" .\"'")


def is_verified(source: str | None, quote: str | None, texts: dict[str, list[str]]) -> bool:
    if not source or not quote:
        return False
    needle = _trim(quote)
    if normalize_source(source).startswith(RECORDS):
        # Record values are short (an email, a bank account), so instead of a word
        # minimum the quote must be a whole 'field: value' line or a whole value.
        # A quote spanning several lines ("a: 1; b: 2") verifies only if every part is a whole line.
        lines = [line for line in texts.get(RECORDS, []) if not line.startswith("[")]  # skip labels
        parts = [_trim(p) for p in re.split(r"\n|;\s+", quote) if _trim(p)]
        return bool(parts) and all(
            any(part in (_trim(line), _trim(line.partition(": ")[2])) for line in lines) for part in parts)
    if len(needle.split()) < MIN_QUOTE_WORDS:
        return False
    return any(needle in normalize(text) for text in texts.get(normalize_source(source), []))


def verify_findings(findings: Findings, context: list[ToolExecutionResult]) -> Findings:
    texts = source_texts(context)
    checked = [f.model_copy(update={"verified": is_verified(f.source, f.quote, texts)}) for f in findings.findings]
    return findings.model_copy(update={"findings": checked})


def extract_citations(markdown: str, context: list[ToolExecutionResult]) -> list[Citation]:
    """Every quote+source pair in the report, in order of appearance, each checked."""
    texts = source_texts(context)
    found = [m for pattern in (LINE_CITATION, BRACKET_CITATION, DASH_CITATION) for m in pattern.finditer(markdown)]
    citations: list[Citation] = []
    for match in sorted(found, key=lambda m: m.start()):
        quote, source = match.group(1).strip(), match.group(2).strip()
        if all((c.quote, c.source) != (quote, source) for c in citations):  # skip repeats
            citations.append(Citation(source=source, quote=quote, verified=is_verified(source, quote, texts)))
    return citations
