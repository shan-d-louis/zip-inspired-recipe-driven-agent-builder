from app.engine.citations import extract_citations, verify_findings
from app.models import Finding, Findings, ToolExecutionResult

CONTEXT = [
    ToolExecutionResult(
        tool_name="document_retrieval",
        ai_readable_string="...",
        raw_output=[{"doc_name": "fjord-dpa", "chunk_id": "fjord-dpa#1",
                     "text": "Data is stored in the United States.\nBackups are   replicated nightly."}],
    )
]


def finding(quote, source="fjord-dpa", verified=None):
    return Finding(severity="high", title="t", detail="d", source=source, quote=quote, verified=verified)


def test_model_supplied_verified_is_overwritten():
    findings = Findings(summary="s", findings=[
        finding("Data is stored in Canada.", verified=True),  # invented quote, model claims verified
        finding("stored in the United States", verified=False),  # real quote, model claims unverified
    ])
    checked = verify_findings(findings, CONTEXT)
    assert [f.verified for f in checked.findings] == [False, True]


def test_matching_ignores_case_whitespace_and_curly_quotes():
    findings = Findings(summary="s", findings=[
        finding("“BACKUPS are replicated\n nightly”"),
        finding("stored in the United States", source="Fjord-DPA.md"),
    ])
    assert all(f.verified for f in verify_findings(findings, CONTEXT).findings)


def test_real_quote_with_wrong_source_or_no_quote_is_not_verified():
    findings = Findings(summary="s", findings=[finding("stored in the United States", source="POL-03"),
                                               finding(None)])
    assert [f.verified for f in verify_findings(findings, CONTEXT).findings] == [False, False]


def test_markdown_citations_extracted_and_checked():
    markdown = ('Verdict.\n\n> "stored in the United States" — fjord-dpa\n'
                '> “stored only in Canada” - fjord-dpa\n')
    citations = extract_citations(markdown, CONTEXT)
    assert [(c.quote, c.verified) for c in citations] == [
        ("stored in the United States", True), ("stored only in Canada", False)]
