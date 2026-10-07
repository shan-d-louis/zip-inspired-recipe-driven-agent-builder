from app.engine.citations import extract_citations, verify_findings
from app.models import Finding, Findings, ToolExecutionResult

CONTEXT = [
    ToolExecutionResult(
        tool_name="document_retrieval",
        ai_readable_string="...",
        raw_output=[{"doc_name": "fjord-dpa", "chunk_id": "fjord-dpa#1",
                     "text": "Data is stored in the United States.\nBackups are   replicated every night."}],
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
        finding("“BACKUPS are replicated\n every night”"),
        finding("stored in the United States", source="Fjord-DPA.md"),
    ])
    assert all(f.verified for f in verify_findings(findings, CONTEXT).findings)


def test_quotes_under_five_words_are_never_verified():
    findings = Findings(summary="s", findings=[finding("the United States"), finding("in the United States")])
    assert [f.verified for f in verify_findings(findings, CONTEXT).findings] == [False, False]


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


def test_api_data_quotes_must_be_a_whole_record_line_or_value():
    records = ToolExecutionResult(tool_name="api_data", ai_readable_string="...", raw_output={
        "vendor": {"name": "Northbeam Analytic Inc.", "bank_account": "Demo Bank C (fictional) ****2290"}})
    quotes = [
        ("bank_account: Demo Bank C (fictional) ****2290", "api_data", True),   # whole line
        ("Demo Bank C (fictional) ****2290", "api_data", True),                 # whole value
        ("Northbeam Analytic Inc.", "api_data", True),                          # short, but a whole value
        ("Demo Bank C", "api_data", False),                                     # part of a value
        ("[vendor]", "api_data", False),                                        # a section heading
        ("bank_account: Demo Bank C (fictional) ****2290", "POL-06", False),    # wrong source
    ]
    findings = Findings(summary="s", findings=[finding(q, source=s) for q, s, _ in quotes])
    checked = verify_findings(findings, [records])
    assert [f.verified for f in checked.findings] == [expected for _, _, expected in quotes]


def test_bracket_style_citations_with_unicode_hyphens_are_extracted_and_checked():
    policy = ToolExecutionResult(tool_name="company_context", ai_readable_string="...", raw_output=[
        {"doc_name": "POL-02", "chunk_id": "POL-02",
         "text": "Any auto-renewal clause must be flagged to the requester and to Procurement before signature."}])
    records = ToolExecutionResult(tool_name="api_data", ai_readable_string="...",
                                  raw_output={"request": {"amount": 64900}})
    markdown = (
        "| POL\u201102 | \u201cAny auto\u2011renewal clause must be flagged to the requester\u201d \u3010POL-02\u3011 |\n"
        "- **Price:** `amount: 64900` (CAD) \u2013 > 64900 CAD \u3010api_data\u2011request#1\u3011\n"
        "- \u201cThe vendor may raise prices freely at any time\u201d \u3010POL-02\u3011\n"
    )
    citations = extract_citations(markdown, [policy, records])
    assert [(c.source, c.verified) for c in citations] == [
        ("POL-02", True), ("api_data\u2011request#1", True), ("POL-02", False)]


def test_dash_style_citations_and_multi_line_record_quotes():
    records = ToolExecutionResult(tool_name="api_data", ai_readable_string="...", raw_output={
        "vendor": {"name": "Northbeam Analytic Inc.", "email_domain": "northbeam-analytics.example"}})
    markdown = ("- Fee: \u201cname: Northbeam Analytic Inc.; email_domain: northbeam-analytics.example\u201d \u2014 `api_data#vendor`\n"
                "- Mixed: \u201cname: Northbeam Analytic Inc.; email_domain: other.example\u201d \u2014 `api_data#vendor`\n")
    assert [c.verified for c in extract_citations(markdown, [records])] == [True, False]
