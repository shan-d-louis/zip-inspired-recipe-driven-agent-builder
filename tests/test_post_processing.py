import pytest

from app.models import Finding, Findings, ToolExecutionResult


@pytest.mark.xfail(strict=True, reason="post-processing is built in Step 4; remove this marker then")
def test_model_supplied_verified_is_overwritten():
    from app.engine.nodes import verify_findings  # does not exist yet

    context = [
        ToolExecutionResult(
            tool_name="document_retrieval",
            ai_readable_string="...",
            raw_output=[{"doc_name": "fjord-dpa", "chunk_id": "fjord-dpa#1", "text": "Data is stored in the United States."}],
        )
    ]
    findings = Findings(
        summary="s",
        findings=[
            Finding(severity="high", title="t", detail="d", source="fjord-dpa",
                    quote="Data is stored in Canada.", verified=True),  # invented quote, model claims verified
            Finding(severity="high", title="t", detail="d", source="fjord-dpa",
                    quote="stored in the United States", verified=False),  # real quote, model claims unverified
        ],
    )
    checked = verify_findings(findings, context)
    assert [f.verified for f in checked.findings] == [False, True]
