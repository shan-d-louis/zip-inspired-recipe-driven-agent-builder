import pytest
from pydantic import ValidationError

from app.models import Findings, OutputFormat, Recipe


def make_recipe(**overrides) -> Recipe:
    fields = {
        "id": "r1",
        "name": "Data Residency Check",
        "prompt": "Flag any conflict between this vendor's data terms and our data residency policy.",
        "tools": ["document_retrieval", "company_context"],
        "output_format": "structured",
    }
    fields.update(overrides)
    return Recipe(**fields)


def test_valid_recipe():
    recipe = make_recipe()
    assert recipe.output_format is OutputFormat.STRUCTURED
    assert recipe.include_citations is True
    assert recipe.is_preset is False


def test_name_and_prompt_are_trimmed():
    recipe = make_recipe(name="  My agent  ", prompt="  Check it.  ")
    assert recipe.name == "My agent"
    assert recipe.prompt == "Check it."


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "x" * 61},
        {"prompt": "x" * 601},
        {"name": "   "},
        {"prompt": ""},
        {"tools": []},
        {"tools": ["api_data", "document_retrieval", "company_context", "api_data"]},
        {"tools": ["web_search"]},
        {"tools": ["api_data", "api_data"]},
        {"output_format": "html"},
    ],
)
def test_invalid_recipes_rejected(overrides):
    with pytest.raises(ValidationError):
        make_recipe(**overrides)


def test_findings_round_trip():
    raw = (
        '{"summary": "One issue.", "findings": [{"severity": "high", "title": "US-only storage",'
        ' "detail": "Conflicts with POL-03.", "source": "fjord-dpa", "quote": "exclusively in data centers"}]}'
    )
    findings = Findings.model_validate_json(raw)
    assert findings.findings[0].severity == "high"
    assert findings.findings[0].verified is None  # only post-processing sets this
    assert Findings.model_validate_json(findings.model_dump_json()) == findings


def test_findings_reject_unknown_severity():
    with pytest.raises(ValidationError):
        Findings.model_validate({"summary": "s", "findings": [{"severity": "critical", "title": "t", "detail": "d"}]})


def test_verified_hidden_from_model_schema():
    finding_schema = Findings.model_json_schema()["$defs"]["Finding"]
    assert "verified" not in finding_schema["properties"]
    assert "verified" not in finding_schema.get("required", [])
