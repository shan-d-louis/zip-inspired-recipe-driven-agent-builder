"""The precompute quality gate, checked against hand-made run outputs (no LLM)."""

import json
from pathlib import Path

import pytest

from app import quality
from app.models import Finding, Findings, RunOutput, TraceStep

DEFAULT_REQUEST = {"renewal-check": "1", "duplicate-vendor-check": "2"}
STALE_DUPLICATE = Path(__file__).parent / "fixtures" / "stale_duplicate_vendor_run.json"


def finding(source, quote, verified=True, title="t", detail="d"):
    return Finding(severity="high", title=title, detail=detail, source=source, quote=quote, verified=verified)


def output(recipe_id, findings, tools=("api_data", "document_retrieval", "company_context"), ok=True):
    trace = [TraceStep(step=i, tool=t, summary="s", duration_ms=1) for i, t in enumerate(tools, start=1)]
    return RunOutput(ok=ok, recipe_id=recipe_id, request_id=DEFAULT_REQUEST[recipe_id], trace=trace,
                     findings=Findings(summary="s", findings=findings), error=None if ok else "boom")


RENEWAL_PASS = [
    finding("northbeam-master-agreement#4", "annual fee increases shall not exceed seven percent (7%)",
            detail="The renewal is 18% above last year, over the 7% cap."),
    finding("northbeam-renewal-order-form#5", "automatically renew for successive twelve (12) month terms",
            detail="Auto-renewal with only 60 days' notice."),
    finding("POL-02", "Any auto-renewal clause must be flagged to the requester"),
]
NAME = finding("api_data#vendor", "name: Northbeam Analytic Inc.", title="New vendor: name")
DOMAIN = finding("api_data#existing_vendors-1", "email_domain: northbeam-analytics.example",
                 title="Existing vendor: same email domain")
BANK = finding("api_data#vendor", "bank_account: Demo Bank C (fictional) ****2290", title="New vendor: bank account")
POLICY = finding("POL-06", "A possible duplicate must be reviewed before setup.", title="Duplicate vendor policy")
DUPLICATE_PASS = [NAME, DOMAIN, BANK, POLICY]


def failed(recipe_id, findings, **kwargs) -> set[str]:
    checks = quality.check(recipe_id, output(recipe_id, findings, **kwargs))
    return {name for name, ok in checks.items() if not ok}


def test_passing_runs_pass_every_check():
    assert failed("renewal-check", RENEWAL_PASS) == set()
    assert failed("duplicate-vendor-check", DUPLICATE_PASS) == set()


# --- Rules for every preset ---


@pytest.mark.parametrize("recipe_id,passing", [("renewal-check", RENEWAL_PASS), ("duplicate-vendor-check", DUPLICATE_PASS)])
@pytest.mark.parametrize("bad,rule", [
    (finding(None, None, verified=None, title="Summary note"), "every finding quoted + verified"),        # no quote
    (finding("POL-02", "an invented line that is in no policy", verified=False), "every finding quoted + verified"),
    (finding("POL-01", "Software and service renewals must not increase", detail="The policy was not found."),
     "no 'not found' claims"),
    (finding("POL-01", "Software and service renewals must not increase", title="No policy applies"),
     "no 'not found' claims"),
    (finding("POL-01", "Software and service renewals must not increase", detail="It could not be quoted."),
     "no 'not found' claims"),
])
def test_rules_for_every_preset(recipe_id, passing, bad, rule):
    assert rule in failed(recipe_id, [*passing, bad])


def test_failed_run_never_passes():
    assert "ok, no error" in failed("renewal-check", RENEWAL_PASS, ok=False)


# --- Renewal Check ---


@pytest.mark.parametrize("findings,rule", [
    (RENEWAL_PASS[1:], "18% vs 7% cap"),
    ([RENEWAL_PASS[0], RENEWAL_PASS[2]], "auto-renewal / 60-day notice"),
    (RENEWAL_PASS[:2], "verified policy quote"),
])
def test_renewal_rules(findings, rule):
    assert rule in failed("renewal-check", findings)


def test_renewal_requires_company_context():
    assert "company_context called" in failed("renewal-check", RENEWAL_PASS, tools=("document_retrieval",))


# --- Duplicate Vendor Check ---


@pytest.mark.parametrize("findings,rule", [
    ([DOMAIN, BANK, POLICY], "name or address quoted"),       # domain alone is not enough
    ([NAME, BANK, POLICY], "email domain quoted"),
    ([NAME, DOMAIN, POLICY], "new vendor's bank line quoted"),
    # The existing vendor's account (****4471) doesn't count: it must be the new vendor's (****2290).
    ([NAME, DOMAIN, finding("api_data#existing_vendors-1", "bank_account: Demo Bank A (fictional) ****4471"), POLICY],
     "new vendor's bank line quoted"),
    ([NAME, DOMAIN, BANK], "verified POL-06 quote"),
    ([NAME, DOMAIN, BANK, finding("POL-08", "Any vendor that will store or process customer data")], "verified POL-06 quote"),
])
def test_duplicate_rules(findings, rule):
    assert rule in failed("duplicate-vendor-check", findings)


def test_duplicate_accepts_a_bare_address_value():
    address = finding("api_data#vendor", "410 King Street West, Suite 810, Toronto, ON M5V 1K2")
    assert failed("duplicate-vendor-check", [address, DOMAIN, BANK, POLICY]) == set()


def test_duplicate_requires_company_context():
    assert "company_context called" in failed("duplicate-vendor-check", DUPLICATE_PASS, tools=("api_data",))


def test_the_old_cached_duplicate_vendor_run_now_fails():
    """The live entry cached before this gate (saved verbatim from cached_runs.json)."""
    stale = RunOutput.model_validate(json.loads(STALE_DUPLICATE.read_text(encoding="utf-8"))["output"])
    checks = quality.check("duplicate-vendor-check", stale)
    assert {name for name, ok in checks.items() if not ok} == {
        "every finding quoted + verified",  # the "policy not found" finding had no quote
        "no 'not found' claims",
        "name or address quoted",           # it quoted only email domains and bank accounts
        "verified POL-06 quote",
        "company_context called",
    }
