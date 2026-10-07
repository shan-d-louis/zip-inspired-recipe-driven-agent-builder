"""Quality gate for live runs that become the cached preset results.

A cached preset is the first thing a visitor sees, so it must show the
planted issues with verified quotes. Each check is a plain yes/no on the
run's findings; a run passes only if every check passes.
"""

import re
from collections.abc import Callable

from app import data_store
from app.engine.citations import normalize_source
from app.models import Finding, RunOutput


def _text(f: Finding) -> str:
    return " ".join(part for part in (f.title, f.detail, f.quote) if part).lower()


def _source(f: Finding) -> str:
    return normalize_source(f.source or "")


def _findings(out: RunOutput) -> list[Finding]:
    return out.findings.findings if out.findings else []


# Claims of absence ("policy not found") mean the model didn't search, or guessed.
ABSENCE_CLAIM = re.compile(r"not found|could ?n[o']t be (quoted|found)|no (relevant )?policy|contains no|does not contain")


def _called(out: RunOutput, tool: str) -> bool:
    return any(step.tool == tool for step in out.trace)


def common_checks(out: RunOutput) -> dict[str, bool]:
    """Rules for every cached preset."""
    findings = _findings(out)
    return {
        "ok, no error": out.ok and not out.error and out.findings is not None,
        "every finding quoted + verified": bool(findings) and all(f.quote and f.verified for f in findings),
        "no 'not found' claims": not any(ABSENCE_CLAIM.search(_text(f)) for f in findings),
    }


def renewal_checks(out: RunOutput) -> dict[str, bool]:
    verified = [f for f in _findings(out) if f.verified]
    return {
        "18% vs 7% cap": any(
            re.search(r"\b18\s?%|eighteen", _text(f)) and re.search(r"\b7\s?%|seven percent", _text(f))
            for f in verified),
        "auto-renewal / 60-day notice": any(
            _source(f) == "northbeam-renewal-order-form"
            and re.search(r"auto|automatic", _text(f)) and re.search(r"\b60\b|sixty", _text(f))
            for f in verified),
        "verified policy quote": any(_source(f).startswith("pol-") for f in verified),
        "company_context called": _called(out, "company_context"),
    }


def _is_record(f: Finding) -> bool:
    return _source(f).startswith("api_data")


def duplicate_checks(out: RunOutput) -> dict[str, bool]:
    verified = [f for f in _findings(out) if f.verified]
    record_quotes = [(f.quote or "").lower() for f in verified if _is_record(f)]
    request = data_store.get_request(out.request_id)
    new_vendor = data_store.get_vendor(request["vendor_id"]) if request else None
    if new_vendor is None:
        return {"request's vendor exists": False}
    # Names and addresses of the new vendor and of every vendor sharing its email domain.
    look_alikes = [v for v in data_store.list_vendors() if v["email_domain"] == new_vendor["email_domain"]]
    name_or_address = {v[field].lower() for v in look_alikes for field in ("name", "address")}
    return {
        # Name or address similarity, not just the shared domain: a quoted name/address line or value.
        "name or address quoted": any(
            re.search(r"\b(name|address)\s*:", q) or any(value in q for value in name_or_address)
            for q in record_quotes),
        "email domain quoted": any(new_vendor["email_domain"] in q for q in record_quotes),
        # The bank finding must quote the NEW vendor's account, not the existing vendor's.
        "new vendor's bank line quoted": any(new_vendor["bank_account"].lower() in q for q in record_quotes),
        "verified POL-06 quote": any(_source(f) == "pol-06" for f in verified),
        "company_context called": _called(out, "company_context"),
    }


def residency_checks(out: RunOutput) -> dict[str, bool]:
    verified = [f for f in _findings(out) if f.verified]
    return {
        "verified vendor quote (fjord-dpa)": any(_source(f) == "fjord-dpa" for f in verified),
        "verified policy quote (POL-03)": any(_source(f) == "pol-03" for f in verified),
    }


GATES: dict[str, Callable[[RunOutput], dict[str, bool]]] = {
    "renewal-check": renewal_checks,
    "duplicate-vendor-check": duplicate_checks,
    "data-residency-check": residency_checks,
}


def check(recipe_id: str, out: RunOutput) -> dict[str, bool]:
    """All checks for this recipe: the common ones plus its own (if it has any)."""
    specific = GATES.get(recipe_id)
    return {**common_checks(out), **(specific(out) if specific else {})}
