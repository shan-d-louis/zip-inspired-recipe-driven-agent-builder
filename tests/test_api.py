"""GraphQL API + guardrails, through the real FastAPI app (fake LLM mode)."""

import asyncio
import json
import logging
import time

import pytest
from fastapi.testclient import TestClient

from app import limits, preset_cache, service
from app.limits import RateLimiter
from app.main import app
from app.models import RunOutput, RunRecord
from app.recipes import RecipeStore
from app.run_store import RunStore

CREATE = """mutation($input: RecipeInput!) { createRecipe(input: $input) { id name tools isPreset } }"""
RUN = """mutation($r: ID!, $q: ID!) { runRecipe(recipeId: $r, requestId: $q) {
  runId ok cached mode provider error trace { step tool summary durationMs }
  findings { summary findings { severity source quote verified } } } }"""
RESIDENCY_INPUT = {
    "name": "Data Residency Check", "outputFormat": "STRUCTURED",
    "prompt": "Flag any conflict between this vendor's data terms and our data residency policy",
    "tools": ["document_retrieval", "company_context"],
}


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    """Every test starts with empty stores and limits."""
    monkeypatch.setenv("LLM_MODE", "fake")
    monkeypatch.setattr(service, "recipes", RecipeStore())
    monkeypatch.setattr(service, "run_store", RunStore())
    monkeypatch.setattr(limits, "run_limiter", RateLimiter(limits.RUNS_PER_IP, limits.RUN_WINDOW_SECONDS))
    monkeypatch.setattr(limits, "run_semaphore", asyncio.Semaphore(limits.MAX_CONCURRENT_RUNS))


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def gql(client, query, variables=None, ip="203.0.113.7"):
    response = client.post("/graphql", json={"query": query, "variables": variables or {}},
                           headers={"x-forwarded-for": ip})
    assert response.status_code == 200
    return response.json()


def create_residency(client, ip="203.0.113.7"):
    return gql(client, CREATE, {"input": RESIDENCY_INPUT}, ip)["data"]["createRecipe"]["id"]


# --- Basics ---


def test_healthz_and_startup_log(caplog):
    caplog.set_level(logging.INFO)
    with TestClient(app) as c:
        assert c.get("/healthz").json() == {"status": "ok"}
    assert "LLM mode: fake; providers in order: fake" in caplog.text


def test_queries(client):
    data = gql(client, "{ tools { name description } recipes { id isPreset } requests { id vendor amount } }")["data"]
    assert {t["name"] for t in data["tools"]} == {"document_retrieval", "api_data", "company_context"}
    assert [r["id"] for r in data["recipes"]] == ["renewal-check", "duplicate-vendor-check"]
    assert [r["vendor"] for r in data["requests"]][2] == "Fjord Cloud Storage"


# --- Create and run ---


def test_create_and_run_custom_recipe_live(client):
    recipe_id = create_residency(client)
    result = gql(client, RUN, {"r": recipe_id, "q": "3"})["data"]["runRecipe"]
    assert result["ok"] and not result["cached"] and result["mode"] == "fake"
    assert [s["tool"] for s in result["trace"]] == RESIDENCY_INPUT["tools"]
    assert all(f["verified"] for f in result["findings"]["findings"])
    assert len(result["runId"]) >= 20  # secrets.token_urlsafe(16)
    assert service.run_store.get(result["runId"]).output.ok


def test_custom_recipes_are_fetched_by_id_not_listed(client):
    recipe_id = create_residency(client)
    data = gql(client, '{ recipes { id } recipe(id: "%s") { name isPreset } missing: recipe(id: "nope") { id } }'
               % recipe_id)["data"]
    assert [r["id"] for r in data["recipes"]] == ["renewal-check", "duplicate-vendor-check"]
    assert data["recipe"] == {"name": "Data Residency Check", "isPreset": False}
    assert data["missing"] is None


@pytest.mark.parametrize("bad,field", [
    ({"name": "x" * 61}, "name"),
    ({"prompt": "x" * 601}, "prompt"),
    ({"tools": ["api_data", "document_retrieval", "company_context", "web_search"]}, "tools"),
    ({"tools": ["web_search"]}, "unknown tools"),
    ({"prompt": "   "}, "prompt"),
])
def test_create_recipe_rejects_bad_input_with_friendly_message(client, bad, field):
    body = gql(client, CREATE, {"input": {**RESIDENCY_INPUT, **bad}})
    message = body["errors"][0]["message"]
    assert field in message
    assert "Traceback" not in message and "pydantic" not in message.lower()


def test_unknown_recipe_or_request_is_friendly_and_not_stored(client):
    for recipe_id, request_id in [("custom-gone", "3"), ("renewal-check", "99")]:
        result = gql(client, RUN, {"r": recipe_id, "q": request_id})["data"]["runRecipe"]
        assert not result["ok"] and result["runId"] is None and result["error"]
    assert len(service.run_store) == 0


def test_unexpected_errors_are_masked(client, monkeypatch):
    async def explode(*args):
        raise RuntimeError("internal detail: /secret/path")

    monkeypatch.setattr(service, "run", explode)
    body = gql(client, RUN, {"r": "renewal-check", "q": "1"})
    assert body["errors"][0]["message"] == "Something went wrong. Please try again."
    assert "secret" not in json.dumps(body)


# --- Preset cache ---


def test_cached_preset_is_instant_labeled_and_stored(client):
    start = time.perf_counter()
    result = gql(client, RUN, {"r": "renewal-check", "q": "1"})["data"]["runRecipe"]
    assert time.perf_counter() - start < 1.0
    assert result["ok"] and result["cached"] and result["provider"] == "fake" and result["mode"] == "fake"
    assert service.run_store.get(result["runId"]).cached


def test_preset_on_another_request_runs_live(client):
    result = gql(client, RUN, {"r": "renewal-check", "q": "4"})["data"]["runRecipe"]
    assert result["ok"] and not result["cached"]


def test_cache_entry_is_served_only_in_its_own_mode(tmp_path):
    preset = service.recipes.get("renewal-check")
    output = RunOutput(ok=True, recipe_id="renewal-check", request_id="1", provider="gemini", markdown="cached")
    path = tmp_path / "cached_runs.json"
    path.write_text(json.dumps({"entries": [{"recipe_id": "renewal-check", "request_id": "1", "mode": "live",
                                             "provider": "gemini", "output": output.model_dump(mode="json")}]}))
    assert preset_cache.lookup(preset, "1", "fake", path) is None
    assert preset_cache.lookup(preset, "1", "live", path).provider == "gemini"
    assert preset_cache.lookup(preset, "4", "live", path) is None  # not its default request


def test_precompute_records_mode_and_provider_and_keeps_other_mode(tmp_path):
    from scripts.precompute_presets import precompute

    path = tmp_path / "cached_runs.json"
    path.write_text(json.dumps({"entries": [{"recipe_id": "renewal-check", "request_id": "1", "mode": "live",
                                             "provider": "gemini", "output": {}}]}))
    asyncio.run(precompute(path))
    entries = json.loads(path.read_text())["entries"]
    assert sorted((e["mode"], e["recipe_id"]) for e in entries) == [
        ("fake", "duplicate-vendor-check"), ("fake", "renewal-check"), ("live", "renewal-check")]
    assert all(e["provider"] == "fake" for e in entries if e["mode"] == "fake")


# --- Guardrails ---


def test_rate_limit_is_per_ip_friendly_and_skips_cached_presets(client):
    recipe_id = create_residency(client)
    for _ in range(limits.RUNS_PER_IP):
        assert gql(client, RUN, {"r": recipe_id, "q": "3"})["data"]["runRecipe"]["ok"]
    blocked = gql(client, RUN, {"r": recipe_id, "q": "3"})["data"]["runRecipe"]
    assert not blocked["ok"] and blocked["error"] == service.RATE_LIMITED and blocked["runId"] is None
    assert gql(client, RUN, {"r": "renewal-check", "q": "1"})["data"]["runRecipe"]["cached"]  # still works
    assert gql(client, RUN, {"r": recipe_id, "q": "3"}, ip="198.51.100.9")["data"]["runRecipe"]["ok"]  # other IP


def test_rate_limiter_window_slides_and_prunes_idle_ips():
    now = [0.0]
    limiter = RateLimiter(2, 60, clock=lambda: now[0])
    assert limiter.allow("a") and limiter.allow("a") and not limiter.allow("a")
    limiter.allow("b")
    now[0] = 61
    assert limiter.allow("a")
    assert len(limiter) == 1  # "b" had no events left in the window, so it was forgotten


@pytest.mark.parametrize("headers,expected", [
    ({"x-forwarded-for": "6.6.6.6, 203.0.113.7"}, "203.0.113.7"),  # left entry is client-supplied (fakeable)
    ({"x-forwarded-for": "203.0.113.7"}, "203.0.113.7"),
    ({}, "testclient"),  # no proxy: the socket address
])
def test_client_ip_uses_rightmost_forwarded_entry(headers, expected):
    from starlette.requests import Request

    scope = {"type": "http", "client": ("testclient", 50000),
             "headers": [(k.encode(), v.encode()) for k, v in headers.items()]}
    assert limits.client_ip(Request(scope)) == expected


async def test_busy_after_waiting_for_a_slot_and_quota_not_used(monkeypatch):
    monkeypatch.setattr(limits, "SLOT_WAIT_SECONDS", 0.05)
    recipe = service.create_recipe(name="x", prompt="p", tools=["api_data"], output_format="markdown")
    for _ in range(limits.MAX_CONCURRENT_RUNS):  # occupy every slot
        await limits.run_semaphore.acquire()
    result = await service.run(recipe.id, "3", "10.0.0.1")
    assert not result.output.ok and result.output.error == service.BUSY and result.run_id is None
    assert len(limits.run_limiter) == 0  # the busy attempt did not count against the visitor


async def test_at_most_two_runs_at_once(monkeypatch):
    active, peak = 0, 0

    async def slow_run(recipe, request_id):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.05)
        active -= 1
        return RunOutput(ok=True, recipe_id=recipe.id, request_id=request_id)

    monkeypatch.setattr(service, "run_recipe", slow_run)
    recipe = service.create_recipe(**{"name": "x", "prompt": "p", "tools": ["api_data"], "output_format": "markdown"})
    results = await asyncio.gather(*(service.run(recipe.id, "3", f"10.0.0.{i}") for i in range(5)))
    assert all(r.output.ok for r in results)  # extra runs waited instead of failing
    assert peak == limits.MAX_CONCURRENT_RUNS


# --- Stores ---


def test_run_store_expires_after_ttl_and_evicts_oldest():
    now = [0.0]
    store = RunStore(ttl_seconds=1800, max_entries=2, clock=lambda: now[0])
    record = RunRecord(run_id=None, recipe_name="r", cached=False, mode="fake",
                       output=RunOutput(ok=True, recipe_id="r", request_id="1"))
    first, second, third = (store.put(record) for _ in range(3))
    assert store.get(first.run_id) is None  # evicted: cap is 2
    assert store.get(third.run_id) is not None
    now[0] = 1801
    assert store.get(second.run_id) is None and len(store) == 0  # expired


def test_custom_recipes_capped_oldest_evicted_presets_kept():
    store = RecipeStore(max_custom=3)
    ids = [store.create(name=f"r{i}", prompt="p", tools=["api_data"], output_format="markdown").id for i in range(4)]
    assert store.get(ids[0]) is None and all(store.get(i) for i in ids[1:])
    assert store.get("renewal-check").is_preset
