"""What the API does, independent of GraphQL: create recipes and run them
behind the free-tier guardrails.
"""

from pydantic import ValidationError

from app import data_store, limits, preset_cache
from app.engine.graph import run_recipe
from app.llm import llm_mode
from app.models import RunOutput, RunRecord
from app.recipes import RecipeStore
from app.run_store import RunStore

recipes = RecipeStore()
run_store = RunStore()

RATE_LIMITED = ("You've run 6 agents in the last 10 minutes. Please wait a few minutes and try again. "
                "The two presets still work instantly.")


class UserError(Exception):
    """A problem with the visitor's input. Its message is shown as-is."""


def friendly_validation_message(error: ValidationError) -> str:
    """'name: String should have at most 60 characters' rather than a raw pydantic dump."""
    parts = []
    for issue in error.errors(include_url=False):
        field = ".".join(str(part) for part in issue["loc"]) or "input"
        parts.append(f"{field}: {issue['msg'].removeprefix('Value error, ')}")
    return "; ".join(parts)


def create_recipe(**fields):
    try:
        return recipes.create(**fields)
    except ValidationError as error:
        raise UserError(friendly_validation_message(error)) from None


def _not_run(recipe_id: str, request_id: str, recipe_name: str, error: str) -> RunRecord:
    """A refusal before anything ran: not stored, so it has no runId."""
    return RunRecord(run_id=None, recipe_name=recipe_name, cached=False, mode=llm_mode(),
                     output=RunOutput(ok=False, recipe_id=recipe_id, request_id=request_id, error=error))


async def run(recipe_id: str, request_id: str, ip: str) -> RunRecord:
    recipe = recipes.get(recipe_id)
    if recipe is None:
        return _not_run(recipe_id, request_id, "Unknown agent",
                        "That agent no longer exists (the demo resets now and then). Please create it again.")
    if data_store.get_request(request_id) is None:
        return _not_run(recipe_id, request_id, recipe.name, "That purchase request doesn't exist.")

    mode = llm_mode()
    cached = preset_cache.lookup(recipe, request_id, mode)
    if cached is not None:
        return run_store.put(RunRecord(run_id=None, recipe_name=recipe.name, cached=True, mode=mode, output=cached))

    if not limits.run_limiter.allow(ip):
        return _not_run(recipe_id, request_id, recipe.name, RATE_LIMITED)

    async with limits.run_semaphore:  # at most 2 runs at once; extra runs wait here
        output = await run_recipe(recipe, request_id)
    return run_store.put(RunRecord(run_id=None, recipe_name=recipe.name, cached=False, mode=mode, output=output))
