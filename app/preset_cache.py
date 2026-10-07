"""Precomputed preset runs (written by scripts/precompute_presets.py).

A cache entry is served only when:
  - the recipe is a preset and runs on its default request, and
  - the entry was produced in the same LLM_MODE the server is running in
    (so a fake-mode result is never shown as a live one).
Custom recipes always run live.
"""

import json
from functools import lru_cache
from pathlib import Path

from app.models import Recipe, RunOutput

CACHE_PATH = Path(__file__).parent / "data" / "cached_runs.json"


@lru_cache
def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["entries"]


def lookup(recipe: Recipe, request_id: str, mode: str, path: Path = CACHE_PATH) -> RunOutput | None:
    if not recipe.is_preset or request_id != recipe.default_request_id:
        return None
    for entry in _load(path):
        if entry["recipe_id"] == recipe.id and entry["request_id"] == request_id and entry["mode"] == mode:
            return RunOutput.model_validate(entry["output"])
    return None
