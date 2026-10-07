"""In-memory recipe store: the two presets (reseeded on every boot) plus up to
50 custom recipes. When full, the oldest custom recipe is evicted.
"""

import secrets
from collections import OrderedDict

from app import data_store
from app.models import Recipe

MAX_CUSTOM_RECIPES = 50


class RecipeStore:
    def __init__(self, max_custom: int = MAX_CUSTOM_RECIPES):
        self.max_custom = max_custom
        self._presets = {r.id: r for r in data_store.load_presets()}
        self._custom: OrderedDict[str, Recipe] = OrderedDict()  # oldest first

    def get(self, recipe_id: str) -> Recipe | None:
        return self._presets.get(recipe_id) or self._custom.get(recipe_id)

    def list(self) -> list[Recipe]:
        """Presets first, then custom recipes, newest first."""
        return [*self._presets.values(), *reversed(self._custom.values())]

    def create(self, **fields) -> Recipe:
        """Validate and store a custom recipe. Raises pydantic.ValidationError on bad input."""
        recipe = Recipe(id=f"custom-{secrets.token_urlsafe(6)}", is_preset=False, default_request_id=None, **fields)
        self._custom[recipe.id] = recipe
        while len(self._custom) > self.max_custom:
            self._custom.popitem(last=False)
        return recipe
