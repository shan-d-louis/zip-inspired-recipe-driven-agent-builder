"""GraphQL schema (Strawberry). A thin layer: it converts between GraphQL
types and app models, and delegates the work to app.service.
"""

import strawberry
from strawberry.extensions import MaskErrors
from strawberry.types import Info

from app import data_store, service
from app.limits import client_ip
from app.models import OutputFormat as OutputFormatModel
from app.models import Recipe as RecipeModel
from app.models import RunRecord
from app.tools import TOOL_REGISTRY

OutputFormat = strawberry.enum(OutputFormatModel, name="OutputFormat")


@strawberry.type
class ToolInfo:
    name: str
    description: str


@strawberry.type
class Recipe:
    id: strawberry.ID
    name: str
    prompt: str
    tools: list[str]
    output_format: OutputFormat
    include_citations: bool
    is_preset: bool
    default_request_id: str | None

    @staticmethod
    def from_model(recipe: RecipeModel) -> "Recipe":
        return Recipe(**recipe.model_dump())


@strawberry.type
class PurchaseRequestSummary:
    id: strawberry.ID
    title: str
    type: str
    vendor: str
    amount: float
    currency: str


@strawberry.type
class Finding:
    severity: str
    title: str
    detail: str
    source: str | None
    quote: str | None
    verified: bool | None


@strawberry.type
class FindingsResult:
    summary: str
    findings: list[Finding]


@strawberry.type
class Citation:
    source: str
    quote: str
    verified: bool


@strawberry.type
class TraceStep:
    step: int
    tool: str
    summary: str
    duration_ms: int


@strawberry.type
class RunResult:
    run_id: strawberry.ID | None  # null when nothing ran (e.g. rate limited)
    ok: bool
    cached: bool
    provider: str | None
    mode: str
    recipe_name: str
    markdown: str | None
    findings: FindingsResult | None
    citations: list[Citation]
    trace: list[TraceStep]
    error: str | None

    @staticmethod
    def from_record(record: RunRecord) -> "RunResult":
        out = record.output
        findings = None
        if out.findings:
            findings = FindingsResult(summary=out.findings.summary,
                                      findings=[Finding(**f.model_dump()) for f in out.findings.findings])
        return RunResult(
            run_id=record.run_id, ok=out.ok, cached=record.cached, provider=out.provider, mode=record.mode,
            recipe_name=record.recipe_name, markdown=out.markdown, findings=findings,
            citations=[Citation(**c.model_dump()) for c in out.citations],
            trace=[TraceStep(**s.model_dump()) for s in out.trace], error=out.error,
        )


@strawberry.input
class RecipeInput:
    name: str
    prompt: str
    tools: list[str]
    output_format: OutputFormat
    include_citations: bool = True


@strawberry.type
class Query:
    @strawberry.field
    def tools(self) -> list[ToolInfo]:
        return [ToolInfo(name=t.name, description=t.description) for t in TOOL_REGISTRY.values()]

    @strawberry.field(description="The preset recipes. Custom recipes are fetched by id.")
    def recipes(self) -> list[Recipe]:
        return [Recipe.from_model(r) for r in service.recipes.presets()]

    @strawberry.field(description="One recipe (preset or custom) by id; null if unknown or evicted.")
    def recipe(self, id: strawberry.ID) -> Recipe | None:
        found = service.recipes.get(str(id))
        return Recipe.from_model(found) if found else None

    @strawberry.field
    def requests(self) -> list[PurchaseRequestSummary]:
        return [
            PurchaseRequestSummary(id=r["id"], title=r["title"], type=r["type"], amount=r["amount"],
                                   currency=r["currency"], vendor=data_store.get_vendor(r["vendor_id"])["name"])
            for r in data_store.list_requests()
        ]


@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_recipe(self, input: RecipeInput) -> Recipe:
        recipe = service.create_recipe(
            name=input.name, prompt=input.prompt, tools=input.tools,
            output_format=input.output_format, include_citations=input.include_citations,
        )
        return Recipe.from_model(recipe)

    @strawberry.mutation
    async def run_recipe(self, info: Info, recipe_id: strawberry.ID, request_id: strawberry.ID) -> RunResult:
        record = await service.run(str(recipe_id), str(request_id), client_ip(info.context["request"]))
        return RunResult.from_record(record)


def _hide_unexpected(error) -> bool:
    """Show UserError messages; replace everything else with a generic message."""
    return not isinstance(error.original_error, service.UserError)


schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    extensions=[lambda: MaskErrors(should_mask_error=_hide_unexpected,
                                   error_message="Something went wrong. Please try again.")],
)
