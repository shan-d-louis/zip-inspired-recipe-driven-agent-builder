"""Command-line runner for the engine.

Preset:  python -m app.engine.run --recipe renewal-check --request 1
Custom:  python -m app.engine.run --request 3 --name "Data Residency Check" \
           --tools document_retrieval company_context --format structured \
           --prompt "Flag any conflict between this vendor's data terms and our data residency policy"

Uses LLM_MODE from the environment or .env (default: fake, no keys needed).
"""

import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv

from app import data_store
from app.engine.graph import run_recipe
from app.llm import log_provider_config
from app.models import KNOWN_TOOLS, Recipe, RunOutput


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an AgentBlocks recipe against a mock purchase request.")
    parser.add_argument("--request", required=True, help="purchase request id (1-4)")
    parser.add_argument("--recipe", help="preset id, e.g. renewal-check or duplicate-vendor-check")
    parser.add_argument("--name", default="Custom agent")
    parser.add_argument("--prompt")
    parser.add_argument("--tools", nargs="+", choices=KNOWN_TOOLS)
    parser.add_argument("--format", choices=["markdown", "structured"], default="markdown")
    parser.add_argument("--no-citations", action="store_true")
    args = parser.parse_args(argv)
    if not args.recipe and not (args.prompt and args.tools):
        parser.error("give --recipe, or --prompt and --tools for a custom recipe")
    return args


def make_recipe(args: argparse.Namespace) -> Recipe:
    if args.recipe:
        presets = {p.id: p for p in data_store.load_presets()}
        if args.recipe not in presets:
            sys.exit(f"Unknown preset '{args.recipe}'. Choose from: {', '.join(presets)}")
        return presets[args.recipe]
    return Recipe(id="cli-custom", name=args.name, prompt=args.prompt, tools=args.tools,
                  output_format=args.format, include_citations=not args.no_citations)


def render(recipe: Recipe, output: RunOutput) -> str:
    mark = {True: "[verified]", False: "[NOT verified]", None: ""}
    lines = [f"Agent: {recipe.name}   Request: {output.request_id}   Provider: {output.provider or '-'}", "",
             "Trace:"]
    lines += [f"  {s.step}. {s.tool} ({s.duration_ms} ms): {s.summary}" for s in output.trace] or ["  (none)"]
    lines.append("")
    if not output.ok:
        lines.append(f"ERROR: {output.error}")
    elif output.findings:
        lines.append(f"Summary: {output.findings.summary}")
        for f in output.findings.findings:
            lines.append(f"- [{f.severity.upper()}] {f.title}: {f.detail}")
            if f.quote:
                lines.append(f'    "{f.quote}" ({f.source}) {mark[f.verified]}')
    else:
        lines.append(output.markdown or "")
        if output.citations:
            lines += ["", "Citations:"] + [f'  "{c.quote}" ({c.source}) {mark[c.verified]}' for c in output.citations]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):  # Windows pipes default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    recipe = make_recipe(args)
    log_provider_config()
    output = asyncio.run(run_recipe(recipe, args.request))
    print(render(recipe, output))
    return 0 if output.ok else 1


if __name__ == "__main__":
    sys.exit(main())
