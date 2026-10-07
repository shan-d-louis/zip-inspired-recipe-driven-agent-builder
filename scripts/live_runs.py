"""Run recipes live, one after another, and report quality checks, tokens and time.

Usage:  python scripts/live_runs.py data-residency-check data-residency-check renewal-check duplicate-vendor-check
        python scripts/live_runs.py --gap 90 renewal-check
Names are preset ids or 'data-residency-check' (the custom agent built in the demo).
Nothing is cached. Every run is logged to logs/ (gitignored).
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from scripts.live_common import ROOT, log_attempt, print_table, utf8_console  # noqa: E402

DATA_RESIDENCY = {
    "id": "data-residency-check", "name": "Data Residency Check", "output_format": "structured",
    "prompt": "Flag any conflict between this vendor's data terms and our data residency policy",
    "tools": ["document_retrieval", "company_context"],
}


async def main(names: list[str], gap: float) -> int:
    from app import data_store, quality
    from app.engine.graph import run_recipe
    from app.models import Recipe

    recipes = {p.id: (p, p.default_request_id) for p in data_store.load_presets()}
    recipes["data-residency-check"] = (Recipe(**DATA_RESIDENCY), "3")
    rows = []
    for i, name in enumerate(names):
        if i:
            await asyncio.sleep(gap)
        recipe, request_id = recipes[name]
        output = await run_recipe(recipe, request_id)
        checks = quality.check(recipe.id, output)
        attempt = sum(1 for r in rows if r["label"] == name) + 1
        path = log_attempt(name, attempt, recipe, request_id, "live", output, checks)
        rows.append({"label": name, "attempt": attempt, "output": output, "checks": checks})
        print(f"\n=== {name} #{attempt}: ok={output.ok} provider={output.provider}  (log: {path.relative_to(ROOT)})")
        for step in output.trace:
            print(f"  trace {step.step}. {step.tool}: {step.summary}")
        if output.error:
            print("  ERROR:", output.error)
        for f in output.findings.findings if output.findings else []:
            mark = {True: "VERIFIED", False: "NOT verified", None: "-"}[f.verified]
            print(f"  - [{f.severity}] {f.title}\n      {f.detail}\n      {mark} {f.source}: {f.quote!r}")
    print_table(rows)
    return 0


if __name__ == "__main__":
    utf8_console()
    load_dotenv(ROOT / ".env")
    os.environ["LLM_MODE"] = "live"
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    parser.add_argument("--gap", type=float, default=60)
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.names, args.gap)))
