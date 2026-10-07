"""Precompute the two presets on their default requests -> app/data/cached_runs.json.

Usage:  python scripts/precompute_presets.py [--only renewal-check] [--clear]
Runs in the current LLM_MODE (from env or .env) and records the mode and
provider in each entry. Entries from the other mode are kept.

Live mode has a quality gate (app/quality.py): up to 4 attempts per preset,
60-70 s apart, stopping at the first attempt that passes every check. A preset
is cached only if an attempt passed; otherwise its existing entry is left as it
was. Every attempt is logged to logs/.
"""

import argparse
import asyncio
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from app import data_store, quality  # noqa: E402
from app.engine.graph import run_recipe  # noqa: E402
from app.llm import llm_mode  # noqa: E402
from app.models import Recipe, RunOutput  # noqa: E402
from app.preset_cache import CACHE_PATH  # noqa: E402
from scripts.live_common import ROOT, log_attempt, print_table, utf8_console  # noqa: E402

MAX_ATTEMPTS = 4
GAP_SECONDS = (60, 70)  # between live calls, to stay under the free tier's per-minute limits


def _entry(preset: Recipe, mode: str, output: RunOutput) -> dict:
    return {
        "recipe_id": preset.id, "request_id": preset.default_request_id, "mode": mode,
        "provider": output.provider, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output": output.model_dump(mode="json"),
    }


async def _live_with_gate(preset: Recipe, rows: list[dict], first_call: bool) -> RunOutput | None:
    """Try up to MAX_ATTEMPTS live runs; return the first that passes the gate."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if not (first_call and attempt == 1):
            await asyncio.sleep(random.uniform(*GAP_SECONDS))
        output = await run_recipe(preset, preset.default_request_id)
        checks = quality.check(preset.id, output)
        log_attempt(preset.id, attempt, preset, preset.default_request_id, "live", output, checks)
        rows.append({"label": preset.id, "attempt": attempt, "output": output, "checks": checks})
        print_table(rows[-1:])
        if all(checks.values()):
            return output
    return None


def _read_entries(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["entries"] if path.exists() else []


def _write_entries(path: Path, entries: list[dict]) -> None:
    path.write_text(json.dumps({"entries": entries}, indent=2, ensure_ascii=False), encoding="utf-8")


def clear(path: Path, mode: str, recipe_ids: set[str]) -> int:
    """Remove these presets' entries for this mode. Returns how many were removed."""
    entries = _read_entries(path)
    kept = [e for e in entries if not (e["mode"] == mode and e["recipe_id"] in recipe_ids)]
    _write_entries(path, kept)
    return len(entries) - len(kept)


async def precompute(path: Path = CACHE_PATH, only: str | None = None, clear_first: bool = False) -> list[dict]:
    mode = llm_mode()
    presets = [p for p in data_store.load_presets() if only is None or p.id == only]
    if not presets:
        raise SystemExit(f"Unknown preset '{only}'.")
    if clear_first:
        removed = clear(path, mode, {p.id for p in presets})
        print(f"cleared {removed} existing {mode}-mode entr{'y' if removed == 1 else 'ies'} for: "
              f"{', '.join(p.id for p in presets)}")
    entries, rows = [], []
    for i, preset in enumerate(presets):
        if mode == "live":
            output = await _live_with_gate(preset, rows, first_call=(i == 0))
            if output is None:
                print(f"!! {preset.id}: no attempt passed the quality gate; its cache entry was not changed.")
                continue
        else:
            output = await run_recipe(preset, preset.default_request_id)
            if not output.ok:
                raise SystemExit(f"{preset.id} failed: {output.error}. Nothing was written.")
        entries.append(_entry(preset, mode, output))

    if mode == "live":
        print_table(rows)
    # Keep entries from the other mode, and same-mode entries for presets that weren't refreshed.
    refreshed = {e["recipe_id"] for e in entries}
    kept = [e for e in _read_entries(path) if e["mode"] != mode or e["recipe_id"] not in refreshed]
    _write_entries(path, kept + entries)
    return entries


if __name__ == "__main__":
    utf8_console()
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Precompute cached preset runs.")
    parser.add_argument("--only", metavar="PRESET_ID", help="only this preset, e.g. renewal-check")
    parser.add_argument("--clear", action="store_true",
                        help="first remove the selected presets' entries for the current mode")
    args = parser.parse_args()
    for entry in asyncio.run(precompute(only=args.only, clear_first=args.clear)):
        print(f"cached {entry['recipe_id']} on request {entry['request_id']} "
              f"(mode={entry['mode']}, provider={entry['provider']})")
