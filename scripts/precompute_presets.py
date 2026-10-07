"""Precompute the two presets on their default requests -> app/data/cached_runs.json.

Usage:  python scripts/precompute_presets.py
Runs in the current LLM_MODE (from env or .env) and records the mode and
provider in each entry. Entries from the same mode are replaced; entries from
the other mode are kept, so one file can hold both fake and live results.
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from app import data_store  # noqa: E402
from app.engine.graph import run_recipe  # noqa: E402
from app.llm import llm_mode  # noqa: E402
from app.preset_cache import CACHE_PATH  # noqa: E402


async def precompute(path: Path = CACHE_PATH) -> list[dict]:
    mode = llm_mode()
    entries = []
    for preset in data_store.load_presets():
        output = await run_recipe(preset, preset.default_request_id)
        if not output.ok:
            raise SystemExit(f"{preset.id} failed: {output.error}. Nothing was written.")
        entries.append({
            "recipe_id": preset.id, "request_id": preset.default_request_id, "mode": mode,
            "provider": output.provider, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "output": output.model_dump(mode="json"),
        })
    kept = []
    if path.exists():
        kept = [e for e in json.loads(path.read_text(encoding="utf-8"))["entries"] if e["mode"] != mode]
    path.write_text(json.dumps({"entries": kept + entries}, indent=2, ensure_ascii=False), encoding="utf-8")
    return entries


if __name__ == "__main__":
    load_dotenv()
    for entry in asyncio.run(precompute()):
        print(f"cached {entry['recipe_id']} on request {entry['request_id']} "
              f"(mode={entry['mode']}, provider={entry['provider']})")
