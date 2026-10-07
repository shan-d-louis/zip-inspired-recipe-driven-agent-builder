"""Helpers shared by the live scripts: UTF-8 output, per-attempt logs, a results table.

Logs go to logs/ (gitignored). They hold the recipe and the run output only:
no API keys, no system prompts, no environment.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.models import Recipe, RunOutput

ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "logs"


def utf8_console() -> None:
    """Windows consoles and pipes default to cp1252; model output can contain any character."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def log_attempt(label: str, attempt: int, recipe: Recipe, request_id: str, mode: str,
                output: RunOutput, checks: dict[str, bool]) -> Path:
    """Write one live attempt to logs/<time>-<label>-<attempt>.json and return the path."""
    LOGS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = LOGS_DIR / f"{stamp}-{label}-attempt{attempt}.json"
    record = {
        "time": stamp, "label": label, "attempt": attempt, "mode": mode, "request_id": request_id,
        "recipe": recipe.model_dump(mode="json", include={"id", "name", "prompt", "tools", "output_format"}),
        "checks": checks, "passed": all(checks.values()),
        "output": output.model_dump(mode="json"),
    }
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def print_table(rows: list[dict]) -> None:
    """rows: label, attempt, output (RunOutput), checks (dict)."""
    print(f"\n{'run':28} {'try':>3} {'ok':>3} {'calls':>5} {'in':>6} {'out':>6} {'total':>6} {'secs':>5}  checks")
    for row in rows:
        out: RunOutput = row["output"]
        usage = out.usage
        tin = usage.input_tokens if usage and usage.input_tokens is not None else 0
        tout = usage.output_tokens if usage and usage.output_tokens is not None else 0
        failed = [name for name, ok in row["checks"].items() if not ok]
        verdict = "PASS" if not failed else "FAIL: " + "; ".join(failed)
        print(f"{row['label']:28} {row['attempt']:>3} {'yes' if out.ok else 'no':>3} "
              f"{usage.llm_calls if usage else 0:>5} {tin:>6} {tout:>6} {tin + tout:>6} "
              f"{(usage.duration_ms / 1000) if usage else 0:>5.1f}  {verdict}")
