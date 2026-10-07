# AgentBlocks

A small, phone-first demo of a recipe-driven agent builder. Pick a few tools ("blocks"), write one
sentence of instructions, and run the agent against a fictional company's purchase requests. You
see the result and a trace of every tool call the agent made. No code per agent: an agent is just a
recipe (name, prompt, tools, output format) that runs on one shared four-step engine.

Inspired by a public engineering blog post on composable custom agents:
https://zip.com/engineering-blog/custom-agents-composable-ai-platform. This is my own small take on
the pattern, not affiliated with that company. All data (Maple Robotics, its vendors and documents)
is fictional.

## Run it locally

Backend (Python 3.11+):

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # Windows (use .venv/bin/python on macOS/Linux)
cp .env.example .env                                         # LLM_MODE=fake works with no keys
.venv/Scripts/python -m uvicorn app.main:app --port 8000     # open http://localhost:8000
```

Frontend (only needed to change the UI; the built app is committed in `app/static/`):

```bash
cd frontend && npm install && npm run dev                    # http://localhost:5173, proxies to :8000
npm run build                                                # type-check, then rebuild app/static/
```

Command line, no UI:

```bash
.venv/Scripts/python -m app.engine.run --recipe renewal-check --request 1
```

Tests (offline, scripted model, never call a real API): `.venv/Scripts/python -m pytest -q`.
Live smoke test against real providers (needs keys): `.venv/Scripts/python scripts/live_smoke.py`.

## LLM providers

- `LLM_MODE=fake` (default) uses a deterministic scripted model: no keys, no network. The UI labels
  these results "Demo mode (scripted model)".
- `LLM_MODE=live` uses every provider that has both a key and a model ID set, in order: Gemini,
  then Groq. Model IDs come from env vars because free-tier model names change.
- **This demo runs on Groq only.** The provider layer supports Gemini as a primary with Groq as a
  fallback (rate limits, timeouts, 5xx and key errors move on to the next provider), and that logic
  is covered by offline tests with fake providers. A real Gemini-to-Groq failover has not been
  verified against the live APIs in this deployment.
- With a single provider, a rate limit (HTTP 429) is retried once after the server's `Retry-After`
  delay (capped at 5 s), inside the overall run timeout (`RUN_TIMEOUT_S`, default 25 s).
- Each run logs its cost (LLM calls and token counts when the API reports them), never prompts or
  document text.

## Layout

```
app/            FastAPI + GraphQL (schema.py), engine/ (the four-node LangGraph), tools/ (the blocks),
                llm.py (providers), service.py + limits.py (guardrails), data/ (fictional mock data)
frontend/       Vite + React + TypeScript + Tailwind source; built into app/static/
scripts/        precompute_presets.py (cached preset runs), live_smoke.py
tests/          pytest suite (offline)
```
