"""FastAPI app: GraphQL at /graphql, a health check, and the built frontend."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # before anything reads env vars

from fastapi import FastAPI  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from strawberry.fastapi import GraphQLRouter  # noqa: E402

from app.llm import log_provider_config  # noqa: E402
from app.schema import schema  # noqa: E402

STATIC_DIR = Path(__file__).parent / "static"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
# HTTP client libraries log full request URLs at INFO; those can include secrets
# (e.g. the email relay URL), so only their warnings are kept.
for noisy in ("httpx", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_provider_config()  # mode + provider names and model IDs, never keys
    yield


app = FastAPI(title="AgentBlocks", lifespan=lifespan)
app.include_router(GraphQLRouter(schema), prefix="/graphql")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


# Must be mounted last: it catches every path not handled above.
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
