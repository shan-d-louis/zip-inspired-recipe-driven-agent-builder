"""FastAPI app: GraphQL at /graphql, a health check, and the built frontend."""

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # before anything reads env vars

from fastapi import FastAPI  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from strawberry.fastapi import GraphQLRouter  # noqa: E402

from app.llm import log_provider_config  # noqa: E402
from app.log_config import configure_logging  # noqa: E402
from app.schema import schema  # noqa: E402

STATIC_DIR = Path(__file__).parent / "static"

configure_logging()


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
