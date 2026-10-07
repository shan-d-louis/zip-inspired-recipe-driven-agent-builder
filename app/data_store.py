"""Read-only access to the mock data in app/data.

Everything is loaded once from JSON/markdown and cached. Nothing here writes.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

from app.models import Recipe

DATA_DIR = Path(__file__).parent / "data"


class Chunk(BaseModel):
    doc_name: str  # e.g. "fjord-dpa" or "POL-03"
    chunk_id: str  # e.g. "fjord-dpa#2"
    text: str


@lru_cache
def _load_json(filename: str):
    return json.loads((DATA_DIR / filename).read_text(encoding="utf-8"))


# --- Requests and vendors ---


def list_requests() -> list[dict]:
    return _load_json("requests.json")


def get_request(request_id: str) -> dict | None:
    return next((r for r in list_requests() if r["id"] == request_id), None)


def list_vendors() -> list[dict]:
    return _load_json("vendors.json")


def get_vendor(vendor_id: str) -> dict | None:
    return next((v for v in list_vendors() if v["id"] == vendor_id), None)


def get_contract(vendor_id: str) -> dict | None:
    contracts = _load_json("contracts.json")["contracts"]
    return next((c for c in contracts if c["vendor_id"] == vendor_id), None)


def get_purchase_history(vendor_id: str) -> list[dict]:
    history = _load_json("contracts.json")["purchase_history"]
    return [p for p in history if p["vendor_id"] == vendor_id]


# --- Documents and policies ---


@lru_cache
def _read_document(name: str) -> str:
    return (DATA_DIR / "documents" / f"{name}.md").read_text(encoding="utf-8")


def get_documents(request_id: str) -> list[dict]:
    """The documents attached to a request: [{name, title, text}]."""
    request = get_request(request_id)
    if request is None:
        return []
    docs = []
    for name in request["documents"]:
        text = _read_document(name)
        title = text.splitlines()[0].lstrip("# ").strip()
        docs.append({"name": name, "title": title, "text": text})
    return docs


@lru_cache
def get_policies() -> list[dict]:
    """Split policies.md on '## POL-XX Title' headings: [{id, title, text}]."""
    text = (DATA_DIR / "policies.md").read_text(encoding="utf-8")
    policies = []
    for section in re.split(r"^## ", text, flags=re.MULTILINE)[1:]:
        heading, _, body = section.partition("\n")
        policy_id, _, title = heading.partition(" ")
        policies.append({"id": policy_id, "title": title.strip(), "text": body.strip()})
    return policies


# --- Preset recipes ---


def load_presets() -> list[Recipe]:
    return [Recipe.model_validate(p) for p in _load_json("presets.json")]


# --- Chunking (used by the retrieval tools) ---


def chunk_document(doc_name: str, text: str, max_words: int = 120) -> list[Chunk]:
    """Group whole paragraphs into chunks of about max_words words.

    Paragraphs are never split, so a short quote from one paragraph always
    sits inside a single chunk. That keeps citation checks simple later.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    groups: list[list[str]] = []
    current: list[str] = []
    count = 0
    for paragraph in paragraphs:
        words = len(paragraph.split())
        if current and count + words > max_words:
            groups.append(current)
            current, count = [], 0
        current.append(paragraph)
        count += words
    if current:
        groups.append(current)
    return [
        Chunk(doc_name=doc_name, chunk_id=f"{doc_name}#{i}", text="\n\n".join(group))
        for i, group in enumerate(groups, start=1)
    ]
