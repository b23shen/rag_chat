"""Embeddings, shared by ingestion (chunks) and search (questions).

gemini-embedding-2 takes the task as a text prefix (not a task_type parameter):
  documents   "title: {title} | text: {text}"
  questions   "task: search result | query: {query}"
The formats live in IndexConfig, so changing one changes the corpus fingerprint.

Every vector is cached by (model, dims, sha256 of the exact input). Rebuilding a corpus
version then only calls the API for text that changed.
"""

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from google.genai import types
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from promo_rag.db.models import EmbeddingCacheEntry
from promo_rag.llm import Pacer, gemini_client

EmbedCall = Callable[[list[str]], list[list[float]]]


class EmbedError(RuntimeError):
    pass


def input_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_input(fmt: str, title: str, text: str) -> str:
    return fmt.format(title=title, text=text)


def query_input(fmt: str, query: str) -> str:
    return fmt.format(query=query)


class EmbeddingCache(Protocol):
    def get_many(self, model: str, dims: int, hashes: Sequence[str]) -> dict[str, list[float]]: ...
    def put_many(self, model: str, dims: int, vectors: dict[str, list[float]]) -> None: ...


class DbEmbeddingCache:
    """The embedding_cache table. Writes join the caller's transaction (the caller commits)."""

    def __init__(self, session: Session):
        self.session = session

    def get_many(self, model, dims, hashes):
        rows = self.session.execute(
            select(EmbeddingCacheEntry.input_hash, EmbeddingCacheEntry.embedding).where(
                EmbeddingCacheEntry.model == model,
                EmbeddingCacheEntry.dims == dims,
                EmbeddingCacheEntry.input_hash.in_(list(hashes)),
            )
        ).all()
        return {h: [float(x) for x in v] for h, v in rows}

    def put_many(self, model, dims, vectors):
        if vectors:
            rows = [
                {"model": model, "dims": dims, "input_hash": h, "embedding": v}
                for h, v in vectors.items()
            ]
            self.session.execute(insert(EmbeddingCacheEntry).values(rows).on_conflict_do_nothing())


@dataclass
class EmbedResult:
    vectors: list[list[float]]  # in the same order as the inputs
    from_cache: int
    from_api: int
    api_calls: int
    fetched: set[str]  # input hashes that came from the API (not the cache)


def embed_inputs(
    inputs: list[str],
    *,
    model: str,
    dims: int,
    cache: EmbeddingCache,
    call: EmbedCall,
    pacer: Pacer | None = None,
    batch_size: int = 50,
) -> EmbedResult:
    hashes = [input_hash(t) for t in inputs]
    found = cache.get_many(model, dims, set(hashes))
    missing = list(dict.fromkeys(h for h in hashes if h not in found))  # unique, in order
    text_for = dict(zip(hashes, inputs, strict=True))

    api_calls = 0
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        if pacer:
            pacer.wait()
        vectors = call([text_for[h] for h in batch])
        api_calls += 1
        # One vector per input, or something went wrong (e.g. inputs merged into one).
        if len(vectors) != len(batch):
            raise EmbedError(f"sent {len(batch)} texts, got {len(vectors)} vectors back")
        if bad := [len(v) for v in vectors if len(v) != dims]:
            raise EmbedError(f"expected {dims} dims, got {bad[0]}")
        new = dict(zip(batch, vectors, strict=True))
        cache.put_many(model, dims, new)
        found.update(new)

    was_missing = set(missing)
    return EmbedResult(
        vectors=[found[h] for h in hashes],
        from_cache=sum(1 for h in hashes if h not in was_missing),
        from_api=len(missing),
        api_calls=api_calls,
        fetched=was_missing,
    )


def gemini_embed(model: str, dims: int) -> EmbedCall:
    """The real API call. One Content per text: a plain list[str] would be read as the parts
    of ONE input and come back as a single blended vector."""
    client = gemini_client()

    def call(texts: list[str]) -> list[list[float]]:
        response = client.models.embed_content(
            model=model,
            contents=[types.Content(parts=[types.Part(text=t)]) for t in texts],
            config=types.EmbedContentConfig(output_dimensionality=dims),
        )
        return [list(e.values) for e in response.embeddings]

    return call


def norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))
