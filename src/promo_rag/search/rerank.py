"""Reranking with Cohere Rerank: a model that reads the question and each chunk together.

Vector search compares two embeddings made separately; a reranker sees the question and the
chunk side by side, so it judges relevance better but is too slow to run on every chunk.
Search uses it on hybrid's top results only.

POST https://api.cohere.com/v2/rerank  {model, query, documents}
  → results: [{index, relevance_score}], best first; scores are in [0, 1].
"""

import time
from collections.abc import Callable

import httpx

from promo_rag.config import get_settings
from promo_rag.llm import Pacer

RerankCall = Callable[[str, list[str]], list[float]]  # (query, documents) → one score each

URL = "https://api.cohere.com/v2/rerank"
# The first call plus up to 2 retries, only on rate limits and server errors.
ATTEMPTS = 3
RETRY_STATUS = {429, 500, 502, 503, 504}


class RerankError(RuntimeError):
    pass


def cohere_rerank(
    model: str | None = None,
    *,
    pacer: Pacer | None = None,
    transport: httpx.BaseTransport | None = None,  # tests pass a fake
    timeout_s: float = 30,
) -> RerankCall:
    settings = get_settings()
    key = settings.cohere_api_key.get_secret_value() if settings.cohere_api_key else ""
    if not key:  # unset, or the empty `COHERE_API_KEY=` line from .env.example
        raise RerankError("COHERE_API_KEY is not set in .env (a free trial key works)")
    model = model or settings.rerank_model
    client = httpx.Client(
        transport=transport,
        timeout=timeout_s,
        headers={"Authorization": f"Bearer {key}"},
    )

    def call(query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        body = {"model": model, "query": query, "documents": documents}
        try:
            response = _post(client, body, pacer)
        except httpx.HTTPError as e:  # timeouts, connection errors
            raise RerankError(f"Cohere rerank: {type(e).__name__}: {e}") from e
        return scores_in_input_order(response.json(), len(documents))

    return call


def scores_in_input_order(payload: dict, n_documents: int) -> list[float]:
    """Cohere returns results best first, by index; put the scores back in input order."""
    scores: list[float | None] = [None] * n_documents
    for result in payload.get("results", []):
        i = result["index"]
        if not 0 <= i < n_documents or scores[i] is not None:
            raise RerankError(f"bad or repeated index {i} for {n_documents} documents")
        scores[i] = float(result["relevance_score"])
    if None in scores:
        raise RerankError(
            f"sent {n_documents} documents, got {len(payload.get('results', []))} scores"
        )
    return scores


def _post(client: httpx.Client, body: dict, pacer: Pacer | None) -> httpx.Response:
    for attempt in range(1, ATTEMPTS + 1):
        if pacer:
            pacer.wait()
        response = client.post(URL, json=body)
        if response.status_code not in RETRY_STATUS or attempt == ATTEMPTS:
            break
        # A trial key allows 10 calls a minute; Retry-After says how long to wait, if sent.
        time.sleep(float(response.headers.get("retry-after", 2 * 3**attempt)))
    if response.status_code != 200:
        raise RerankError(f"Cohere rerank: HTTP {response.status_code}: {response.text[:300]}")
    return response
