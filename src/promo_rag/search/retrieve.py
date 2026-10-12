"""Search a corpus version's chunks: vector, keyword, hybrid, or rerank.

  vector   cosine distance between the question's embedding and each chunk's (pgvector)
  keyword  Postgres full-text search on the segmented text (chunks.tsv), any word may match
  hybrid   both lists merged with reciprocal rank fusion (RRF): each chunk scores
           sum(weight / (rrf_k + rank)) over the lists it appears in. Ranks, not scores, so
           the two very different score scales never need to be compared.
  rerank   vector's top results (or hybrid's), put in a new order by a reranker model
           (search/rerank.py). The default: on the golden set it ranks as well as vector,
           and its scores can tell "nothing relevant" apart, which vector's can't.

On the golden set, keyword search made hybrid worse than vector alone (it helped 1 question
and hurt 6 to 13; evals/results/), so keyword and hybrid stay as modes for experiments only.

The question is embedded and segmented with the settings the version was built with (its
config), so a question and the chunks it is compared with are always processed the same way.
"""

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import text
from sqlalchemy.orm import Session

from promo_rag.search.embed import DbEmbeddingCache, embed_inputs, gemini_embed, query_input
from promo_rag.search.rerank import RerankCall, RerankError, cohere_rerank
from promo_rag.search.segment import segment, word_list_version

Mode = Literal["vector", "keyword", "hybrid", "rerank"]
EmbedQuery = Callable[[str], list[float]]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchSettings:
    """The knobs the search eval tunes. The defaults are what search uses."""

    # Each list's top N goes into the fusion, and the top N candidates into the reranker.
    candidates: int = 20
    # Where the reranker's candidates come from.
    rerank_from: Literal["vector", "hybrid"] = "vector"
    # The usual RRF constant; a larger value flattens the gap between rank 1 and rank 2.
    rrf_k: int = 60
    # How much a vector rank counts in RRF compared with a keyword rank (1 = equal).
    vector_weight: float = 1.0
    # Keyword search ignores question words found in more than this share of the version's
    # chunks (2026, 活動, 的 …). Postgres ranking has no IDF, so without this a chunk full of
    # common words outranks one that holds the rare word the question is about. 1.0 = off.
    max_word_share: float = 0.2


DEFAULT_SETTINGS = SearchSettings()


class SearchError(RuntimeError):
    pass


@dataclass
class Version:
    id: uuid.UUID
    config: dict
    n_chunks: int


@dataclass
class Hit:
    rank: int  # 1 = best
    chunk_id: uuid.UUID
    source_url: str
    title: str
    text: str
    # vector: cosine similarity; keyword: ts_rank; hybrid: RRF score; rerank: relevance 0-1
    # (or, if the reranker failed, the candidates' own scores: see `reranked`)
    score: float
    vector_rank: int | None = None  # when fused with keyword: the chunk's rank in each list
    keyword_rank: int | None = None
    candidate_rank: int | None = None  # rerank only: the chunk's rank before reranking
    reranked: bool = False  # rerank only: False if the reranker failed and order was kept


def search(
    session: Session,
    query: str,
    *,
    mode: Mode = "rerank",
    k: int = 5,
    version_id: uuid.UUID | None = None,
    embed_query: EmbedQuery | None = None,
    rerank: RerankCall | None = None,
    settings: SearchSettings = DEFAULT_SETTINGS,
    fallback: bool = True,
) -> list[Hit]:
    """The top k chunks for a question. Reads the active version unless one is given.

    If the reranker fails (no key, a trial key's limit, an outage), rerank keeps its
    candidates' order and marks the hits reranked=False, so search keeps working; pass
    fallback=False to raise instead (the eval does, so a failure can't pass as a result).

    A new question embedding goes into the embedding cache in the caller's transaction;
    commit to keep it.
    """
    version = load_version(session, version_id)
    fused = mode == "hybrid" or (mode == "rerank" and settings.rerank_from == "hybrid")
    n = max(k, settings.candidates) if mode in ("hybrid", "rerank") else k
    vector_ids, keyword_ids = [], []
    if mode != "keyword":
        embed = embed_query or cached_query_embedder(session, version)
        vector_ids = _vector(session, version, embed(query), n)
    if mode == "keyword" or fused:
        keyword_ids = _keyword(session, version, query, n, settings.max_word_share)

    if fused:
        lists = [[i for i, _ in vector_ids], [i for i, _ in keyword_ids]]
        ranked = rrf(lists, settings.rrf_k, [settings.vector_weight, 1.0])[:n]
    else:
        ranked = keyword_ids if mode == "keyword" else vector_ids
    rows = _chunk_rows(session, [chunk_id for chunk_id, _ in ranked])
    before = {chunk_id: r for r, (chunk_id, _) in enumerate(ranked, 1)}
    reranked = False
    if mode == "rerank":
        try:
            scores = (rerank or cohere_rerank())(query, [rows[c].text for c, _ in ranked])
        except RerankError as e:
            if not fallback:
                raise
            log.warning("rerank failed; keeping the %s order: %s", settings.rerank_from, e)
        else:
            reranked = True
            rescored = [(c, s) for (c, _), s in zip(ranked, scores, strict=True)]
            ranked = sorted(rescored, key=lambda cs: (-cs[1], before[cs[0]]))
    ranked = ranked[:k]

    v_rank = {chunk_id: r for r, (chunk_id, _) in enumerate(vector_ids, 1)}
    kw_rank = {chunk_id: r for r, (chunk_id, _) in enumerate(keyword_ids, 1)}
    return [
        Hit(
            rank=r,
            chunk_id=chunk_id,
            source_url=rows[chunk_id].source_url,
            title=rows[chunk_id].title,
            text=rows[chunk_id].text,
            score=score,
            vector_rank=v_rank.get(chunk_id) if fused else None,
            keyword_rank=kw_rank.get(chunk_id) if fused else None,
            candidate_rank=before[chunk_id] if mode == "rerank" else None,
            reranked=reranked,
        )
        for r, (chunk_id, score) in enumerate(ranked, 1)
    ]


def rrf(
    lists: list[list[uuid.UUID]], rrf_k: int, weights: list[float] | None = None
) -> list[tuple[uuid.UUID, float]]:
    """Reciprocal rank fusion. Ties go to the chunk with the better best rank, then by id."""
    weights = weights or [1.0] * len(lists)
    scores: dict[uuid.UUID, float] = {}
    best: dict[uuid.UUID, int] = {}
    for ids, weight in zip(lists, weights, strict=True):
        for rank, chunk_id in enumerate(ids, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (rrf_k + rank)
            best[chunk_id] = min(best.get(chunk_id, rank), rank)
    order = sorted(scores, key=lambda c: (-scores[c], best[c], str(c)))
    return [(c, scores[c]) for c in order]


def keyword_query(
    words: list[str], word_counts: dict[str, int], n_chunks: int, max_word_share: float
) -> str:
    """The question's words, minus the too-common ones, joined with OR (websearch syntax).

    OR, not AND: a question has many words, and a chunk rarely holds all of them.
    """
    limit = max_word_share * n_chunks
    kept = [w for w in dict.fromkeys(words) if word_counts.get(w, 0) <= limit]
    return " OR ".join(kept)


def load_version(session: Session, version_id: uuid.UUID | None = None) -> Version:
    where = "id = :id" if version_id else "status = 'active'"
    row = session.execute(
        text(f"SELECT id, config, n_chunks FROM corpus_versions WHERE {where}"),
        {"id": version_id},
    ).one_or_none()
    if row is None:
        raise SearchError(f"no corpus version {version_id}" if version_id else "no active version")
    version = Version(*row)
    if (
        version.config["segmenter"] == "jieba"
        and version.config["word_list"] != word_list_version()
    ):
        raise SearchError(
            f"version {version.id} was built with another POYA word list; questions would be "
            "split differently from its chunks. Rebuild it (ingest chunk)."
        )
    return version


def find_version(session: Session, prefix: str) -> uuid.UUID:
    """A corpus version by its id or the first characters of it (e.g. 02fc002a)."""
    ids = session.scalars(
        text("SELECT id FROM corpus_versions WHERE id::text LIKE :p"), {"p": prefix + "%"}
    ).all()
    if len(ids) != 1:
        raise SearchError(f"{len(ids)} corpus versions start with {prefix!r}")
    return ids[0]


def cached_query_embedder(session: Session, version: Version) -> EmbedQuery:
    model, dims = version.config["embed_model"], version.config["embed_dims"]
    call = gemini_embed(model, dims)

    def embed(query: str) -> list[float]:
        text_in = query_input(version.config["embed_query_format"], query)
        result = embed_inputs(
            [text_in], model=model, dims=dims, cache=DbEmbeddingCache(session), call=call
        )
        return result.vectors[0]

    return embed


def _vector(session, version, vector, n) -> list[tuple[uuid.UUID, float]]:
    rows = session.execute(
        text(
            "SELECT id, 1 - (embedding <=> CAST(:q AS vector)) FROM chunks "
            "WHERE corpus_version_id = :v "
            "ORDER BY embedding <=> CAST(:q AS vector), id LIMIT :n"
        ),
        {"q": str(vector), "v": version.id, "n": n},
    ).all()
    return [(r[0], float(r[1])) for r in rows]


def _keyword(session, version, query, n, max_word_share) -> list[tuple[uuid.UUID, float]]:
    words = segment(query, version.config["segmenter"]).split()
    q = keyword_query(words, _word_counts(session, version), version.n_chunks, max_word_share)
    if not q:
        return []
    rows = session.execute(
        text(
            "SELECT c.id, ts_rank(c.tsv, q, 1) AS score "
            "FROM chunks c, websearch_to_tsquery('simple', :q) q "
            "WHERE c.corpus_version_id = :v AND c.tsv @@ q "
            "ORDER BY score DESC, c.id LIMIT :n"
        ),
        {"q": q, "v": version.id, "n": n},
    ).all()
    return [(r[0], float(r[1])) for r in rows]


# A version's chunks never change once built, so its word counts can be kept for good.
_WORD_COUNTS: dict[uuid.UUID, dict[str, int]] = {}


def _word_counts(session, version) -> dict[str, int]:
    """How many of the version's chunks hold each word (Postgres ts_stat)."""
    if version.id not in _WORD_COUNTS:
        inner = f"SELECT tsv FROM chunks WHERE corpus_version_id = '{uuid.UUID(str(version.id))}'"
        rows = session.execute(text("SELECT word, ndoc FROM ts_stat(:inner)"), {"inner": inner})
        _WORD_COUNTS[version.id] = dict(rows.all())
    return _WORD_COUNTS[version.id]


def _chunk_rows(session, chunk_ids):
    if not chunk_ids:
        return {}
    rows = session.execute(
        text(
            "SELECT c.id, d.source_url, d.title, c.text FROM chunks c "
            "JOIN documents d ON d.id = c.document_id WHERE c.id = ANY(:ids)"
        ),
        {"ids": chunk_ids},
    ).all()
    return {r.id: r for r in rows}
