"""Search a corpus version's chunks: vector, keyword, hybrid, or rerank.

  vector   cosine distance between the question's embedding and each chunk's (pgvector)
  keyword  Postgres full-text search on the segmented text (chunks.tsv), any word may match
  hybrid   both lists merged with reciprocal rank fusion (RRF): each chunk scores
           sum(1 / (rrf_k + rank)) over the lists it appears in. Ranks, not scores, so the
           two very different score scales never need to be compared.
  rerank   hybrid's top results, put in a new order by a reranker model (search/rerank.py)

The question is embedded and segmented with the settings the version was built with (its
config), so a question and the chunks it is compared with are always processed the same way.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import text
from sqlalchemy.orm import Session

from promo_rag.search.embed import DbEmbeddingCache, embed_inputs, gemini_embed, query_input
from promo_rag.search.rerank import RerankCall, cohere_rerank
from promo_rag.search.segment import segment, word_list_version

Mode = Literal["vector", "keyword", "hybrid", "rerank"]
EmbedQuery = Callable[[str], list[float]]

# Each list's top N goes into the fusion, and the fused top N into the reranker.
CANDIDATES = 20
# The usual RRF constant; a larger value flattens the gap between rank 1 and rank 2.
RRF_K = 60
# Keyword search ignores question words found in more than this share of the version's
# chunks (2026, 活動, 的 …). Postgres ranking has no IDF, so without this a chunk full of
# common words outranks one that holds the rare word the question is about.
MAX_WORD_SHARE = 0.2


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
    score: float
    vector_rank: int | None = None  # hybrid and rerank: the chunk's rank in each list
    keyword_rank: int | None = None
    hybrid_rank: int | None = None  # rerank only: the chunk's rank before reranking


def search(
    session: Session,
    query: str,
    *,
    mode: Mode = "hybrid",
    k: int = 5,
    version_id: uuid.UUID | None = None,
    embed_query: EmbedQuery | None = None,
    rerank: RerankCall | None = None,
) -> list[Hit]:
    """The top k chunks for a question. Reads the active version unless one is given.

    A new question embedding goes into the embedding cache in the caller's transaction;
    commit to keep it.
    """
    version = load_version(session, version_id)
    fused = mode in ("hybrid", "rerank")
    n = max(k, CANDIDATES) if fused else k
    vector_ids, keyword_ids = [], []
    if mode != "keyword":
        embed = embed_query or cached_query_embedder(session, version)
        vector_ids = _vector(session, version, embed(query), n)
    if mode != "vector":
        keyword_ids = _keyword(session, version, query, n)

    if mode == "vector":
        ranked = vector_ids
    elif mode == "keyword":
        ranked = keyword_ids
    else:
        ranked = rrf([[i for i, _ in vector_ids], [i for i, _ in keyword_ids]], RRF_K)[:n]
    rows = _chunk_rows(session, [chunk_id for chunk_id, _ in ranked])
    hybrid_rank = {chunk_id: r for r, (chunk_id, _) in enumerate(ranked, 1)}
    if mode == "rerank":
        scores = (rerank or cohere_rerank())(query, [rows[c].text for c, _ in ranked])
        rescored = [(c, s) for (c, _), s in zip(ranked, scores, strict=True)]
        ranked = sorted(rescored, key=lambda cs: (-cs[1], hybrid_rank[cs[0]]))
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
            hybrid_rank=hybrid_rank[chunk_id] if mode == "rerank" else None,
        )
        for r, (chunk_id, score) in enumerate(ranked, 1)
    ]


def rrf(lists: list[list[uuid.UUID]], rrf_k: int) -> list[tuple[uuid.UUID, float]]:
    """Reciprocal rank fusion. Ties go to the chunk with the better best rank, then by id."""
    scores: dict[uuid.UUID, float] = {}
    best: dict[uuid.UUID, int] = {}
    for ids in lists:
        for rank, chunk_id in enumerate(ids, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            best[chunk_id] = min(best.get(chunk_id, rank), rank)
    order = sorted(scores, key=lambda c: (-scores[c], best[c], str(c)))
    return [(c, scores[c]) for c in order]


def keyword_query(words: list[str], word_counts: dict[str, int], n_chunks: int) -> str:
    """The question's words, minus the too-common ones, joined with OR (websearch syntax).

    OR, not AND: a question has many words, and a chunk rarely holds all of them.
    """
    limit = MAX_WORD_SHARE * n_chunks
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


def _keyword(session, version, query, n) -> list[tuple[uuid.UUID, float]]:
    words = segment(query, version.config["segmenter"]).split()
    q = keyword_query(words, _word_counts(session, version), version.n_chunks)
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
