"""Search: rank fusion and keyword queries (pure), plus searches on the real database."""

import json
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from promo_rag.search.rerank import RerankError
from promo_rag.search.retrieve import SearchSettings, keyword_query, load_version, rrf, search

A, B, C, D = (uuid.UUID(int=i) for i in range(1, 5))

# ---------------------------------------------------------------- rank fusion


def test_rrf_rewards_chunks_found_by_both_lists():
    # B is second in both lists: 2/62 beats A's single first place, 1/61.
    fused = rrf([[A, B], [C, B]], rrf_k=60)
    assert [c for c, _ in fused] == [B, A, C]
    assert fused[0][1] == pytest.approx(2 / 62)


def test_rrf_with_one_empty_list_keeps_the_other_order():
    assert [c for c, _ in rrf([[C, A, B], []], rrf_k=60)] == [C, A, B]


def test_rrf_weights_favor_one_list():
    # Equal weights: A (first in list 1) and C (first in list 2) tie, A wins on id.
    # Weight 2 on list 2 puts C ahead.
    assert [c for c, _ in rrf([[A], [C]], rrf_k=60, weights=[1.0, 2.0])] == [C, A]
    assert rrf([[A], [C]], rrf_k=60, weights=[1.0, 2.0])[0][1] == pytest.approx(2 / 61)


def test_rrf_ties_go_to_the_better_best_rank_then_id():
    # A: ranks 1 and 3; B: ranks 2 and 2. Nearly equal sums; A has the better best rank.
    fused = rrf([[A, B], [D, B, A]], rrf_k=0)  # k=0 makes the sums easy: A 1+1/3, B 1/2+1/2
    assert [c for c, _ in fused][:2] == [A, D]
    assert [c for c, _ in rrf([[A], [B]], rrf_k=60)] == [A, B]  # exact tie: A has the lower id


# ---------------------------------------------------------------- keyword query


def test_keyword_query_drops_common_words_and_repeats():
    counts = {"的": 71, "活動": 140, "寵物": 1}
    assert keyword_query(["寵物", "的", "活動", "寵物", "推車"], counts, 232, 0.2) == "寵物 OR 推車"
    assert keyword_query(["寵物", "的"], counts, 232, 1.0) == "寵物 OR 的"  # filter off


def test_keyword_query_is_empty_when_every_word_is_common():
    assert keyword_query(["的", "活動"], {"的": 71, "活動": 140}, 232, 0.2) == ""


# ---------------------------------------------------------------- the real database


@pytest.fixture(scope="module")
def session():
    from promo_rag.db.session import new_session

    s = new_session()
    try:
        load_version(s)
    except OperationalError:
        pytest.skip("no database")
    yield s
    s.rollback()
    s.close()


def test_keyword_search_finds_the_pets_faq(session):
    hits = search(session, "寵物可以進去嗎", mode="keyword", k=3)
    assert hits[0].source_url.endswith("/faq/8") and "寵物" in hits[0].text


def test_vector_search_returns_a_chunk_first_for_its_own_vector(session):
    chunk_id, vector = session.execute(
        text(
            "SELECT c.id, c.embedding::text FROM chunks c JOIN corpus_versions v "
            "ON v.id = c.corpus_version_id WHERE v.status = 'active' ORDER BY c.id LIMIT 1"
        )
    ).one()
    hits = search(session, "-", mode="vector", k=3, embed_query=lambda _: json.loads(vector))
    assert hits[0].chunk_id == chunk_id and hits[0].score == pytest.approx(1.0)
    assert [h.rank for h in hits] == [1, 2, 3]


def test_hybrid_reports_each_lists_rank(session):
    chunk_id, vector = session.execute(
        text(
            "SELECT c.id, c.embedding::text FROM chunks c JOIN corpus_versions v "
            "ON v.id = c.corpus_version_id JOIN documents d ON d.id = c.document_id "
            "WHERE v.status = 'active' AND d.source_url LIKE '%/faq/8' AND c.text LIKE '%寵物%'"
        )
    ).one()
    hits = search(session, "寵物", mode="hybrid", k=5, embed_query=lambda _: json.loads(vector))
    assert hits[0].chunk_id == chunk_id
    assert hits[0].vector_rank == 1 and hits[0].keyword_rank == 1


@pytest.fixture
def pets(session):
    """The pets FAQ chunk and its own embedding (a query vector that finds it first)."""
    chunk_id, vector = session.execute(
        text(
            "SELECT c.id, c.embedding::text FROM chunks c JOIN corpus_versions v "
            "ON v.id = c.corpus_version_id JOIN documents d ON d.id = c.document_id "
            "WHERE v.status = 'active' AND d.source_url LIKE '%/faq/8' AND c.text LIKE '%寵物%'"
        )
    ).one()
    return chunk_id, json.loads(vector)


def _backwards(sent):
    def call(query, documents):  # a fake reranker that prefers the last candidate
        sent.append(documents)
        return [i / len(documents) for i in range(len(documents))]

    return call


def test_rerank_reorders_vectors_top_results_by_default(session, pets):
    pets_chunk_id, vector = pets
    sent = []
    hits = search(session, "寵物", k=3, embed_query=lambda _: vector, rerank=_backwards(sent))
    n = len(sent[0])
    assert n == 20  # vector's top 20 go to the reranker, with their full text
    assert [h.candidate_rank for h in hits] == [n, n - 1, n - 2]
    assert hits[0].score == pytest.approx((n - 1) / n) and hits[0].reranked
    assert hits[0].vector_rank is None  # no keyword list was fused
    vector_first = search(session, "-", mode="vector", k=1, embed_query=lambda _: vector)
    assert vector_first[0].chunk_id == pets_chunk_id
    assert sent[0][0] == vector_first[0].text


def test_rerank_can_take_hybrids_candidates(session, pets):
    _, vector = pets
    sent = []
    settings = SearchSettings(rerank_from="hybrid")
    hits = search(
        session,
        "寵物",
        k=3,
        embed_query=lambda _: vector,
        rerank=_backwards(sent),
        settings=settings,
    )
    hybrid = search(session, "寵物", mode="hybrid", k=20, embed_query=lambda _: vector)
    assert sent[0] == [h.text for h in hybrid]
    assert hits[0].keyword_rank is not None or hits[0].vector_rank is not None


def test_rerank_failure_falls_back_to_the_candidates_order(session, pets):
    pets_chunk_id, vector = pets

    def broken(query, documents):
        raise RerankError("HTTP 429: trial limit")

    hits = search(session, "寵物", k=3, embed_query=lambda _: vector, rerank=broken)
    assert hits[0].chunk_id == pets_chunk_id and not hits[0].reranked
    assert [h.candidate_rank for h in hits] == [1, 2, 3]
    with pytest.raises(RerankError):
        search(session, "寵物", k=3, embed_query=lambda _: vector, rerank=broken, fallback=False)
