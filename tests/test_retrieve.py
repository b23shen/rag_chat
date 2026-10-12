"""Search: rank fusion and keyword queries (pure), plus searches on the real database."""

import json
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from promo_rag.search.retrieve import keyword_query, load_version, rrf, search

A, B, C, D = (uuid.UUID(int=i) for i in range(1, 5))

# ---------------------------------------------------------------- rank fusion


def test_rrf_rewards_chunks_found_by_both_lists():
    # B is second in both lists: 2/62 beats A's single first place, 1/61.
    fused = rrf([[A, B], [C, B]], rrf_k=60)
    assert [c for c, _ in fused] == [B, A, C]
    assert fused[0][1] == pytest.approx(2 / 62)


def test_rrf_with_one_empty_list_keeps_the_other_order():
    assert [c for c, _ in rrf([[C, A, B], []], rrf_k=60)] == [C, A, B]


def test_rrf_ties_go_to_the_better_best_rank_then_id():
    # A: ranks 1 and 3; B: ranks 2 and 2. Nearly equal sums; A has the better best rank.
    fused = rrf([[A, B], [D, B, A]], rrf_k=0)  # k=0 makes the sums easy: A 1+1/3, B 1/2+1/2
    assert [c for c, _ in fused][:2] == [A, D]
    assert [c for c, _ in rrf([[A], [B]], rrf_k=60)] == [A, B]  # exact tie: A has the lower id


# ---------------------------------------------------------------- keyword query


def test_keyword_query_drops_common_words_and_repeats():
    counts = {"的": 71, "活動": 140, "寵物": 1}
    assert keyword_query(["寵物", "的", "活動", "寵物", "推車"], counts, 232) == "寵物 OR 推車"


def test_keyword_query_is_empty_when_every_word_is_common():
    assert keyword_query(["的", "活動"], {"的": 71, "活動": 140}, 232) == ""


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
