"""The golden question set: its shape, and that every labeled page is a real, active page."""

import csv
from collections import Counter, defaultdict
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from promo_rag.evals.golden import SITE, GoldenSet, Question, load_golden, normalize

SOURCES_CSV = Path(__file__).resolve().parents[1] / "data" / "sources.csv"


@pytest.fixture(scope="module")
def golden() -> GoldenSet:
    return load_golden()


def test_categories_and_languages(golden):
    assert len(golden.questions) == 102
    assert Counter(q.category for q in golden.questions) == {
        "rule": 26,
        "exact": 15,
        "date": 12,
        "list": 11,
        "count": 8,
        "unanswerable": 20,
        "multi_turn": 10,
    }
    n_english = sum(q.lang == "en" for q in golden.questions)
    assert 30 <= n_english <= 45  # a third or more


def test_every_page_is_in_the_snapshot(golden):
    with open(SOURCES_CSV, encoding="utf-8", newline="") as f:
        known = {row["url"] for row in csv.DictReader(f)}
    missing = {
        (q.id, url) for q in golden.questions for url in q.relevant_urls() if url not in known
    }
    assert not missing


@pytest.fixture(scope="module")
def active_chunks() -> dict[str, list[str]]:
    """Chunk texts of the active corpus version, by page URL. Needs Postgres."""
    from promo_rag.db.session import get_engine

    sql = text("""
        SELECT d.source_url, c.text FROM rag.chunks c
        JOIN rag.documents d ON d.id = c.document_id
        JOIN rag.corpus_versions v ON v.id = c.corpus_version_id
        WHERE v.status = 'active' AND d.status = 'active' AND d.is_latest
    """)
    try:
        with get_engine().connect() as conn:
            rows = conn.execute(sql).all()
    except OperationalError:
        pytest.skip("no database")
    chunks = defaultdict(list)
    for url, chunk_text in rows:
        chunks[url].append(chunk_text)
    return chunks


def test_every_page_is_active_in_the_database(golden, active_chunks):
    """Catches labels on pages that parse skipped (duplicates, image-only)."""
    inactive = {
        (q.id, url)
        for q in golden.questions
        for url in q.relevant_urls()
        if url not in active_chunks
    }
    assert not inactive


def test_every_quote_is_inside_one_chunk_of_its_page(golden, active_chunks):
    """A quote that is misspelled, or cut in two by a chunk boundary, would never match."""
    not_found = {
        (q.id, url, quote)
        for q in golden.questions
        for url, evidence in q.relevant_urls().items()
        for quote in evidence.quote
        if not any(normalize(quote) in normalize(c) for c in active_chunks[url])
    }
    assert not not_found


def test_chunk_grade():
    q = _question(relevant={"/faq/8": {"grade": 2, "quote": ["無配合大來卡", "x"]}})
    url = SITE + "/faq/8"
    assert q.chunk_grade(url, "現金、信用卡,無配合 大來卡。") == 2  # spaces and width ignored
    assert q.chunk_grade(url, "寵物入店規定") == 0  # right page, wrong chunk
    assert q.chunk_grade(SITE + "/faq/7", "無配合大來卡") == 0  # quote on an unlabeled page


def _question(**changes) -> Question:
    return GoldenSet.model_validate(_one(**changes)).questions[0]


def _one(**changes) -> dict:
    question = {
        "id": "rule-99",
        "category": "rule",
        "lang": "zh",
        "question": "?",
        "relevant": {"/faq/8": {"grade": 2, "quote": "無配合大來卡"}},
        "answer": "ok",
    }
    return {"as_of": "2026-10-10", "questions": [question | changes]}


@pytest.mark.parametrize(
    "changes",
    [
        {"id": "date-99"},  # prefix doesn't match category
        {"relevant": {"/faq/8": {"grade": 1, "quote": "a"}}},  # nothing graded 2
        {"relevant": {"/faq/8": {"grade": 3, "quote": "a"}}},  # grade out of range
        {"relevant": {"/faq/8": {"grade": 2}}},  # no quote
        {"relevant": {"/faq/8": {"grade": 2, "quote": " "}}},  # empty quote
        {"relevant": {"/faq/8": {"grade": 2, "quote": "長" * 41}}},  # quote too long
        {"relevant": {SITE + "/faq/8": {"grade": 2, "quote": "a"}}},  # full URL, not a path
        {"answer": None},  # answerable but no answer
        {"history": [{"role": "user", "content": "hi"}]},  # history on a non-multi_turn question
        {"id": "none-99", "category": "unanswerable"},  # unanswerable with pages
        {"id": "multi-99", "category": "multi_turn"},  # multi_turn without history
    ],
)
def test_bad_entries_are_rejected(changes):
    with pytest.raises(ValidationError):
        GoldenSet.model_validate(_one(**changes))


def test_duplicate_ids_are_rejected():
    data = _one()
    data["questions"] *= 2
    with pytest.raises(ValidationError):
        GoldenSet.model_validate(data)
