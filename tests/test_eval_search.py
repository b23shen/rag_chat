"""The search eval: metrics, the searched text, the report, and one real run (keyword only)."""

import math
import uuid

import pytest
from sqlalchemy.exc import OperationalError

from promo_rag.evals.golden import load_golden
from promo_rag.evals.metrics import hit_at, ndcg_at, percentile, reciprocal_rank
from promo_rag.evals.search_eval import QuestionResult, report, run, search_query, summarize
from promo_rag.search.retrieve import Version, load_version

# ---------------------------------------------------------------- metrics


def test_hit_counts_only_chunks_that_answer():
    assert hit_at([0, 1, 2], 3) == 1.0
    assert hit_at([0, 1, 2], 2) == 0.0  # a grade-1 chunk helps but isn't a hit
    assert hit_at([], 5) == 0.0


def test_reciprocal_rank_of_the_first_answer():
    assert reciprocal_rank([0, 2, 2], 10) == 0.5
    assert reciprocal_rank([0, 0, 2], 2) == 0.0  # beyond the cutoff


def test_ndcg():
    assert ndcg_at([2, 1, 0], [2, 1], 5) == pytest.approx(1.0)  # best possible order
    swapped = ndcg_at([1, 2, 0], [2, 1], 5)
    assert 0 < swapped < 1
    # DCG = 3/log2(2) + 1/log2(3) best; swapped = 1/log2(2) + 3/log2(3)
    assert swapped == pytest.approx((1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3)))
    # Three answering chunks exist; finding one at the top is not a perfect score.
    assert ndcg_at([2, 0, 0], [2, 2, 2], 5) < 1
    assert ndcg_at([0, 0], [], 5) == 0.0


def test_percentile_nearest_rank():
    values = list(range(1, 21))  # 1..20
    assert percentile(values, 50) == 10
    assert percentile(values, 95) == 19
    assert percentile([7.0], 95) == 7.0


# ---------------------------------------------------------------- what is searched


def test_follow_ups_are_searched_with_the_earlier_user_turns():
    questions = {q.id: q for q in load_golden().questions}
    assert search_query(questions["rule-03"]) == questions["rule-03"].question
    query = search_query(questions["multi-02"])
    assert query.startswith("What's the LINE Pay offer at POYA?")
    assert query.endswith("Does it work at the Taipei Main Station store?")
    assert "3% back" not in query  # assistant turns are left out


# ---------------------------------------------------------------- the report


def _result(qid, mode, grades, category="rule"):
    return QuestionResult(
        qid, category, "zh", mode, f"{qid}?", grades, ["u"] * len(grades), [2], 5.0
    )


def test_report_has_a_row_per_mode_and_lists_the_best_modes_misses():
    results = [
        _result("rule-01", "vector", [2]),
        _result("rule-02", "vector", [0, 0, 0, 0, 0, 2]),  # first answer at #6
        _result("rule-01", "keyword", [0]),
        _result("rule-02", "keyword", [0]),
    ]
    assert summarize(results)["vector"]["hit@5"] == 0.5
    version = Version(uuid.UUID(int=1), {"segmenter": "jieba"}, 232)
    text = report(results, version, n_skipped=6, settings={"rrf_k": 60})
    assert "| vector | 0.50 | 1.00 |" in text and "| keyword | 0.00 | 0.00 |" in text
    assert "Misses at 5 for the best mode (vector): 1" in text
    assert "rule-02 (zh): first answer at #6" in text
    assert "rrf_k=60" in text


# ---------------------------------------------------------------- one real run


def test_keyword_run_on_the_database():
    from promo_rag.db.session import new_session

    questions = [q for q in load_golden().questions if q.id in ("rule-03", "list-01")]
    with new_session() as session:
        try:
            version = load_version(session)
        except OperationalError:
            pytest.skip("no database")
        results = run(session, version, questions, ("keyword",))
        session.rollback()
    assert [r.question_id for r in results] == ["rule-03", "list-01"]
    assert all(len(r.grades) <= 20 and r.ms > 0 for r in results)
    assert results[1].ideal.count(2) == 3  # list-01: three store pages, one chunk each
