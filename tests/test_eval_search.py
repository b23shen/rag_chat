"""The search eval: metrics, the searched text, the report, and one real run (keyword only)."""

import math
import uuid

import pytest
from sqlalchemy.exc import OperationalError

from promo_rag.evals.golden import load_golden
from promo_rag.evals.metrics import (
    best_cutoff,
    hit_at,
    ndcg_at,
    paired_range,
    percentile,
    reciprocal_rank,
)
from promo_rag.evals.search_eval import (
    QuestionResult,
    make_variants,
    report,
    run,
    search_query,
    summarize,
)
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


# ---------------------------------------------------------------- variants


def _version(n, segmenter="jieba"):
    return Version(uuid.UUID(int=n), {"segmenter": segmenter}, 232)


def test_variants_name_only_what_varies():
    labels = [
        v.label for v in make_variants([_version(1)], ["vector", "hybrid"], [0.2], [1, 2], [60])
    ]
    assert labels == ["vector", "hybrid vw=1", "hybrid vw=2"]  # vector ignores the weight
    two = make_variants([_version(1), _version(2, "bigram")], ["keyword"], [0.1, 1.0], [1], [60])
    assert [v.label for v in two] == [
        "keyword jieba:00000000 share=0.1",
        "keyword jieba:00000000 share=1",
        "keyword bigram:00000000 share=0.1",
        "keyword bigram:00000000 share=1",
    ]
    assert two[1].settings.max_word_share == 1.0
    rerank = make_variants([_version(1)], ["rerank"], [0.2], [1, 2], [60], ["vector", "hybrid"])
    # The vector-fed reranker has no hybrid knobs; the hybrid-fed one varies its weight.
    assert [v.label for v in rerank] == [
        "rerank from=vector",
        "rerank from=hybrid vw=1",
        "rerank from=hybrid vw=2",
    ]
    assert rerank[2].settings.rerank_from == "hybrid" and rerank[2].settings.vector_weight == 2


# ---------------------------------------------------------------- the report


def _result(qid, variant, grades, category="rule", top=0.8):
    scores = [top] + [0.1] * (len(grades) - 1)
    urls = ["u"] * len(grades)
    return QuestionResult(qid, category, "zh", variant, f"{qid}?", grades, scores, urls, [2], 5.0)


def test_report_compares_with_the_baseline_and_lists_the_best_variants_misses():
    variants = make_variants([_version(1)], ["vector", "keyword"], [0.2], [1], [60])
    results = [
        _result("rule-01", "vector", [2]),
        _result("rule-02", "vector", [0, 0, 0, 0, 0, 2]),  # first answer at #6
        _result("none-01", "vector", [0], category="unanswerable", top=0.3),
        _result("rule-01", "keyword", [0]),
        _result("rule-02", "keyword", [0]),
        _result("none-01", "keyword", [0], category="unanswerable"),
    ]
    assert summarize(results)["vector"]["hit@5"] == 0.5  # the unanswerable one is left out
    text = report(results, variants, {"rrf_k": 60})
    assert "| vector | 0.50 | 1.00 | 0.58 | " in text and "| baseline | baseline |" in text
    assert "| keyword | 0.00 | 0.00 | 0.00 | 0.00 | -0.58 (" in text
    assert "Misses at 5 for the best variant (vector): 1" in text
    assert "rule-02 (zh): first answer at #6" in text
    # vector has a fixed score scale: a cutoff between 0.3 and 0.8 sorts all three right
    assert "| vector | 0.800 | 2/2 | 1/1 |" in text
    assert "rrf_k=60" in text


def test_paired_range():
    mean, low, high = paired_range([0.5] * 20)
    assert mean == low == high == 0.5  # no spread, no doubt
    mean, low, high = paired_range([1.0, -1.0] * 10)
    assert mean == 0 and low < 0 < high


def test_best_cutoff_prefers_keeping_answerable_questions():
    c = best_cutoff(answerable=[0.9, 0.8, 0.5], unanswerable=[0.6, 0.3])
    # 0.8: keeps 2, catches 2 (4 right); 0.5: keeps 3, catches 1 (4 right) → keep more
    assert (c["cutoff"], c["kept"], c["caught"]) == (0.5, 3, 1)


# ---------------------------------------------------------------- one real run


def test_keyword_run_on_the_database():
    from promo_rag.db.session import new_session

    questions = [q for q in load_golden().questions if q.id in ("rule-03", "list-01", "none-01")]
    with new_session() as session:
        try:
            version = load_version(session)
        except OperationalError:
            pytest.skip("no database")
        variants = make_variants([version], ["keyword"], [0.2], [1], [60])
        results = run(session, questions, variants)
        session.rollback()
    assert [r.question_id for r in results] == ["rule-03", "list-01", "none-01"]
    assert all(len(r.grades) <= 20 and r.ms > 0 for r in results)
    assert results[1].ideal.count(2) == 3  # list-01: three store pages, one chunk each
    assert results[2].ideal == [] and not results[2].answerable
