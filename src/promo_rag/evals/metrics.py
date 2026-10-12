"""Search metrics over one question's results.

`grades` is the grade of each result, best first: 2 = the chunk answers the question,
1 = it helps, 0 = not relevant (see Question.chunk_grade). A "hit" is a grade-2 chunk.
"""

import math
import random
import statistics
from collections.abc import Sequence


def hit_at(grades: Sequence[int], k: int) -> float:
    """1 if a chunk in the top k answers the question, else 0."""
    return 1.0 if 2 in grades[:k] else 0.0


def reciprocal_rank(grades: Sequence[int], k: int) -> float:
    """1 / the rank of the first chunk that answers the question (0 if none in the top k)."""
    for rank, grade in enumerate(grades[:k], 1):
        if grade == 2:
            return 1.0 / rank
    return 0.0


def ndcg_at(grades: Sequence[int], ideal: Sequence[int], k: int) -> float:
    """DCG of the results over DCG of the best possible order, both cut at k.

    `ideal` is every relevant chunk's grade in the corpus version, so a question with three
    answering chunks only scores 1.0 when all three are at the top.
    """
    best = _dcg(sorted(ideal, reverse=True), k)
    return _dcg(grades, k) / best if best else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile: the smallest value with at least p% of values at or below it."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def paired_range(
    diffs: Sequence[float], n: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """The mean of per-question differences, and its 95% range (bootstrap).

    Draw len(diffs) questions at random, with repeats, n times; the middle 95% of those
    means is the range. A range that includes 0 means the two variants can't be told apart
    on this many questions.
    """
    if not diffs:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return statistics.mean(diffs), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def best_cutoff(answerable: Sequence[float], unanswerable: Sequence[float]) -> dict:
    """The top-score cutoff that sorts the most questions right: a question whose best
    result scores at least the cutoff is treated as answerable. Ties keep more answerable."""
    best = None
    for cutoff in sorted(set(answerable) | set(unanswerable)):
        kept = sum(s >= cutoff for s in answerable)
        caught = sum(s < cutoff for s in unanswerable)
        if best is None or (kept + caught, kept) > (best["kept"] + best["caught"], best["kept"]):
            best = {"cutoff": cutoff, "kept": kept, "caught": caught}
    return {**best, "n_answerable": len(answerable), "n_unanswerable": len(unanswerable)}


def _dcg(grades: Sequence[int], k: int) -> float:
    return sum((2**g - 1) / math.log2(rank + 1) for rank, g in enumerate(grades[:k], 1))
