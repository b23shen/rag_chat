"""Search metrics over one question's results.

`grades` is the grade of each result, best first: 2 = the chunk answers the question,
1 = it helps, 0 = not relevant (see Question.chunk_grade). A "hit" is a grade-2 chunk.
"""

import math
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


def _dcg(grades: Sequence[int], k: int) -> float:
    return sum((2**g - 1) / math.log2(rank + 1) for rank, g in enumerate(grades[:k], 1))
