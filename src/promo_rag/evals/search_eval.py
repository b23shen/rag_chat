"""Search eval: every golden question, searched in every mode, graded chunk by chunk.

Per mode it reports Hit@5, Hit@20, MRR@10, nDCG@5 and latency (see metrics.py). Hit@20
of hybrid is the reranker's ceiling: the reranker only re-orders hybrid's top 20, so it
can't find a chunk that isn't there.

Latency is search time only. Question embeddings are fetched in one batch before the
timed runs (warm cache), each mode runs once untimed first (one-time loading), and the
reranker's rate limit is waited out before the timer starts. So neither the embedding API
nor a trial key's pacing is counted; the reranker's own API call is.
"""

import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from promo_rag.evals.golden import Question
from promo_rag.evals.metrics import hit_at, ndcg_at, percentile, reciprocal_rank
from promo_rag.llm import Pacer
from promo_rag.search.embed import DbEmbeddingCache, embed_inputs, gemini_embed, query_input
from promo_rag.search.rerank import RerankCall
from promo_rag.search.retrieve import (
    CANDIDATES,
    MAX_WORD_SHARE,
    RRF_K,
    EmbedQuery,
    Mode,
    Version,
    cached_query_embedder,
    search,
)

MODES: tuple[Mode, ...] = ("vector", "keyword", "hybrid", "rerank")
DEPTH = 20  # results kept per search: enough for Hit@20


@dataclass
class QuestionResult:
    question_id: str
    category: str
    lang: str
    mode: Mode
    query: str  # the text that was searched
    grades: list[int]  # each result's grade, best first
    urls: list[str]  # each result's page
    ideal: list[int]  # the grades of every relevant chunk in the version, for nDCG
    ms: float


def search_query(q: Question) -> str:
    """The text to search. Follow-ups: the earlier user turns, then the question.

    A simple baseline: "Does it work at the Taipei Main Station store?" alone doesn't say
    what "it" is. The M3 agent will rewrite follow-ups into standalone questions instead.
    """
    return " ".join([t.content for t in q.history if t.role == "user"] + [q.question])


def ideal_grades(session: Session, version: Version, questions: list[Question]) -> dict:
    """Each question's grade for every chunk in the version, keeping the relevant ones."""
    rows = session.execute(
        text(
            "SELECT d.source_url, c.text FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE c.corpus_version_id = :v"
        ),
        {"v": version.id},
    ).all()
    ideal = {}
    for q in questions:
        grades = [q.chunk_grade(url, chunk_text) for url, chunk_text in rows]
        ideal[q.id] = sorted((g for g in grades if g), reverse=True)
    return ideal


def warm_embeddings(session: Session, version: Version, queries: list[str]) -> int:
    """Embed every question in one batch (cached), so timed searches skip the API.

    Returns the number of API calls made (0 when all were cached already)."""
    model, dims = version.config["embed_model"], version.config["embed_dims"]
    inputs = [query_input(version.config["embed_query_format"], q) for q in queries]
    result = embed_inputs(
        inputs,
        model=model,
        dims=dims,
        cache=DbEmbeddingCache(session),
        call=gemini_embed(model, dims),
    )
    session.commit()
    return result.api_calls


def run(
    session: Session,
    version: Version,
    questions: list[Question],
    modes: tuple[Mode, ...] = MODES,
    *,
    embed_query: EmbedQuery | None = None,
    rerank: RerankCall | None = None,
    rerank_pacer: Pacer | None = None,
    progress=lambda msg: None,
) -> list[QuestionResult]:
    embed_query = embed_query or cached_query_embedder(session, version)
    ideal = ideal_grades(session, version, questions)
    missing = [q.id for q in questions if 2 not in ideal[q.id]]
    if missing:
        raise ValueError(f"no chunk in version {version.id} holds a grade-2 quote for {missing}")

    # One untimed search per mode first: loading jieba and the version's word counts happens
    # once per process and isn't search time. (Not rerank: that would spend an API call.)
    for mode in modes:
        if mode != "rerank":
            search(
                session,
                search_query(questions[0]),
                mode=mode,
                k=1,
                version_id=version.id,
                embed_query=embed_query,
            )

    results = []
    for mode in modes:
        for i, q in enumerate(questions, 1):
            query = search_query(q)
            if mode == "rerank" and rerank_pacer:
                rerank_pacer.wait()  # before the timer: pacing isn't search time
            start = time.perf_counter()
            hits = search(
                session,
                query,
                mode=mode,
                k=DEPTH,
                version_id=version.id,
                embed_query=embed_query,
                rerank=rerank,
            )
            ms = (time.perf_counter() - start) * 1000
            results.append(
                QuestionResult(
                    question_id=q.id,
                    category=q.category,
                    lang=q.lang,
                    mode=mode,
                    query=query,
                    grades=[q.chunk_grade(h.source_url, h.text) for h in hits],
                    urls=[h.source_url for h in hits],
                    ideal=ideal[q.id],
                    ms=ms,
                )
            )
            progress(f"{mode} {i}/{len(questions)}")
    return results


def summarize(results: list[QuestionResult]) -> dict[str, dict[str, float]]:
    """Per mode: the mean of each metric over the questions, and latency percentiles."""
    by_mode = defaultdict(list)
    for r in results:
        by_mode[r.mode].append(r)
    return {mode: _metrics(rs) for mode, rs in by_mode.items()}


def by_category(results: list[QuestionResult], metric: str = "hit@5") -> dict:
    """Per mode, per category: the mean of one metric."""
    groups = defaultdict(list)
    for r in results:
        groups[(r.mode, r.category)].append(r)
    table = defaultdict(dict)
    for (mode, category), rs in groups.items():
        table[mode][category] = _metrics(rs)[metric]
    return dict(table)


def report(results: list[QuestionResult], version: Version, n_skipped: int, settings: dict) -> str:
    """The results as Markdown tables."""
    summary = summarize(results)
    modes = list(summary)
    n_questions = len({r.question_id for r in results})
    lines = [
        f"# Search eval, corpus version {str(version.id)[:8]} ({version.config['segmenter']})",
        "",
        (
            f"{n_questions} answerable questions ({n_skipped} unanswerable skipped). "
            "A hit is a chunk that answers the question (grade 2). Latency is search time "
            "with the question's embedding already cached."
        ),
        "",
        "| Mode | Hit@5 | Hit@20 | MRR@10 | nDCG@5 | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode in modes:
        m = summary[mode]
        lines.append(
            f"| {mode} | {m['hit@5']:.2f} | {m['hit@20']:.2f} | {m['mrr@10']:.2f} | "
            f"{m['ndcg@5']:.2f} | {m['p50_ms']:.0f} | {m['p95_ms']:.0f} |"
        )

    categories = sorted({r.category for r in results})
    table = by_category(results)
    lines += [
        "",
        "Hit@5 by question type:",
        "",
        "| Mode | " + " | ".join(categories) + " |",
        "|---|" + "---|" * len(categories),
    ]
    for mode in modes:
        cells = [f"{table[mode].get(c, 0):.2f}" for c in categories]
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")

    best = max(modes, key=lambda m: (summary[m]["ndcg@5"], summary[m]["hit@5"]))
    misses = [r for r in results if r.mode == best and not hit_at(r.grades, 5)]
    lines += ["", f"Misses at 5 for the best mode ({best}): {len(misses)}", ""]
    for r in misses:
        first = next((i for i, g in enumerate(r.grades, 1) if g == 2), None)
        where = f"first answer at #{first}" if first else "no answer in the top 20"
        lines.append(f"- {r.question_id} ({r.lang}): {where}. {r.query}")

    lines += ["", "Settings: " + ", ".join(f"{k}={v}" for k, v in settings.items())]
    return "\n".join(lines) + "\n"


def run_settings(version: Version, rerank_model: str) -> dict:
    """Everything that decides the results, saved with them."""
    return {
        "version": str(version.id),
        "segmenter": version.config["segmenter"],
        "embed_model": version.config["embed_model"],
        "embed_query_format": version.config["embed_query_format"],
        "candidates": CANDIDATES,
        "rrf_k": RRF_K,
        "max_word_share": MAX_WORD_SHARE,
        "rerank_model": rerank_model,
    }


def as_records(results: list[QuestionResult]) -> list[dict]:
    return [asdict(r) for r in results]


def _metrics(rs: list[QuestionResult]) -> dict[str, float]:
    ms = [r.ms for r in rs]
    return {
        "hit@5": statistics.mean(hit_at(r.grades, 5) for r in rs),
        "hit@20": statistics.mean(hit_at(r.grades, 20) for r in rs),
        "mrr@10": statistics.mean(reciprocal_rank(r.grades, 10) for r in rs),
        "ndcg@5": statistics.mean(ndcg_at(r.grades, r.ideal, 5) for r in rs),
        "p50_ms": percentile(ms, 50),
        "p95_ms": percentile(ms, 95),
    }
