"""Search eval: every golden question, searched with every variant, graded chunk by chunk.

A variant is a mode on a corpus version with some settings, e.g. "hybrid, bigram version,
vector_weight=2". Running many variants on the same questions lets each one be compared
with the first (the baseline) question by question, with a 95% range for the difference.

Per variant it reports Hit@5, Hit@20, MRR@10, nDCG@5 and latency (see metrics.py). Hit@20
of hybrid is the reranker's ceiling: the reranker only re-orders hybrid's top 20.

Unanswerable questions are searched too. They are left out of the metrics, but they show
whether a score cutoff can tell "nothing relevant" apart (vector and rerank scores have a
fixed scale; keyword and hybrid scores don't).

Latency is search time only. Question embeddings are fetched in one batch before the
timed runs (warm cache), each variant runs once untimed first (one-time loading), and the
reranker's rate limit is waited out before the timer starts. So neither the embedding API
nor a trial key's pacing is counted; the reranker's own API call is.
"""

import itertools
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from promo_rag.evals.golden import Question
from promo_rag.evals.metrics import (
    best_cutoff,
    hit_at,
    ndcg_at,
    paired_range,
    percentile,
    reciprocal_rank,
)
from promo_rag.llm import Pacer
from promo_rag.search.embed import DbEmbeddingCache, embed_inputs, gemini_embed, query_input
from promo_rag.search.rerank import RerankCall
from promo_rag.search.retrieve import (
    DEFAULT_SETTINGS,
    EmbedQuery,
    Mode,
    SearchSettings,
    Version,
    cached_query_embedder,
    search,
)

MODES: tuple[Mode, ...] = ("vector", "keyword", "hybrid", "rerank")
DEPTH = 20  # results kept per search: enough for Hit@20
FIXED_SCALE = ("vector", "rerank")  # modes whose scores can be compared across questions


@dataclass
class Variant:
    label: str  # e.g. "hybrid vw=2"; unique within a run
    mode: Mode
    version: Version
    settings: SearchSettings


@dataclass
class QuestionResult:
    question_id: str
    category: str
    lang: str
    variant: str
    query: str  # the text that was searched
    grades: list[int]  # each result's grade, best first
    scores: list[float]  # each result's score, best first
    urls: list[str]  # each result's page
    ideal: list[int]  # the grades of every relevant chunk in the version, for nDCG
    ms: float

    @property
    def answerable(self) -> bool:
        return self.category != "unanswerable"


def make_variants(
    versions: list[Version],
    modes: list[Mode],
    max_word_shares: list[float],
    vector_weights: list[float],
    rrf_ks: list[int],
    rerank_froms: list[str] = ("vector",),
) -> list[Variant]:
    """Every mode on every version, with every value of the settings that mode uses.

    Labels name only what varies in this run: the mode, then the version if there are
    several, then each setting that takes more than one value.
    """
    values = {
        "max_word_share": max_word_shares,
        "vector_weight": vector_weights,
        "rrf_k": rrf_ks,
        "rerank_from": rerank_froms,
    }
    short = {"max_word_share": "share", "vector_weight": "vw", "rrf_k": "k", "rerank_from": "from"}
    keyword_knobs = ["max_word_share"]
    hybrid_knobs = ["max_word_share", "vector_weight", "rrf_k"]
    variants = []
    for version, mode in itertools.product(versions, modes):
        if mode == "rerank":
            knob_sets = [
                ["rerank_from"] + (hybrid_knobs if source == "hybrid" else [])
                for source in rerank_froms
            ]
            fixed = [{"rerank_from": source} for source in rerank_froms]
        else:
            knobs = {"vector": [], "keyword": keyword_knobs, "hybrid": hybrid_knobs}[mode]
            knob_sets, fixed = [knobs], [{}]
        for knobs, pinned in zip(knob_sets, fixed, strict=True):
            free = [k for k in knobs if k not in pinned]
            for combo in itertools.product(*(values[k] for k in free)):
                chosen = {**pinned, **dict(zip(free, combo, strict=True))}
                parts = [mode]
                if len(versions) > 1:
                    parts.append(f"{version.config['segmenter']}:{str(version.id)[:8]}")
                parts += [
                    f"{short[k]}={v:g}" if isinstance(v, float) else f"{short[k]}={v}"
                    for k, v in chosen.items()
                    if len(values[k]) > 1
                ]
                settings = SearchSettings(**chosen)
                variants.append(Variant(" ".join(parts), mode, version, settings))
    return variants


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
    questions: list[Question],
    variants: list[Variant],
    *,
    embed_query: EmbedQuery | None = None,
    rerank: RerankCall | None = None,
    rerank_pacer: Pacer | None = None,
    progress=lambda msg: None,
) -> list[QuestionResult]:
    versions = {v.version.id: v.version for v in variants}
    embedders = {
        vid: embed_query or cached_query_embedder(session, version)
        for vid, version in versions.items()
    }
    ideal = {vid: ideal_grades(session, version, questions) for vid, version in versions.items()}
    for vid, grades in ideal.items():
        missing = [
            q.id for q in questions if q.category != "unanswerable" and 2 not in grades[q.id]
        ]
        if missing:
            raise ValueError(f"no chunk in version {vid} holds a grade-2 quote for {missing}")

    def one_search(variant: Variant, query: str, k: int):
        return search(
            session,
            query,
            mode=variant.mode,
            k=k,
            version_id=variant.version.id,
            embed_query=embedders[variant.version.id],
            rerank=rerank,
            settings=variant.settings,
            fallback=False,  # a reranker failure must stop the eval, not pass as a result
        )

    # One untimed search per variant first: loading jieba and the version's word counts
    # happens once per process and isn't search time. (Not rerank: it would cost a call.)
    for variant in variants:
        if variant.mode != "rerank":
            one_search(variant, search_query(questions[0]), 1)

    results = []
    for variant in variants:
        for i, q in enumerate(questions, 1):
            query = search_query(q)
            if variant.mode == "rerank" and rerank_pacer:
                rerank_pacer.wait()  # before the timer: pacing isn't search time
            start = time.perf_counter()
            hits = one_search(variant, query, DEPTH)
            ms = (time.perf_counter() - start) * 1000
            results.append(
                QuestionResult(
                    question_id=q.id,
                    category=q.category,
                    lang=q.lang,
                    variant=variant.label,
                    query=query,
                    grades=[q.chunk_grade(h.source_url, h.text) for h in hits],
                    scores=[h.score for h in hits],
                    urls=[h.source_url for h in hits],
                    ideal=ideal[variant.version.id][q.id],
                    ms=ms,
                )
            )
            progress(f"{variant.label} {i}/{len(questions)}")
    return results


def summarize(results: list[QuestionResult]) -> dict[str, dict[str, float]]:
    """Per variant: each metric's mean over the answerable questions, and latency."""
    return {label: _metrics(rs) for label, rs in _by_variant(results).items()}


def compare(results: list[QuestionResult], baseline: str) -> dict[str, dict[str, tuple]]:
    """Each variant minus the baseline, question by question: (mean, low, high) per metric."""
    groups = _by_variant(results)
    base = {r.question_id: r for r in groups[baseline]}
    out = {}
    for label, rs in groups.items():
        out[label] = {
            name: paired_range([metric(r) - metric(base[r.question_id]) for r in rs])
            for name, metric in (("mrr@10", _rr), ("ndcg@5", _ndcg))
        }
    return out


def by_category(results: list[QuestionResult], metric: str = "hit@5") -> dict:
    """Per variant, per category: the mean of one metric (answerable questions only)."""
    groups = defaultdict(list)
    for r in results:
        if r.answerable:
            groups[(r.variant, r.category)].append(r)
    table = defaultdict(dict)
    for (label, category), rs in groups.items():
        table[label][category] = _metrics(rs)[metric]
    return dict(table)


def no_answer_cutoffs(results: list[QuestionResult], variants: list[Variant]) -> dict:
    """For fixed-scale variants: the top-score cutoff that best tells the unanswerable apart.

    Fitted on the same questions it is scored on, so it is optimistic; treat it as a hint.
    """
    modes = {v.label: v.mode for v in variants}
    out = {}
    for label, rs in _by_variant(results, answerable_only=False).items():
        if modes[label] not in FIXED_SCALE:
            continue
        answerable = [r.scores[0] for r in rs if r.answerable and r.scores]
        unanswerable = [r.scores[0] for r in rs if not r.answerable and r.scores]
        if unanswerable:
            out[label] = {
                **best_cutoff(answerable, unanswerable),
                "answerable_min": min(answerable),
                "unanswerable_max": max(unanswerable),
            }
    return out


def report(results: list[QuestionResult], variants: list[Variant], base_settings: dict) -> str:
    """The results as Markdown tables."""
    summary = summarize(results)
    labels = [v.label for v in variants]
    baseline = labels[0]
    deltas = compare(results, baseline)
    n_answerable = len({r.question_id for r in results if r.answerable})
    n_unanswerable = len({r.question_id for r in results if not r.answerable})
    versions = {v.version.id: v.version for v in variants}.values()
    lines = [
        "# Search eval",
        "",
        "Corpus version(s): "
        + ", ".join(f"{str(v.id)[:8]} ({v.config['segmenter']})" for v in versions),
        "",
        (
            f"{n_answerable} answerable questions; {n_unanswerable} unanswerable ones are "
            "searched but left out of the metrics. A hit is a chunk that answers the "
            "question (grade 2). Δ columns: the variant minus the baseline "
            f"(**{baseline}**), question by question, with the 95% range. A range that "
            "includes 0 means no clear difference. Latency is search time with the "
            "question's embedding already cached."
        ),
        "",
        "| Variant | Hit@5 | Hit@20 | MRR@10 | nDCG@5 | ΔMRR@10 | ΔnDCG@5 | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label in labels:
        m, d = summary[label], deltas[label]
        lines.append(
            f"| {label} | {m['hit@5']:.2f} | {m['hit@20']:.2f} | {m['mrr@10']:.2f} | "
            f"{m['ndcg@5']:.2f} | {_delta(d['mrr@10'], label == baseline)} | "
            f"{_delta(d['ndcg@5'], label == baseline)} | {m['p50_ms']:.0f} | {m['p95_ms']:.0f} |"
        )

    categories = sorted({r.category for r in results if r.answerable})
    table = by_category(results)
    lines += [
        "",
        "Hit@5 by question type:",
        "",
        "| Variant | " + " | ".join(categories) + " |",
        "|---|" + "---|" * len(categories),
    ]
    for label in labels:
        cells = [f"{table[label].get(c, 0):.2f}" for c in categories]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    cutoffs = no_answer_cutoffs(results, variants)
    if cutoffs:
        lines += [
            "",
            (
                "Telling unanswerable questions apart by the top result's score (cutoff "
                "fitted on these same questions, so optimistic):"
            ),
            "",
            (
                "| Variant | Best cutoff | Answerable kept | Unanswerable caught | "
                "Lowest answerable top score | Highest unanswerable top score |"
            ),
            "|---|---|---|---|---|---|",
        ]
        for label, c in cutoffs.items():
            lines.append(
                f"| {label} | {c['cutoff']:.3f} | {c['kept']}/{c['n_answerable']} | "
                f"{c['caught']}/{c['n_unanswerable']} | {c['answerable_min']:.3f} | "
                f"{c['unanswerable_max']:.3f} |"
            )

    best = max(labels, key=lambda lb: (summary[lb]["ndcg@5"], summary[lb]["hit@5"]))
    misses = [r for r in results if r.variant == best and r.answerable and not hit_at(r.grades, 5)]
    lines += ["", f"Misses at 5 for the best variant ({best}): {len(misses)}", ""]
    for r in misses:
        first = next((i for i, g in enumerate(r.grades, 1) if g == 2), None)
        where = f"first answer at #{first}" if first else "no answer in the top 20"
        lines.append(f"- {r.question_id} ({r.lang}): {where}. {r.query}")

    lines += ["", "Base settings: " + ", ".join(f"{k}={v}" for k, v in base_settings.items())]
    return "\n".join(lines) + "\n"


def base_settings(rerank_model: str) -> dict:
    """What every variant shares unless its label says otherwise; saved with the results."""
    return {**asdict(DEFAULT_SETTINGS), "rerank_model": rerank_model, "depth": DEPTH}


def variant_records(variants: list[Variant]) -> list[dict]:
    return [
        {
            "label": v.label,
            "mode": v.mode,
            "version": str(v.version.id),
            "segmenter": v.version.config["segmenter"],
            "embed_query_format": v.version.config["embed_query_format"],
            **asdict(v.settings),
        }
        for v in variants
    ]


def as_records(results: list[QuestionResult]) -> list[dict]:
    return [asdict(r) for r in results]


def _by_variant(results, answerable_only=True) -> dict[str, list[QuestionResult]]:
    groups = defaultdict(list)
    for r in results:
        if r.answerable or not answerable_only:
            groups[r.variant].append(r)
    return dict(groups)


def _rr(r: QuestionResult) -> float:
    return reciprocal_rank(r.grades, 10)


def _ndcg(r: QuestionResult) -> float:
    return ndcg_at(r.grades, r.ideal, 5)


def _delta(d: tuple[float, float, float], is_baseline: bool) -> str:
    if is_baseline:
        return "baseline"
    mean, low, high = d
    return f"{mean:+.2f} ({low:+.2f} to {high:+.2f})"


def _metrics(rs: list[QuestionResult]) -> dict[str, float]:
    ms = [r.ms for r in rs]
    return {
        "hit@5": statistics.mean(hit_at(r.grades, 5) for r in rs),
        "hit@20": statistics.mean(hit_at(r.grades, 20) for r in rs),
        "mrr@10": statistics.mean(_rr(r) for r in rs),
        "ndcg@5": statistics.mean(_ndcg(r) for r in rs),
        "p50_ms": percentile(ms, 50),
        "p95_ms": percentile(ms, 95),
    }
