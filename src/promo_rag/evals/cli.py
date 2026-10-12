"""Run the search eval from the terminal.

uv run eval-search                                   # every mode, the active corpus version
uv run eval-search --modes vector hybrid             # some modes (rerank is the slow one)
uv run eval-search --only rule-01 multi-02           # some questions
uv run eval-search --version 5da896fe 02fc002a       # compare corpus versions (jieba, bigram)
uv run eval-search --modes vector hybrid --vector-weight 1 2 3    # compare settings
uv run eval-search --modes vector rerank --rerank-from vector hybrid

Flags that take several values make one variant per combination. The first variant is
the baseline every other one is compared with, so list vector (or the current default)
first. Writes evals/results/search_<time>.md (the tables) and .json (every question's
results, the metrics and each variant's settings).
"""

import argparse
import json
import sys
from datetime import datetime

from promo_rag.config import REPO_ROOT, get_settings
from promo_rag.db.session import new_session
from promo_rag.evals.golden import load_golden
from promo_rag.evals.search_eval import (
    MODES,
    as_records,
    base_settings,
    by_category,
    compare,
    make_variants,
    no_answer_cutoffs,
    report,
    run,
    search_query,
    summarize,
    variant_records,
    warm_embeddings,
)
from promo_rag.llm import Pacer
from promo_rag.search.rerank import RerankError, cohere_rerank
from promo_rag.search.retrieve import DEFAULT_SETTINGS, SearchError, find_version, load_version

RESULTS_DIR = REPO_ROOT / "evals" / "results"


def main() -> int:
    parser = argparse.ArgumentParser(prog="eval-search")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--only", nargs="+", metavar="ID", help="question ids")
    parser.add_argument(
        "--version", nargs="+", help="corpus version ids, or their first characters"
    )
    parser.add_argument(
        "--max-word-share", nargs="+", type=float, default=[DEFAULT_SETTINGS.max_word_share]
    )
    parser.add_argument(
        "--vector-weight", nargs="+", type=float, default=[DEFAULT_SETTINGS.vector_weight]
    )
    parser.add_argument("--rrf-k", nargs="+", type=int, default=[DEFAULT_SETTINGS.rrf_k])
    parser.add_argument(
        "--rerank-from",
        nargs="+",
        choices=["vector", "hybrid"],
        default=[DEFAULT_SETTINGS.rerank_from],
    )
    args = parser.parse_args()

    questions = load_golden().questions
    if args.only:
        unknown = set(args.only) - {q.id for q in questions}
        if unknown:
            print(f"error: unknown question ids {sorted(unknown)}", file=sys.stderr)
            return 1
        questions = [q for q in questions if q.id in args.only]
    settings = get_settings()

    with new_session() as session:
        try:
            ids = [find_version(session, p) for p in args.version] if args.version else [None]
            versions = [load_version(session, i) for i in ids]
            rerank = cohere_rerank() if "rerank" in args.modes else None
        except (SearchError, RerankError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        variants = make_variants(
            versions,
            args.modes,
            args.max_word_share,
            args.vector_weight,
            args.rrf_k,
            args.rerank_from,
        )
        print(f"{len(variants)} variant(s) × {len(questions)} questions.")

        for version in versions:
            calls = warm_embeddings(session, version, [search_query(q) for q in questions])
            print(f"Question embeddings for {str(version.id)[:8]}: {calls} API call(s).")
        n_rerank = sum(v.mode == "rerank" for v in variants) * len(questions)
        if n_rerank:
            minutes = n_rerank / settings.rerank_rpm
            print(f"Rerank: {n_rerank} calls at {settings.rerank_rpm}/min ≈ {minutes:.0f} min.")

        results = run(
            session,
            questions,
            variants,
            rerank=rerank,
            rerank_pacer=Pacer(settings.rerank_rpm),
            progress=lambda msg: print(f"\r  {msg}      ", end="", file=sys.stderr, flush=True),
        )
        print(file=sys.stderr)
        session.commit()

    shared = base_settings(settings.rerank_model)
    markdown = report(results, variants, shared)
    stem = f"search_{datetime.now():%Y-%m-%dT%H%M%S}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{stem}.md").write_text(markdown, encoding="utf-8")
    payload = {
        "base_settings": shared,
        "variants": variant_records(variants),
        "summary": summarize(results),
        "vs_baseline": compare(results, variants[0].label),
        "hit@5_by_category": by_category(results),
        "no_answer_cutoffs": no_answer_cutoffs(results, variants),
        "results": as_records(results),
    }
    (RESULTS_DIR / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(markdown)
    print(f"Saved evals/results/{stem}.md and .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
