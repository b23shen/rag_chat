"""Run the search eval from the terminal.

uv run eval-search                                  # every mode, the active corpus version
uv run eval-search --modes vector hybrid            # some modes (rerank is the slow one)
uv run eval-search --only rule-01 multi-02          # some questions
uv run eval-search --version 02fc002a               # another corpus version, by id prefix

Writes evals/results/search_<time>_<version>.md (the tables) and .json (every question's
results, the metrics and the settings).
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
    by_category,
    report,
    run,
    run_settings,
    search_query,
    summarize,
    warm_embeddings,
)
from promo_rag.llm import Pacer
from promo_rag.search.rerank import RerankError, cohere_rerank
from promo_rag.search.retrieve import SearchError, find_version, load_version

RESULTS_DIR = REPO_ROOT / "evals" / "results"


def main() -> int:
    parser = argparse.ArgumentParser(prog="eval-search")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--only", nargs="+", metavar="ID", help="question ids")
    parser.add_argument("--version", help="corpus version id or its first characters")
    args = parser.parse_args()

    golden = load_golden()
    answerable = [q for q in golden.questions if q.category != "unanswerable"]
    if args.only:
        unknown = set(args.only) - {q.id for q in golden.questions}
        if unknown:
            print(f"error: unknown question ids {sorted(unknown)}", file=sys.stderr)
            return 1
        answerable = [q for q in answerable if q.id in args.only]
    n_skipped = len(golden.questions) - len(answerable) if not args.only else 0
    settings = get_settings()

    with new_session() as session:
        try:
            version = load_version(
                session, find_version(session, args.version) if args.version else None
            )
            rerank = cohere_rerank() if "rerank" in args.modes else None
        except (SearchError, RerankError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

        calls = warm_embeddings(session, version, [search_query(q) for q in answerable])
        print(f"Question embeddings: {calls} API call(s), the rest from the cache.")
        if rerank:
            minutes = len(answerable) / settings.rerank_rpm
            print(
                f"Rerank: {len(answerable)} calls at {settings.rerank_rpm}/min ≈ {minutes:.0f} min."
            )

        results = run(
            session,
            version,
            answerable,
            tuple(args.modes),
            rerank=rerank,
            rerank_pacer=Pacer(settings.rerank_rpm),
            progress=lambda msg: print(f"\r  {msg}   ", end="", file=sys.stderr, flush=True),
        )
        print(file=sys.stderr)
        session.commit()

    run_info = run_settings(version, settings.rerank_model)
    markdown = report(results, version, n_skipped, run_info)
    stem = f"search_{datetime.now():%Y-%m-%dT%H%M}_{str(version.id)[:8]}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{stem}.md").write_text(markdown, encoding="utf-8")
    payload = {
        "settings": run_info,
        "summary": summarize(results),
        "hit@5_by_category": by_category(results),
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
