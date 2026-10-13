"""Run the answer eval from the terminal.

uv run eval-answers                          # every golden question (about 15 minutes)
uv run eval-answers --only count-04 multi-02 # some questions
CHAT_MODEL=gemini-3.5-flash-lite uv run eval-answers     # another chat model, to compare
uv run eval-answers --rejudge evals/results/answers_<time>.json   # grade saved answers again

Writes evals/results/answers_<time>.md (the tables, and every question not fully correct)
and .json (every question's answer, tool calls, grade and checks, and the settings).
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

from promo_rag.agents.models import model_for_role
from promo_rag.agents.tracing import flush_traces
from promo_rag.config import REPO_ROOT, get_settings
from promo_rag.evals.answer_eval import as_records, by_category, rejudge, report, run, summarize
from promo_rag.evals.golden import load_golden

RESULTS_DIR = REPO_ROOT / "evals" / "results"


def main() -> int:
    parser = argparse.ArgumentParser(prog="eval-answers")
    parser.add_argument("--only", nargs="+", metavar="ID", help="question ids")
    parser.add_argument(
        "--pace",
        type=float,
        default=None,
        help="seconds between questions (default: 60 / RERANK_RPM, for a trial Cohere key)",
    )
    parser.add_argument(
        "--rejudge",
        type=Path,
        metavar="JSON",
        help="grade the answers saved in a results file again, without asking the assistant",
    )
    args = parser.parse_args()
    # The judge model ignores temperature and warns on every call; the setting doesn't matter.
    warnings.filterwarnings("ignore", message=".*uses fixed sampling defaults.*")

    golden = load_golden()
    questions = golden.questions
    if args.only:
        unknown = set(args.only) - {q.id for q in questions}
        if unknown:
            print(f"error: unknown question ids {sorted(unknown)}", file=sys.stderr)
            return 1
        questions = [q for q in questions if q.id in args.only]
    settings = get_settings()
    pace = args.pace if args.pace is not None else 60 / settings.rerank_rpm
    run_settings = {
        "run_at": f"{datetime.now():%Y-%m-%d %H:%M}",
        "chat_model": model_for_role("chat"),
        "chat_fallback_model": settings.chat_fallback_model,
        "judge_model": model_for_role("judge"),
        "rerank_model": settings.rerank_model,
        "today": golden.as_of.isoformat(),
        "questions": len(questions),
    }

    if args.rejudge:
        saved = json.loads(args.rejudge.read_text(encoding="utf-8"))
        wanted = {q.id for q in questions}
        records = [r for r in saved["results"] if r["id"] in wanted]
        run_settings = saved["settings"] | {
            "run_at": run_settings["run_at"],
            "judge_model": run_settings["judge_model"],
            "rejudged_from": args.rejudge.name,
            "questions": len(records),
        }
        print(f"Grading {len(records)} saved answers from {args.rejudge.name} again.")
        results = rejudge(golden, records, progress=progress)
    else:
        print(
            f"{len(questions)} questions, chat {run_settings['chat_model']}, judge "
            f"{run_settings['judge_model']}, today = {golden.as_of}, one question every "
            f"{pace:.0f} s."
        )
        results = run(
            golden,
            questions,
            chat_model=run_settings["chat_model"],
            pace_s=pace,
            progress=progress,
        )
    print(file=sys.stderr)
    flush_traces()

    markdown = report(results, run_settings)
    stem = f"answers_{datetime.now():%Y-%m-%dT%H%M%S}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{stem}.md").write_text(markdown, encoding="utf-8")
    payload = {
        "settings": run_settings,
        "summary": summarize(results),
        "by_category": by_category(results),
        "results": as_records(results),
    }
    (RESULTS_DIR / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(markdown)
    print(f"Saved evals/results/{stem}.md and .json")
    return 0


def progress(message: str) -> None:
    print(f"\r  {message}          ", end="", file=sys.stderr, flush=True)


if __name__ == "__main__":
    sys.exit(main())
