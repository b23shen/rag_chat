"""Answer eval: the chat assistant on every golden question, graded by an LLM judge and by code.

For each question, the assistant answers with the golden history before it (multi-turn) and
"today" pinned to the golden set's as_of date, so "now" and "this month" match the reference
answers. Then:

  the judge     compares the answer with the reference: correct | partial | wrong, and
                whether it declined. Unanswerable questions are correct when declined.
  code checks   the reply's language matches the question's; the answer cites pages; every
                cited URL was returned by a tool in that turn (no invented sources); a cited
                page, and a page the tools found, is one the golden set names.

Also recorded: tool calls, tokens, cost, latency, and anything that would make a result
unfair (the fallback model answered, the reranker was unavailable, a call limit was hit).
"""

import re
import statistics
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from promo_rag.agents import AgentError, AgentResult, TraceInfo, invoke_agent
from promo_rag.agents.tools.dates import pinned_today, today
from promo_rag.evals.golden import GoldenSet, Question
from promo_rag.evals.metrics import percentile
from promo_rag.llm import cost_usd
from promo_rag.workflows.chat.agents import ANSWER_JUDGE
from promo_rag.workflows.chat.assistant import ask

URL = re.compile(r"https?://[A-Za-z0-9._~:/?#@!$&'*+,;=%-]+")
CJK = re.compile(r"[㐀-鿿豈-﫿]")
LATIN = re.compile(r"[A-Za-z]")
SOURCES = re.compile(r"^\W*(sources|來源|資料來源|參考資料)\W*$", re.IGNORECASE | re.MULTILINE)
POINTS = {"correct": 1.0, "partial": 0.5, "wrong": 0.0, "error": 0.0}

Ask = Callable[[str, list[BaseMessage]], AgentResult]
Judge = Callable[[Question, str], tuple["AnswerGrade", AgentResult | None]]


class AnswerGrade(BaseModel):
    verdict: Literal["correct", "partial", "wrong"]
    declined: bool = Field(description="The answer says it couldn't find it or can't answer.")
    reason: str = Field(description="One short sentence on what decided the verdict.")


@dataclass
class AnswerResult:
    id: str
    category: str
    lang: str
    question: str
    reference: str | None
    answer: str = ""
    verdict: str = "error"  # correct | partial | wrong | error
    declined: bool = False
    reason: str = ""
    tool_calls: list[str] = field(default_factory=list)
    tool_urls: list[str] = field(default_factory=list)
    cited: list[str] = field(default_factory=list)
    lang_ok: bool = False
    grounded: bool = False  # every cited URL came from a tool in this turn
    cites_golden: bool | None = None  # None for unanswerable questions
    found_golden: bool | None = None
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    judge_model: str = ""
    judge_tokens_in: int = 0
    judge_tokens_out: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return self.category != "unanswerable"


# ---------------------------------------------------------------- running


def run(
    golden: GoldenSet,
    questions: list[Question],
    *,
    chat_model: str,
    ask_fn: Ask | None = None,
    judge_fn: Judge | None = None,
    pace_s: float = 0.0,
    progress: Callable[[str], None] = lambda _: None,
) -> list[AnswerResult]:
    ask_fn = ask_fn or _ask
    judge_fn = judge_fn or judge
    results, last = [], 0.0
    with pinned_today(golden.as_of):
        for i, q in enumerate(questions, 1):
            time.sleep(max(0.0, last + pace_s - time.monotonic()))
            last = time.monotonic()
            results.append(evaluate(q, ask_fn, judge_fn, chat_model))
            r = results[-1]
            progress(f"[{i}/{len(questions)}] {q.id}: {r.verdict}")
    return results


def rejudge(
    golden: GoldenSet,
    records: list[dict],
    *,
    judge_fn: Judge | None = None,
    progress: Callable[[str], None] = lambda _: None,
) -> list[AnswerResult]:
    """Grade saved answers again (after a judge change), without asking the assistant again,
    so any change in the numbers comes from the judge alone."""
    judge_fn = judge_fn or judge
    questions = {q.id: q for q in golden.questions}
    results = []
    with pinned_today(golden.as_of):
        for i, record in enumerate(records, 1):
            r = AnswerResult(**record)
            if r.answer:  # an answer the assistant failed to give stays an error
                _grade(r, questions[r.id], judge_fn)
            results.append(r)
            progress(f"[{i}/{len(records)}] {r.id}: {r.verdict}")
    return results


def evaluate(q: Question, ask_fn: Ask, judge_fn: Judge, chat_model: str) -> AnswerResult:
    r = AnswerResult(q.id, q.category, q.lang, q.question, q.answer)
    history = [
        HumanMessage(t.content) if t.role == "user" else AIMessage(t.content) for t in q.history
    ]
    try:
        result = ask_fn(q.question, history)
    except AgentError as e:
        r.reason = f"assistant failed: {e}"
        return r
    turn = result.messages[len(history) + 1 :]  # this turn's messages, after the question
    r.answer, r.model = result.text, result.model
    r.tokens_in, r.tokens_out, r.latency_ms = result.tokens_in, result.tokens_out, result.latency_ms
    r.tool_calls = [
        f"{c['name']}({_args(c['args'])})"
        for m in turn
        if isinstance(m, AIMessage)
        for c in m.tool_calls
    ]
    r.tool_urls = sorted({u for m in turn if isinstance(m, ToolMessage) for u in urls_in(m.text)})
    r.cited = cited_urls(r.answer)
    r.lang_ok = reply_language(r.answer) == q.lang
    r.grounded = bool(r.cited) and set(r.cited) <= set(r.tool_urls)
    golden_urls = {_norm(u) for u in q.relevant_urls()}
    if r.answerable:
        r.cites_golden = bool(golden_urls & set(r.cited))
        r.found_golden = bool(golden_urls & set(r.tool_urls))
    r.warnings = _warnings(turn, result, chat_model)
    _grade(r, q, judge_fn)
    return r


def _grade(r: AnswerResult, q: Question, judge_fn: Judge) -> None:
    try:
        grade, judged = judge_fn(q, r.answer)
    except AgentError as e:
        r.verdict, r.declined, r.reason = "error", False, f"judge failed: {e}"
        return
    r.verdict, r.declined, r.reason = grade.verdict, grade.declined, grade.reason
    if judged:
        r.judge_model = judged.model
        r.judge_tokens_in, r.judge_tokens_out = judged.tokens_in, judged.tokens_out


def _ask(question: str, history: list[BaseMessage]) -> AgentResult:
    return ask(question, history=history, session_id=f"eval-{uuid.uuid4().hex[:8]}")


def judge(q: Question, answer: str) -> tuple[AnswerGrade, AgentResult]:
    convo = "\n".join(f"{t.role}: {t.content}" for t in q.history)
    reference = q.answer or "none: POYA's pages don't answer this question."
    message = (
        (f"<conversation>\n{convo}\n</conversation>\n" if convo else "")
        + f"<question>{q.question}</question>\n<reference>{reference}</reference>\n"
        + f"<answer>\n{answer}\n</answer>"
    )
    result = invoke_agent(
        ANSWER_JUDGE.name,
        message,
        variables={"today": today().isoformat()},  # the golden as_of date, inside run()
        response_schema=AnswerGrade,
        trace=TraceInfo(workflow="eval", run_id=f"judge-{q.id}", metadata={"question": q.id}),
    )
    return result.output, result


# ---------------------------------------------------------------- code checks


def urls_in(text: str) -> list[str]:
    return [_norm(u) for u in URL.findall(text)]


def cited_urls(answer: str) -> list[str]:
    """The URLs the answer cites, in order, without repeats."""
    return list(dict.fromkeys(urls_in(answer)))


def reply_language(answer: str) -> str:
    """'en' or 'zh', from the body (without the sources list and URLs): English answers still
    quote Chinese names (台北北車店), so count letters against Chinese characters."""
    body = SOURCES.split(answer, maxsplit=1)[0]
    body = URL.sub("", body)
    return "en" if len(LATIN.findall(body)) > len(CJK.findall(body)) else "zh"


def _norm(url: str) -> str:
    return url.rstrip(".,;:)]").rstrip("/")


def _args(args: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def _warnings(turn: list[BaseMessage], result: AgentResult, chat_model: str) -> list[str]:
    """Anything that makes this result unlike a normal run."""
    found = []
    if result.model and result.model != chat_model:
        found.append(f"answered by {result.model}")
    for m in turn:
        if isinstance(m, ToolMessage):
            if "reranker was unavailable" in m.text:
                found.append("reranker unavailable")
            if getattr(m, "status", None) == "error":
                found.append(f"tool error: {m.text[:80]}")
        elif (
            isinstance(m, AIMessage) and "limit" in m.text.lower() and "exceeded" in m.text.lower()
        ):
            found.append("call limit hit")
    return list(dict.fromkeys(found))


# ---------------------------------------------------------------- summary and report


def summarize(results: list[AnswerResult]) -> dict:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    verdicts = Counter(r.verdict for r in answerable)
    cost = sum(
        cost_usd(r.model, r.tokens_in, r.tokens_out)
        + cost_usd(r.judge_model, r.judge_tokens_in, r.judge_tokens_out)
        for r in results
    )
    return {
        "questions": len(results),
        "score": _mean(POINTS[r.verdict] for r in results),
        "answerable": {
            "n": len(answerable),
            **{v: verdicts[v] for v in ("correct", "partial", "wrong", "error")},
            "score": _mean(POINTS[r.verdict] for r in answerable),
            "declined": sum(r.declined for r in answerable),
            "cited": _rate(bool(r.cited) for r in answerable),
            "cites_golden_page": _rate(r.cites_golden for r in answerable),
            "tools_found_golden_page": _rate(r.found_golden for r in answerable),
        },
        "unanswerable": {
            "n": len(unanswerable),
            "declined_correctly": sum(r.verdict == "correct" for r in unanswerable),
            "score": _mean(POINTS[r.verdict] for r in unanswerable),
        },
        "grounded_citations": _rate(r.grounded for r in results if r.cited),
        "language_ok": _rate(r.lang_ok for r in results if r.answer),
        "tool_calls_mean": _mean(len(r.tool_calls) for r in results),
        "tool_calls_max": max((len(r.tool_calls) for r in results), default=0),
        "latency_ms_p50": percentile([r.latency_ms for r in results if r.answer], 50),
        "latency_ms_p95": percentile([r.latency_ms for r in results if r.answer], 95),
        "tokens_in": sum(r.tokens_in for r in results),
        "tokens_out": sum(r.tokens_out for r in results),
        "cost_usd": round(cost, 4),
        "warnings": dict(Counter(w.split(":")[0] for r in results for w in r.warnings)),
    }


def by_category(results: list[AnswerResult]) -> dict[str, dict]:
    out = {}
    for category in dict.fromkeys(r.category for r in results):
        rs = [r for r in results if r.category == category]
        verdicts = Counter(r.verdict for r in rs)
        out[category] = {
            "n": len(rs),
            **{v: verdicts[v] for v in ("correct", "partial", "wrong", "error")},
            "score": _mean(POINTS[r.verdict] for r in rs),
        }
    return out


def report(results: list[AnswerResult], settings: dict) -> str:
    s = summarize(results)
    a, u = s["answerable"], s["unanswerable"]
    lines = [
        f"# Answer eval, {settings['run_at']}",
        "",
        *(
            [f"Saved answers from `{settings['rejudged_from']}`, graded again.", ""]
            if settings.get("rejudged_from")
            else []
        ),
        (
            f"Chat model `{settings['chat_model']}`, judge `{settings['judge_model']}`, "
            f"today pinned to {settings['today']}, {s['questions']} questions. "
            f"Score: correct = 1, partial = 0.5, wrong or error = 0."
        ),
        "",
        "| | n | correct | partial | wrong | error | score |",
        "|---|---|---|---|---|---|---|",
        (
            f"| answerable | {a['n']} | {a['correct']} | {a['partial']} | {a['wrong']} | "
            f"{a['error']} | {a['score']:.2f} |"
        ),
        (
            f"| unanswerable (correct = declined) | {u['n']} | {u['declined_correctly']} | | | | "
            f"{u['score']:.2f} |"
        ),
        f"| **all** | {s['questions']} | | | | | **{s['score']:.2f}** |",
        "",
        "## By category",
        "",
        "| category | n | correct | partial | wrong | error | score |",
        "|---|---|---|---|---|---|---|",
    ]
    for category, c in by_category(results).items():
        lines.append(
            f"| {category} | {c['n']} | {c['correct']} | {c['partial']} | {c['wrong']} | "
            f"{c['error']} | {c['score']:.2f} |"
        )
    lines += [
        "",
        "## Sources, language, cost",
        "",
        (
            f"- Answerable questions answered with citations: {a['cited']:.0%}; citing a page the "
            f"golden set names: {a['cites_golden_page']:.0%}; the tools found such a page: "
            f"{a['tools_found_golden_page']:.0%}."
        ),
        (
            f"- Answers whose cited URLs all came from a tool in that turn: "
            f"{s['grounded_citations']:.0%}."
        ),
        f"- Declined although answerable: {a['declined']}.",
        f"- Reply in the question's language: {s['language_ok']:.0%}.",
        (
            f"- Tool calls per question: {s['tool_calls_mean']:.1f} on average, at most "
            f"{s['tool_calls_max']}. Latency p50 {s['latency_ms_p50'] / 1000:.1f} s, p95 "
            f"{s['latency_ms_p95'] / 1000:.1f} s."
        ),
        (
            f"- Tokens: {s['tokens_in']:,} in, {s['tokens_out']:,} out (assistant); cost with the "
            f"judge ≈ US${s['cost_usd']:.2f} at paid prices."
        ),
        f"- Warnings: {s['warnings'] or 'none'}.",
        "",
        "## Not fully correct",
        "",
        "| id | verdict | judge's reason | tools |",
        "|---|---|---|---|",
    ]
    for r in results:
        if r.verdict != "correct":
            reason = r.reason.replace("|", "/").replace("\n", " ")
            tools = "; ".join(c.split("(")[0] for c in r.tool_calls) or "none"
            lines.append(f"| {r.id} | {r.verdict} | {reason} | {tools} |")
    return "\n".join(lines) + "\n"


def as_records(results: list[AnswerResult]) -> list[dict]:
    return [asdict(r) for r in results]


def _mean(values) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def _rate(flags) -> float:
    return _mean(float(bool(f)) for f in flags)
