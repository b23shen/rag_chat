"""The answer eval's checks and summary, with a fake assistant and judge (no API calls)."""

from datetime import date

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from promo_rag.agents import AgentError, AgentResult
from promo_rag.agents.tools.dates import today
from promo_rag.evals.answer_eval import (
    AnswerGrade,
    as_records,
    cited_urls,
    evaluate,
    rejudge,
    reply_language,
    report,
    run,
    summarize,
)
from promo_rag.evals.golden import GoldenSet, Question

PAGE = "https://www.poya.com.tw/news-detail/357/"
OTHER = "https://www.poya.com.tw/faq/7"
QUESTION = Question(
    id="multi-02",
    category="multi_turn",
    lang="en",
    question="Does it work at the Taipei Main Station store?",
    history=[
        {"role": "user", "content": "What's the LINE Pay offer at POYA?"},
        {"role": "assistant", "content": "Every Wednesday, 3% back in LINE POINTS."},
    ],
    relevant={"/news-detail/357/": {"grade": 2, "quote": "寶雅台北北車店"}},
    answer="No. 台北北車店 is one of the excluded stores.",
)
UNANSWERABLE = Question(
    id="none-01",
    category="unanswerable",
    lang="en",
    question="Can I collect Starbucks stars at POYA?",
    relevant={},
    answer=None,
)
ANSWER = (
    f"No, the 台北北車店 store is excluded from this offer [1].\n\nSources:\n[1] LINE Pay — {PAGE}"
)


def _fake_ask(answer=ANSWER, tool_text=f'{{"results": [{{"url": "{PAGE}"}}]}}', model="chat-m"):
    def ask_fn(question, history):
        call = {
            "name": "search_promotion_text",
            "args": {"query": "LINE Pay 台北北車店"},
            "id": "c1",
        }
        turn = [
            AIMessage("", tool_calls=[call]),
            ToolMessage(tool_text, tool_call_id="c1"),
            AIMessage(answer),
        ]
        return AgentResult(
            output=answer,
            text=answer,
            messages=[*history, HumanMessage(question), *turn],
            agent="promo_assistant",
            model=model,
            prompt_version="p1",
            tokens_in=1000,
            tokens_out=100,
            latency_ms=2000,
        )

    return ask_fn


def _fake_judge(verdict="correct", declined=False):
    def judge_fn(q, answer):
        return AnswerGrade(verdict=verdict, declined=declined, reason="matches"), None

    return judge_fn


# ---------------------------------------------------------------- code checks


def test_cited_urls_in_order_without_repeats_or_trailing_punctuation():
    text = f"見 {PAGE}。也見 {OTHER}, 和 {PAGE}"
    assert cited_urls(text) == [PAGE.rstrip("/"), OTHER]


def test_reply_language_counts_letters_against_chinese_characters():
    assert reply_language("No, the 台北北車店 store is excluded [1].") == "en"
    assert reply_language("每週三用 LINE Pay 單筆滿 650 元享 3% 回饋。") == "zh"
    # the sources list (Chinese page titles) doesn't count
    assert (
        reply_language(f"Yes, it works.\n\nSources:\n[1] 【寶雅門市】每週三用LINE Pay — {PAGE}")
        == "en"
    )


# ---------------------------------------------------------------- one question


def test_a_good_answer_is_grounded_and_cites_the_golden_page():
    r = evaluate(QUESTION, _fake_ask(), _fake_judge(), chat_model="chat-m")
    assert (r.verdict, r.lang_ok, r.grounded, r.cites_golden, r.found_golden) == (
        "correct",
        True,
        True,
        True,
        True,
    )
    assert r.tool_calls == ["search_promotion_text(query='LINE Pay 台北北車店')"]
    assert r.warnings == []


def test_a_cited_url_no_tool_returned_is_not_grounded():
    invented = f"No [1].\n\nSources:\n[1] {PAGE}\n[2] https://www.poya.com.tw/made-up"
    r = evaluate(QUESTION, _fake_ask(answer=invented), _fake_judge(), chat_model="chat-m")
    assert not r.grounded


def test_warnings_flag_a_fallback_model_and_a_missing_reranker():
    tool_text = '{"results": [], "note": "The reranker was unavailable: ..."}'
    ask_fn = _fake_ask(tool_text=tool_text, model="fallback-m")
    r = evaluate(QUESTION, ask_fn, _fake_judge(), chat_model="chat-m")
    assert r.warnings == ["answered by fallback-m", "reranker unavailable"]


def test_unanswerable_questions_have_no_golden_page_checks():
    r = evaluate(UNANSWERABLE, _fake_ask(), _fake_judge(declined=True), chat_model="chat-m")
    assert (r.cites_golden, r.found_golden, r.declined) == (None, None, True)


def test_an_assistant_failure_is_an_error_not_a_crash():
    def failing(question, history):
        raise AgentError("promo_assistant", RuntimeError("503"))

    r = evaluate(QUESTION, failing, _fake_judge(), chat_model="chat-m")
    assert r.verdict == "error" and "503" in r.reason


# ---------------------------------------------------------------- a run


def test_run_pins_today_to_the_golden_date_and_summarizes():
    seen = []

    def ask_fn(question, history):
        seen.append(today())
        return _fake_ask()(question, history)

    golden = GoldenSet(as_of=date(2026, 10, 10), questions=[QUESTION, UNANSWERABLE])
    results = run(
        golden, golden.questions, chat_model="chat-m", ask_fn=ask_fn, judge_fn=_fake_judge()
    )
    assert seen == [date(2026, 10, 10)] * 2

    s = summarize(results)
    assert s["answerable"]["correct"] == 1 and s["unanswerable"]["declined_correctly"] == 1
    assert s["score"] == 1.0 and s["grounded_citations"] == 1.0
    settings = {"run_at": "now", "chat_model": "chat-m", "judge_model": "j", "today": "2026-10-10"}
    assert "| **all** | 2 |" in report(results, settings)


def test_the_judge_is_told_the_golden_date():
    from promo_rag.agents.config import get_agent_config
    from promo_rag.agents.prompts import load_prompt

    prompt = load_prompt(get_agent_config("answer_judge").prompt)
    assert prompt.variables == {"today"}
    assert "took place on 2026-10-10" in prompt.render({"today": "2026-10-10"})


def test_rejudge_grades_saved_answers_again_without_the_assistant():
    golden = GoldenSet(as_of=date(2026, 10, 10), questions=[QUESTION, UNANSWERABLE])
    first = run(
        golden,
        golden.questions,
        chat_model="chat-m",
        ask_fn=_fake_ask(),
        judge_fn=_fake_judge("partial"),
    )
    seen = []

    def judge_fn(q, answer):
        seen.append((q.id, today()))
        return AnswerGrade(verdict="correct", declined=False, reason="ok"), None

    again = rejudge(golden, as_records(first), judge_fn=judge_fn)
    assert seen == [("multi-02", date(2026, 10, 10)), ("none-01", date(2026, 10, 10))]
    assert [r.verdict for r in again] == ["correct", "correct"]
    assert again[0].answer == first[0].answer and again[0].tool_calls == first[0].tool_calls
