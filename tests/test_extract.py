"""The extraction retry loop, with a fake invoke_agent (no network)."""

from datetime import date

from langchain_core.messages import AIMessage, HumanMessage

from promo_rag.agents import AgentError, AgentResult, TraceInfo
from promo_rag.db.models import Document
from promo_rag.workflows.ingestion.actions.extract import Pacer, _shown_text, extract_with_agent
from promo_rag.workflows.ingestion.facts import PromotionFacts, check_facts

DOC = Document(
    title="滿$299送濕巾",
    category="檔期活動",
    published_date=date(2026, 10, 8),
    full_text="活動期間:2026/10/08-11/04\n55大品牌商品,贈送濕巾乙包",
)
GOOD = PromotionFacts(
    promo_type="gift",
    period_text="活動期間:2026/10/08-11/04",
    start_date=date(2026, 10, 8),
    end_date=date(2026, 11, 4),
    min_spend=299,
    discount_text="滿$299送濕巾",
)
WRONG_YEAR = GOOD.model_copy(update={"end_date": date(2028, 11, 4)})


class FakeInvoke:
    """Stands in for invoke_agent: returns the given answers (or raises them) in order."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, name, messages, *, variables, response_schema, trace):
        self.calls.append({"name": name, "messages": list(messages), "variables": variables})
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        reply = AIMessage(answer.model_dump_json())
        return AgentResult(
            output=answer,
            text=reply.text,
            messages=[*messages, reply],
            agent=name,
            model="fake-model",
            prompt_version="abc123",
            tokens_in=100,
            tokens_out=20,
            latency_ms=5,
        )


def _run(*answers):
    invoke = FakeInvoke(*answers)
    outcome = extract_with_agent(
        DOC, PromotionFacts, TraceInfo(workflow="test"), Pacer(rpm=10**6), invoke=invoke
    )
    return outcome, invoke


def test_good_answer_passes_first_time():
    outcome, invoke = _run(GOOD)
    assert (outcome.status, outcome.attempts, outcome.model) == ("ok", 1, "fake-model")
    assert outcome.facts.min_spend == 299
    assert invoke.calls[0]["name"] == "fact_extractor"
    assert invoke.calls[0]["variables"] == {"published_date": "2026-10-08"}


def test_failed_check_is_sent_back_and_the_retry_can_pass():
    outcome, invoke = _run(WRONG_YEAR, GOOD)
    assert (outcome.status, outcome.attempts) == ("ok", 2)
    assert "more than a year" in outcome.errors[0][0]
    second = invoke.calls[1]["messages"]
    # The retry continues the conversation: page, first answer, then the check errors.
    assert [type(m) for m in second] == [HumanMessage, AIMessage, HumanMessage]
    assert "failed these checks" in second[-1].text and "more than a year" in second[-1].text


def test_two_failures_mean_needs_review_and_no_facts():
    outcome, _ = _run(WRONG_YEAR, WRONG_YEAR)
    assert (outcome.status, outcome.attempts, outcome.facts) == ("needs_review", 2, None)


def test_agent_error_is_recorded_as_failed():
    outcome, _ = _run(AgentError("fact_extractor", RuntimeError("503 high demand")))
    assert outcome.status == "failed"
    assert "503" in outcome.errors[0][0]


def test_checks_search_the_title_too():
    """events-48: the numbers were only in the title, and a body-only check rejected them."""
    assert check_facts(GOOD, DOC.full_text, DOC.published_date)  # body only: false alarm
    assert check_facts(GOOD, _shown_text(DOC), DOC.published_date) == []
