"""The chat assistant's wiring (no model calls): config, prompt, tools, history."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from promo_rag.agents.config import get_agent_config
from promo_rag.agents.middleware import MIDDLEWARE
from promo_rag.agents.prompts import load_prompt
from promo_rag.agents.tools import build_tools
from promo_rag.workflows.chat.assistant import transcript


def test_the_assistant_has_every_tool_and_its_limits():
    config = get_agent_config("promo_assistant")
    assert {t.name for t in build_tools(config)} == {
        "count_promotions",
        "list_promotions",
        "get_promotion",
        "resolve_dates",
        "search_promotion_text",
    }
    assert set(config.middleware) <= set(MIDDLEWARE)
    assert {"tool_call_limit", "model_call_limit"} <= set(config.middleware)


def test_the_prompt_needs_today_and_its_weekday():
    prompt = load_prompt(get_agent_config("promo_assistant").prompt)
    assert prompt.variables == {"today", "weekday"}
    assert "Today is 2026-10-12 (Monday)" in prompt.render(
        {"today": "2026-10-12", "weekday": "Monday"}
    )


def test_the_history_keeps_questions_and_answers_only():
    call = {"name": "search_promotion_text", "args": {"query": "退貨"}, "id": "c1"}
    messages = [
        HumanMessage("可以退貨嗎?"),
        AIMessage("", tool_calls=[call]),
        ToolMessage('{"results": []}', tool_call_id="c1"),
        AIMessage("7 天內可以退 [1]。"),
    ]
    assert [m.text for m in transcript(messages)] == ["可以退貨嗎?", "7 天內可以退 [1]。"]


def test_the_cli_answers_one_question_and_shows_tool_calls(monkeypatch, capsys):
    from promo_rag.agents import AgentResult
    from promo_rag.workflows.chat import cli

    def fake_ask(question, history, session_id):
        call = {"name": "list_promotions", "args": {"store": "新店中正店"}, "id": "c1"}
        turn = [AIMessage("", tool_calls=[call]), ToolMessage("{}", tool_call_id="c1")]
        answer = AIMessage("10 月 17 日開幕 [1]。")
        return AgentResult(
            output=answer.text,
            text=answer.text,
            messages=[*history, HumanMessage(question), *turn, answer],
            agent="promo_assistant",
            model="m",
            prompt_version="p",
            tokens_in=1,
            tokens_out=1,
            latency_ms=1,
        )

    monkeypatch.setattr(cli, "ask", fake_ask)
    monkeypatch.setattr("sys.argv", ["chat", "--tools", "新店中正店哪天開幕?"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "list_promotions(store='新店中正店')" in out and "10 月 17 日開幕" in out
