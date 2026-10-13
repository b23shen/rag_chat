"""Agents in the chat workflow: the assistant that answers customers from POYA's pages, and
the judge that grades its answers in the answer eval."""

from pathlib import Path

from promo_rag.agents import AgentConfig, register_agent

PROMPTS = Path(__file__).parent / "prompts"

ASSISTANT = register_agent(
    AgentConfig(
        name="promo_assistant",
        description="Answers questions about POYA's promotions and stores, with sources.",
        prompt=PROMPTS / "assistant.md",
        model="chat",
        tools=(
            "count_promotions",
            "list_promotions",
            "get_promotion",
            "resolve_dates",
            "search_promotion_text",
        ),
        middleware=("model_fallback", "tool_call_limit", "model_call_limit"),
        timeout_s=60.0,
    )
)

ANSWER_JUDGE = register_agent(
    AgentConfig(
        name="answer_judge",
        description="Grades an assistant answer against the golden reference answer.",
        prompt=PROMPTS / "judge_answer.md",
        model="judge",
        # The eval passes the schema (AnswerGrade) with each call.
        response_schema=None,
        middleware=("model_fallback",),
    )
)
