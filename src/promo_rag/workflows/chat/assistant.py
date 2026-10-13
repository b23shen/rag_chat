"""One chat turn: the conversation so far + a new question → an answer with sources.

    result = ask("週五用街口支付有什麼優惠?")
    result = ask("那要先領券嗎?", history=transcript(result.messages))

The history keeps the questions and answers, not the tool calls and results: the answers
already carry the facts and their sources, a turn fetches again what it needs, and the
prompt stays small as the conversation grows.
"""

import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from promo_rag.agents import AgentResult, TraceInfo, invoke_agent
from promo_rag.agents.tools.dates import today
from promo_rag.workflows.chat.agents import ASSISTANT


def ask(
    question: str,
    history: list[BaseMessage] | None = None,
    *,
    session_id: str | None = None,
) -> AgentResult:
    now = today()
    return invoke_agent(
        ASSISTANT.name,
        [*(history or []), HumanMessage(question)],
        variables={"today": now.isoformat(), "weekday": now.strftime("%A")},
        trace=TraceInfo(
            workflow="chat", run_id=f"turn-{uuid.uuid4().hex[:8]}", session_id=session_id
        ),
    )


def transcript(messages: list[BaseMessage]) -> list[BaseMessage]:
    """The questions and the final answers, without tool calls and tool results."""
    return [
        m
        for m in messages
        if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls and m.text)
    ]
