"""The shared agent machinery: prompts, the registry, and invoke_agent's wrapper (no network)."""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from promo_rag.agents import invoke as invoke_module
from promo_rag.agents.config import all_agents, get_agent_config
from promo_rag.agents.invoke import AgentError, TraceInfo, invoke_agent
from promo_rag.agents.middleware import MIDDLEWARE
from promo_rag.agents.models import ROLE_SETTINGS
from promo_rag.agents.prompts import Prompt, PromptError, load_prompt
from promo_rag.agents.tools import TOOLS

# ---------------------------------------------------------------- prompts


def test_prompt_fills_slots_and_rejects_missing_or_extra_variables():
    prompt = Prompt(path=Path("p.md"), text='Published {{ published_date }}. JSON: {"a": 1}')
    assert prompt.variables == {"published_date"}
    assert prompt.render({"published_date": "2026-10-07"}) == 'Published 2026-10-07. JSON: {"a": 1}'
    with pytest.raises(PromptError, match="missing"):
        prompt.render({})
    with pytest.raises(PromptError, match="unexpected"):
        prompt.render({"published_date": "x", "store": "y"})


def test_prompt_version_changes_with_the_text():
    a = Prompt(path=Path("p.md"), text="v1 {{x}}")
    b = Prompt(path=Path("p.md"), text="v2 {{x}}")
    assert a.version != b.version and len(a.version) == 12


# ---------------------------------------------------------------- the registry


def test_registry_finds_the_workflow_agents():
    assert "fact_extractor" in {c.name for c in all_agents()}


@pytest.mark.parametrize("config", all_agents(), ids=lambda c: c.name)
def test_every_agent_is_fully_wired(config):
    """Catches a typo'd prompt path, model role, tool or middleware name before any run."""
    assert load_prompt(config.prompt).text
    assert config.model in ROLE_SETTINGS
    assert set(config.tools) <= set(TOOLS)
    assert set(config.middleware) <= set(MIDDLEWARE)


# ---------------------------------------------------------------- invoke_agent


class Dates(BaseModel):
    start: str


class FakeAgent:
    """What create_agent returns: .invoke(state, config) → state."""

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.seen = reply, error, None

    def invoke(self, state, config):
        self.seen = (state, config)
        if self.error:
            raise self.error
        return {
            "messages": [*state["messages"], self.reply],
            "structured_response": Dates(start="10/01"),
        }


def _reply():
    return AIMessage(
        '{"start": "10/01"}',
        usage_metadata={"input_tokens": 40, "output_tokens": 9, "total_tokens": 49},
        response_metadata={"model_name": "gemini-3.5-flash-lite"},  # e.g. the fallback answered
    )


def test_invoke_agent_renders_the_prompt_and_returns_one_result(monkeypatch):
    fake = FakeAgent(reply=_reply())
    monkeypatch.setattr(invoke_module, "_build_agent", lambda config, schema: fake)
    result = invoke_agent(
        "fact_extractor",
        "<page>…</page>",
        variables={"published_date": "2026-10-07"},
        response_schema=Dates,
        trace=TraceInfo(workflow="ingestion", run_id="run-1", metadata={"document_id": "d1"}),
    )
    state, config = fake.seen
    system, user = state["messages"]
    assert isinstance(system, SystemMessage) and "2026-10-07" in system.text
    assert isinstance(user, HumanMessage)
    assert config["tags"] == ["ingestion", "fact_extractor"]
    assert config["metadata"]["document_id"] == "d1"
    assert config["metadata"]["langfuse_session_id"] == "run-1"

    assert result.output == Dates(start="10/01")
    assert result.model == "gemini-3.5-flash-lite"
    assert (result.tokens_in, result.tokens_out) == (40, 9)
    assert [type(m) for m in result.messages] == [HumanMessage, AIMessage]  # no system prompt
    assert result.prompt_version == invoke_module.prompt_version(
        get_agent_config("fact_extractor"), Dates
    )


def test_provider_errors_become_agent_errors(monkeypatch):
    fake = FakeAgent(error=invoke_module.genai_errors.APIError(503, {"error": {"message": "busy"}}))
    monkeypatch.setattr(invoke_module, "_build_agent", lambda config, schema: fake)
    with pytest.raises(AgentError, match="fact_extractor"):
        invoke_agent(
            "fact_extractor", "hi", variables={"published_date": "x"}, response_schema=Dates
        )
