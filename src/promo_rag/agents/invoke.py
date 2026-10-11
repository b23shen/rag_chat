"""invoke_agent: the one way any workflow calls an LLM agent.

    result = invoke_agent(
        "fact_extractor",
        messages,                         # str, or the conversation so far
        variables={"published_date": "2026-10-07"},   # fills the prompt's {{slots}}
        response_schema=PromotionFacts,   # optional; overrides the config's default
        trace=TraceInfo(workflow="ingestion", run_id=..., metadata={"document_id": ...}),
    )                                     # trace: labels for Langfuse, never sent to the model
    result.output        # a PromotionFacts instance (or the reply text if no schema)

It looks up the agent's config, renders its prompt, builds the agent (model + shared tools +
shared middleware, through LangChain's create_agent), runs it with tracing, and returns one
AgentResult with the output, the conversation, token usage, latency and prompt version.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import httpx
from google.genai import errors as genai_errors
from langchain.agents import create_agent
from langchain.agents.structured_output import StructuredOutputError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from pydantic import BaseModel

from promo_rag.agents.config import AgentConfig, get_agent_config
from promo_rag.agents.middleware import build_middleware
from promo_rag.agents.models import chat_model, model_for_role
from promo_rag.agents.prompts import load_prompt
from promo_rag.agents.tools import build_tools
from promo_rag.agents.tracing import langfuse_handler

PROVIDER_ERRORS = (ChatGoogleGenerativeAIError, genai_errors.APIError, httpx.HTTPError)


@dataclass
class TraceInfo:
    """Labels for tracing and logs. Never sent to the model.

    They let Langfuse group and search calls: every call of one ingestion run, every turn
    of one conversation, the document a call was about. Anything the model or its tools
    need (e.g. today's date) goes in `variables` or the agent's runtime context instead.
    """

    workflow: str | None = None
    run_id: str | None = None  # one batch run, or one chat turn
    session_id: str | None = None  # a conversation; groups its turns in Langfuse
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    output: Any  # an instance of the response schema, or the reply text if there is none
    text: str  # the model's last reply as text
    messages: list[BaseMessage]  # the conversation without the system prompt; append to continue
    agent: str
    model: str  # the model that actually answered (the fallback, if it was used)
    prompt_version: str
    tokens_in: int
    tokens_out: int
    latency_ms: int


class AgentError(RuntimeError):
    """The model call failed after its retries and fallback (rate limit, server error, ...)."""

    def __init__(self, agent: str, cause: Exception):
        super().__init__(f"{agent}: {type(cause).__name__}: {cause}")
        self.agent, self.cause = agent, cause


class AgentOutputError(AgentError):
    """The reply didn't match the response schema."""


def invoke_agent(
    name: str,
    messages: str | list[BaseMessage],
    *,
    variables: dict[str, object] | None = None,
    response_schema: type[BaseModel] | None = None,
    trace: TraceInfo | None = None,
) -> AgentResult:
    config = get_agent_config(name)
    schema = response_schema or config.response_schema
    prompt = load_prompt(config.prompt)
    history = [HumanMessage(messages)] if isinstance(messages, str) else list(messages)
    trace = trace or TraceInfo()
    version = prompt_version(config, schema)
    handler = langfuse_handler()
    run_config = {
        "run_name": name,
        "tags": [t for t in (trace.workflow, name, *trace.tags) if t],
        "metadata": {
            **trace.metadata,
            "agent": name,
            "prompt_version": version,
            "run_id": trace.run_id,
            "langfuse_session_id": trace.session_id or trace.run_id,
            "langfuse_tags": [t for t in (trace.workflow, name, *trace.tags) if t],
        },
        "callbacks": [handler] if handler else [],
    }

    agent = _build_agent(config, schema)
    started = time.monotonic()
    try:
        state = agent.invoke(
            {"messages": [SystemMessage(prompt.render(variables)), *history]}, config=run_config
        )
    except StructuredOutputError as e:
        raise AgentOutputError(name, e) from e
    except PROVIDER_ERRORS as e:
        raise AgentError(name, e) from e
    latency_ms = int((time.monotonic() - started) * 1000)

    conversation = state["messages"][1:]  # without the system prompt
    replies = [m for m in conversation[len(history) :] if isinstance(m, AIMessage)]
    last = replies[-1]
    usage = [m.usage_metadata or {} for m in replies]
    return AgentResult(
        output=state.get("structured_response") if schema else last.text,
        text=last.text,
        messages=conversation,
        agent=name,
        model=last.response_metadata.get("model_name") or model_for_role(config.model),
        prompt_version=version,
        tokens_in=sum(u.get("input_tokens", 0) for u in usage),
        tokens_out=sum(u.get("output_tokens", 0) for u in usage),
        latency_ms=latency_ms,
    )


def prompt_version(config: AgentConfig, schema: type[BaseModel] | None) -> str:
    """Changes when the prompt text or the output schema changes."""
    prompt = load_prompt(config.prompt)
    blob = prompt.text + (json.dumps(schema.model_json_schema(), sort_keys=True) if schema else "")
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


@lru_cache
def _build_agent(config: AgentConfig, schema: type[BaseModel] | None):
    model = chat_model(model_for_role(config.model), config.temperature, config.timeout_s)
    return create_agent(
        model,
        tools=build_tools(config),
        middleware=build_middleware(config),
        response_format=schema,
        name=config.name,
    )
