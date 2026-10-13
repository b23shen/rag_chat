"""Shared middleware, by name. An AgentConfig lists names; invoke_agent builds them in order.

The call limits bound one run (one chat turn), so a model stuck in a loop can't run up costs.
"""

from collections.abc import Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ToolCallLimitMiddleware,
)

from promo_rag.agents.config import AgentConfig, AgentConfigError
from promo_rag.agents.models import chat_model
from promo_rag.config import get_settings


def model_fallback(config: AgentConfig) -> AgentMiddleware:
    """If the model still fails after its retries (e.g. 503 "high demand"), ask the fallback."""
    name = get_settings().chat_fallback_model
    if not name:
        raise AgentConfigError("CHAT_FALLBACK_MODEL is not set in .env")
    return ModelFallbackMiddleware(chat_model(name, config.temperature, config.timeout_s))


def tool_call_limit(config: AgentConfig) -> AgentMiddleware:
    """After 8 tool calls in one run, further calls are refused with a message, and the model
    has to answer with what it has."""
    return ToolCallLimitMiddleware(run_limit=8, exit_behavior="continue")


def model_call_limit(config: AgentConfig) -> AgentMiddleware:
    """A hard stop: at most 10 model calls in one run."""
    return ModelCallLimitMiddleware(run_limit=10, exit_behavior="end")


MIDDLEWARE: dict[str, Callable[[AgentConfig], AgentMiddleware]] = {
    "model_fallback": model_fallback,
    "tool_call_limit": tool_call_limit,
    "model_call_limit": model_call_limit,
}


def build_middleware(config: AgentConfig) -> list[AgentMiddleware]:
    unknown = [n for n in config.middleware if n not in MIDDLEWARE]
    if unknown:
        raise AgentConfigError(f"{config.name}: unknown middleware {unknown}")
    return [MIDDLEWARE[n](config) for n in config.middleware]
