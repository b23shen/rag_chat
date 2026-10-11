"""Shared middleware, by name. An AgentConfig lists names; invoke_agent builds them in order.

M3 adds the input guardrail, summarization and call limits here.
"""

from collections.abc import Callable

from langchain.agents.middleware import AgentMiddleware, ModelFallbackMiddleware

from promo_rag.agents.config import AgentConfig, AgentConfigError
from promo_rag.agents.models import chat_model
from promo_rag.config import get_settings


def model_fallback(config: AgentConfig) -> AgentMiddleware:
    """If the model still fails after its retries (e.g. 503 "high demand"), ask the fallback."""
    name = get_settings().chat_fallback_model
    if not name:
        raise AgentConfigError("CHAT_FALLBACK_MODEL is not set in .env")
    return ModelFallbackMiddleware(chat_model(name, config.temperature, config.timeout_s))


MIDDLEWARE: dict[str, Callable[[AgentConfig], AgentMiddleware]] = {
    "model_fallback": model_fallback,
}


def build_middleware(config: AgentConfig) -> list[AgentMiddleware]:
    unknown = [n for n in config.middleware if n not in MIDDLEWARE]
    if unknown:
        raise AgentConfigError(f"{config.name}: unknown middleware {unknown}")
    return [MIDDLEWARE[n](config) for n in config.middleware]
