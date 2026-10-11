"""Shared agent machinery: configs, prompts, models, tools, middleware, tracing, invoke_agent."""

from promo_rag.agents.config import AgentConfig, register_agent
from promo_rag.agents.invoke import (
    AgentError,
    AgentOutputError,
    AgentResult,
    TraceInfo,
    invoke_agent,
)

__all__ = [
    "AgentConfig",
    "AgentError",
    "AgentOutputError",
    "AgentResult",
    "TraceInfo",
    "invoke_agent",
    "register_agent",
]
