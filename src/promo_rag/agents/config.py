"""Agent configs and the registry that finds them.

Each workflow defines its agents in `workflows/<name>/agents.py`: the model role, the prompt
and the output schema. Tools and middleware are shared and referenced by name.
"""

import importlib
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

ModelRole = Literal["chat", "helper", "judge"]


class AgentConfigError(LookupError):
    pass


@dataclass(frozen=True)
class AgentConfig:
    name: str
    prompt: Path  # a Markdown file in the workflow's prompts/ folder
    model: ModelRole  # resolved from settings (CHAT_MODEL, HELPER_MODEL, JUDGE_MODEL)
    # Default output schema. invoke_agent(..., response_schema=...) can pass one per call.
    response_schema: type[BaseModel] | None = None
    tools: tuple[str, ...] = ()  # names in promo_rag.agents.tools.TOOLS
    middleware: tuple[str, ...] = ()  # names in promo_rag.agents.middleware.MIDDLEWARE, in order
    temperature: float = 0.0
    timeout_s: float = 30.0
    description: str = ""


_REGISTRY: dict[str, AgentConfig] = {}
_discovered = False


def register_agent(config: AgentConfig) -> AgentConfig:
    if config.name in _REGISTRY and _REGISTRY[config.name] != config:
        raise AgentConfigError(f"two agents are named {config.name!r}")
    _REGISTRY[config.name] = config
    return config


def get_agent_config(name: str) -> AgentConfig:
    _discover()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise AgentConfigError(f"no agent {name!r}; known: {sorted(_REGISTRY)}") from None


def all_agents() -> list[AgentConfig]:
    _discover()
    return sorted(_REGISTRY.values(), key=lambda c: c.name)


def _discover() -> None:
    """Import every workflows/<name>/agents.py once, so each registers its agents."""
    global _discovered
    if _discovered:
        return
    from promo_rag import workflows

    for module in pkgutil.iter_modules(workflows.__path__):
        if module.ispkg:
            try:
                importlib.import_module(f"promo_rag.workflows.{module.name}.agents")
            except ModuleNotFoundError as e:
                if not e.name.endswith(".agents"):  # a real import error inside agents.py
                    raise
    _discovered = True
