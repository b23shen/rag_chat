"""Shared tools, by name. An AgentConfig lists the names it may use.

Each tool is a plain function, wrapped once here, so the chat agent and the MCP server (M5)
share the same code. M3 adds count_promotions, list_promotions, get_promotion,
search_promotion_text and resolve_dates.
"""

from langchain_core.tools import BaseTool

from promo_rag.agents.config import AgentConfig, AgentConfigError

TOOLS: dict[str, BaseTool] = {}


def build_tools(config: AgentConfig) -> list[BaseTool]:
    unknown = [n for n in config.tools if n not in TOOLS]
    if unknown:
        raise AgentConfigError(f"{config.name}: unknown tools {unknown}")
    return [TOOLS[n] for n in config.tools]
