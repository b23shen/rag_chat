"""Shared tools, by name. An AgentConfig lists the names it may use.

Each tool is a plain function, wrapped once here, so the chat agent and the MCP server (M5)
share the same code. Each wrapper returns JSON; the database tools open their own session.
"""

import json
from collections.abc import Callable

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from promo_rag.agents.config import AgentConfig, AgentConfigError
from promo_rag.agents.tools.dates import DateQuery, resolve_dates, today
from promo_rag.agents.tools.promotions import (
    GetPromotionArgs,
    ListPromotionsArgs,
    PromotionFilters,
    count_promotions,
    get_promotion,
    list_promotions,
)
from promo_rag.agents.tools.search_text import SearchArgs, search_promotion_text
from promo_rag.db.session import new_session


def _db_tool(
    fn: Callable, args_schema: type[BaseModel], name: str, description: str
) -> StructuredTool:
    def run(**kwargs) -> str:
        with new_session() as session:
            result = fn(session, args_schema(**kwargs))
            session.commit()  # search adds new question embeddings to the cache
        return json.dumps(result, ensure_ascii=False)

    return StructuredTool.from_function(
        func=run, name=name, description=description, args_schema=args_schema
    )


def _resolve_dates(**kwargs) -> str:
    return json.dumps(resolve_dates(DateQuery(**kwargs), today()), ensure_ascii=False)


TOOLS: dict[str, BaseTool] = {
    t.name: t
    for t in (
        _db_tool(
            count_promotions,
            PromotionFilters,
            "count_promotions",
            "Count promotions and store events (openings, reopenings, clearance sales) that "
            "match filters on kind, dates, store, city, payment method or channel. Use it for "
            "'how many' questions. Returns the count and the count per kind.",
        ),
        _db_tool(
            list_promotions,
            ListPromotionsArgs,
            "list_promotions",
            "List promotions and store events that match filters on kind, dates, store, city, "
            "payment method or channel, earliest first, with each one's dates, summary and "
            "page URLs. Use it for 'which' questions and to back up a count. Returns the "
            "total, so you can tell when the list was cut at the limit. For rules and fine "
            "print, search the page text instead.",
        ),
        _db_tool(
            get_promotion,
            GetPromotionArgs,
            "get_promotion",
            "Get one promotion by its id (from list_promotions): every fact, and the full "
            "text of each page it comes from, with the page URL to cite. Use it for the "
            "details of one promotion: its conditions, dates, exclusions or address.",
        ),
        _db_tool(
            search_promotion_text,
            SearchArgs,
            "search_promotion_text",
            "Search the text of every page (campaigns, store notices, FAQ and policy pages) "
            "for passages that answer a question, best first, each with its page URL. Use it "
            "for rules, conditions and fine print, and for anything the promotion tools "
            "don't cover. It returns only the top few passages, so don't count or list "
            "with it. A 'note' says when the passages may not answer the question.",
        ),
        StructuredTool.from_function(
            func=_resolve_dates,
            name="resolve_dates",
            description="Turn a period in words (today, this week, next month, October, this "
            "year, Fridays this month) into dates: give the parts, and it returns from/to "
            "(and each date, if weekdays are given) counted from today in Taiwan. Use it "
            "before filtering promotions by a period instead of working dates out yourself.",
            args_schema=DateQuery,
        ),
    )
}


def build_tools(config: AgentConfig) -> list[BaseTool]:
    unknown = [n for n in config.tools if n not in TOOLS]
    if unknown:
        raise AgentConfigError(f"{config.name}: unknown tools {unknown}")
    return [TOOLS[n] for n in config.tools]
